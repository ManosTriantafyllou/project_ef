"""
PPO fine-tuning via Stable-Baselines3, training a residual correction
on top of the frozen BC policy (see residual_bc_env.py for the
architecture).

Usage
-----
    python train_ppo.py --bc-checkpoint checkpoints/bc_policy.pt \
                         --total-timesteps 5_000_000 \
                         --out checkpoints/ppo_residual

Monitor training with TensorBoard:
    tensorboard --logdir checkpoints/ppo_residual/tb
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from residual_bc_env import ResidualBCQuadrupedEnv


def warm_up_quadruped_assets():
    """
    Create and immediately discard one QuadrupedEnv instance in the
    main process before spawning SubprocVecEnv workers.

    Root cause this works around: on Windows, SubprocVecEnv uses
    'spawn' (not 'fork'), so each of the n_envs worker processes
    independently triggers gym_quadruped's first-time asset loading
    for the go2-flat.xml scene file. If gym_quadruped does any
    lazy write/cache/copy of that asset on first load (rather than
    just reading a file that already fully exists on disk), multiple
    processes hitting this simultaneously can race -- one process
    reads the file mid-write and sees it as empty, causing
    'ParseXML: empty file' errors. Loading it once here, sequentially,
    before any subprocess starts, ensures the asset is fully
    materialized on disk first.
    """
    print("Warming up gym_quadruped assets in main process before spawning workers...")
    # We instantiate the underlying QuadrupedEnv directly (not the full
    # ResidualBCQuadrupedEnv wrapper) since that's the object actually
    # doing the asset loading, and a BC checkpoint isn't needed just to
    # warm up the MuJoCo scene file.
    from gym_quadruped.quadruped_env import QuadrupedEnv
    from gym_quadruped.sensors.imu import IMU
    import mujoco as _mujoco

    _original_imu_init = IMU.__init__

    def _patched_imu_init(self, mj_model, mj_data, *args, **kwargs):
        _mujoco.mj_forward(mj_model, mj_data)
        _original_imu_init(self, mj_model, mj_data, *args, **kwargs)
        self.step()

    IMU.__init__ = _patched_imu_init

    tmp_env = QuadrupedEnv(
        robot="go2",
        scene="flat",
        base_vel_command_type="random",
        state_obs_names=("qpos_js",),
        sensors=(IMU,),
        sensors_kwargs=({"accel_name": "imu_acc", "gyro_name": "imu_gyro", "imu_site_name": "imu"},),
    )
    tmp_env.close()
    print("Warm-up complete.")


def make_env_fn(bc_checkpoint, residual_scale, seed, log_dir):
    def _init():
        env = ResidualBCQuadrupedEnv(
            bc_checkpoint_path=bc_checkpoint,
            residual_scale=residual_scale,
        )
        env = Monitor(env, filename=str(Path(log_dir) / f"monitor_{seed}"))
        return env
    return _init


def train(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tb").mkdir(exist_ok=True)
    (out_dir / "monitor_logs").mkdir(exist_ok=True)

    # NOTE: SubprocVecEnv triggers a Windows-specific race condition in
    # gym_quadruped's MuJoCo asset loading (multiple spawned processes
    # opening go2-flat.xml simultaneously intermittently produces
    # 'ParseXML: empty file' errors, even after warming up the asset in
    # the main process first). DummyVecEnv runs all envs sequentially
    # in a single process, which is slower per wall-clock step but
    # completely avoids the multiprocessing-related race. If this is
    # ever run on Linux/Mac where the issue may not reproduce (fork()
    # semantics differ from Windows' spawn()), SubprocVecEnv can be
    # re-enabled by setting force_dummy_vec_env=False below.
    force_dummy_vec_env = True

    if args.n_envs > 1 and not force_dummy_vec_env:
        warm_up_quadruped_assets()

    env_fns = [
        make_env_fn(args.bc_checkpoint, args.residual_scale, seed=i,
                    log_dir=out_dir / "monitor_logs")
        for i in range(args.n_envs)
    ]
    # SubprocVecEnv parallelizes across processes -- much faster wall-
    # clock time for MuJoCo sims, which are CPU-bound. Falls back to
    # DummyVecEnv (single process) if n_envs == 1, useful for debugging
    # since stack traces are much clearer without subprocess boundaries.
    if args.n_envs > 1 and not force_dummy_vec_env:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    eval_env = DummyVecEnv([make_env_fn(args.bc_checkpoint, args.residual_scale,
                                          seed=999, log_dir=out_dir / "monitor_logs")])

    # Policy network sizing: kept modest since this only has to learn a
    # SMALL residual correction, not a full gait from scratch -- a
    # large network here would be solving an easier problem with more
    # capacity than it needs, which mainly costs sample efficiency.
    #
    # log_std_init: SB3's default is 0.0 (std=1.0), which combined with
    # the bounded [-residual_scale, residual_scale] action space means
    # early rollouts are essentially uniform random noise at the full
    # residual_scale magnitude every step -- confirmed via
    # diagnose_action_rate.py to keep torque jumping step-to-step
    # indefinitely (vs. pure BC, which settles within ~10 steps).
    # Starting log_std_init very negative makes the initial policy
    # output close to its mean (near-zero residual, since the policy
    # head is randomly initialized near zero) with very low variance,
    # so early PPO rollouts closely resemble the pure BC policy. PPO is
    # free to increase std again later via the learned log_std
    # parameter as it discovers useful corrections worth exploring.
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        log_std_init=args.log_std_init,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        target_kl=0.02,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(out_dir / "tb"),
        verbose=1,
        seed=args.seed,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path=str(out_dir / "checkpoints"),
        name_prefix="ppo_residual",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "best_model"),
        log_path=str(out_dir / "eval_logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    final_path = out_dir / "ppo_residual_final"
    model.save(str(final_path))
    print(f"\nTraining complete. Final model saved to {final_path}.zip")
    print(f"Best model (by eval reward) saved to {out_dir / 'best_model'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-checkpoint", type=str, default="checkpoints/bc_policy.pt")
    parser.add_argument("--out", type=str, default="checkpoints/ppo_residual")
    parser.add_argument("--residual-scale", type=float, default=0.05,
                         help="Max magnitude (rad) of the PPO residual "
                              "correction added to the BC joint target. "
                              "Start SMALL (e.g. 0.03-0.05) so early, "
                              "high-variance PPO exploration stays close "
                              "to the already-stable BC gait instead of "
                              "immediately producing large torque jumps "
                              "every step (confirmed via diagnose_action_rate.py: "
                              "pure BC settles to action_rate_penalty in the "
                              "~10-35 raw range within ~10 steps; residual_scale=0.08 "
                              "with log_std_init=-2.0 still produced "
                              "raw_action_rate_sq in the ~1000-2700 range "
                              "indefinitely -- too high. At residual_scale=0.05 "
                              "with log_std_init=-3.0, expected per-step torque "
                              "noise is roughly kp * residual_scale * std "
                              "~= 90 * 0.05 * 0.05 ~= 0.225 Nm per joint, giving "
                              "raw_action_rate_sq ~ 12*0.225^2 ~ 0.6 -- comparable "
                              "to or better than pure BC's steady state.). Can "
                              "be raised in a later run once the policy has "
                              "learned to use small corrections well.")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=2048,
                         help="Rollout length per env before each PPO update")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4,
                         help="PPO learning rate. Lowered from SB3's typical "
                              "3e-4 default because with a very small "
                              "log_std_init, the policy distribution is tight "
                              "around its mean -- the same gradient step size "
                              "that's fine for std~1.0 produces a much larger "
                              "change in the probability ratio (and thus "
                              "approx_kl/clip_fraction) when std is small. "
                              "Confirmed empirically: lr=3e-4 with "
                              "log_std_init=-3.0 produced approx_kl~0.097 and "
                              "clip_fraction~0.55 (both far above the usual "
                              "~0.01-0.02 / ~0.1-0.2 healthy range).")
    parser.add_argument("--save-freq", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--log-std-init", type=float, default=-3.0,
                         help="Initial log std for the PPO Gaussian policy "
                              "(SB3 default is 0.0, i.e. std=1.0). A very "
                              "negative value (e.g. -3.0, std~0.05) keeps "
                              "early rollouts close to the BC baseline's "
                              "deterministic behavior; PPO can still learn "
                              "to increase std later if useful.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()