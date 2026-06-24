"""
train_ppo_cpg.py
------------------
Εκπαιδεύει PPO με CPG (Central Pattern Generator) residual base — βλέπε
env_wrapper_cpg.py για πλήρη εξήγηση.

Χρήση:
    python train_ppo_cpg.py --timesteps 5000000 --n_envs 16

Στόχος: λύση στο πρόβλημα "static balance reward hacking" που είδαμε
στα προηγούμενα runs, δίνοντας στον PPO μια ΗΔΗ ΚΙΝΟΥΜΕΝΗ βάση αντί
για στατική standing pose.
"""

import os
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn

from env_wrapper_cpg import make_env_cpg
from pretrain_bc import load_bc_weights_into_ppo


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum Learning Callback
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """
    Implements a simple linear curriculum over two axes:

      1. residual_scale: starts small (agent mostly follows CPG) and
         grows to the configured final value (agent gains more freedom).
         Schedule: 0.3 -> final_residual_scale over the first 70% of training.

      2. ent_coef: starts high (lots of exploration) and decays to a
         small value (exploitation focus).
         Schedule: 0.05 -> 0.005 over the full training duration.

    Both are logged to TensorBoard under curriculum/.
    """

    def __init__(self, total_timesteps: int,
                 final_residual_scale: float = 1.5,
                 residual_scale_start: float = 0.3,
                 residual_ramp_fraction: float = 0.70,
                 ent_coef_start: float = 0.05,
                 ent_coef_end: float = 0.005,
                 verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps       = total_timesteps
        self.final_residual_scale  = final_residual_scale
        self.residual_scale_start  = residual_scale_start
        self.residual_ramp_fraction = residual_ramp_fraction
        self.ent_coef_start        = ent_coef_start
        self.ent_coef_end          = ent_coef_end

    def _get_wrapped_env(self, monitor_env):
        """Unpack Monitor -> Go2WrapperCPG."""
        env = monitor_env
        while hasattr(env, "env"):
            env = env.env
            if hasattr(env, "residual_scale"):
                return env
        return None

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)

        # 1. residual_scale: ramp from start to final over first ramp_fraction
        ramp = min(progress / self.residual_ramp_fraction, 1.0)
        current_scale = (self.residual_scale_start
                         + (self.final_residual_scale - self.residual_scale_start) * ramp)

        # Update residual_scale in every training env
        try:
            # self.training_env is VecNormalize -> .venv is DummyVecEnv
            for monitor_env in self.training_env.venv.envs:
                wrapped = self._get_wrapped_env(monitor_env)
                if wrapped is not None:
                    wrapped.residual_scale = current_scale
        except Exception:
            pass

        # 2. ent_coef: linear decay
        current_ent = self.ent_coef_start + (self.ent_coef_end - self.ent_coef_start) * progress
        self.model.ent_coef = float(current_ent)

        # Log to TensorBoard every 10K steps
        if self.num_timesteps % 10_000 < self.training_env.num_envs:
            self.logger.record("curriculum/residual_scale", current_scale)
            self.logger.record("curriculum/ent_coef",       current_ent)
            self.logger.record("curriculum/progress",       progress)

        return True


def train(total_timesteps=5_000_000, n_envs=16, seed=42,
          base_frequency=1.2, base_stride_amplitude=0.4, lift_amplitude=0.25,
          duty_cycle=0.65, velocity_gain=1.0, residual_scale=1.5,
          action_smoothing=0.3, bc_weights=None, use_curriculum=True):
    print("=" * 60)
    print("PPO Training — Go2 Velocity Tracking (Velocity-aware CPG + Residual)")
    print("=" * 60)
    print(f"  Timesteps             : {total_timesteps:,}")
    print(f"  Parallel envs         : {n_envs}")
    print(f"  Base frequency        : {base_frequency} Hz")
    print(f"  Base stride amplitude : {base_stride_amplitude} rad")
    print(f"  Lift amplitude        : {lift_amplitude} rad")
    print(f"  Duty cycle            : {duty_cycle}")
    print(f"  Velocity gain         : {velocity_gain}")
    print(f"  Residual scale        : {residual_scale}")
    print(f"  Action smoothing      : {action_smoothing}")
    print(f"  BC weights            : {bc_weights or 'None (training from scratch)'}")
    print()

    suffix = f"_as{action_smoothing}" if action_smoothing > 0 else ""
    models_dir = f"models_cpg{suffix}"
    logs_dir   = f"logs_cpg{suffix}"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir,   exist_ok=True)

    def monitored_env():
        def _init():
            env = make_env_cpg(base_frequency, base_stride_amplitude, lift_amplitude,
                                duty_cycle, velocity_gain, residual_scale,
                                action_smoothing)()
            return Monitor(env)
        return _init

    train_env = DummyVecEnv([monitored_env() for _ in range(n_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=10.0, clip_reward=10.0)

    eval_env = DummyVecEnv([monitored_env()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

    lr_schedule = get_linear_fn(start=3e-4, end=1e-5, end_fraction=0.8)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=lr_schedule,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=logs_dir + "/",
        verbose=1,
        seed=seed,
    )

    print(f"  Parameters     : {sum(p.numel() for p in model.policy.parameters()):,}\n")

    # --- BC Warm-Start ---------------------------------------------------
    # Αν υπάρχουν pretrained BC weights, φορτώνουμε τα πριν το PPO training.
    # Ο agent ξεκινά ΗΔΗ από μια policy που "ξέρει να ακολουθεί το CPG"
    # αντί από τυχαία βάρη.
    if bc_weights and os.path.exists(bc_weights):
        load_bc_weights_into_ppo(model, bc_weights, net_arch=[256, 256])
    elif bc_weights:
        print(f"[WARNING] BC weights path '{bc_weights}' δεν βρέθηκε — training from scratch.")
    # ---------------------------------------------------------------------

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"{models_dir}/best/",
        log_path=f"{logs_dir}/eval/",
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path=f"{models_dir}/checkpoints/",
        name_prefix="ppo_go2_cpg",
        save_vecnormalize=True,
    )

    callbacks = [eval_cb, ckpt_cb]

    if use_curriculum:
        curriculum_cb = CurriculumCallback(
            total_timesteps=total_timesteps,
            final_residual_scale=residual_scale,  # ramp UP to the configured value
            residual_scale_start=0.3,             # start very conservative
            residual_ramp_fraction=0.70,           # reach full scale at 70% of training
            ent_coef_start=0.05,                  # start exploring a lot
            ent_coef_end=0.005,                   # end exploiting
            verbose=1,
        )
        callbacks.append(curriculum_cb)
        print("[Curriculum] Enabled: residual_scale 0.3 -> "
              f"{residual_scale} over first 70% of training.")
        print("[Curriculum] ent_coef 0.05 -> 0.005 over full training.\n")
    else:
        print("[Curriculum] Disabled.\n")

    print("Ξεκινάει το training (CPG)...")
    print(f"Παρακολούθηση: tensorboard --logdir {logs_dir}/\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=True,
    )

    model.save(f"{models_dir}/ppo_go2_cpg_final")
    train_env.save(f"{models_dir}/vecnormalize.pkl")

    print("\nTraining (CPG) ολοκληρώθηκε!")
    print(f"  Final model : {models_dir}/ppo_go2_cpg_final.zip")
    print(f"  Best model  : {models_dir}/best/best_model.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps",             type=int,   default=5_000_000)
    parser.add_argument("--n_envs",                type=int,   default=16)
    parser.add_argument("--seed",                  type=int,   default=42)
    parser.add_argument("--base_frequency",        type=float, default=1.2,
                         help="Βασική συχνότητα βάδισης σε Hz (όταν target velocity≈0)")
    parser.add_argument("--base_stride_amplitude", type=float, default=0.4,
                         help="Βασικό πλάτος εμπρός-πίσω κίνησης thigh σε rad")
    parser.add_argument("--lift_amplitude",        type=float, default=0.25,
                         help="Πλάτος ανύψωσης ποδιού κατά το swing σε rad")
    parser.add_argument("--duty_cycle",             type=float, default=0.65,
                         help="Κλάσμα κύκλου σε stance phase (0.65 = 65%% στο έδαφος)")
    parser.add_argument("--velocity_gain",          type=float, default=1.0,
                         help="Πόσο επηρεάζει η ζητούμενη ταχύτητα το CPG amplitude/frequency")
    parser.add_argument("--residual_scale",         type=float, default=1.5,
                         help="Max torque correction from PPO (1.5 = balanced, 2.0=original)")
    parser.add_argument("--action_smoothing",       type=float, default=0.3,
                         help="EMA alpha for PPO residual smoothing (0=off, 0.3=recommended)")
    parser.add_argument("--bc_weights",             type=str,   default=None,
                         help="Path σε BC pretrained weights .pt (από pretrain_bc.py). Αν δοθεί, PPO ξεκινά από εκεί.")
    parser.add_argument("--no_curriculum",          action="store_true", default=False,
                         help="Disable curriculum learning (use fixed residual_scale from start).")
    args = parser.parse_args()

    train(args.timesteps, args.n_envs, args.seed,
          args.base_frequency, args.base_stride_amplitude, args.lift_amplitude,
          args.duty_cycle, args.velocity_gain, args.residual_scale,
          args.action_smoothing, args.bc_weights,
          use_curriculum=not args.no_curriculum)