"""
Interactive MuJoCo viewer for the Go2 velocity-tracking project.

Three modes:
    --mode cpg     : runs the CPG+PD baseline directly (no learned
                      policy) -- use this to visually sanity-check the
                      trot gait itself, independent of any BC/PPO model.
    --mode bc       : loads a trained BC policy checkpoint and runs it
                      in closed loop, predicting q_des directly from the
                      proprioceptive observation (the PD torque
                      conversion happens internally exactly as during
                      data collection, so the comparison with --mode cpg
                      is apples-to-apples).
    --mode bc_ppo   : loads BOTH a frozen BC checkpoint and a trained
                      SB3 PPO residual checkpoint, and runs them
                      together exactly as during PPO training/eval:
                      q_des = a_BC(obs) + a_PPO(obs) (deterministic PPO
                      action). Use this to visually check PPO training
                      progress -- re-run periodically with the latest
                      checkpoint while training continues in the
                      background (PPO checkpoints are written every
                      --save-freq steps by CheckpointCallback, plus a
                      running best_model/best_model.zip from EvalCallback).

In all modes, the velocity command is read from the environment
itself (base_vel_command_type="random", as required by the benchmark)
-- we never override it, consistent with collect_data.py.

Usage
-----
    # Visualize the CPG baseline directly (no model needed):
    python view_rollout.py --mode cpg --episode-len 8.0

    # Visualize a trained BC policy:
    python view_rollout.py --mode bc --checkpoint checkpoints/bc_policy.pt --episode-len 8.0

    # Visualize BC + PPO residual together (check training progress):
    python view_rollout.py --mode bc_ppo \
        --bc-checkpoint checkpoints/bc_policy.pt \
        --ppo-checkpoint checkpoints/ppo_residual/checkpoints/ppo_residual_500000_steps.zip \
        --episode-len 15.0

Controls in the MuJoCo viewer window: space = pause/resume, the usual
mouse drag/scroll for camera, Tab cycles through render modes.
"""

import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer

from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.sensors.imu import IMU

from trot_cpg import TrotCPG, TrotCPGParams, joint_angles_dict_to_vector

_original_imu_init = IMU.__init__


def _patched_imu_init(self, mj_model, mj_data, *args, **kwargs):
    mujoco.mj_forward(mj_model, mj_data)
    _original_imu_init(self, mj_model, mj_data, *args, **kwargs)
    self.step()


IMU.__init__ = _patched_imu_init


PROPRIOCEPTIVE_OBS = (
    "gravity_vector:base",
    "imu_acc",
    "imu_gyro",
    "qpos_js",
    "qvel_js",
    "base_lin_vel_err:base",
    "base_ang_vel_err:base",
)

IMU_KWARGS = {"accel_name": "imu_acc", "gyro_name": "imu_gyro", "imu_site_name": "imu"}


def make_env():
    return QuadrupedEnv(
        robot="go2",
        scene="flat",
        base_vel_command_type="random",
        state_obs_names=PROPRIOCEPTIVE_OBS,
        sensors=(IMU,),
        sensors_kwargs=(IMU_KWARGS,),
    )


def get_joint_state(obs_dict_or_array):
    if isinstance(obs_dict_or_array, dict):
        qpos = np.asarray(obs_dict_or_array["qpos_js"], dtype=np.float64).flatten()
        qvel = np.asarray(obs_dict_or_array["qvel_js"], dtype=np.float64).flatten()
    else:
        arr = np.asarray(obs_dict_or_array, dtype=np.float64).flatten()
        qpos = arr[9:21]
        qvel = arr[21:33]
    return qpos, qvel


def flatten_obs(obs_dict_or_array):
    if isinstance(obs_dict_or_array, dict):
        parts = [np.atleast_1d(np.asarray(obs_dict_or_array[k], dtype=np.float32))
                  for k in PROPRIOCEPTIVE_OBS]
        return np.concatenate(parts)
    return np.asarray(obs_dict_or_array, dtype=np.float32)


def compute_pd_torque(q_des, q, q_dot, kp=90.0, kd=5.0, torque_limit=50.0):
    tau = kp * (q_des - q) - kd * q_dot
    return np.clip(tau, -torque_limit, torque_limit).astype(np.float32)


def read_velocity_command(env):
    lin_vel, ang_vel = env.target_base_vel()
    lin_vel = np.asarray(lin_vel, dtype=np.float32).flatten()
    ang_vel = np.asarray(ang_vel, dtype=np.float32).flatten()
    return float(lin_vel[0]), float(lin_vel[1]), float(ang_vel[2])


def load_ppo_residual_policy(checkpoint_path):
    """
    Loads a trained SB3 PPO checkpoint (a .zip file produced by
    model.save() or CheckpointCallback) and returns a callable mapping
    the 39-dim proprioceptive observation -> 12-dim residual action,
    using deterministic (mean) actions rather than sampling, since
    this is for visualization of the policy's learned behavior, not
    for collecting exploration data.
    """
    from stable_baselines3 import PPO

    model = PPO.load(checkpoint_path, device="cpu")

    def policy_fn(obs_flat):
        action, _ = model.predict(obs_flat, deterministic=True)
        return action.astype(np.float64)

    return policy_fn


def load_bc_policy(checkpoint_path):
    """
    Loads a trained BC policy. Expects a PyTorch checkpoint containing
    at minimum a state_dict for an MLP mapping obs (39-dim
    proprioceptive) -> q_des (12-dim). Adjust the model class import
    below to match whatever architecture the BC training script
    actually defines -- this is a thin loader, not the model
    definition itself.
    """
    import torch
    from bc_train import BCPolicy  # expects bc_train.py defining BCPolicy

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    obs_mean = checkpoint.get("obs_mean", None)
    obs_std = checkpoint.get("obs_std", None)

    model = BCPolicy(obs_dim=39, act_dim=12)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def policy_fn(obs_flat):
        x = obs_flat.copy()
        if obs_mean is not None and obs_std is not None:
            x = (x - obs_mean) / (obs_std + 1e-8)
        with torch.no_grad():
            q_des = model(torch.from_numpy(x.astype(np.float32)).unsqueeze(0))
        return q_des.squeeze(0).numpy().astype(np.float64)

    return policy_fn


def run(args):
    env = make_env()
    control_dt = getattr(env, "dt", None) or (1.0 / 50.0)
    steps = int(round(args.episode_len / control_dt))

    bc_policy_fn = None
    ppo_policy_fn = None
    cpg = None

    if args.mode == "cpg":
        cpg = TrotCPG(TrotCPGParams())
    elif args.mode == "bc":
        bc_policy_fn = load_bc_policy(args.checkpoint)
    elif args.mode == "bc_ppo":
        bc_policy_fn = load_bc_policy(args.bc_checkpoint)
        ppo_policy_fn = load_ppo_residual_policy(args.ppo_checkpoint)
        print(f"[diag] loaded BC checkpoint: {args.bc_checkpoint}")
        print(f"[diag] loaded PPO residual checkpoint: {args.ppo_checkpoint}")

    obs = env.reset(seed=args.seed)
    if cpg is not None:
        cpg.reset(seed=args.seed)

    mj_model = env.mjModel
    mj_data = env.mjData
    mujoco.mj_forward(mj_model, mj_data)

    print(f"[diag] mj_model.nq={mj_model.nq} nv={mj_model.nv} nbody={mj_model.nbody}")
    print(f"[diag] base qpos[:7] (pos+quat)={np.round(mj_data.qpos[:7], 3)}")
    print(f"[diag] mj_data.qpos full size={mj_data.qpos.shape}")

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Point the camera at the robot's base explicitly -- default
        # camera placement can otherwise be far away, zoomed out, or
        # centered on the world origin instead of the robot.
        viewer.cam.lookat[:] = mj_data.qpos[:3]
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.sync()

        t = 0
        ep_start = time.time()
        while viewer.is_running():
            step_start = time.time()

            vx, vy, wz = read_velocity_command(env)
            q, q_dot = get_joint_state(obs)

            if cpg is not None:
                out = cpg.step(control_dt, vx, vy, wz)
                q_des = joint_angles_dict_to_vector(out["joint_angles"]).astype(np.float64)
            elif ppo_policy_fn is not None:
                obs_flat = flatten_obs(obs)
                a_bc = bc_policy_fn(obs_flat)
                a_residual = ppo_policy_fn(obs_flat)
                q_des = a_bc + a_residual
            else:
                obs_flat = flatten_obs(obs)
                q_des = bc_policy_fn(obs_flat)

            torque = compute_pd_torque(q_des, q, q_dot)
            step_result = env.step(torque)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result

            # Keep the camera following the robot's base each frame so
            # it never drifts out of view as it walks.
            viewer.cam.lookat[:] = mj_data.qpos[:3]
            viewer.sync()

            # Real-time pacing: sleep so wall-clock time matches sim time.
            elapsed = time.time() - step_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            t += 1
            if done or t >= steps:
                print(f"Episode ended at t={t} ({'terminated' if done else 'truncated/max-len'}), "
                      f"resetting. cmd was ({vx:.2f},{vy:.2f},{wz:.2f})")
                obs = env.reset(seed=args.seed + t)  # vary seed so it's not identical every reset
                if cpg is not None:
                    cpg.reset(seed=args.seed + t)
                t = 0

    env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cpg", "bc", "bc_ppo"], default="cpg")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to BC policy checkpoint (.pt). Required if --mode bc.")
    parser.add_argument("--bc-checkpoint", type=str, default=None,
                         help="Path to BC policy checkpoint (.pt). Required if --mode bc_ppo.")
    parser.add_argument("--ppo-checkpoint", type=str, default=None,
                         help="Path to SB3 PPO checkpoint (.zip), e.g. "
                              "checkpoints/ppo_residual/checkpoints/ppo_residual_500000_steps.zip "
                              "or checkpoints/ppo_residual/best_model/best_model.zip. "
                              "Required if --mode bc_ppo. Re-run with a newer "
                              "checkpoint path periodically while training is "
                              "still running in the background to check progress.")
    parser.add_argument("--episode-len", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "bc" and args.checkpoint is None:
        parser.error("--checkpoint is required when --mode bc")
    if args.mode == "bc_ppo" and (args.bc_checkpoint is None or args.ppo_checkpoint is None):
        parser.error("--bc-checkpoint and --ppo-checkpoint are both required when --mode bc_ppo")

    run(args)


if __name__ == "__main__":
    main()