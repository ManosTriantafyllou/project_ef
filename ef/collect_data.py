"""
Data collection for Behavioral Cloning: runs the verified trot CPG inside
the real gym-quadruped MuJoCo simulation, with the CPG's joint angle
targets tracked by the environment's built-in PD controller (same
interface as the residual-RL experiment). Records (proprioceptive
observation, joint position action) pairs for every control step,
labeled by the commanded velocity, ready for BC training.

Usage
-----
    python collect_data.py --episodes 5000 --episode-len 5.0 --out data/bc_dataset.npz

Resumable: periodically writes partial shards to --out-dir so a crash
or interruption doesn't lose completed episodes. Run with --resume to
continue from the last shard.

Output format
--------------
Saved as a set of .npz shards in --out-dir, each containing:
    observations : (N, obs_dim) float32 -- proprioceptive observation
    actions      : (N, 12) float32      -- CPG joint position targets q_des (rad),
                                            i.e. the BC label. NOTE: the PD torque
                                            computed from q_des is what's actually
                                            sent to env.step() (since this env's
                                            action space expects torques), but we
                                            log q_des as the label so BC/PPO learn
                                            in joint-position-target space.
    commands     : (N, 3) float32       -- [vx, vy, wz] active at that step
    episode_ids  : (N,) int32           -- which episode each row belongs to

A manifest.json tracks how many episodes/shards have been completed so
collection can resume.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import mujoco

from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.sensors.imu import IMU

from trot_cpg import TrotCPG, TrotCPGParams, joint_angles_dict_to_vector


# ----------------------------------------------------------------------
# Patch: ensure the IMU sensor doesn't read uninitialized sim data on
# first step (same fix used in the project's reference environment
# usage example).
# ----------------------------------------------------------------------
_original_imu_init = IMU.__init__


def _patched_imu_init(self, mj_model, mj_data, *args, **kwargs):
    mujoco.mj_forward(mj_model, mj_data)
    _original_imu_init(self, mj_model, mj_data, *args, **kwargs)
    self.step()


IMU.__init__ = _patched_imu_init


# ----------------------------------------------------------------------
# Observation variant: proprioceptive (matches the benchmark's main
# evaluation setting -- see project spec).
# ----------------------------------------------------------------------
PROPRIOCEPTIVE_OBS = (
    "gravity_vector:base",
    "imu_acc",
    "imu_gyro",
    "qpos_js",
    "qvel_js",
    "base_lin_vel_err:base",
    "base_ang_vel_err:base",
)

IMU_KWARGS = {
    "accel_name": "imu_acc",
    "gyro_name": "imu_gyro",
    "imu_site_name": "imu",
}


# NOTE: command sampling is owned entirely by QuadrupedEnv via
# base_vel_command_type="random" (see project spec reference example).
# We only ever READ the active command via read_velocity_command()
# below -- never sample or set it ourselves, since the benchmark rules
# prohibit modifying how evaluation commands are generated.


# ----------------------------------------------------------------------
# Main collection loop
# ----------------------------------------------------------------------
def make_env():
    env = QuadrupedEnv(
        robot="go2",
        scene="flat",
        base_vel_command_type="random",  # benchmark owns command sampling -- see project spec
        state_obs_names=PROPRIOCEPTIVE_OBS,
        sensors=(IMU,),
        sensors_kwargs=(IMU_KWARGS,),
    )
    return env


def flatten_obs(obs_dict_or_array):
    """
    gym-quadruped's reset/step observation may already be a flat array
    matching state_obs_names order, or a dict depending on version --
    handle both defensively.
    """
    if isinstance(obs_dict_or_array, dict):
        parts = [np.atleast_1d(np.asarray(obs_dict_or_array[k], dtype=np.float32))
                  for k in PROPRIOCEPTIVE_OBS]
        return np.concatenate(parts)
    return np.asarray(obs_dict_or_array, dtype=np.float32)


def get_joint_state(obs_dict_or_array):
    """Extract qpos_js and qvel_js from either dict or flat array observation."""
    if isinstance(obs_dict_or_array, dict):
        qpos = np.asarray(obs_dict_or_array["qpos_js"], dtype=np.float64).flatten()
        qvel = np.asarray(obs_dict_or_array["qvel_js"], dtype=np.float64).flatten()
    else:
        arr = np.asarray(obs_dict_or_array, dtype=np.float64).flatten()
        # PROPRIOCEPTIVE_OBS order: gravity_vector:base (3), imu_acc (3), imu_gyro (3),
        # qpos_js (12), qvel_js (12), base_lin_vel_err:base (3), base_ang_vel_err:base (3)
        qpos = arr[9:21]
        qvel = arr[21:33]
    return qpos, qvel


def compute_pd_torque(q_des, q, q_dot, kp=90.0, kd=5.0, torque_limit=50.0):
    """Convert desired joint angles to torque using a joint-level PD controller."""
    tau = kp * (q_des - q) - kd * q_dot
    return np.clip(tau, -torque_limit, torque_limit).astype(np.float32)


def read_velocity_command(env):
    """
    Read the velocity command the environment itself sampled this
    episode (via base_vel_command_type="random", as in the project's
    reference usage example). We must NOT set this ourselves -- the
    benchmark owns the command-sampling logic, and teams may not
    modify it. We only read it so the CPG can track the SAME target
    the reward function is scoring against.

    Confirmed (via inspect_env_command_attr.py against the installed
    gym-quadruped version) that env.target_base_vel() is the official
    public API and returns a tuple (lin_vel, ang_vel), each a 3-vector:
        lin_vel = [vx, vy, vz]
        ang_vel = [wx, wy, wz]
    We want vx, vy from lin_vel and wz from ang_vel.
    """
    lin_vel, ang_vel = env.target_base_vel()
    lin_vel = np.asarray(lin_vel, dtype=np.float32).flatten()
    ang_vel = np.asarray(ang_vel, dtype=np.float32).flatten()
    return float(lin_vel[0]), float(lin_vel[1]), float(ang_vel[2])


def collect(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        start_episode = manifest["episodes_completed"]
        shard_idx = manifest["next_shard_idx"]
        print(f"Resuming from episode {start_episode}, shard {shard_idx}")
    else:
        manifest = {"episodes_completed": 0, "next_shard_idx": 0,
                     "episodes_per_shard": args.episodes_per_shard}
        start_episode = 0
        shard_idx = 0

    env = make_env()
    print(f"action_space: low={env.action_space.low}, high={env.action_space.high}")
    print("  (CPG outputs raw joint angles in radians -- if the bounds "
          "above look like [-1, 1] rather than joint-limit-sized radian "
          "ranges, this env expects NORMALIZED actions and the CPG "
          "output must be rescaled before calling env.step(). Check this "
          "before trusting the collected dataset.)")
    cpg_params = TrotCPGParams()
    cpg = TrotCPG(cpg_params)

    control_dt = getattr(env, "dt", None) or (1.0 / 50.0)
    steps_per_episode = int(round(args.episode_len / control_dt))

    rng = np.random.default_rng(args.seed + start_episode)

    obs_buf, act_buf, cmd_buf, epi_buf = [], [], [], []
    t_start = time.time()

    for ep in range(start_episode, args.episodes):
        seed = args.seed + ep
        obs = env.reset(seed=seed)
        cpg.reset(seed=seed)
        vx, vy, wz = read_velocity_command(env)

        episode_failed = False
        height_trace = []
        for t in range(steps_per_episode):
            # Re-read each step: some gym-quadruped configurations vary
            # the command within an episode (time-varying commands),
            # so the CPG must always track whatever is currently active
            # rather than assuming it's fixed for the whole episode.
            vx, vy, wz = read_velocity_command(env)
            out = cpg.step(control_dt, vx, vy, wz)
            q_des = joint_angles_dict_to_vector(out["joint_angles"]).astype(np.float64)
            q, q_dot = get_joint_state(obs)
            torque = compute_pd_torque(q_des, q, q_dot)

            obs_flat = flatten_obs(obs)
            obs_buf.append(obs_flat)
            act_buf.append(q_des.astype(np.float32))  # BC label = joint position target, not torque
            cmd_buf.append(np.array([vx, vy, wz], dtype=np.float32))
            epi_buf.append(ep)

            step_result = env.step(torque)
            # Support both 4-tuple and 5-tuple gym step() return signatures.
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result

            if hasattr(env, "data"):
                try:
                    height_trace.append(float(env.data.qpos[2]))
                except Exception:
                    pass

            if done:
                episode_failed = True
                try:
                    base_pos = env.data.qpos[:3] if hasattr(env, "data") else None
                except Exception:
                    base_pos = None
                print(f"    [diag] step {t}: reward={reward:.3f} "
                      f"terminated={locals().get('terminated', 'n/a')} "
                      f"truncated={locals().get('truncated', 'n/a')} "
                      f"info={info} base_pos={base_pos}")
                if height_trace:
                    n = len(height_trace)
                    sample_idx = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
                    trace_str = ", ".join(f"t={i}:{height_trace[i]:.3f}" for i in sample_idx)
                    print(f"    [diag] base height (z) trace: {trace_str}")
                break

        status = "FAILED (terminated early)" if episode_failed else "ok"
        if (ep + 1) % 25 == 0 or episode_failed:
            elapsed = time.time() - t_start
            print(f"episode {ep+1}/{args.episodes} cmd=({vx:.2f},{vy:.2f},{wz:.2f}) "
                  f"steps={t+1}/{steps_per_episode} [{status}] "
                  f"elapsed={elapsed:.1f}s")

        if (ep + 1) % manifest["episodes_per_shard"] == 0 or (ep + 1) == args.episodes:
            shard_path = out_dir / f"shard_{shard_idx:05d}.npz"
            np.savez_compressed(
                shard_path,
                observations=np.stack(obs_buf).astype(np.float32),
                actions=np.stack(act_buf).astype(np.float32),
                commands=np.stack(cmd_buf).astype(np.float32),
                episode_ids=np.array(epi_buf, dtype=np.int32),
            )
            print(f"  wrote {shard_path} ({len(obs_buf)} steps)")
            obs_buf, act_buf, cmd_buf, epi_buf = [], [], [], []
            shard_idx += 1
            manifest["episodes_completed"] = ep + 1
            manifest["next_shard_idx"] = shard_idx
            manifest_path.write_text(json.dumps(manifest, indent=2))

    env.close()
    print(f"\nDone. {manifest['episodes_completed']} episodes, "
          f"{shard_idx} shards written to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--episode-len", type=float, default=5.0,
                         help="Episode duration in seconds")
    parser.add_argument("--out-dir", type=str, default="data/bc_dataset")
    parser.add_argument("--episodes-per-shard", type=int, default=100,
                         help="Write a shard to disk every N episodes "
                              "(controls checkpoint granularity)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                         help="Resume from the last shard in --out-dir "
                              "if a manifest.json is present")
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()