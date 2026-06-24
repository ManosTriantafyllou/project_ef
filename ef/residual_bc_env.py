"""
Residual-on-BC environment wrapper for PPO fine-tuning.

Architecture
------------
    a_final = a_BC(obs) + scale * a_PPO

where a_BC is the frozen, pre-trained Behavioral Cloning policy (warm
start from the CPG demonstrations), and a_PPO is the small residual
correction that Stable-Baselines3's PPO learns on top. a_final is the
joint position target (q_des) that gets converted to torque via the
same PD controller used during data collection, then sent to the
underlying QuadrupedEnv.

Why residual-on-BC instead of fine-tuning the BC network directly:
    - The BC policy already produces a stable walking gait. A residual
      formulation means PPO starts with a_PPO == 0 (assuming the
      residual head is zero-initialized), so the very first PPO
      rollouts are IDENTICAL to the pure BC policy's behavior --
      avoiding the catastrophic-forgetting risk of directly fine-tuning
      BC weights with early, noisy policy-gradient updates (which is
      a major contributor to the collapse-to-falling failure mode
      observed in the original from-scratch PPO experiments).
    - The 'scale' parameter bounds how large a correction PPO can ever
      apply per step, which keeps early exploration from immediately
      producing the large, erratic joint targets that caused training
      instability before.

This wrapper applies the project-specified reward function (via the
QuadrupedEnv._compute_reward monkey-patch below, copied verbatim from
the project description's reference example) and does not alter its
logic, weighting, or any termination/cost/seed behavior -- it only
adds an action-space transformation (the BC+residual sum) on top of
the unmodified QuadrupedEnv physics and scoring.
"""

import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.sensors.imu import IMU
import mujoco

from bc_train import BCPolicy, OBS_DIM, ACT_DIM


_original_imu_init = IMU.__init__


def _patched_imu_init(self, mj_model, mj_data, *args, **kwargs):
    mujoco.mj_forward(mj_model, mj_data)
    _original_imu_init(self, mj_model, mj_data, *args, **kwargs)
    self.step()


IMU.__init__ = _patched_imu_init


# ----------------------------------------------------------------------
# Reward function: exactly as specified in the project description's
# reference example. QuadrupedEnv's own default _compute_reward is NOT
# the benchmark's intended reward -- it must be monkey-patched in, the
# same way the professor's reference script does. This was missing
# from collect_data.py and bc_train.py too, but since BC trains from
# demonstrations (not reward) it didn't matter there. It matters here:
# PPO has no learning signal at all without this.
# ----------------------------------------------------------------------
_reward_debug_printed = {"count": 0}


def _compute_reward(self):
    lin_vel_err_B = self.base_lin_vel_err(frame="base")
    ang_vel_err_B = self.base_ang_vel_err(frame="base")

    sigma_lin_vel = 0.05
    sigma_ang_vel = 0.05
    tracking_lin_vel = np.exp(-np.sum(lin_vel_err_B[:2] ** 2) / (2 * sigma_lin_vel**2))
    tracking_yaw_rate = np.exp(-(ang_vel_err_B[2] ** 2) / (2 * sigma_ang_vel**2))

    gravity_B = self.gravity_vector
    upright_penalty = np.sum(gravity_B[:2] ** 2)

    base_lin_vel_B = self.base_lin_vel(frame="base")
    base_ang_vel_B = self.base_ang_vel(frame="base")

    z_vel_penalty = base_lin_vel_B[2] ** 2
    roll_pitch_ang_vel_penalty = np.sum(base_ang_vel_B[:2] ** 2)

    tau = self.torque_ctrl_setpoint
    torque_penalty = np.sum(tau**2)

    current_action = self.mjData.ctrl.copy()
    if not hasattr(self, "_last_action_for_reward"):
        self._last_action_for_reward = np.zeros_like(current_action)

    action_rate_penalty = np.sum((current_action - self._last_action_for_reward) ** 2)
    self._last_action_for_reward = current_action

    terms = {
        "tracking_lin_vel": 2.0 * tracking_lin_vel,
        "tracking_yaw_rate": 1.0 * tracking_yaw_rate,
        "upright_penalty": -0.5 * upright_penalty,
        "z_vel_penalty": -0.2 * z_vel_penalty,
        "roll_pitch_ang_vel_penalty": -0.1 * roll_pitch_ang_vel_penalty,
        "torque_penalty": -1e-4 * torque_penalty,
        "action_rate_penalty": -0.01 * action_rate_penalty,
    }
    reward = float(sum(terms.values()))

    if os.environ.get("REWARD_DEBUG") and _reward_debug_printed["count"] < 10:
        i = _reward_debug_printed["count"]
        print(f"[reward diag step {i}] tau (torque_ctrl_setpoint)={np.round(tau, 1)}")
        print(f"[reward diag step {i}] mjData.ctrl directly        ={np.round(self.mjData.ctrl, 1)}")
        print(f"[reward diag step {i}] current_action={np.round(current_action, 1)}")
        print(f"[reward diag step {i}] raw_action_rate_sq={action_rate_penalty:.2f}  "
              f"terms={ {k: round(v, 3) for k, v in terms.items()} }  total={reward:.3f}")
        _reward_debug_printed["count"] += 1

    return reward


QuadrupedEnv._compute_reward = _compute_reward


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


def flatten_obs(obs_dict_or_array):
    if isinstance(obs_dict_or_array, dict):
        parts = [np.atleast_1d(np.asarray(obs_dict_or_array[k], dtype=np.float32))
                  for k in PROPRIOCEPTIVE_OBS]
        return np.concatenate(parts)
    return np.asarray(obs_dict_or_array, dtype=np.float32)


def get_joint_state(obs_flat):
    qpos = obs_flat[9:21].astype(np.float64)
    qvel = obs_flat[21:33].astype(np.float64)
    return qpos, qvel


def compute_pd_torque(q_des, q, q_dot, kp=90.0, kd=5.0, torque_limit=50.0):
    tau = kp * (q_des - q) - kd * q_dot
    return np.clip(tau, -torque_limit, torque_limit).astype(np.float32)


class ResidualBCQuadrupedEnv(gym.Env):
    """
    Gymnasium env wrapping QuadrupedEnv with a frozen BC base policy
    and a learnable residual action on top, in joint-position-target
    space.

    Action space: Box(-residual_scale, residual_scale, shape=(12,))
        The PPO policy outputs a residual in this bounded range; it
        is added to the frozen BC policy's q_des prediction.

    Observation space: same 39-dim proprioceptive vector used by BC,
        unchanged -- PPO sees exactly what the BC policy saw.
    """

    metadata = {"render_modes": []}

    def __init__(self, bc_checkpoint_path, residual_scale=0.05,
                 kp=90.0, kd=5.0, torque_limit=50.0, device="cpu"):
        super().__init__()
        self.env = QuadrupedEnv(
            robot="go2",
            scene="flat",
            base_vel_command_type="random",
            state_obs_names=PROPRIOCEPTIVE_OBS,
            sensors=(IMU,),
            sensors_kwargs=(IMU_KWARGS,),
        )
        self.control_dt = getattr(self.env, "dt", None) or (1.0 / 50.0)
        self.kp, self.kd, self.torque_limit = kp, kd, torque_limit
        self.residual_scale = residual_scale
        self.device = torch.device(device)

        checkpoint = torch.load(bc_checkpoint_path, map_location=self.device, weights_only=False)
        self.bc_model = BCPolicy(obs_dim=OBS_DIM, act_dim=ACT_DIM).to(self.device)
        self.bc_model.load_state_dict(checkpoint["model_state_dict"])
        self.bc_model.eval()
        for p in self.bc_model.parameters():
            p.requires_grad_(False)
        self.obs_mean = checkpoint["obs_mean"].astype(np.float32)
        self.obs_std = checkpoint["obs_std"].astype(np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-residual_scale, high=residual_scale, shape=(ACT_DIM,), dtype=np.float32
        )

        self._last_obs_flat = None

    def _bc_action(self, obs_flat):
        x = (obs_flat - self.obs_mean) / (self.obs_std + 1e-8)
        with torch.no_grad():
            q_des = self.bc_model(torch.from_numpy(x).unsqueeze(0).to(self.device))
        return q_des.squeeze(0).cpu().numpy().astype(np.float64)

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset(seed=seed)
        # Fix: _last_action_for_reward (used by the action_rate_penalty
        # term in _compute_reward) persists on the QuadrupedEnv instance
        # across episodes since it's lazily created via hasattr() and
        # never cleared on reset. Without resetting it here, the first
        # step of every new episode compares the fresh torque against
        # whatever torque was active at the END of the PREVIOUS episode
        # (or zero on episode 0), producing a spurious, huge
        # action_rate_penalty that has nothing to do with this episode's
        # actual action smoothness. This only resets bookkeeping state
        # our wrapper depends on for a sane reward signal -- it does not
        # change the reward function's logic, weights, or the
        # underlying QuadrupedEnv's reset/termination/cost behavior.
        if hasattr(self.env, "mjData"):
            self.env._last_action_for_reward = np.zeros_like(self.env.mjData.ctrl)
        obs_flat = flatten_obs(obs)
        self._last_obs_flat = obs_flat
        return obs_flat, {}

    def step(self, action_residual):
        obs_flat = self._last_obs_flat
        a_bc = self._bc_action(obs_flat)
        a_residual = np.clip(action_residual, -self.residual_scale, self.residual_scale)
        q_des = a_bc + a_residual.astype(np.float64)

        q, q_dot = get_joint_state(obs_flat)
        torque = compute_pd_torque(q_des, q, q_dot, self.kp, self.kd, self.torque_limit)

        step_result = self.env.step(torque)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated, truncated = done, False

        obs_flat = flatten_obs(obs)
        self._last_obs_flat = obs_flat

        info = dict(info)
        info["a_bc"] = a_bc
        info["a_residual"] = a_residual
        info["q_des"] = q_des

        return obs_flat, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.env.close()