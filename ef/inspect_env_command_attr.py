"""
One-off diagnostic: run this BEFORE collect_data.py to find the exact
attribute name your installed gym-quadruped version uses to expose the
actively-sampled base velocity command (when base_vel_command_type="random").

Usage
-----
    python inspect_env_command_attr.py

If none of the candidate attribute names in read_velocity_command()
(collect_data.py) match what's printed here, add the correct one to
that function's `for attr in (...)` tuple.
"""

import mujoco
import numpy as np

from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.sensors.imu import IMU

_original_imu_init = IMU.__init__


def _patched_imu_init(self, mj_model, mj_data, *args, **kwargs):
    mujoco.mj_forward(mj_model, mj_data)
    _original_imu_init(self, mj_model, mj_data, *args, **kwargs)
    self.step()


IMU.__init__ = _patched_imu_init


def main():
    env = QuadrupedEnv(
        robot="go2",
        scene="flat",
        base_vel_command_type="random",
        state_obs_names=("qpos_js",),  # minimal, we just want to inspect env
        sensors=(IMU,),
        sensors_kwargs=({"accel_name": "imu_acc", "gyro_name": "imu_gyro", "imu_site_name": "imu"},),
    )
    env.reset(seed=0)

    print("=" * 70)
    print("Attributes on env containing 'vel', 'cmd', or 'command':")
    print("=" * 70)
    candidates = [a for a in dir(env) if
                  ("vel" in a.lower() or "cmd" in a.lower() or "command" in a.lower())
                  and not a.startswith("__")]
    for a in candidates:
        try:
            val = getattr(env, a)
            if callable(val):
                print(f"  {a}  (method/callable)")
            else:
                print(f"  {a}  = {val}")
        except Exception as e:
            print(f"  {a}  (error reading: {e})")

    print()
    print("=" * 70)
    print("Calling the candidate METHODS to see their return values:")
    print("=" * 70)
    method_names = ["target_base_vel", "base_lin_vel", "base_ang_vel",
                     "base_lin_vel_err", "base_ang_vel_err", "_sample_ref_vel"]
    for name in method_names:
        if hasattr(env, name):
            method = getattr(env, name)
            try:
                result = method()
                print(f"  {name}()  = {result}")
            except TypeError as e:
                # might require an argument like frame="base"
                try:
                    result = method(frame="base")
                    print(f"  {name}(frame='base')  = {result}")
                except Exception as e2:
                    print(f"  {name}()  raised: {e}  /  with frame='base': {e2}")
            except Exception as e:
                print(f"  {name}()  raised: {e}")

    print()
    print("=" * 70)
    print("Raw _ref_base_lin_vel_H across 3 steps (for comparison):")
    print("=" * 70)
    for t in range(3):
        action = env.action_space.sample()
        env.step(action)
        print(f"  step {t}: _ref_base_lin_vel_H = {env._ref_base_lin_vel_H}")
        if hasattr(env, "target_base_vel"):
            print(f"           target_base_vel() = {env.target_base_vel()}")

    env.close()
    print()
    print("Update read_velocity_command() in collect_data.py with the "
          "attribute name that printed a stable 3-element [vx, vy, wz]-like "
          "vector above.")


if __name__ == "__main__":
    main()