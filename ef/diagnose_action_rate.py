"""
Diagnostic: run the FROZEN BC policy with a_residual forced to zero
through the reward-patched environment, and print the same reward term
breakdown as the training diagnostic. This isolates whether the huge
action_rate_penalty values seen during PPO sanity checks are:

    (a) inherent to the reward function's scale given this env's
        torque-control setup (i.e. even the smooth, trained BC gait
        produces large step-to-step torque deltas, in which case the
        0.01 weight on action_rate_penalty may simply be miscalibrated
        for this benchmark instance), or

    (b) specific to PPO's random, untrained exploration noise at the
        very start of training (in which case it's expected to shrink
        rapidly as soon as PPO's std collapses below ~1.0).

Usage
-----
    set REWARD_DEBUG=1
    python diagnose_action_rate.py --bc-checkpoint checkpoints/bc_policy.pt --steps 30
"""

import argparse
import os

import numpy as np
import torch

# Import after setting REWARD_DEBUG so the patched _compute_reward's
# debug prints are active for this run.
from residual_bc_env import ResidualBCQuadrupedEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-checkpoint", type=str, default="checkpoints/bc_policy.pt")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    env = ResidualBCQuadrupedEnv(bc_checkpoint_path=args.bc_checkpoint, residual_scale=0.05)
    obs, info = env.reset(seed=0)

    print("=" * 70)
    print(f"Running PURE BC policy (a_residual forced to exactly zero) for {args.steps} steps")
    print("=" * 70)

    rewards = []
    terminated_at = None
    for t in range(args.steps):
        zero_residual = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(zero_residual)
        rewards.append(reward)
        if terminated or truncated:
            terminated_at = t
            print(f"Episode ended at step {t} ({'terminated' if terminated else 'truncated'})")
            break

    rewards = np.array(rewards)
    print()
    print(f"Steps completed: {len(rewards)}" + (f" (ended early at {terminated_at})" if terminated_at is not None else ""))
    print(f"Mean reward: {rewards.mean():.4f}")
    print(f"Min/Max reward: {rewards.min():.4f} / {rewards.max():.4f}")
    print(f"Sum reward (= what ep_rew_mean would show for this episode): {rewards.sum():.2f}")
    print()
    print("Reward over time (every 25th step):")
    for i in range(0, len(rewards), 25):
        print(f"  step {i}: reward={rewards[i]:.3f}")
    print()
    worst_idx = np.argsort(rewards)[:10]
    print("10 worst steps (index, reward):")
    for idx in sorted(worst_idx):
        print(f"  step {idx}: reward={rewards[idx]:.3f}")

    env.close()


if __name__ == "__main__":
    main()