"""
pretrain_bc.py
--------------
Behavioral Cloning (BC) Warm-Start for the PPO agent.

Idea:
    The CPG already produces a reasonable gait (no training needed).
    We use it as an "expert": collect (observation, target_residual=0)
    pairs and train the PPO policy network with supervised learning (MSE loss).

    Before PPO even starts:
        - The agent ALREADY knows to "follow the CPG" (residual ~= 0)
        - Does NOT start from random weights that cause falls/standing still
        - PPO then learns SMALL CORRECTIONS on top of CPG — much easier

Usage:
    py pretrain_bc.py                      # default: 200K steps, 512 batch
    py pretrain_bc.py --n_steps 500000     # more demos
    py pretrain_bc.py --epochs 20          # more training epochs

Then:
    py train_ppo_cpg.py --bc_weights models_cpg/bc_pretrained/bc_weights.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env_wrapper_cpg import make_env_cpg


# ─────────────────────────────────────────────────────────────────────────────
# 1. Collect demonstrations from CPG (expert = zero residual)
# ─────────────────────────────────────────────────────────────────────────────

def collect_demonstrations(n_steps: int, residual_scale: float = 1.0,
                            action_smoothing: float = 0.3,
                            n_envs: int = 8):
    """
    Runs the CPG with zero PPO residual and collects (obs, target_action).

    target_action = zeros(12) => "don't correct anything, just follow the CPG".

    Returns:
        obs_buf    : (N, obs_dim) float32
        action_buf : (N, 12)     float32  -- always zero (BC target)
    """
    print(f"[BC] Collecting {n_steps:,} demonstration steps from CPG expert...")

    envs = DummyVecEnv([make_env_cpg(residual_scale=residual_scale,
                                      action_smoothing=action_smoothing)
                        for _ in range(n_envs)])

    obs_list, action_list = [], []
    obs = envs.reset()
    steps_per_env = n_steps // n_envs

    for step in range(steps_per_env):
        # Expert action = zero residual (CPG does the work)
        zero_action = np.zeros((n_envs, 12), dtype=np.float32)
        obs_list.append(obs.copy())
        action_list.append(zero_action.copy())
        obs, _, dones, _ = envs.step(zero_action)

        if (step + 1) % (steps_per_env // 5) == 0:
            pct = (step + 1) / steps_per_env * 100
            print(f"  ... {pct:.0f}%  ({(step+1)*n_envs:,} / {n_steps:,} steps)", flush=True)

    envs.close()

    obs_buf    = np.concatenate(obs_list,    axis=0).astype(np.float32)
    action_buf = np.concatenate(action_list, axis=0).astype(np.float32)

    print(f"[BC] Collected {len(obs_buf):,} pairs | obs_dim={obs_buf.shape[1]}")
    return obs_buf, action_buf


# ─────────────────────────────────────────────────────────────────────────────
# 2. Supervised pretraining of the PPO actor network
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_bc(obs_buf: np.ndarray, action_buf: np.ndarray,
                obs_dim: int, action_dim: int,
                net_arch: list, epochs: int, batch_size: int,
                lr: float, save_dir: str) -> str:
    """
    Trains an MLP (same architecture as PPO policy) with MSE loss
    between predicted action and target action (=zeros).

    Saves state_dict to save_dir/bc_weights.pt and returns the path.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BC] Training device: {device}")

    # Build MLP matching the PPO actor head
    layers = []
    in_dim = obs_dim
    for hidden in net_arch:
        layers += [nn.Linear(in_dim, hidden), nn.Tanh()]
        in_dim = hidden
    layers += [nn.Linear(in_dim, action_dim), nn.Tanh()]  # Tanh = bounded [-1,1]
    model = nn.Sequential(*layers).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    obs_t    = torch.tensor(obs_buf,    dtype=torch.float32)
    action_t = torch.tensor(action_buf, dtype=torch.float32)
    dataset  = TensorDataset(obs_t, action_t)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"[BC] Pretraining: {epochs} epochs x {len(loader)} batches/epoch")
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for obs_batch, action_batch in loader:
            obs_batch    = obs_batch.to(device)
            action_batch = action_batch.to(device)

            pred = model(obs_batch)
            loss = criterion(pred, action_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            print(f"  Epoch {epoch:3d}/{epochs} | MSE loss: {avg_loss:.6f} (best: {best_loss:.6f})")

    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, "bc_weights.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"\n[BC] Weights saved: {weights_path}")
    return weights_path


# ─────────────────────────────────────────────────────────────────────────────
# 3. Load BC weights into the PPO policy (called from train_ppo_cpg.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_bc_weights_into_ppo(ppo_model: PPO, bc_weights_path: str,
                              net_arch: list) -> None:
    """
    Copies BC-pretrained weights into the actor (mlp_extractor + action_net)
    of the PPO policy.

    SB3 MlpPolicy actor structure:
        policy.mlp_extractor.policy_net  -- hidden layers for actor
        policy.action_net                -- final linear layer

    We match BC MLP layers with those layer-by-layer.
    """
    device = next(ppo_model.policy.parameters()).device
    bc_state = torch.load(bc_weights_path, map_location=device, weights_only=True)

    policy = ppo_model.policy

    bc_linear_layers = [(k, v) for k, v in bc_state.items() if "weight" in k or "bias" in k]

    # mlp_extractor hidden layers
    actor_layers = list(policy.mlp_extractor.policy_net.children())
    bc_idx = 0
    for layer in actor_layers:
        if isinstance(layer, nn.Linear):
            _, w_val = bc_linear_layers[bc_idx]
            _, b_val = bc_linear_layers[bc_idx + 1]
            with torch.no_grad():
                layer.weight.copy_(w_val)
                layer.bias.copy_(b_val)
            bc_idx += 2

    # action_net (final linear layer)
    _, w_val = bc_linear_layers[bc_idx]
    _, b_val = bc_linear_layers[bc_idx + 1]
    with torch.no_grad():
        policy.action_net.weight.copy_(w_val)
        policy.action_net.bias.copy_(b_val)

    print(f"[BC] Weights loaded into PPO policy from: {bc_weights_path}")
    print(f"     Actor initialized: already knows to follow CPG pattern.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BC Pretraining for Go2 CPG+PPO")
    parser.add_argument("--n_steps",          type=int,   default=200_000,
                        help="Total demonstration steps (default: 200K)")
    parser.add_argument("--n_envs",           type=int,   default=8,
                        help="Parallel envs for demo collection")
    parser.add_argument("--epochs",           type=int,   default=15,
                        help="Epochs for supervised training")
    parser.add_argument("--batch_size",       type=int,   default=512)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--residual_scale",   type=float, default=1.0)
    parser.add_argument("--action_smoothing", type=float, default=0.3)
    parser.add_argument("--save_dir",         type=str,   default="models_cpg/bc_pretrained",
                        help="Directory for saved BC weights")
    args = parser.parse_args()

    print("=" * 60)
    print("Behavioral Cloning Pretraining -- Go2 CPG Expert")
    print("=" * 60)
    print(f"  Demo steps      : {args.n_steps:,}")
    print(f"  Parallel envs   : {args.n_envs}")
    print(f"  BC epochs       : {args.epochs}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Learning rate   : {args.lr}")
    print(f"  Save dir        : {args.save_dir}")
    print()

    # Architecture -- must match train_ppo_cpg.py
    NET_ARCH   = [256, 256]
    ACTION_DIM = 12

    # Discover obs_dim from env
    tmp_env = make_env_cpg(residual_scale=args.residual_scale,
                            action_smoothing=args.action_smoothing)()
    obs, _ = tmp_env.reset()
    OBS_DIM = obs.shape[0]
    tmp_env.close()
    print(f"  Obs dim         : {OBS_DIM}")
    print(f"  Action dim      : {ACTION_DIM}")
    print(f"  Net arch        : {NET_ARCH}\n")

    # Phase 1: Collect demonstrations
    obs_buf, action_buf = collect_demonstrations(
        n_steps=args.n_steps,
        residual_scale=args.residual_scale,
        action_smoothing=args.action_smoothing,
        n_envs=args.n_envs,
    )

    # Phase 2: Supervised pretraining
    weights_path = pretrain_bc(
        obs_buf=obs_buf,
        action_buf=action_buf,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        net_arch=NET_ARCH,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
    )

    print("\n" + "=" * 60)
    print("BC Pretraining complete!")
    print("Run next:")
    print(f"  py train_ppo_cpg.py --bc_weights {weights_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
