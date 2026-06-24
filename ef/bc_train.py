"""
Behavioral Cloning training: supervised regression from proprioceptive
observation -> CPG joint position target (q_des), using the dataset
collected by collect_data.py.

Architecture: simple MLP. This is intentionally not fancy -- the goal
of BC here is a warm-start for PPO fine-tuning, not a final controller,
so a compact, fast-to-train network is the right choice over something
more elaborate (e.g. recurrent/transformer).

Usage
-----
    python bc_train.py --data data/bc_dataset --epochs 50 --out checkpoints/bc_policy.pt

Expects the dataset directory to contain shard_*.npz files as produced
by collect_data.py (after the fix saving q_des, not torque, as 'actions').
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


OBS_DIM = 39  # gravity(3) + imu_acc(3) + imu_gyro(3) + qpos(12) + qvel(12) + lin_err(3) + ang_err(3)
ACT_DIM = 12


class BCPolicy(nn.Module):
    """
    MLP: proprioceptive observation -> joint position targets (q_des).
    Same input/output spec the PPO residual policy will eventually use,
    so the BC-trained weights can seed (or be distilled into) the PPO
    network's initial layers if desired.
    """

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=(256, 256, 128)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


class BCDataset(Dataset):
    def __init__(self, observations, actions, obs_mean, obs_std):
        self.obs = (observations - obs_mean) / (obs_std + 1e-8)
        self.act = actions

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx]


def load_dataset(data_dir):
    data_dir = Path(data_dir)
    combined_path = data_dir / "bc_dataset_combined.npz"
    if combined_path.exists():
        print(f"Loading pre-combined dataset: {combined_path}")
        d = np.load(combined_path)
        return d["observations"], d["actions"], d["commands"], d["episode_ids"]

    shard_paths = sorted(glob.glob(str(data_dir / "shard_*.npz")))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.npz files found in {data_dir}")
    print(f"Loading {len(shard_paths)} shards from {data_dir}")

    obs_list, act_list, cmd_list, epi_list = [], [], [], []
    for p in shard_paths:
        d = np.load(p)
        obs_list.append(d["observations"])
        act_list.append(d["actions"])
        cmd_list.append(d["commands"])
        epi_list.append(d["episode_ids"])

    return (np.concatenate(obs_list), np.concatenate(act_list),
            np.concatenate(cmd_list), np.concatenate(epi_list))


def sanity_check_dataset(obs, act, act_dim=ACT_DIM):
    """
    Catches the exact class of bug we hit earlier (torques saved
    instead of joint position targets) before wasting a training run
    on mislabeled data. Go2 joint angles are physically bounded to
    roughly [-1.5, 4.5] rad depending on the joint; torques from a
    PD controller with the gains used here can reach +-50. If any
    action dimension's range looks torque-like, abort loudly.
    """
    assert obs.shape[1] == OBS_DIM, f"obs dim {obs.shape[1]} != expected {OBS_DIM}"
    assert act.shape[1] == act_dim, f"act dim {act.shape[1]} != expected {act_dim}"
    assert not np.isnan(obs).any(), "NaN found in observations"
    assert not np.isnan(act).any(), "NaN found in actions"

    act_abs_max = np.abs(act).max()
    if act_abs_max > 6.0:
        raise ValueError(
            f"action magnitude max={act_abs_max:.1f} rad is far outside "
            f"plausible Go2 joint-angle range (~[-1.5, 4.5] rad). This "
            f"looks like torques were saved instead of joint position "
            f"targets (q_des) -- check collect_data.py saves q_des as "
            f"the 'actions' array, not the PD torque output. Aborting "
            f"before training on mislabeled data."
        )
    print(f"Dataset sanity check passed: obs={obs.shape}, act={act.shape}, "
          f"act range=[{act.min():.2f}, {act.max():.2f}] rad")


def train(args):
    obs, act, cmd, epi = load_dataset(args.data)
    sanity_check_dataset(obs, act)

    obs_mean = obs.mean(axis=0)
    obs_std = obs.std(axis=0)

    dataset = BCDataset(obs.astype(np.float32), act.astype(np.float32), obs_mean, obs_std)
    n_val = int(0.1 * len(dataset))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                        generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}, train={n_train} val={n_val} samples")

    model = BCPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for obs_batch, act_batch in train_loader:
            obs_batch, act_batch = obs_batch.to(device), act_batch.to(device)
            pred = model(obs_batch)
            loss = loss_fn(pred, act_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for obs_batch, act_batch in val_loader:
                obs_batch, act_batch = obs_batch.to(device), act_batch.to(device)
                pred = model(obs_batch)
                val_losses.append(loss_fn(pred, act_batch).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        print(f"epoch {epoch+1}/{args.epochs}  train_mse={train_loss:.5f}  val_mse={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "obs_mean": obs_mean,
                "obs_std": obs_std,
                "epoch": epoch,
                "val_loss": val_loss,
            }, out_path)
            print(f"  -> saved new best checkpoint (val_mse={val_loss:.5f}) to {out_path}")

    print(f"\nDone. Best val_mse={best_val_loss:.5f}. Checkpoint at {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="data/bc_dataset")
    parser.add_argument("--out", type=str, default="checkpoints/bc_policy.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
