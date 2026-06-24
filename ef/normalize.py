import numpy as np
from pathlib import Path

out_dir = Path('method/data/bc_dataset')
obs_list, act_list, cmd_list, epi_list = [], [], [], []

for i in range(50):
    shard = np.load(out_dir / f'shard_{i:05d}.npz')
    obs_list.append(shard['observations'])
    act_list.append(shard['actions'])
    cmd_list.append(shard['commands'])
    epi_list.append(shard['episode_ids'])

obs = np.concatenate(obs_list)
acts = np.concatenate(act_list)
cmds = np.concatenate(cmd_list)
epis = np.concatenate(epi_list)

print(f'Final: obs={obs.shape} acts={acts.shape}')

np.savez_compressed(out_dir / 'bc_dataset_combined.npz', 
                     observations=obs, actions=acts, 
                     commands=cmds, episode_ids=epis)