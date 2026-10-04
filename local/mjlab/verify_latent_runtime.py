"""Reload an existing smoke policy and verify inference; never run PPO updates."""
import argparse
import faulthandler
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
import active_adaptation as aa

faulthandler.enable()

p = argparse.ArgumentParser()
p.add_argument('--dataset', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--report', type=Path, required=True)
p.add_argument('--steps', type=int, default=128)
p.add_argument('--filenames-path', default='train.txt')
a = p.parse_args()
a.report.parent.mkdir(parents=True, exist_ok=True)
checkpoint = a.checkpoint.resolve(strict=True)
saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
initial = torch.load(checkpoint.with_name('checkpoint_0.pt'), map_location='cpu', weights_only=False)
changed = sum(not torch.equal(v, initial['policy']['actor'][k]) for k, v in saved['policy']['actor'].items())
assert changed > 0
retained = a.report.parent / 'latent-checkpoints'
retained.mkdir(exist_ok=True)
for name in ['checkpoint_0.pt', checkpoint.name]:
    shutil.copy2(checkpoint.with_name(name), retained / name)
cfg = saved['cfg']
OmegaConf.set_struct(cfg, False)
cfg.checkpoint_path = str(retained / checkpoint.name)
cfg.task.command.motion_cfgs = {'rvq': dict(path=str(a.dataset), weight=1.0, full_motion=False, filenames_path=a.filenames_path)}
cfg.task.command.rvq_latent_path = str(a.dataset / 'latent')
cfg.task.num_envs = 4
cfg.task.max_episode_length = 32
cfg.algo.compile = False
assert list(cfg.task.observation.command) == ['rvq_latent']
aa.init(cfg, auto_rank=True)
from active_adaptation.helpers import make_env_policy
from torchrl.envs.utils import set_exploration_type, ExplorationType
from active_adaptation.learning.modules.vecnorm import VecNorm

env, policy = make_env_policy(cfg)
assert policy.load_state_dict(saved['policy']) == []
for module, state in saved['policy'].items():
    if isinstance(state, dict):
        for k, v in state.items():
            assert torch.equal(policy.state_dict()[module][k].cpu(), v.cpu())
c = env.base_env.command_manager
store = c.rvq_latent_store
assert len(c.dataset.motion_paths) == len((a.dataset/a.filenames_path).read_text().splitlines())

def finite(td):
    for key, v in td.items(True, True):
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            assert torch.isfinite(v).all(), key

def alignment():
    routes = c.motion_ids
    mids = c.dataset._env_current_motion_id[routes].cpu().numpy()
    local = np.minimum(np.maximum(c.obs_motion_t.cpu().numpy(), 0), c.dataset._env_current_window_len[routes].cpu().numpy()-1)
    steps = c.dataset._env_current_source_start_t[routes].cpu().numpy() + local
    lookup = {str(s): i for i, s in enumerate(store.sample_ids)}
    frames = (steps*20)//50
    idx = np.array([lookup[Path(c.dataset.motion_paths[m]).stem] for m in mids])
    tokens = store.token_offsets[idx] + frames//4
    np.testing.assert_array_equal(c.current_rvq_latent.cpu().numpy(), store.latent[tokens])
    np.testing.assert_array_equal(c.current_rvq_phase.cpu().numpy().ravel(), (frames%4)/4)

with torch.inference_mode(), set_exploration_type(ExplorationType.MODE), VecNorm.freeze():
    carry = env.reset()
    rollout = policy.get_rollout_policy('eval')
    for i in range(a.steps):
        finite(carry)
        alignment()
        assert carry['command'].shape[-1] == 513
        td, carry = env.step_and_maybe_reset(rollout(carry))
        finite(td)
    carry = env.reset()
    alignment()
    finite(carry)
report = dict(result='PASS', dataset=str(a.dataset), motions=len(c.dataset.motion_paths),
              filenames_path=a.filenames_path,
              inference_steps=a.steps, num_envs=4, command_dim=513,
              prior_smoke_changed_actor_tensors=changed, new_optimizer_updates=0,
              reload='all saved module tensors exact', alignment='checked every inference step',
              gpu_peak_MiB=torch.cuda.max_memory_allocated()/2**20)
a.report.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
env.close()
