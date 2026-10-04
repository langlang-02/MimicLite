"""Verify trained weights changed, exact reload, reset, and finite GPU steps."""
import json
import os
import shutil
from pathlib import Path

import torch
from omegaconf import OmegaConf
import active_adaptation as aa

root = Path(os.environ['MIMICLITE_ROOT'])
run = root / 'local/mjlab/records/train'
final_path = (run / 'checkpoint_latest.pt').resolve(strict=True)
initial_path = final_path.parent / 'checkpoint_0.pt'
# W&B disabled mode writes under TMPDIR; retain the two smoke checkpoints in records.
artifacts = root / 'local/mjlab/records/checkpoints'
artifacts.mkdir(exist_ok=True)
for source in (initial_path, final_path):
    target = artifacts / source.name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
initial_path = artifacts / initial_path.name
final_path = artifacts / final_path.name
initial = torch.load(initial_path, map_location='cpu', weights_only=False)
final = torch.load(final_path, map_location='cpu', weights_only=False)
changed = [k for k, v in final['policy']['actor'].items()
           if not torch.equal(v, initial['policy']['actor'][k])]
assert changed, 'PPO did not change any actor weights'
cfg = final['cfg']
cfg.checkpoint_path = str(final_path)
OmegaConf.set_struct(cfg, False)
aa.init(cfg, auto_rank=True)
from active_adaptation.helpers import make_env_policy
from torchrl.envs.utils import set_exploration_type, ExplorationType

env, policy = make_env_policy(cfg)
# Upstream loader warns instead of raising on missing keys; explicitly enforce it.
assert policy.load_state_dict(final['policy']) == []
for module, saved in final['policy'].items():
    if isinstance(saved, dict):
        for k, tensor in saved.items():
            assert torch.equal(policy.state_dict()[module][k].cpu(), tensor.cpu()), (module, k)

def check_finite(td):
    for key, value in td.items(True, True):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            assert torch.isfinite(value).all(), f'Non-finite tensor: {key}'

with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
    carry = env.reset()
    check_finite(carry)
    rollout = policy.get_rollout_policy('eval')
    for _ in range(16):
        td, carry = env.step_and_maybe_reset(rollout(carry))
        check_finite(td)
    env.reset()
torch.cuda.synchronize()
report = {
    'result': 'PASS', 'device': str(env.device), 'gpu': torch.cuda.get_device_name(),
    'num_envs': env.num_envs, 'reload_steps': 16,
    'changed_actor_tensors': len(changed), 'checkpoint': str(final_path),
    'checkpoint_bytes': final_path.stat().st_size,
    'max_torch_allocated_MiB': torch.cuda.max_memory_allocated() / 2**20,
    'scope': 'synthetic standing; no tracking quality or long-run stability claim',
}
(root / 'local/mjlab/records/verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
env.close()
