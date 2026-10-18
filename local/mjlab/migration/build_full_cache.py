"""Build the full training FK cache on CPU without reserving any GPU."""
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['ANY4HDMI_CACHE_BUILD_DEVICE'] = 'cpu'

from any4hdmi.dataset.loading import resolve_input_paths, resolve_motion_filenames
from mimic_lite.tasks.windowed_cache import _DirectCache

root = Path(os.environ['RVQ_DATA_ROOT'])
base = Path(os.environ['MIMICLITE_MOTION_CACHE_BASE'])
base.mkdir(parents=True, exist_ok=True)
names = resolve_motion_filenames(root, filenames_path='flat_ground_v1/train.txt')
assert len(names) == 159777, f'Unexpected full training split size: {len(names)}'
cache = _DirectCache.from_inputs(input_paths=resolve_input_paths(base, str(root)),
                               target_fps=50, base_dir=base, motion_filenames=names)
cache.get_or_build()
report = dict(result='PASS', motions=len(names), cache=str(cache.cache_entry_dir), device='cpu')
(base / 'full-cache-verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
