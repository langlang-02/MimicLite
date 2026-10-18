#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$MIMICLITE_ROOT"
export PYTHONPATH="$MIMICLITE_ROOT/active-adaptation:$MIMICLITE_ROOT/mimic-lite${PYTHONPATH:+:$PYTHONPATH}"
export ANY4HDMI_CACHE_BUILD_BATCH_SIZE=2048
export TORCHINDUCTOR_COMPILE_THREADS=2
mkdir -p "$RL_RUN_DIR/preflight"
exec > >(tee -a "$RL_RUN_DIR/launcher.log") 2>&1
echo "Stage: EGL/GL system libraries (before framework imports)"
# active_adaptation imports torchrl, which imports MuJoCo and loads EGL.
check_gl_libraries() {
  python -c "import ctypes; ctypes.CDLL('libEGL.so.1'); ctypes.CDLL('libGL.so.1')"
}
if ! check_gl_libraries; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libegl1 libgl1 libglib2.0-0
fi
# Recheck in a fresh process after installation; fail here if still unavailable.
check_gl_libraries
python - <<'PY'
import mujoco
from OpenGL import EGL
assert callable(EGL.eglQueryString)
print('EGL symbols and MuJoCo import: PASS', mujoco.__version__)
PY
echo "Stage: environment and frozen code verification"
python - <<'PY'
import hashlib, importlib.util, json, os, sys
from pathlib import Path
root = Path(os.environ['MIMICLITE_ROOT'])
for name, expected in json.loads((root/'local/mjlab/code.lock.json').read_text())['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
import active_adaptation
assert Path(active_adaptation.__file__).is_relative_to(root)
assert Path(importlib.util.find_spec('mimic_lite_learning').origin).is_relative_to(root)
print('Frozen code verified:', root, 'Python:', sys.version)
from active_adaptation.cli import aa_discover_projects
aa_discover_projects()
p = root/'active-adaptation/.cache/projects.json'
d = json.loads(p.read_text())
for group in d.values():
    for name, project in group.items():
        project['enabled'] = name == 'mimic_lite'
p.write_text(json.dumps(d, indent=2)+'\n')
PY
nvidia-smi
echo "Stage: four-GPU NCCL preflight"
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  local/mjlab/verify_nccl.py
echo "Stage: single-process filtered FK cache and inference preflight (no PPO updates)"
TORCH_COMPILE_DISABLE=1 python local/mjlab/verify_latent_runtime.py \
  --dataset "$RVQ_DATA_ROOT" --filenames-path flat_ground_v1/train.txt \
  --checkpoint "$MIMICLITE_SMOKE_CHECKPOINT" \
  --report "$RL_RUN_DIR/preflight/latent-runtime-filtered.json"
echo "Stage: PPO training, four GPUs, 2048 environments per GPU"
exec bash local/mjlab/platform_entrypoint.sh
