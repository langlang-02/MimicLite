#!/usr/bin/env bash
# Persistent validation job: wait for data and free GPUs, never terminate other jobs.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
source "$MIMICLITE_ROOT/local/mjlab/.env.local"
base="${MIMICLITE_TARGET_BASE:-/data/langlang}"
record="$base/transfer/validation"
mkdir -p "$record"
exec > >(tee -a "$record/console.log") 2>&1
echo 'Waiting for verified dataset and runtime artifacts'
until [[ -f "$base/transfer/rvq-dataset-20261006/verification.json" && -f "$base/transfer/runtime-artifacts.verified" ]]; do sleep 30; done
echo 'Building full FK cache on CPU'
python "$MIMICLITE_ROOT/local/mjlab/migration/build_full_cache.py"
python "$MIMICLITE_ROOT/local/mjlab/test_motion_index.py"
python "$MIMICLITE_ROOT/mimic-lite/tests/test_ppo_diagnostics.py"
checkpoint="$base/checkpoints/mimiclite/train1024-entropy-2500.pt"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MIMICLITE_NPROC_PER_NODE=4
export RL_RUN_DIR="$base/runs/mimiclite/full-data-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
export PPO_OVERRIDES_FILE="$record/smoke.txt"
export WANDB_MODE=disabled
python "$MIMICLITE_ROOT/local/mjlab/migration/full_training.py" --checkpoint "$checkpoint" --output "$PPO_OVERRIDES_FILE" --iterations 10 --wandb-mode disabled
python "$MIMICLITE_ROOT/local/mjlab/migration/full_training.py" --checkpoint "$checkpoint" --output "$record/full-2500.txt"
echo 'Waiting for GPUs 0–3 to have no compute processes and <1 GiB allocated'
while ! python - <<'PY'
import subprocess
rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'], text=True)
selected = [row.split(',') for row in rows.splitlines() if int(row.split(',')[0]) < 4]
assert len(selected) == 4
busy = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'], text=True)
raise SystemExit(0 if all(int(mem) < 1024 and uuid.strip() not in busy for _,uuid,mem in selected) else 1)
PY
do sleep 60; done
echo 'Starting 4 × 8192 full-data smoke training'
bash "$MIMICLITE_ROOT/local/mjlab/train_local.sh"
python "$MIMICLITE_ROOT/local/mjlab/migration/full_training.py" --checkpoint "$checkpoint" --output "$record/result.json" --verify-run "$RL_RUN_DIR" --iterations 10
echo "Validation complete. Formal training profile: $record/full-2500.txt"
