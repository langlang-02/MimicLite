#!/usr/bin/env bash
# Persistent validation job: wait for data and free GPUs, never terminate other jobs.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
source "$MIMICLITE_ROOT/local/mjlab/.env.local"
base="${MIMICLITE_TARGET_BASE:-/data/langlang}"
record="$base/transfer/validation"
mkdir -p "$record"
export CUDA_VISIBLE_DEVICES="${MIMICLITE_GPU_IDS:-4,5,6,7}"
wait_free_gpus() {
echo "Waiting for GPUs $CUDA_VISIBLE_DEVICES to have no compute processes and <1 GiB allocated"
while ! python - <<'PY'
import os, subprocess
ids = set(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'], text=True)
selected = [row.split(',') for row in rows.splitlines() if int(row.split(',')[0]) in ids]
assert len(selected) == 4
busy = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'], text=True)
raise SystemExit(0 if all(int(mem) < 1024 and uuid.strip() not in busy for _,uuid,mem in selected) else 1)
PY
do sleep 60; done
}
exec > >(tee -a "$record/console.log") 2>&1
echo 'Waiting for verified dataset and runtime artifacts'
until [[ -f "$base/transfer/rvq-dataset-20261006/verification.json" && -f "$base/transfer/runtime-artifacts.verified" ]]; do sleep 30; done
cache_device="${MIMICLITE_CACHE_BUILD_DEVICE:-cpu}"
if [[ "$cache_device" == cuda ]]; then wait_free_gpus; fi
echo "Building full FK cache on $cache_device"
python "$MIMICLITE_ROOT/local/mjlab/migration/build_full_cache.py" --device "$cache_device"
python "$MIMICLITE_ROOT/local/mjlab/test_motion_index.py"
python "$MIMICLITE_ROOT/mimic-lite/tests/test_ppo_diagnostics.py"
checkpoint="$base/checkpoints/mimiclite/train1024-entropy-2500.pt"
export CUDA_VISIBLE_DEVICES="${MIMICLITE_GPU_IDS:-4,5,6,7}"
export MIMICLITE_NPROC_PER_NODE=4
export RL_RUN_DIR="$base/runs/mimiclite/full-data-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
export PPO_OVERRIDES_FILE="$record/smoke.txt"
export WANDB_MODE=disabled
python "$MIMICLITE_ROOT/local/mjlab/migration/full_training.py" --checkpoint "$checkpoint" --output "$PPO_OVERRIDES_FILE" --iterations 10 --wandb-mode disabled
python "$MIMICLITE_ROOT/local/mjlab/migration/full_training.py" --checkpoint "$checkpoint" --output "$record/full-2500.txt" --wandb-mode "${MIMICLITE_TRAIN_WANDB_MODE:-offline}"

wait_free_gpus
echo 'Starting 4 × 8192 full-data smoke training'
bash "$MIMICLITE_ROOT/local/mjlab/train_local.sh"
python "$MIMICLITE_ROOT/local/mjlab/migration/full_training.py" --checkpoint "$checkpoint" --output "$record/result.json" --verify-run "$RL_RUN_DIR" --iterations 10
echo "Validation complete. Formal training profile: $record/full-2500.txt"
if [[ "${MIMICLITE_START_FULL_TRAINING:-0}" == 1 ]]; then
  wait_free_gpus
  export PPO_OVERRIDES_FILE="$record/full-2500.txt"
  export RL_RUN_DIR="$base/runs/mimiclite/full-data-2500-$(date -u +%Y%m%dT%H%M%SZ)"
  export WANDB_MODE="${MIMICLITE_TRAIN_WANDB_MODE:-offline}"
  printf '%s\n' "$RL_RUN_DIR" > "$record/formal-run-path.txt"
  bash "$MIMICLITE_ROOT/local/mjlab/train_local.sh"
fi
