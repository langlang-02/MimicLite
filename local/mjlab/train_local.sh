#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
[[ ! -f "$MIMICLITE_ROOT/local/mjlab/.env.local" ]] || source "$MIMICLITE_ROOT/local/mjlab/.env.local"
cd "$MIMICLITE_ROOT"

# Select allocated GPUs explicitly; do not accidentally use every visible GPU.
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the allocated GPU IDs}"
IFS=',' read -r -a selected_gpus <<< "$CUDA_VISIBLE_DEVICES"
export MIMICLITE_NPROC_PER_NODE="${MIMICLITE_NPROC_PER_NODE:-${#selected_gpus[@]}}"
# Match the validated local runtime; enable compilation only after benchmarking.
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export PPO_OVERRIDES_FILE="${PPO_OVERRIDES_FILE:-$MIMICLITE_ROOT/local/mjlab/ppo_local_a800.txt}"
export RL_RUN_DIR="${RL_RUN_DIR:-${MIMICLITE_RUNS_ROOT:-$MIMICLITE_ROOT/local/mjlab/records}/$(date +%Y%m%d-%H%M%S-%N)}"
mkdir -p "$RL_RUN_DIR"
bash "$MIMICLITE_ROOT/local/mjlab/platform_entrypoint.sh" 2>&1 | tee "$RL_RUN_DIR/console.log"
