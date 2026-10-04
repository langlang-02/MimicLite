# Source this file from any directory. All writable caches stay in MimicLite.
MIMICLITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MIMICLITE_ROOT
export UV_CACHE_DIR="$MIMICLITE_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$MIMICLITE_ROOT/.cache/python"
export XDG_CACHE_HOME="$MIMICLITE_ROOT/.cache"
export HF_HOME="$MIMICLITE_ROOT/.cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export WANDB_DIR="$MIMICLITE_ROOT/local/mjlab/records"
export WANDB_CONFIG_DIR="$MIMICLITE_ROOT/.cache/wandb"
export MUJOCO_GL=egl
export WARP_CACHE_PATH="$MIMICLITE_ROOT/.cache/warp"
export TORCHINDUCTOR_CACHE_DIR="$MIMICLITE_ROOT/.cache/torchinductor"
export CUDA_CACHE_PATH="$MIMICLITE_ROOT/.cache/cuda"
export MPLCONFIGDIR="$MIMICLITE_ROOT/.cache/matplotlib"
export TMPDIR="$MIMICLITE_ROOT/.cache/tmp"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export ANY4HDMI_CACHE_BUILD_NUM_WORKERS="${ANY4HDMI_CACHE_BUILD_NUM_WORKERS:-0}"
export ANY4HDMI_CACHE_BUILD_BATCH_SIZE="${ANY4HDMI_CACHE_BUILD_BATCH_SIZE:-256}"
# Upstream stages 10 GiB by default before concatenating another copy.
export ANY4HDMI_CACHE_BUILD_WRITE_BUFFER_BYTES="${ANY4HDMI_CACHE_BUILD_WRITE_BUFFER_BYTES:-536870912}"
mkdir -p "$TMPDIR"
export PATH="$MIMICLITE_ROOT/local/mjlab/.venv/bin:$PATH"
