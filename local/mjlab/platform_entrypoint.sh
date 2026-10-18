#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$MIMICLITE_ROOT"
: "${RVQ_DATA_ROOT:?Set the absolute mounted dataset path}"
: "${RL_RUN_DIR:?Set a persistent unique run directory}"
: "${MIMICLITE_MOTION_CACHE_BASE:?Set a persistent motion cache directory}"
# A user-supplied file is mandatory; this entry point has no chosen PPO settings.
: "${PPO_OVERRIDES_FILE:?Supply the PPO overrides file after choosing training parameters}"
test -f "$RVQ_DATA_ROOT/conversion.json"
test -f "$RVQ_DATA_ROOT/flat_ground_v1/train.txt"
test -f "$PPO_OVERRIDES_FILE"
mkdir -p "$RL_RUN_DIR" "$MIMICLITE_MOTION_CACHE_BASE"
export TMPDIR="$RL_RUN_DIR/tmp"
export WANDB_DIR="$RL_RUN_DIR/wandb"
mkdir -p "$TMPDIR" "$WANDB_DIR"
# Editable projects must be discovered again at the image's actual path.
python - <<'PY'
import json
from pathlib import Path
import os
from active_adaptation.cli import aa_discover_projects
aa_discover_projects()
p = Path(os.environ['MIMICLITE_ROOT'])/'active-adaptation/.cache/projects.json'
d = json.loads(p.read_text())
for group in d.values():
    for name, project in group.items():
        project['enabled'] = name == 'mimic_lite'
p.write_text(json.dumps(d, indent=2)+'\n')
PY
overrides=()
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  overrides+=("$line")
done < "$PPO_OVERRIDES_FILE"
[[ ${#overrides[@]} -gt 0 ]] || { echo 'PPO overrides file is empty' >&2; exit 2; }
cp "$PPO_OVERRIDES_FILE" "$RL_RUN_DIR/ppo-overrides.txt"
# Single-node launch; each worker receives its own LOCAL_RANK/CUDA device.
workers="${MIMICLITE_NPROC_PER_NODE:-1}"
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid MIMICLITE_NPROC_PER_NODE' >&2; exit 2; }
launcher=(python)
if (( workers > 1 )); then
  launcher+=( -m torch.distributed.run --standalone "--nproc_per_node=$workers" )
fi
exec "${launcher[@]}" mimic-lite/scripts/train.py \
  backend=mjlab headless=true "wandb.mode=$WANDB_MODE" \
  task=tracking-base task/observation/command=rvq_latent '~task/motion' \
  "task.command.motion_cfgs={rvq:{path:$RVQ_DATA_ROOT,weight:1.0,full_motion:false,filenames_path:flat_ground_v1/train.txt}}" \
  "+task.command.rvq_latent_path=$RVQ_DATA_ROOT/latent" \
  "hydra.run.dir=$RL_RUN_DIR" "${overrides[@]}"
