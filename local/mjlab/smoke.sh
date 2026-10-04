#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$MIMICLITE_ROOT"
# Tiny smoke: avoid costly Torch autotuning while sharing the L4.
export TORCH_COMPILE_DISABLE=1
mkdir -p local/mjlab/records
rm -f local/mjlab/records/verification.json
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
python local/mjlab/prepare_reference.py
python mimic-lite/scripts/train.py \
  backend=mjlab headless=true wandb.mode=disabled task=tracking-base '~task/motion' \
  task.num_envs=4 task.max_episode_length=32 total_iters=2 \
  algo.train_every=8 algo.ppo_epochs=1 algo.num_minibatches=1 \
  'algo.actor_hidden_dims=[64,64]' 'algo.critic_hidden_dims=[64,64]' \
  algo.compile=false \
  "task.command.motion_cfgs={synthetic:{path:$MIMICLITE_ROOT/local/mjlab/synthetic,weight:1.0,full_motion:true}}" \
  "hydra.run.dir=$MIMICLITE_ROOT/local/mjlab/records/train" \
  > local/mjlab/records/train.log 2>&1
python local/mjlab/verify_reload.py > local/mjlab/records/reload.log 2>&1
cat local/mjlab/records/verification.json
