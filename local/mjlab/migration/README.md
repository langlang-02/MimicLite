# digua_4090_48 migration

Develop as root in `/data/langlang/Project/MimicLite`. Data, checkpoints, runs,
archives, transfer staging and FK caches live under `/data/langlang` in separate
directories. Do not use the small root filesystem for Python or CUDA caches.

`receive_bos_dataset.py --base /data/langlang` downloads the existing BOS export
and validates both archive and per-file SHA256. It extracts only dataset entries.
`bootstrap.sh` recreates the pinned Python 3.12.13 environment; robot assets must
be restored into `.cache/huggingface` before offline imports.

Set `.env.local` with RVQ_DATA_ROOT, MIMICLITE_MOTION_CACHE_BASE and
MIMICLITE_RUNS_ROOT. This machine-specific file is ignored by Git.

Run `bash local/mjlab/migration/validate_target.sh` in a persistent Zellij session.
It waits for transfer verification, builds the 159777-motion FK cache (CPU by default, or an allocated GPU with
MIMICLITE_CACHE_BUILD_DEVICE=cuda),
waits for GPUs selected by MIMICLITE_GPU_IDS (default 4–7) to be unoccupied, and runs ten PPO iterations with 8192
environments per rank. A PASS requires finite metrics and changed actor weights.
Set MIMICLITE_START_FULL_TRAINING=1 to chain the 2500-iteration formal run after PASS.

After PASS, review `/data/langlang/transfer/validation/full-2500.txt` and launch:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PPO_OVERRIDES_FILE=/data/langlang/transfer/validation/full-2500.txt \
WANDB_MODE=offline bash local/mjlab/train_local.sh
```

The generated profile uses offline W&B until the target has the intended account
configured. For online logging, change `wandb.mode=online` in the profile and set
WANDB_MODE=online. Credentials are not included in Git or BOS archives.
The profile preserves the 512³ actor, no explicit XY feedback, root-position
failure as termination, 32-step rollouts, three PPO epochs and eight minibatches.
The entropy schedule accounts for the checkpoint's restored last_iter.

Historical experiment scripts in active-adaptation retain old paths for provenance;
use these portable entry points for new runs. FK caches are rebuilt because cache
identity includes paths and model metadata. Existing target projects are untouched.
