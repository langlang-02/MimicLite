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
It waits for transfer verification, builds the 159777-motion FK cache on CPU,
waits for GPUs 0–3 to be unoccupied, and runs ten PPO iterations with 8192
environments per rank. A PASS requires finite metrics and changed actor weights.
It does not start the 2500-iteration formal run automatically.

After PASS, review `/data/langlang/transfer/validation/full-2500.txt` and launch:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
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
