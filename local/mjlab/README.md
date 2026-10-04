# Local mjlab environment

Scope: MimicLite only; no base Python edits, no momask edits, no remote jobs.
Training submodules remain detached at the integration gitlinks:

- integration: `3963976de8778d9292305fc8efacbcae79ed6685`
- active-adaptation: `faea67554b318798f5e8b1ca4f4d0f3d28df12e6`
- mimic-lite: `089d0ad07619e47a6ce61393dcd69037d6aa32c9`

## Reproduce

Run from `/DRoboticsAILab/zhaohui.yang/MimicLite`:

```bash
bash local/mjlab/bootstrap.sh
source local/mjlab/env.sh
HF_HUB_OFFLINE=0 python local/mjlab/fetch_assets.py
bash local/mjlab/smoke.sh
```

`uv` must already be available. Python is downloaded into `.cache/python`;
interpreter: `local/mjlab/.venv/bin/python` (CPython 3.12.13).
`env.sh` sets all runtime caches inside this checkout and disables W&B and HF
network access during training. `fetch_assets.py` is the explicit network step.
Both G1 and Atom robot assets are needed because the pinned package eagerly
imports both asset modules. `assets.lock.json` pins their HF revisions.
No reference motion dataset is downloaded: `prepare_reference.py` generates a
256-frame, 50 Hz G1 standing clip in the real any4hdmi qpos manifest format.
This verifies plumbing, not motion tracking quality.

`requirements.lock` records all resolved dependency versions and Git commits.
It uses editable local training repos and pinned Git sources for `mjhub` and
`any4hdmi`. The original mjlab manifest requires Python 3.12 despite the framework
README's older 3.11 guidance. The package extra pins mjlab 1.3.0, Torch 2.10.0,
and Warp 1.12.0; `mujoco<3.8` follows the MimicLite compatibility note.

any4hdmi is pinned to `bc83d6c96935028c5448e6f2c7bb918021e26790`, the last commit
before its DatasetIndex API removal. Current main is incompatible with this
MimicLite gitlink. The smoke deletes the default Hydra motion config group
(`~task/motion`) before setting the local synthetic reference.

The upstream uv project source for any4hdmi assumes a different sibling layout.
To avoid modifying either submodule or writing outside this tree, installation
uses `uv pip` with `--no-sources` when generating the lock. Do not run `uv sync`
in the original submodule expecting it to use this environment.

To intentionally regenerate the dependency lock (not needed for bootstrap):

```bash
source local/mjlab/env.sh
uv pip compile --python local/mjlab/.venv/bin/python --no-sources \
  local/mjlab/requirements.in -o local/mjlab/requirements.lock
```

## Smoke scope and artifacts

`smoke.sh` runs the unmodified upstream PPO training entry point with four GPU
environments, two iterations, eight steps per rollout, one PPO epoch, one
minibatch, and 64x64 actor/critic networks. It saves initial/final checkpoints.
`verify_reload.py` checks that actor weights changed, verifies exact checkpoint
reload (including rejecting upstream loader warning-only failures), resets the
real tracking environment, and checks all floating tensors for finiteness for
16 policy steps, followed by another reset.

Logs and artifacts are in `records/`; `verification.json` is only written after
all assertions pass. `synthetic/`, `records/`, `.venv/`, and root `.cache/` are
ignored. Re-running the smoke uses the same output directory and replaces logs;
archive it first if retaining several trials.

The upstream PPO checkpoint contains module state and `last_iter`; it does not
save optimizer state. Reload is suitable for policy verification/warm start,
not evidence of bit-exact interrupted-training resume.

## Verified on 2026-09-30

- NVIDIA L4; Python 3.12.13; Torch 2.10.0 + CUDA 12.8; mjlab 1.3.0;
  MuJoCo 3.7.0; mujoco-warp 3.7.0.1; Warp 1.12.0.
- Bootstrap rerun and task discovery succeeded; `uv pip check` passed for all
  174 installed packages. `records/versions.json` records exact Git dependencies.
- The actual upstream training entry point completed 2 PPO iterations on
  4 GPU environments. 11 actor parameter tensors changed.
- Checkpoint reload matched all saved module tensors. Reset, 16 deterministic
  policy simulation steps, and another reset passed finite-value checks.
- `records/verification.json` is the machine-readable PASS report;
  `records/train.log` and `records/reload.log` contain runtime evidence.
- `verify_reload.py` copies both smoke checkpoints from W&B's disabled-mode
  temporary directory into `records/checkpoints/` before loading the final one.
  The final artifact is `records/checkpoints/checkpoint_2.pt` (750,343 bytes).
- Both training submodules remain clean at the pinned commits. No RL algorithms
  or observations were changed. This standing-clip smoke verifies the runtime,
  not learned tracking quality or latent conditioning.

## RVQ latent integration

The completed RVQ export is at
`/EFS/users/zhaohui.yang/dataset/momask-rvq-tracking/rot6d68_ep499_v1`:
187,777 feature-bank windows, 93 shards, 20 Hz original references, 5 Hz
512D RVQ conditions. Its export/validation code and documentation are in
`../momask-codes/dataset/bones-seed_prepare/g1_preprocess_rvq.py` and
`../momask-codes/docs/g1_rvq_tracking_preprocess.md`.

The any4hdmi conversion and latent observation integration are now implemented.
The converted dataset is at
`/EFS/users/zhaohui.yang/dataset/mimiclite-rvq-tracking/rot6d68_ep499_v1`.
See [LATENT_RL.md](LATENT_RL.md) for data locations, reset/time alignment,
verification commands, memory handling and the platform handover procedure.
The clean-submodule statements above describe the original standing smoke only;
the current mimic-lite working tree contains the latent integration changes.

For Volcano training, reproduce this isolated environment from the locked
requirements and pinned Git/assets in the training image. Mount the prepared
RVQ export and a persistent run directory explicitly; development-machine
mount paths are not automatically training-container paths. Checkpoint
optimizer/resume support must be addressed before treating long cloud runs as
fully resumable. No cloud task has been submitted by this setup.
