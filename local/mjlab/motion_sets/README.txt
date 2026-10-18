Frozen diagnostic motion sets v1
===============================
eval_fixed_v1.txt: 20 motions from flat_ground_v1/val.txt.
train_small_v1.txt: 12 motions from flat_ground_v1/train.txt.
Each line is a path relative to RVQ_DATA_ROOT. Order is fixed. Numeric labels
E01..E20/T01..T12 are presentation labels; sample_id is the stable identity.
annotations.csv describes category, source group, duration and caveats.
validation.json records hashes and structural checks, not physical feasibility.
Same-source clips occur within eval; use per-motion and category results.
The two selected sets do not share source groups (including mirrored variants).
This does not certify that the entire historical train/val split is source-disjoint.

Representation: g1_local_rot6d68_v1, 20 Hz; 29 joint angles, 29 per-frame
joint differences, 6D root roll/pitch rotation, yaw increment, 2D XY increment
in current heading frame, and root height. No absolute XY or yaw input.
Normalize by recorded mean/std; six residual codebook embeddings (512D each)
are summed. Tokens are 5 Hz. RL command is 512D latent + phase.
Reference qpos is separate: absolute XYZ + wxyz quaternion + 29 joint angles.

Run from project root after:
  source local/mjlab/env.sh
  source local/mjlab/.env.local

Validate (no GPU simulation):
  python local/mjlab/validate_motion_sets.py --dataset "$RVQ_DATA_ROOT" --report local/mjlab/motion_sets/validation.json

Batch evaluation (when explicitly launching an evaluation):
  CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1 python local/mjlab/evaluate_latent_motion.py --dataset "$RVQ_DATA_ROOT" --checkpoint /path/to/checkpoint.pt --index-file local/mjlab/motion_sets/eval_fixed_v1.txt --split val --output /path/to/new-output
Append --validate-only to check data without loading checkpoint or simulation.
For training-set evaluation, use train_small_v1.txt and --split train.
For relaxed termination evaluation append both:
  --disable-root-terminations --disable-body-ori-termination
Default retains checkpoint termination settings. Always report which conditions
were disabled: finishing under relaxed conditions is not strict tracking success.

The batch runner runs motions sequentially, one child process per motion, using
inherited CUDA_VISIBLE_DEVICES. Each gets its own video/metrics/config/summary;
logs are in the batch root. batch-summary.json stores ordered results, macro
(per-motion) means and category means using the adjacent annotations.csv.
Use --annotations to supply another complete category CSV. No auto-resume or
output overwrite: use a fresh output directory. Failures stop the batch and
preserve completed results. Check GPU occupancy before starting simulations.

Small-set training integration (not launched by this change): replace the
filenames_path within task.command.motion_cfgs.rvq with the absolute path to
local/mjlab/motion_sets/train_small_v1.txt. Keep dataset root and latent path
unchanged. Do not train on eval_fixed_v1.txt. The training entrypoint's final
Hydra override can replace task.command.motion_cfgs with the selected index.
