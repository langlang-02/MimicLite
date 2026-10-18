# Train1024 v1

1024 relative motion paths; first 128 exactly retain train128 order, including original 12 anchors.
One window per mirror-canonicalized original source. Disjoint from all flat-ground val/test sources.
Ambiguous review sources excluded. Existing reference and latent reused; no data copy or re-encoding.

Quotas: forward 64, backward 64, sideways 128, turn 160, start_stop 128,
stationary 128, dance 192, run_jump 96, crouch_exercise 64.
Sideways and turn directions are balanced by signed lateral velocity / net yaw.
Labels are filename-based with kinematic checks; not manual semantic certification.

Seed 42. Candidate caps max(600, 6 * category quota); at most two candidate windows per source.
Selection covers robust-scaled duration, speed, yaw, height, joint variation and direction descriptors,
with mild preference for the candidate median. Not population-proportional or policy-success selection.
New candidates: duration 2–15 s, pelvis height 0.45–1.30 m, peak XY speed <=5 m/s,
peak yaw rate <=10 rad/s, peak joint speed <=60 rad/s. Original128 retained without re-screening.
Compared with train128, this intentionally broadens dynamics and includes dance/run/jump/body exercises.
Conservative interaction-name exclusions supplement the existing flat-ground filter.
These checks do not certify physical feasibility or absence of all environmental interactions.

Training override (do not change dataset root or latent path):
`task.command.motion_cfgs.rvq.filenames_path=/pfs/pfs-oHNwH0/langlang/home/Project/MimicLite/local/mjlab/motion_sets/train1024_v1/train_1024_v1.txt`

Reproduce into a NEW directory:
`local/mjlab/.venv/bin/python local/mjlab/select_train1024.py --dataset DATASET_ROOT --output NEW_DIRECTORY`

selection.json records source/split isolation, checksums, per-motion descriptors and qpos/latent validation.
annotations.csv records ordered labels and category. No training was started by dataset selection.
