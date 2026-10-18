Train128 v1: 128 motions, first 12 retained in original order.
Quotas: forward24 backward16 sideways24 turn24 start_stop16 stationary24.
TXT entries are relative to the original dataset root; no data copy or latent re-encoding.
Set task.command.motion_cfgs.rvq.filenames_path to the absolute train_128_v1.txt path.
Source groups are unique, mirror-canonicalized, disjoint from entire flat-ground val/test.
Sideways and turn groups each contain 12 positive and 12 negative signed directions. Turn candidates may use mirrored versions but never both versions of one source.
New motions: seeded candidate sampling, filename and kinematic filters, robust-scaled descriptor coverage with center preference; maximum 350 candidates per category, so this is a curriculum subset, not a population-proportional sample.
Original12 are retained even if outside new candidate kinematic screens.
No selection by policy success. Annotations are provisional; no per-motion visual or physical certification has been completed.
selection.json stores descriptors, source groups, checksums and qpos/latent checks. Generator: local/mjlab/select_train128.py --dataset DATASET --output NEW_DIRECTORY.
