"""Create a synthetic G1 standing clip; only for plumbing smoke tests."""
import json
import os
import re
from pathlib import Path

import mujoco
import numpy as np
from mjhub import resolve_asset_reference

root = Path(os.environ['MIMICLITE_ROOT'])
revision = json.loads((root / 'local/mjlab/assets.lock.json').read_text())['elijahgalahad/g1_xmls']
ref = f'hf://elijahgalahad/g1_xmls@{revision}/g1-mode_13_15.xml'
model = mujoco.MjModel.from_xml_path(str(resolve_asset_reference(ref)))
qpos = model.qpos0.copy()
qpos[:7] = [0, 0, 0.76, 1, 0, 0, 0]
pose = {
    '.*_hip_pitch_joint': -0.312, '.*_knee_joint': 0.669,
    '.*_ankle_pitch_joint': -0.363, '.*_elbow_joint': 0.6,
    'left_shoulder_roll_joint': 0.2, 'right_shoulder_roll_joint': -0.2,
    '.*_shoulder_pitch_joint': 0.2,
}
names = ['root_tx', 'root_ty', 'root_tz', 'root_qw', 'root_qx', 'root_qy', 'root_qz']
for j in range(model.njnt):
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
        continue
    name = model.joint(j).name
    names.append(name)
    for pattern, value in pose.items():
        if re.fullmatch(pattern, name):
            qpos[model.jnt_qposadr[j]] = value
assert len(names) == model.nq == 36
out = root / 'local/mjlab/synthetic'
(out / 'motions').mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / 'motions/standing.npz', qpos=np.tile(qpos.astype(np.float32), (256, 1)))
(out / 'manifest.json').write_text(json.dumps({
    'format_version': 2, 'dataset_name': 'synthetic-standing-smoke',
    'mjcf': ref, 'motions_subdir': 'motions', 'timestep': 0.02,
    'qpos_dim': model.nq, 'qpos_names': names, 'num_motions': 1,
    'source': {'synthetic': True, 'purpose': 'environment plumbing only'},
    'total_hours': 256 * 0.02 / 3600,
}, indent=2) + '\n')
print(f'Synthetic reference: {out}; frames=256 nq={model.nq}')
