"""Validate stable motion paths against split membership and RVQ sidecars."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_group(sample_id):
    return re.sub(r'_seg\d+_win\d+$', '', sample_id).removesuffix('_M')


def read_index(path):
    motions = [s.strip() for s in Path(path).read_text().splitlines() if s.strip()]
    if not motions or len(motions) != len(set(motions)):
        raise ValueError('Index must be nonempty and contain no duplicate paths')
    if len({Path(s).stem for s in motions}) != len(motions):
        raise ValueError('Index sample IDs must be unique')
    return motions


def validate_motions(dataset, motions, split):
    dataset = Path(dataset).resolve()
    if not motions or len(motions) != len(set(motions)):
        raise ValueError('Motion selection must be nonempty and unique')
    members = set((dataset / f'flat_ground_v1/{split}.txt').read_text().splitlines())
    manifest = json.loads((dataset / 'latent/manifest.json').read_text())
    arrays = manifest['arrays']
    ids = np.load(dataset / 'latent' / arrays['sample_ids'], allow_pickle=False)
    offsets = np.load(dataset / 'latent' / arrays['token_offsets'], mmap_mode='r')
    latent = np.load(dataset / 'latent' / arrays['latent'], mmap_mode='r')
    lookup = {str(s): i for i, s in enumerate(ids)}
    if len(lookup) != len(ids) or offsets.shape != (len(ids) + 1,):
        raise ValueError('Invalid sidecar IDs or offsets')
    if offsets[0] != 0 or offsets[-1] != len(latent) or np.any(np.diff(offsets) <= 0):
        raise ValueError('Invalid sidecar token ranges')
    if latent.ndim != 2 or latent.shape[1] != manifest['code_dim']:
        raise ValueError('Invalid latent shape')
    rows = []
    for motion in motions:
        relative = Path(motion)
        if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.npz':
            raise ValueError(f'Expected a relative npz path: {motion}')
        if motion not in members:
            raise ValueError(f'Motion not in {split} split: {motion}')
        path = (dataset / relative).resolve()
        if not path.is_relative_to(dataset):
            raise ValueError(f'Motion escapes dataset: {motion}')
        sid = path.stem
        if sid not in lookup:
            raise ValueError(f'Missing latent: {sid}')
        with np.load(path, allow_pickle=False) as archive:
            qpos = archive['qpos']
        if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 2 or not np.isfinite(qpos).all():
            raise ValueError(f'Invalid qpos: {motion}')
        i = lookup[sid]
        tokens = latent[offsets[i]:offsets[i + 1]]
        expected = (len(qpos) + manifest['token_stride_frames'] - 1) // manifest['token_stride_frames']
        if len(tokens) != expected or not np.isfinite(tokens).all():
            raise ValueError(f'Invalid latent values or token count: {motion}')
        if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=-1), 1, atol=1e-3):
            raise ValueError(f'Non-unit root quaternion: {motion}')
        rows.append(dict(motion=motion, sample_id=sid, split=split,
                         source_group=source_group(sid), frames=len(qpos),
                         duration_s=len(qpos) / manifest['reference_fps'], tokens=len(tokens),
                         motion_sha256=sha256(path),
                         latent_sha256=hashlib.sha256(tokens.tobytes()).hexdigest()))
    return rows
