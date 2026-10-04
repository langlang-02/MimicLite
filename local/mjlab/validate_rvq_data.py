"""Exhaustive comparison against RVQ export; does not modify datasets."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--dataset', type=Path, required=True)
p.add_argument('--report', type=Path, required=True)
a = p.parse_args()
t0 = time.time()
source = json.loads((a.source / 'manifest.json').read_text())
complete = json.loads((a.dataset / 'conversion.json').read_text())
assert source['complete'] and complete['complete']
assert complete['identity']['source_manifest_sha256'] == hashlib.sha256((a.source / 'manifest.json').read_bytes()).hexdigest()
ids = np.load(a.dataset / 'latent/sample_ids.npy', mmap_mode='r')
offsets = np.load(a.dataset / 'latent/token_offsets.npy', mmap_mode='r')
latent = np.load(a.dataset / 'latent/latent.npy', mmap_mode='r')
assert latent.shape == (source['tokens'], source['code_dim'])
assert offsets.shape == (source['samples'] + 1,) and offsets[0] == 0
assert offsets[-1] == len(latent) and np.all(np.diff(offsets) > 0)
assert len(set(ids.tolist())) == len(ids) == source['samples']
assert {x.stem for x in (a.dataset / 'motions').glob('*.npz')} == set(ids.tolist())
splits = {s: (a.dataset / f'{s}.txt').read_text().splitlines() for s in source['split_counts']}
assert sum(map(len, splits.values())) == len(ids)
assert len(set().union(*map(set, splits.values()))) == len(ids)
expected_splits = {s: [] for s in splits}
n = frames = tokens = 0
max_rotation_error = 0.0
for info in source['shards']:
    path = a.source / info['path']
    with path.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == info['sha256'], path
    with np.load(path) as z:
        sid, fo, to = z['sample_ids'], z['frame_offsets'], z['token_offsets']
        ref, lat = z['reference_native38'], z['latent']
        assert np.isfinite(lat).all()
        np.testing.assert_array_equal(ids[n:n+len(sid)], sid)
        np.testing.assert_array_equal(offsets[n:n+len(sid)+1], to + tokens)
        np.testing.assert_array_equal(latent[tokens:tokens+len(lat)], lat)
        for i, sample in enumerate(sid):
            expected_splits[info['split']].append(f'motions/{sample}.npz')
            r = ref[fo[i]:fo[i+1]]
            with np.load(a.dataset / 'motions' / f'{sample}.npz') as motion:
                q = motion['qpos']
            assert q.shape == (len(r), 36) and np.isfinite(q).all(), sample
            np.testing.assert_array_equal(q[:, :3], r[:, :3])
            np.testing.assert_array_equal(q[:, 7:], r[:, 9:])
            np.testing.assert_allclose(np.linalg.norm(q[:, 3:7], axis=1), 1, atol=2e-6)
            mat = Rotation.from_quat(q[:, [4, 5, 6, 3]]).as_matrix()
            columns = np.concatenate([mat[:, :, 0], mat[:, :, 1]], axis=-1)
            error = float(np.abs(columns-r[:, 3:9]).max())
            assert error < 3e-6, (sample, error)
            max_rotation_error = max(max_rotation_error, error)
            assert to[i+1]-to[i] == (len(r)+3)//4
        n += len(sid)
        frames += len(ref)
        tokens += len(lat)
    print(f'{info["path"]}: {n}/{source["samples"]} samples verified', flush=True)
assert splits == expected_splits
assert (n, frames, tokens) == (source['samples'], source['frames'], source['tokens'])
report = dict(result='PASS', samples=n, frames=frames, tokens=tokens,
              splits={s: len(v) for s, v in splits.items()},
              max_rotation_column_error=max_rotation_error,
              latent_comparison='all values exact', qpos_comparison='all frames',
              source=str(a.source), dataset=str(a.dataset), seconds=time.time()-t0)
a.report.parent.mkdir(parents=True, exist_ok=True)
a.report.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
