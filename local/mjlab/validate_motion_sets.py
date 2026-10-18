"""Validate the frozen diagnostic motion sets and record their identities."""
import argparse
import csv
import json
from pathlib import Path

from motion_index import read_index, validate_motions, sha256

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', type=Path, required=True)
parser.add_argument('--sets-dir', type=Path, default=Path(__file__).parent / 'motion_sets')
parser.add_argument('--report', type=Path)
a = parser.parse_args()
annotations_path = a.sets_dir / 'annotations.csv'
with annotations_path.open() as stream:
    annotations = list(csv.DictReader(stream))
lookup = {(r['set'], r['motion']): r for r in annotations}
if len(lookup) != len(annotations):
    raise ValueError('Duplicate annotation rows')
report = dict(dataset=str(a.dataset.resolve()), annotations_sha256=sha256(annotations_path), sets={})
groups = []
used = set()
for name, split, count in [('eval_fixed_v1', 'val', 20), ('train_small_v1', 'train', 12)]:
    path = a.sets_dir / f'{name}.txt'
    motions = read_index(path)
    if len(motions) != count:
        raise ValueError(f'Expected {count} motions in {name}')
    rows = validate_motions(a.dataset, motions, split)
    for r in rows:
        key = (name, r['motion']); used.add(key)
        meta = lookup[key]
        for field in ['sample_id', 'split', 'source_group']:
            if meta[field] != r[field]:
                raise ValueError(f'Annotation mismatch: {key}: {field}')
        if abs(float(meta['duration_s']) - r['duration_s']) > 1e-6:
            raise ValueError(f'Duration mismatch: {key}')
        r.update(category=meta['category'], notes=meta['notes'], id=meta['id'])
    groups.append({r['source_group'] for r in rows})
    report['sets'][name] = dict(index_sha256=sha256(path), split_sha256=sha256(a.dataset / f'flat_ground_v1/{split}.txt'), motions=rows)
if used != set(lookup):
    raise ValueError('Unexpected annotation rows')
if groups[0] & groups[1]:
    raise ValueError(f'Source overlap between selected sets: {groups[0] & groups[1]}')
report.update(status='passed', selected_sets_source_disjoint=True,
              scope='Structural validation only; not physical trackability certification',
              dataset_manifest_sha256=sha256(a.dataset / 'manifest.json'),
              latent_manifest_sha256=sha256(a.dataset / 'latent/manifest.json'))
if a.report:
    a.report.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'eval_count': 20, 'train_count': 12, 'source_overlap': []}))
