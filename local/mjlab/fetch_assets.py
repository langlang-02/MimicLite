"""Explicit network refresh of the two small robot repositories imported by MimicLite."""
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(os.environ['MIMICLITE_ROOT'])
for repo, revision in json.loads((root / 'local/mjlab/assets.lock.json').read_text()).items():
    path = Path(snapshot_download(repo_id=repo, revision=revision, max_workers=4))
    # Upstream configs request @main at import time. Point the offline cache ref
    # at our recorded revision, so subsequent smoke runs cannot drift.
    (path.parent.parent / 'refs').mkdir(exist_ok=True)
    (path.parent.parent / 'refs/main').write_text(revision)
    print(repo, revision, path, flush=True)
