#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$MIMICLITE_ROOT"
git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init active-adaptation mimic-lite
for module in active-adaptation mimic-lite; do
  expected_commit="$(git rev-parse "HEAD:$module")"
  [[ "$(git -C "$module" rev-parse HEAD)" == "$expected_commit" ]]
done
uv venv --python 3.12.13 --allow-existing local/mjlab/.venv
uv pip sync --python local/mjlab/.venv/bin/python local/mjlab/requirements.lock
# Upstream console entry point returns a dict, which sys.exit treats as failure.
python -c "from active_adaptation.cli import aa_discover_projects; aa_discover_projects()"
python - <<'PY'
import json
from pathlib import Path
import os
p = Path(os.environ['MIMICLITE_ROOT']) / 'active-adaptation/.cache/projects.json'
data = json.loads(p.read_text())
for group in data.values():
    for name, project in group.items():
        project['enabled'] = name == 'mimic_lite'
p.write_text(json.dumps(data, indent=2) + '\n')
PY
aa-list-tasks
