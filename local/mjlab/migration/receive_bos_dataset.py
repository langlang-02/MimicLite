"""Receive and verify the existing chunked RVQ dataset export through bcecmd."""
import argparse, concurrent.futures, hashlib, json, subprocess, tarfile, time
from pathlib import Path, PurePosixPath

parser = argparse.ArgumentParser()
parser.add_argument('--base', type=Path, required=True)
parser.add_argument('--prefix', default='bos:/langlang/mimiclite/migration-20261006/')
a = parser.parse_args()
staging = a.base / 'transfer/rvq-dataset-20261006'
staging.mkdir(parents=True, exist_ok=True)
def digest(path):
 h = hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
 return h.hexdigest()
def download(name):
 dest = staging / name
 for attempt in range(5):
  with dest.with_suffix(dest.suffix+'.download.log').open('ab') as log:
   result = subprocess.run(['bcecmd','bos','cp',a.prefix.rstrip('/')+'/'+name,str(dest),'--disable-bar','--yes'],stdout=log,stderr=subprocess.STDOUT)
  if result.returncode == 0:
   break
  print('RETRY', name, 'attempt', attempt + 1, 'exit', result.returncode, flush=True)
  if attempt == 4:
   result.check_returncode()
  time.sleep(min(5 * 2**attempt, 60))
 return dest
def receive(item):
 meta = json.loads(download(item['archive'].replace('.tar','.json')).read_text())
 assert meta['sha256'] == item['sha256'] and meta['bytes'] == item['bytes']
 archive = staging / item['archive']
 if not archive.exists() or archive.stat().st_size != item['bytes'] or digest(archive) != item['sha256']:
  archive = download(item['archive'])
 assert archive.stat().st_size == item['bytes'] and digest(archive) == item['sha256']
 wanted = {f['path']: f for f in meta['files'] if f['path'].startswith('dataset/mimiclite-rvq-tracking/rot6d68_ep499_v1/')}
 with tarfile.open(archive) as tar:
  for member in tar:
   if member.name not in wanted: continue
   rel = PurePosixPath(member.name)
   assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
   dest = a.base / member.name
   expected = wanted[member.name]
   if dest.exists():
    assert dest.stat().st_size == expected['bytes'] and digest(dest) == expected['sha256'], dest
   else:
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tar.extractfile(member) as src, dest.with_suffix(dest.suffix+'.partial').open('wb') as out:
     for b in iter(lambda: src.read(8*1024*1024), b''): out.write(b)
    dest.with_suffix(dest.suffix+'.partial').replace(dest)
   assert dest.stat().st_size == expected['bytes'] and digest(dest) == expected['sha256'], dest
 print('VERIFIED',item['archive'],len(wanted),flush=True)
 return len(wanted)
complete=json.loads(download('COMPLETE.json').read_text())
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 counts=list(pool.map(receive,complete['archives']))
report=dict(result='PASS',files=sum(counts),archives=len(counts),prefix=a.prefix,checks='archive and extracted file SHA256')
(staging/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
