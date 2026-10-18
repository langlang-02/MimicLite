"""Build a reproducible, source-disjoint 128-motion curriculum from flat train."""
import argparse,csv,hashlib,json,re
from pathlib import Path
from collections import Counter
import numpy as np
from motion_index import source_group,read_index,validate_motions,sha256

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset',type=Path,required=True)
parser.add_argument('--output',type=Path,default=Path(__file__).parent/'motion_sets/train128_v1')
a=parser.parse_args();p=a.dataset;out=a.output
quota=dict(forward=24,backward=16,sideways=24,turn=24,start_stop=16,stationary=24)
patterns=dict(forward=r'^(Loop_Forward_Walk|walk_forward_(loop|professional|normal))',backward=r'^walk_backward_loop',sideways=r'^walk_sideway_.*loop',turn=r'^(turn_walk_|walk_arc_(cw|ccw)_(loop|start))',start_stop=r'^walk_.*_(start|stop)_',stationary=r'^(wave_R_|dance_jazz_hands_|idle_turn_000_|idle_normal_|idle_standard_|idle_loop_|neutral_idle_loop_|idle_hands_on_back_loop_|clap_enthusiastic_|clap_forced_|salute_|looking_around_horizontally_)')
def category(s):
 for k,pat in patterns.items():
  if re.search(pat,s):return k
 return None
seed=42
train=read_index(p/'flat_ground_v1/train.txt')
held=set()
for split in ['val','test']:
 held.update(source_group(Path(x).stem) for x in read_index(p/f'flat_ground_v1/{split}.txt'))
review={json.loads(x)['motion'] for x in (p/'flat_ground_v1/review.jsonl').read_text().splitlines() if x.strip()}
anchors=read_index(Path(__file__).parent/'motion_sets/train_small_v1.txt')
used={source_group(Path(x).stem) for x in anchors}
assert not used & held,'Original 12 overlap held-out sources'
assert len(used)==12

def features(m, screen=True):
 with np.load(p/m) as d:q=d['qpos']
 if q.ndim!=2 or q.shape[1]!=36 or not np.isfinite(q).all():return None
 n=len(q);dur=n/20
 if screen and (not 2<=dur<=10 or q[:,2].min()<.50 or q[:,2].max()>1.05):return None
 w,x,y,z=q[:,3:7].T;yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
 delta=np.diff(q[:,:2],axis=0);c=np.cos(yaw[:-1]);s=np.sin(yaw[:-1]);v=np.stack([c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1]],axis=1)*20
 speed=np.linalg.norm(v,axis=1);rate=np.diff(yaw)*20
 if screen and (speed.max()>3.5 or np.abs(rate).max()>8 or np.abs(np.diff(q[:,7:],axis=0)*20).max()>45):return None
 f=[dur,float(speed.mean()),float(np.percentile(speed,90)),float(np.mean(np.abs(rate))),float(yaw[-1]-yaw[0]),float(np.ptp(q[:,2])),float(np.mean(np.std(q[:,7:],axis=0))),float(np.mean(v[:,0])),float(np.mean(v[:,1]))]
 return dict(duration_s=dur,path_m=float(np.linalg.norm(delta,axis=1).sum()),height_min_m=float(q[:,2].min()),height_max_m=float(q[:,2].max()),descriptor=f)

selected=[]
for m in anchors:
 f=features(m, screen=False);assert f is not None
 selected.append(dict(motion=m,category=category(Path(m).stem),anchor=True,source_group=source_group(Path(m).stem),**f))
assert all(r['category'] for r in selected)
candidates={k:[] for k in quota};rejected=Counter()
# Sample candidates by stable seeded hash, not file order or policy success.
ordered=sorted(train,key=lambda m:hashlib.sha256(f'{seed}:{m}'.encode()).hexdigest())
pergroup=Counter()
for m in ordered:
 sid=Path(m).stem;cat=category(sid);group=source_group(sid)
 if not cat:continue
 if ('_M_' in sid and cat != 'turn') or group in held or group in used or m in review:continue
 if len(candidates[cat])>=350 or pergroup[group]>=2:continue
 if re.search(r'injured|limp|edge|stair|chair|wall|ladder|crutch|handstand|jump|run|jog|dog|handshake|heavy|carry|push|pull|box|sitting|kneel|crawl',sid,re.I):continue
 f=features(m)
 if f is None:rejected['structural_or_kinematic']+=1;continue
 dur,mean,p90,turn,yaw,h,j,forward,left=f['descriptor']
 if cat in ['forward','backward','sideways','turn'] and mean<.15:rejected['near_static_locomotion']+=1;continue
 if cat=='stationary' and (mean>.35 or f['path_m']>1.5):rejected['moving_stationary']+=1;continue
 if cat=='forward' and (forward<.10 or abs(left)>abs(forward)):continue
 if cat=='backward' and (forward>-.10 or abs(left)>abs(forward)):continue
 if cat=='sideways' and abs(left)<.10:continue
 candidates[cat].append(dict(motion=m,category=cat,anchor=False,source_group=group,**f));pergroup[group]+=1

for cat,count in quota.items():
 pool=candidates[cat];base=[r for r in selected if r['category']==cat]
 assert len(pool)>=count-len(base),(cat,len(pool))
 X=np.array([r['descriptor'] for r in pool]);low=np.percentile(X,5,axis=0);scale=np.maximum(np.percentile(X,95,axis=0)-low,1e-3)
 norm=lambda x:np.clip((np.array(x)-low)/scale,0,1)
 X=norm(X);chosen=[norm(r['descriptor']) for r in base]
 # Cover dynamics and clip lengths while mildly preferring the candidate median.
 center=np.median(X,axis=0)
 while len(base)<count:
  valid=[i for i,r in enumerate(pool) if r['source_group'] not in used]
  assert valid,cat
  if cat in ['sideways', 'turn']:
   dimension = 8 if cat == 'sideways' else 4
   sign = 1 if sum(r['descriptor'][dimension] > 0 for r in base) < count//2 else -1
   balanced = [i for i in valid if pool[i]['descriptor'][dimension]*sign > 0]
   assert balanced, 'Insufficient direction-balanced candidates'
   valid = balanced
  scores=[(min(np.linalg.norm(X[i]-v) for v in chosen) if chosen else -np.linalg.norm(X[i]-center))-.10*np.linalg.norm(X[i]-center) for i in valid]
  i=valid[int(np.argmax(scores))];r=pool[i];selected.append(r);base.append(r);chosen.append(X[i]);used.add(r['source_group'])
print('Selected',dict(Counter(r['category'] for r in selected)),flush=True)
assert len(selected)==128 and len(used)==128
validated=validate_motions(p,[r['motion'] for r in selected],'train')
out.mkdir(parents=True,exist_ok=False)
index=out/'train_128_v1.txt';index.write_text('\n'.join(r['motion'] for r in selected)+'\n')
with (out/'annotations.csv').open('w') as stream:
 fields=['id','set','sample_id','motion','split','category','duration_s','source_group','notes']
 writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
 for i,r in enumerate(selected,1):
  writer.writerow(dict(id=f'X{i:03}',set='train_128_v1',sample_id=Path(r['motion']).stem,motion=r['motion'],split='train',category=r['category'],duration_s=r['duration_s'],source_group=r['source_group'],notes='Retained T%02d anchor.'%i if r['anchor'] else 'Kinematic screening and descriptor coverage; not yet visually/physically certified.'))
report=dict(status='passed',seed=seed,quota=quota,candidate_counts={k:len(v) for k,v in candidates.items()},rejected=dict(rejected),source_disjoint_from_entire_flat_val_test=True,index_sha256=sha256(index),anchors_sha256=sha256(Path(__file__).parent/'motion_sets/train_small_v1.txt'),dataset_manifest_sha256=sha256(p/'manifest.json'),split_hashes={s:sha256(p/f'flat_ground_v1/{s}.txt') for s in ['train','val','test']},descriptor_fields=['duration_s','mean_speed_mps','p90_speed_mps','mean_abs_yawrate_radps','net_yaw_rad','height_range_m','mean_joint_std_rad','mean_forward_mps','mean_left_mps'],selected=selected,validated=validated)
(out/'selection.json').write_text(json.dumps(report,indent=2)+'\n');print('Validated',index,flush=True)
