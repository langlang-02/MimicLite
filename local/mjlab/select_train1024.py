"""Select a reproducible 1024-motion curriculum, retaining the frozen 128 prefix."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
from motion_index import read_index, sha256, source_group, validate_motions

QUOTA = dict(forward=64, backward=64, sideways=128, turn=160,
             start_stop=128, stationary=128, dance=192, run_jump=96, crouch_exercise=64)
PATTERNS = dict(
    forward=r'^(Loop_Forward_Walk|walk_forward_(loop|professional|normal))',
    backward=r'^walk_backward_loop', sideways=r'^walk_sideway_.*loop',
    turn=r'^(turn_walk_|walk_arc_(cw|ccw)_(loop|start))',
    start_stop=r'^walk_.*_(start|stop)_',
    stationary=r'^(wave_R_|dance_jazz_hands_|idle_turn_000_|idle_normal_|idle_standard_|idle_loop_|neutral_idle_loop_|idle_hands_on_back_loop_|clap_enthusiastic_|clap_forced_|salute_|looking_around_horizontally_)',
    dance=r'^(dance_|dancing_)', run_jump=r'^(jog_|run_|jump_)',
    crouch_exercise=r'^(crouch_|exercise_|body_)')
EXCLUDE = re.compile(r'injured|limp|edge|stair|chair|wall|ladder|crutch|handstand|dog|handshake|heavy|carry|push|pull|box|sitting|kneel|crawl|weapon|sword|rifle|gun', re.I)
FIELDS = ['duration_s','mean_speed_mps','p90_speed_mps','mean_abs_yawrate_radps',
          'net_yaw_rad','height_range_m','mean_joint_std_rad','mean_forward_mps','mean_left_mps']


def category(sid):
    return next((k for k, pattern in PATTERNS.items() if re.search(pattern, sid)), None)


def features(root, motion):
    with np.load(root / motion, allow_pickle=False) as data:
        q = data['qpos']
    if q.ndim != 2 or q.shape[1] != 36 or len(q) < 2 or not np.isfinite(q).all():
        return None
    if not np.allclose(np.linalg.norm(q[:,3:7], axis=1), 1, atol=1e-3):
        return None
    w,x,y,z = q[:,3:7].T
    yaw = np.unwrap(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    delta = np.diff(q[:,:2], axis=0)
    c,s = np.cos(yaw[:-1]), np.sin(yaw[:-1])
    v = np.stack([c*delta[:,0]+s*delta[:,1], -s*delta[:,0]+c*delta[:,1]], axis=1)*20
    speed = np.linalg.norm(v, axis=1)
    rate = np.diff(yaw)*20
    return dict(duration_s=len(q)/20, path_m=float(np.linalg.norm(delta,axis=1).sum()),
        height_min_m=float(q[:,2].min()), height_max_m=float(q[:,2].max()),
        max_speed_mps=float(speed.max()), max_yawrate_radps=float(np.abs(rate).max()),
        max_joint_speed_radps=float(np.abs(np.diff(q[:,7:],axis=0)*20).max()),
        descriptor=[len(q)/20,float(speed.mean()),float(np.percentile(speed,90)),
            float(np.abs(rate).mean()),float(yaw[-1]-yaw[0]),float(np.ptp(q[:,2])),
            float(np.mean(np.std(q[:,7:],axis=0))),float(v[:,0].mean()),float(v[:,1].mean())])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'motion_sets/train1024_v1')
    args=parser.parse_args(); root=args.dataset; out=args.output
    if out.exists(): raise FileExistsError(out)
    seed=42
    previous=Path(__file__).parent/'motion_sets/train128_v1'
    anchors=read_index(previous/'train_128_v1.txt')
    old=json.loads((previous/'selection.json').read_text())
    train=read_index(root/'flat_ground_v1/train.txt')
    held={source_group(Path(m).stem) for split in ['val','test'] for m in read_index(root/f'flat_ground_v1/{split}.txt')}
    review_groups={source_group(Path(json.loads(line)['motion']).stem)
        for line in (root/'flat_ground_v1/review.jsonl').read_text().splitlines() if line.strip()}
    used={source_group(Path(m).stem) for m in anchors}
    assert len(used)==128 and not used & held and not used & review_groups
    selected=[]
    for motion, prior in zip(anchors,old['selected'],strict=True):
        assert motion == prior['motion']
        f=features(root,motion); assert f is not None
        selected.append(dict(motion=motion,category=prior['category'],retained_128=True,
                             source_group=source_group(Path(motion).stem),**f))
    pool={k:[] for k in QUOTA}; rejected=Counter(); pergroup=Counter()
    ordered=sorted(train,key=lambda m:hashlib.sha256(f'{seed}:{m}'.encode()).hexdigest())
    for motion in ordered:
        sid=Path(motion).stem; cat=category(sid); group=source_group(sid)
        if cat is None: rejected['outside_curriculum_categories']+=1; continue
        if group in held: rejected['heldout_source']+=1; continue
        if group in used: continue
        if group in review_groups: rejected['ambiguous_source']+=1; continue
        if EXCLUDE.search(sid): rejected['conservative_name_screen']+=1; continue
        if len(pool[cat])>=max(600,QUOTA[cat]*6) or pergroup[group]>=2: continue
        f=features(root,motion)
        if f is None: rejected['invalid_qpos']+=1; continue
        if not (2<=f['duration_s']<=15 and f['height_min_m']>=.45 and f['height_max_m']<=1.3
                and f['max_speed_mps']<=5 and f['max_yawrate_radps']<=10 and f['max_joint_speed_radps']<=60):
            rejected['kinematic_screen']+=1; continue
        dur,mean,p90,rate,yaw,height,joint,forward,left=f['descriptor']
        if cat in ['forward','backward','sideways','turn'] and mean<.15: continue
        if cat=='stationary' and (mean>.35 or f['path_m']>1.5): continue
        if cat=='forward' and (forward<.1 or abs(left)>abs(forward)): continue
        if cat=='backward' and (forward>-.1 or abs(left)>abs(forward)): continue
        if cat=='sideways' and abs(left)<.1: continue
        pool[cat].append(dict(motion=motion,category=cat,retained_128=False,source_group=group,**f))
        pergroup[group]+=1
    print('Candidates:',{k:len(v) for k,v in pool.items()},flush=True)
    for cat,target in QUOTA.items():
        candidates=pool[cat]; base=[r for r in selected if r['category']==cat]
        assert candidates,cat
        raw=np.array([r['descriptor'] for r in candidates]); low=np.percentile(raw,5,axis=0)
        scale=np.maximum(np.percentile(raw,95,axis=0)-low,1e-3)
        norm=lambda x:np.clip((np.array(x)-low)/scale,0,1)
        X=norm(raw); center=np.median(X,axis=0)
        chosen=[norm(r['descriptor']) for r in base]
        distance=np.min(np.stack([np.linalg.norm(X-v,axis=1) for v in chosen]),axis=0) if chosen else -np.linalg.norm(X-center,axis=1)
        while len(base)<target:
            valid=np.array([r['source_group'] not in used for r in candidates])
            if cat in ['sideways','turn']:
                dim=8 if cat=='sideways' else 4
                sign=1 if sum(r['descriptor'][dim]>0 for r in base)<target//2 else -1
                valid &= raw[:,dim]*sign>0
            assert valid.any(),(cat,'not enough distinct sources/directions',len(base),target)
            score=distance-.1*np.linalg.norm(X-center,axis=1); score[~valid]=-np.inf
            i=int(np.argmax(score)); row=candidates[i]
            selected.append(row);base.append(row);used.add(row['source_group'])
            d=np.linalg.norm(X-X[i],axis=1)
            distance=np.minimum(distance,d) if chosen else d
            chosen.append(X[i])
        print('Selected',cat,len(base),flush=True)
    motions=[r['motion'] for r in selected]
    assert len(motions)==len(used)==1024 and motions[:128]==anchors
    assert not used & held and not used & review_groups
    validated=validate_motions(root,motions,'train')
    out.mkdir(parents=True,exist_ok=False)
    index=out/'train_1024_v1.txt';index.write_text('\n'.join(motions)+'\n')
    with (out/'annotations.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['id','set','sample_id','motion','split','category','duration_s','source_group','notes'])
        writer.writeheader()
        for i,row in enumerate(selected,1):
            writer.writerow(dict(id=f'X{i:04}',set='train_1024_v1',sample_id=Path(row['motion']).stem,
                motion=row['motion'],split='train',category=row['category'],duration_s=row['duration_s'],source_group=row['source_group'],
                notes=f'Retained train128 row {i}.' if row['retained_128'] else 'Name/kinematic screening; not visually or physically certified.'))
    report=dict(status='passed',seed=seed,quota=QUOTA,counts=dict(Counter(r['category'] for r in selected)),
        candidate_counts={k:len(v) for k,v in pool.items()},rejected=dict(rejected),
        source_disjoint_from_entire_flat_val_test=True,ambiguous_review_sources_excluded=True,
        unique_source_groups=len(used),retained_prefix_count=128,
        duration_total_s=sum(r['duration_s'] for r in selected),
        duration_range_s=[min(r['duration_s'] for r in selected),max(r['duration_s'] for r in selected)],
        index_sha256=sha256(index),generator_sha256=sha256(__file__),anchors_sha256=sha256(previous/'train_128_v1.txt'),
        dataset_manifest_sha256=sha256(root/'manifest.json'),
        split_hashes={s:sha256(root/f'flat_ground_v1/{s}.txt') for s in ['train','val','test']},
        descriptor_fields=FIELDS,selected=selected,validated=validated)
    (out/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'README.md').write_text('''# Train1024 v1

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
''')
    print(json.dumps({k:report[k] for k in ['status','counts','unique_source_groups','duration_total_s','duration_range_s','index_sha256']},indent=2),flush=True)

if __name__=='__main__': main()
