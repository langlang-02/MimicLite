"""Evaluate selected motions without auto-reset; record reference/policy videos."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from motion_index import read_index, validate_motions, sha256
from tracking_diagnostics import TrackingDiagnostics

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch
from omegaconf import OmegaConf
import active_adaptation as aa

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', type=Path, required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
selection = parser.add_mutually_exclusive_group(required=True)
selection.add_argument('--motion', help='Relative motion path in the selected split')
selection.add_argument('--index-file', type=Path, help='TXT with one relative motion path per line')
parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
parser.add_argument('--validate-only', action='store_true', help='Validate selected data without loading policy or simulation')
parser.add_argument('--annotations', type=Path, help='Optional category CSV; defaults to annotations.csv beside the index')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--disable-root-terminations', action='store_true',
                    help='Disable root position/orientation termination for this evaluation only')
parser.add_argument('--disable-body-ori-termination', action='store_true',
                    help='Disable local body orientation termination for this evaluation only')
a = parser.parse_args()
a.dataset = a.dataset.resolve()
a.output = a.output.resolve()
motions = read_index(a.index_file) if a.index_file else [a.motion]
validated = validate_motions(a.dataset, motions, a.split)
categories = {}
annotations = a.annotations or (a.index_file.parent / 'annotations.csv' if a.index_file else None)
if annotations and (annotations.exists() or a.annotations):
    with annotations.open() as stream:
        for row in csv.DictReader(stream):
            if row['motion'] in motions and row['split'] == a.split:
                if row['motion'] in categories:
                    parser.error(f'Duplicate annotation: {row["motion"]}')
                categories[row['motion']] = row['category']
    if set(categories) != set(motions):
        parser.error('Annotations must cover every selected motion in the selected split')
if a.validate_only:
    print(json.dumps(validated, indent=2))
    raise SystemExit(0)
if not a.checkpoint.is_file():
    parser.error(f'Checkpoint does not exist: {a.checkpoint}')
if a.output.exists() and any(a.output.iterdir()):
    parser.error(f'Output must be new or empty: {a.output}')
a.output.mkdir(parents=True, exist_ok=True)
if a.index_file:
    manifest = dict(index_file=str(a.index_file.resolve()), index_sha256=sha256(a.index_file),
                    checkpoint=str(a.checkpoint.resolve()), split=a.split, motions=validated,
                    annotations_sha256=sha256(annotations) if categories else None,
                    categories=categories,
                    status='running', results=[])
    manifest_path = a.output / 'batch-summary.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    for motion in motions:
        output = a.output / Path(motion).stem
        command = [sys.executable, str(Path(__file__).resolve()), '--dataset', str(a.dataset),
                   '--checkpoint', str(a.checkpoint.resolve()), '--motion', motion,
                   '--split', a.split, '--output', str(output)]
        for enabled, flag in [(a.disable_root_terminations, '--disable-root-terminations'),
                              (a.disable_body_ori_termination, '--disable-body-ori-termination')]:
            if enabled:
                command.append(flag)
        with (a.output / (Path(motion).stem + '.log')).open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            manifest.update(status='failed', failed_motion=motion, exit_code=result.returncode)
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
            raise SystemExit(result.returncode)
        manifest['results'].append(json.loads((output / 'summary.json').read_text()))
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    reports = manifest['results']
    manifest.update(status='completed', aggregate=dict(
        motion_count=len(reports),
        completion_fraction_mean=float(np.mean([r['completion_fraction'] for r in reports])),
        completed_under_selected_conditions_rate=float(np.mean([r['completed_without_tracking_termination'] for r in reports])),
        mean_errors={k: float(np.mean([r['mean_errors'][k] for r in reports])) for k in reports[0]['mean_errors']}))
    manifest['by_category'] = {}
    for category in sorted(set(categories.values())):
        subset = [r for r in reports if categories[r['motion']] == category]
        manifest['by_category'][category] = dict(
            motion_count=len(subset),
            completion_fraction_mean=float(np.mean([r['completion_fraction'] for r in subset])),
            completed_under_selected_conditions_rate=float(np.mean([r['completed_without_tracking_termination'] for r in subset])),
            mean_errors={k: float(np.mean([r['mean_errors'][k] for r in subset])) for k in subset[0]['mean_errors']})
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['aggregate'], indent=2))
    raise SystemExit(0)
saved = torch.load(a.checkpoint.resolve(), map_location='cpu', weights_only=False)
cfg = saved['cfg']
OmegaConf.set_struct(cfg, False)
cfg.checkpoint_path = str(a.checkpoint.resolve())
cfg.headless = True
cfg.task.num_envs = 1
cfg.task.max_episode_length = 10000
cfg.task.randomization = {}
disabled_terminations = []
if a.disable_root_terminations:
    disabled_terminations.extend(['root_pos_error', 'root_ori_error'])
if a.disable_body_ori_termination:
    disabled_terminations.append('body_ori_error')
for name in disabled_terminations:
    cfg.task.termination[name].enabled = False
cfg.task.command.start_from_zero = True
cfg.task.command.rewind_prob = 0.0
cfg.task.command.motion_cfgs = {'single': dict(path=str(a.dataset), weight=1.0, full_motion=True, filenames=[a.motion])}
cfg.task.command.rvq_latent_path = str(a.dataset / 'latent')
cfg.algo.compile = False
cfg.wandb.mode = 'disabled'
OmegaConf.save(cfg, a.output / 'config.yaml')
aa.init(cfg, auto_rank=True)
from active_adaptation.helpers import make_env_policy
from active_adaptation.learning.modules.vecnorm import VecNorm
from torchrl.envs.utils import set_exploration_type, ExplorationType

env, policy = make_env_policy(cfg)
assert policy.load_state_dict(saved['policy']) == [], 'Checkpoint load failed'
for name, state in saved['policy'].items():
    if isinstance(state, dict):
        for key, tensor in state.items():
            assert torch.equal(policy.state_dict()[name][key].cpu(), tensor), (name, key)
base = env.base_env
base.eval()
c = base.command_manager
assert len(c.dataset.motion_paths) == 1
assert Path(c.dataset.motion_paths[0]).stem == Path(a.motion).stem
model = base.sim.mj_model
model.vis.global_.offwidth = 640
model.vis.global_.offheight = 480
renderer = mujoco.Renderer(model, height=480, width=640)
actual_data, reference_data = mujoco.MjData(model), mujoco.MjData(model)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.azimuth, cam.elevation, cam.distance = 135, -18, 3.2
free_ids = np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
assert len(free_ids) == 1
root_adr = int(model.jnt_qposadr[free_ids[0]])
joints = {}
for j in range(model.njnt):
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j).split('/')[-1]
        joints[name] = int(model.jnt_qposadr[j])
assert set(c.asset.joint_names) <= set(joints)
store = c.rvq_latent_store
sidecar_id = list(map(str, store.sample_ids)).index(Path(a.motion).stem)
latent_offset = int(store.token_offsets[sidecar_id])
latent_count = int(store.token_offsets[sidecar_id + 1]) - latent_offset
rows, tokens, terminations = [], [], []
diagnostic_rows = []
ori_term = base.termination_funcs['body_ori_error']
diagnostics = TrackingDiagnostics(c.tracking_body_names, ori_term.body_indices_tracking, ori_term.threshold)
torso_index = c.tracking_body_names.index('torso_link')
trajectory, reference_trajectory = [], []
video = a.output / 'reference_vs_policy.mp4'
writer = imageio.get_writer(str(video), fps=25, codec='libx264', quality=8, macro_block_size=16)

def frame(step, row):
    actual_data.qpos[:] = base.sim.data.qpos[0].cpu().numpy()
    reference_data.qpos[:] = actual_data.qpos
    ref = c.current_ref_motion
    reference_data.qpos[root_adr:root_adr+3] = (ref.body_pos_w[0, c.root_body_idx_motion] + base.scene.env_origins[0]).cpu().numpy()
    reference_data.qpos[root_adr+3:root_adr+7] = ref.body_quat_w[0, c.root_body_idx_motion].cpu().numpy()
    joint_values = ref.joint_pos[0, c.asset_joint_idx_motion].cpu().numpy()
    for name, value in zip(c.asset.joint_names, joint_values):
        reference_data.qpos[joints[name]] = value
    trajectory.append(actual_data.qpos.copy())
    reference_trajectory.append(reference_data.qpos.copy())
    cam.lookat[:] = reference_data.qpos[root_adr:root_adr+3]
    cam.lookat[2] = 0.8
    panels = []
    for data in [reference_data, actual_data]:
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        panels.append(renderer.render().copy())
    canvas = Image.fromarray(np.concatenate(panels, axis=1))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0,0,1280,44),fill=(18,22,30))
    draw.text((12,10), 'REFERENCE (original motion)', fill='white')
    policy_label = 'POLICY (RVQ latent + proprioception'
    if 'actor_feedback' in cfg.algo.get('actor_extra_in_keys', []):
        policy_label += ' + XY feedback'
    draw.text((652,10), policy_label + ')', fill='white')
    draw.rectangle((0,444,1280,480),fill=(18,22,30))
    draw.text((12,453), f't={row["time_s"]:.2f}s  root error={row["root_position_error_m"]:.3f}m  joint MAE={row["joint_mae_rad"]:.3f}rad', fill='white')
    if step == 0: canvas.save(a.output / 'preview.png')
    writer.append_data(np.asarray(canvas))

try:
    with torch.inference_mode(), set_exploration_type(ExplorationType.DETERMINISTIC), VecNorm.freeze():
        carry = env.reset()
        rollout = policy.get_rollout_policy('eval')
        length = int(c.motion_len[0])
        initial_step = int(c.obs_motion_t[0])
        for step in range(length + 2):
            source_step = int(c.obs_motion_t[0])
            source_frame = source_step * 20 // 50
            token = min(source_frame // 4, latent_count - 1)
            expected = store.latent[latent_offset + token]
            np.testing.assert_array_equal(c.current_rvq_latent[0].cpu().numpy(), expected)
            expected_phase = (source_frame % 4) / 4 if source_frame // 4 < latent_count else 1.0
            assert float(c.current_rvq_phase[0,0]) == expected_phase
            # The upstream reset observation is zero; retain that behavior on step 0.
            if step > 0:
                np.testing.assert_array_equal(carry['command'][0,:512].cpu().numpy(), expected)
            tokens.append(token)
            transition = env.step(rollout(carry))
            next_td = transition['next']
            for key, value in next_td.items(True, True):
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    assert torch.isfinite(value).all(), key
            reference_root = c.current_ref_motion.body_pos_w[0, c.root_body_idx_motion] + base.scene.env_origins[0]
            row = dict(time_s=(step+1)*float(env.step_dt), latent_token=token,
                       root_position_error_m=float((c.robot_root_pos_w[0]-reference_root).norm()),
                       body_position_error_world_m=float(c.body_pos_error.mean()),
                       body_position_error_local_m=float(c.body_pos_error_local.mean()),
                       joint_mae_rad=float(c.joint_pos_error.mean()),
                       joint_rmse_rad=float(c.joint_pos_error.square().mean().sqrt()))
            rows.append(row)
            cpu = lambda tensor: tensor.detach().cpu().numpy()
            diagnostic_rows.append(diagnostics.record(
                row['time_s'], cpu(c.robot_root_pos_w[0]), cpu(reference_root),
                cpu(c.robot_root_quat_w[0]), cpu(c.current_ref_motion.body_quat_w[0, c.root_body_idx_motion]),
                cpu(c.body_ori_error_local[0]), cpu(c.body_pos_error[0]),
                cpu(c.body_pos_error_local[0]), torso_index))
            if step % 2 == 0: frame(step, row)
            if bool(next_td['done'].any()):
                for name in base.termination_funcs:
                    if bool(next_td['stats','termination',name].any()): terminations.append(name)
                break
            carry = next_td
finally:
    writer.close()
    renderer.close()
    env.close()

with (a.output/'metrics.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
with (a.output/'diagnostics.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=diagnostic_rows[0].keys());w.writeheader();w.writerows(diagnostic_rows)
(a.output/'diagnostics-metadata.json').write_text(json.dumps(dict(
    axes='reference pelvis yaw: forward +X, left +Y, up +Z; error=robot-reference',
    velocity='finite differences of root-link position and wrapped heading at control rate; first row blank',
    reference='same current_ref_motion as reward/termination, after each environment step',
    root='pelvis', root_position_termination_body=cfg.task.shared.termination_root_body_name,
    body_ori_threshold_rad=ori_term.threshold, body_ori_min_steps=ori_term.min_steps,
    body_ori_links=[c.tracking_body_names[i] for i in ori_term.body_indices_tracking]), indent=2)+'\n')
np.savez_compressed(a.output/'trajectory.npz',policy_qpos=np.asarray(trajectory),reference_qpos=np.asarray(reference_trajectory),fps=25)
complete = 'motion_timeout' in terminations and not any(t != 'motion_timeout' for t in terminations)
report = dict(checkpoint=str(a.checkpoint.resolve()),motion=a.motion,split=f'flat_ground_v1/{a.split}.txt',
              settings=dict(seed=int(cfg.seed),num_envs=1,deterministic=True,domain_randomization=False,auto_reset=False,reference_fps=20,latent_fps=5,control_fps=50,upstream_zero_first_observation=True),
              motion_steps_50hz=length,initial_source_step=initial_step,executed_steps=len(rows),
              elapsed_motion_seconds=len(rows)*float(env.step_dt),completion_fraction=min(1.0,len(rows)/max(1,length-initial_step)),
              completed_without_tracking_termination=complete,termination_reasons=terminations,
              disabled_terminations=disabled_terminations,
              latent_tokens_available=latent_count,latent_tokens_used=len(set(tokens)),latent_alignment='exact on every step',
              mean_errors={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k.endswith(('_m','_rad'))},
              video=str(video),scope=f'One {a.split}-split motion, one seed; not aggregate policy quality.')
(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
