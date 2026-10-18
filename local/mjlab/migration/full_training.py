"""Create a portable full-data warm-start profile or verify a completed smoke run."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=8192)
    parser.add_argument('--iterations', type=int, default=2500)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='offline')
    parser.add_argument('--verify-run', type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    policy = checkpoint['policy']
    last_iter = int(policy.get('last_iter', 0))
    assert args.iterations > 0 and args.num_envs > 0 and args.workers > 0
    if args.verify_run:
        records = [json.loads(line) for line in (args.verify_run / 'metrics.jsonl').read_text().splitlines()]
        assert records and records[-1]['iteration'] == args.iterations
        assert all(not row['nonfinite_metrics'] for row in records)
        assert records[-1]['env_frames'] == args.workers * args.num_envs * 32 * args.iterations
        final_path = (args.verify_run / 'checkpoint_latest.pt').resolve()
        initial = torch.load(final_path.with_name('checkpoint_0.pt'), map_location='cpu', weights_only=False)
        matched = 0
        for module in ['actor', 'critic', 'vecnorms']:
            assert policy[module].keys() == initial['policy'][module].keys()
            for key, value in policy[module].items():
                assert torch.equal(value, initial['policy'][module][key]), (module, key)
                matched += 1
        final = torch.load(final_path, map_location='cpu', weights_only=False)
        changed = [name for name, value in policy['actor'].items()
                   if not torch.equal(value, final['policy']['actor'][name])]
        assert changed, 'Actor weights did not change'
        assert int(final['cfg'].task.num_envs) == args.num_envs
        assert final['cfg'].task.command.motion_cfgs.rvq.filenames_path == 'flat_ground_v1/train.txt'
        assert not final['cfg'].algo.actor_extra_in_keys
        assert int(final['policy']['last_iter']) == last_iter + args.iterations - 1
        report = dict(result='PASS', iterations=args.iterations, num_envs_per_rank=args.num_envs,
                      changed_actor_tensors=len(changed), checkpoint=str(args.checkpoint),
                      matched_warmstart_tensors=matched, workers=args.workers,
                      scope='full-data loading and finite PPO updates; not convergence')
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
        return
    template = Path(__file__).resolve().parents[1] / 'ppo_local_a800_double.txt'
    values = dict(line.split('=', 1) for line in template.read_text().splitlines()
                  if line and not line.startswith('#'))
    values.update({
        'task.num_envs': str(args.num_envs), 'total_iters': str(args.iterations),
        'total_frames': str(args.workers * args.num_envs * 32 * args.iterations),
        'checkpoint_path': str(args.checkpoint.resolve()),
        'algo.entropy_coef_start': '0.01', 'algo.entropy_coef_end': '0.002',
        'algo.entropy_decay_start': str(last_iter + max(0, args.iterations * 3 // 4 - 1)),
        'algo.entropy_decay_end': str(last_iter + max(1, args.iterations - 1)),
        'algo.actor_extra_in_keys': '[]', 'algo.in_keys': '[command,policy,priv]',
        'task.termination.root_pos_error.is_timeout': 'false',
        'task.command.motion_cfgs.rvq.filenames_path': 'flat_ground_v1/train.txt',
        'wandb.mode': args.wandb_mode, 'exp_name': 'latent-full-no-xy-4090',
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(f'{key}={value}\n' for key, value in values.items()))
    print(f'Wrote {args.output}; checkpoint last_iter={last_iter}')


if __name__ == '__main__':
    main()
