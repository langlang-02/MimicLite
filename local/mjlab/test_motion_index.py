"""CPU checks for index validation and batch dispatch; no simulation is launched."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from motion_index import read_index, validate_motions, source_group


class MotionIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'dataset'
        for d in ['motions', 'latent', 'flat_ground_v1']:
            (self.dataset / d).mkdir(parents=True)
        self.motions = ['motions/walk__A001_seg000_win000.npz', 'motions/wave__A002_seg000_win000.npz']
        for name in self.motions:
            q = np.zeros((8, 36), dtype=np.float32); q[:, 3] = 1
            np.savez(self.dataset / name, qpos=q)
        (self.dataset / 'flat_ground_v1/train.txt').write_text('\n'.join(self.motions) + '\n')
        (self.dataset / 'flat_ground_v1/val.txt').write_text('')
        p = self.dataset / 'latent'
        np.save(p / 'sample_ids.npy', [Path(x).stem for x in self.motions])
        np.save(p / 'token_offsets.npy', [0, 2, 4])
        np.save(p / 'latent.npy', np.zeros((4, 512), dtype=np.float32))
        (p / 'manifest.json').write_text(json.dumps(dict(code_dim=512, reference_fps=20, token_stride_frames=4,
            arrays=dict(sample_ids='sample_ids.npy', token_offsets='token_offsets.npy', latent='latent.npy'))))
        self.index = self.root / 'selection.txt'
        self.index.write_text('\n'.join(self.motions) + '\n')
        self.checkpoint = self.root / 'checkpoint.pt'; self.checkpoint.touch()

    def test_validation_and_invalid_data(self):
        self.assertEqual(len(validate_motions(self.dataset, read_index(self.index), 'train')), 2)
        with self.assertRaisesRegex(ValueError, 'not in val'):
            validate_motions(self.dataset, self.motions, 'val')
        self.index.write_text('\n'.join([self.motions[0]] * 2))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            read_index(self.index)
        np.save(self.dataset / 'latent/latent.npy', np.full((4, 512), np.nan))
        with self.assertRaisesRegex(ValueError, 'Invalid latent'):
            validate_motions(self.dataset, self.motions, 'train')
        self.assertEqual(source_group('walk__A001_M_seg000_win000'), 'walk__A001')

    def run_batch(self, failure=False):
        output = self.root / 'output'
        argv = ['evaluate_latent_motion.py', '--dataset', str(self.dataset), '--checkpoint', str(self.checkpoint),
                '--index-file', str(self.index), '--split', 'train', '--output', str(output),
                '--disable-root-terminations', '--disable-body-ori-termination']
        calls = []
        def fake_run(command, **kwargs):
            calls.append(command)
            self.assertIn('--disable-root-terminations', command)
            self.assertIn('--disable-body-ori-termination', command)
            self.assertEqual(command[command.index('--split') + 1], 'train')
            if failure and len(calls) == 2:
                return subprocess.CompletedProcess(command, 7)
            directory = Path(command[command.index('--output') + 1]); directory.mkdir()
            report = dict(motion=command[command.index('--motion') + 1], completion_fraction=1.0,
                          completed_without_tracking_termination=True, mean_errors={'root_position_error_m': 0.2})
            (directory / 'summary.json').write_text(json.dumps(report))
            return subprocess.CompletedProcess(command, 0)
        with patch.object(sys, 'argv', argv), patch('subprocess.run', side_effect=fake_run), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                runpy.run_path(str(Path(__file__).with_name('evaluate_latent_motion.py')), run_name='__main__')
        self.assertEqual(result.exception.code, 7 if failure else 0)
        result = json.loads((output / 'batch-summary.json').read_text())
        self.assertEqual([c[c.index('--motion') + 1] for c in calls], self.motions)
        return result

    def test_batch_dispatch(self):
        result = self.run_batch()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['aggregate']['motion_count'], 2)

    def test_batch_failure_preserves_results(self):
        result = self.run_batch(failure=True)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(result['results']), 1)
        self.assertEqual(result['failed_motion'], self.motions[1])


if __name__ == '__main__':
    unittest.main()
