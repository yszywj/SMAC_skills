import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from decision_tree.scripts.linux import run_zerg_whole_ac as cli


class WholeEpisodeLauncherTest(unittest.TestCase):
    def test_exact_dry_run_cli(self):
        if Path(sys.executable).resolve() != Path(cli.RSI_PYTHON).resolve():
            self.skipTest(f'requires historical interpreter {cli.RSI_PYTHON}')
        from bridge.scenarios import validate_manifest
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()), patch.object(cli, 'HISTORY', Path(tmp)/'reservations'), \
                patch.object(cli.subprocess, 'run', side_effect=AssertionError('launched in dry-run')):
            out = Path(tmp)/'first'
            cli.main(['--out', str(out), '--dry-run'])
            design = cli.read_json(out/'design.json')
            self.assertEqual((design['workers'], design['per_roster'], len(design['pairs'])), (48, 10, 120))
            self.assertEqual(design['arms'], ['A', 'C'])
            self.assertTrue(design['exploratory_only'])
            rosters, seen = cli.originals()
            self.assertEqual(design['rosters'], rosters)
            for p in design['pairs']:
                for k in cli.KEYS:
                    self.assertNotIn(p[k], seen[k])
                    seen[k].add(p[k])
                m = cli.read_json(out/p['manifest'])
                validate_manifest(m)
                row, = m['episodes']
                cfg = row['episode_config']
                self.assertEqual((m['seed'], row['sc2_seed']), (p['seed'], p['seed']))
                self.assertEqual((m['sha256'], row['sha256']), (p['manifest_sha256'], p['scenario_sha256']))
                self.assertEqual(cli.digest([cfg[f'{s}_start_positions']['item'] for s in ('ally','enemy')]),
                                 p['layout_sha256'])
                for side in ('ally','enemy'):
                    self.assertEqual(cfg['team_gen'][f'{side}_team'], rosters[p['roster']])
                for arm in cli.ARMS:
                    self.assertEqual(list((out/p['pair_id']/arm).iterdir()), [])
            # History is shared even if the output root changes. Both archived
            # batches are independently excluded, including when no arm ran.
            other = Path(tmp)/'elsewhere'/'repeat'
            argv = ['--out', str(other), '--per-roster', '1', '--workers', '1', '--dry-run', '--seed']
            for bad in (design['seed'], 20261123, 20262123):
                with self.assertRaises(SystemExit):
                    cli.main(argv + [str(bad)])
                self.assertFalse(other.exists())
            cli.main(argv + ['11'])
            self.assertEqual(len(cli.read_json(other/'design.json')['pairs']), 12)

    def test_worker_uses_only_the_arm_and_shared_manifest(self):
        pair = dict(pair_id='pair-001', manifest='pair-001/manifest.json', roster='frozen-team')
        with tempfile.TemporaryDirectory() as tmp, patch.object(cli.subprocess, 'run') as run, \
                patch.dict(os.environ, {'ZERG_FORK_SPEC': 'bad', 'ZERG_MODEL': 'bad', 'PYTHONPATH': 'bad'}):
            root = Path(tmp)
            for arm in cli.ARMS:
                work = root/pair['pair_id']/arm
                work.mkdir(parents=True)
                cli.play(root, pair, arm)
                cmd, = run.call_args.args
                self.assertEqual(cmd[cmd.index('--manifest')+1], str(root/pair['manifest']))
                self.assertEqual(cmd[cmd.index('--output-dir')+1], str(work/'runner'))
                self.assertEqual(cmd[5:11], ['--method', 'adaptive-skill', '--skill-version', 'tactical_v1',
                                              '--execution-mode', 'micro8-all'])
                env = run.call_args.kwargs['env']
                self.assertNotIn('PYTHONPATH', env)
                self.assertEqual({k:v for k,v in env.items() if k.startswith('ZERG_')}, dict(
                    ZERG_WHOLE_AC_ARM=arm, ZERG_SHADOW_ROSTER='frozen-team', ZERG_SHADOW_DIR=str(work/'shadow')))

    def test_compare_is_independent_and_incomplete_is_not_zero(self):
        module = 'decision_tree.scripts.linux.compare_zerg_whole_ac'
        callback = Mock(return_value=dict(accepted_pairs=0, excluded=[]))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()), patch.dict(sys.modules, {module: Mock(compare=callback)}):
            run = Path(tmp)
            cli.write_json(run/'design.json', dict(pairs=[{}], dry_run=False))
            with self.assertRaises(SystemExit):
                cli.main(['--compare', tmp, '--dry-run'])
            callback.assert_not_called()
            with self.assertRaises(SystemExit):
                cli.main(['--compare', tmp])
            callback.assert_called_once_with(run)
            callback.return_value = dict(accepted_pairs=1, excluded=[])
            cli.main(['--compare', tmp])


if __name__ == '__main__':
    unittest.main()
