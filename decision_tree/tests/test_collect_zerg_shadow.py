import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from decision_tree.scripts.linux import collect_zerg_shadow as cli


class ZergShadowBatchTest(unittest.TestCase):
    def test_dry_design(self):
        # Integration case against the same historical scenario dependencies as production.
        try:
            cli.version('pysc2')
        except cli.PackageNotFoundError:
            self.skipTest(f'requires RSI interpreter {cli.RSI_PYTHON}')
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(cli, 'play', side_effect=AssertionError('dry-run started worker')):
            root = Path(tmp)
            argv = ['--dry-run', '--output-root', tmp]
            cli.main(argv)
            batch, = root.iterdir()
            design = cli.read_json(batch / 'design.json')
            jobs, rosters = design['jobs'], design['rosters']
            self.assertEqual(len(set(tuple(sorted(t)) for t in rosters.values())), 6)
            for field in ('worker', 'layout_sha256', 'sc2_seed'):
                self.assertEqual(len({j[field] for j in jobs}), 240)
            for name, team in rosters.items():
                self.assertEqual(len(team), 10)
                group = [j for j in jobs if j['roster'] == name]
                self.assertEqual(len(group), 40)
                for job in group:
                    work = batch / job['worker']
                    self.assertEqual([p.name for p in work.iterdir()], ['manifest.json'])
                    manifest = cli.read_json(work / 'manifest.json')
                    row, = manifest['episodes']
                    self.assertEqual(row['sc2_seed'], job['sc2_seed'])
                    teams = row['episode_config']['team_gen']
                    self.assertEqual((teams['ally_team'], teams['enemy_team']), (team, team))
            cli.main(argv + ['--episodes-per-roster', '1'])
            smoke, = set(root.iterdir()) - {batch}
            self.assertEqual(cli.read_json(smoke / 'design.json')['jobs'], jobs[::40])
            self.assertEqual(cli.read_json(batch / 'design.json'), design)

    def test_worker_limit(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()), \
                patch.object(cli, 'historical_config', side_effect=AssertionError('late validation')):
            cli.main(['--workers', '61'])

    def test_worker_protocol_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            batch = Path(tmp)
            job = dict(worker='workers/roster1/episode-001-seed-47', roster='roster1',
                       sc2_seed=47, scenario_sha256='scenario')
            work = batch / job['worker']
            work.mkdir(parents=True)

            def fake_run(cmd, *, cwd, env, stdout, stderr, check):
                self.assertEqual(cmd[:5], [cli.sys.executable, '-B', '-u', str(cli.ADAPTER / 'run.py'), 'run'])
                self.assertEqual(dict(zip(cmd[5::2], cmd[6::2])), {
                    '--method': 'adaptive-skill', '--skill-version': 'tactical_v1',
                    '--execution-mode': 'micro8-all', '--manifest': str(work / 'manifest.json'),
                    '--output-dir': str(work / 'runner')})
                self.assertEqual(env['ZERG_SHADOW_ROSTER'], 'roster1')
                self.assertEqual(env['ZERG_SHADOW_DIR'], str(work / 'shadow'))
                self.assertEqual(stdout.name, str(work / 'worker.log'))
                self.assertTrue(check)
                (work / 'runner').mkdir()
                cli.write_json(work / 'runner/episodes.json', [dict(steps=1, scenario_sha256='scenario')])

            with patch.object(cli.subprocess, 'run', side_effect=fake_run) as runner:
                with self.assertRaisesRegex(ValueError, 'Incomplete'):
                    cli.play(batch, job, {})  # successful process without Shadow records is a failure
                with self.assertRaises(FileExistsError):
                    cli.play(batch, job, {})  # never overwrite a previous worker
                runner.assert_called_once()


if __name__ == '__main__':
    unittest.main()
