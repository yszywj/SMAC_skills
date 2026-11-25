import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from decision_tree.scripts.linux import run_zerg_forks as cli


class ForkLauncherTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.out = self.root / 'batch'
        self.path = self.root / 'input.json'
        self.pair = dict(episode_group='run:1', run_id='run', episode_id='1', roster='roster',
                         seed=7, snapshot_id=4, game_loop=26, manifest=str(self.root / 'original/manifest.json'),
                         A=dict(candidate_id='a', source_plan_id='p0', actions={'u0': 0}),
                         B=dict(candidate_id='b', source_plan_id='p1', actions={'u0': 1}))
        self.plan = dict(source_hash='source', collection_context='ctx', pairs=[{}, self.pair])
        cli.write_json(self.path, self.plan)
        self.argv = ['--out', str(self.out), '--plan', str(self.path), '--pair-index', '2']

    def invoke(self, *extra):
        with patch.object(cli, 'historical_config', return_value=({}, {})), \
                patch.object(cli, 'validate_plan'), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            cli.main(self.argv + list(extra))

    def test_smoke_dry_run_freezes_whole_plan_and_keeps_original_index(self):
        with patch.object(cli.subprocess, 'run', side_effect=AssertionError('started a game')) as run:
            self.invoke('--dry-run')
            run.assert_not_called()
        self.assertEqual((self.out / 'plan.json').read_bytes(), self.path.read_bytes())
        design = cli.read_json(self.out / 'design.json')
        self.assertEqual(design['pair_indices'], [2])
        self.assertEqual(design['workers'], 48)
        self.assertTrue(design['dry_run'])
        self.assertFalse((self.out / 'pair-001').exists())
        for arm in ('A', 'B'):
            self.assertEqual(list((self.out / 'pair-002' / arm).iterdir()), [])
        with self.assertRaises(SystemExit):
            self.invoke('--dry-run')  # cannot overwrite or reuse a dry design

    def test_bad_indices_and_workers_do_not_create_output(self):
        for option, value in [('--pair-index', 0), ('--pair-index', 3), ('--workers', 0)]:
            with self.subTest(option=option, value=value), self.assertRaises(SystemExit):
                self.invoke(option, str(value))
            self.assertFalse(self.out.exists())

    def test_distinct_concurrent_subprocess_calls_with_pinned_inputs(self):
        barrier = threading.Barrier(2, timeout=5)

        def fake_run(cmd, *, env, cwd, stdout, stderr, check):
            spec = json.loads(Path(env['ZERG_FORK_SPEC']).read_text())
            arm = spec['arm']
            work = self.out / 'pair-002' / arm
            self.assertEqual(spec, dict(arm=arm, pair=self.pair, source_hash='source', collection_context='ctx'))
            self.assertEqual(cmd, [cli.RSI_PYTHON, '-B', '-u', str(cli.ADAPTER / 'run.py'), 'run',
                '--method', 'adaptive-skill', '--skill-version', 'tactical_v1', '--execution-mode', 'micro8-all',
                '--manifest', self.pair['manifest'], '--output-dir', str(work / 'runner')])
            self.assertEqual((cwd, stdout.name, stderr, check), (cli.ROOT, str(work / 'worker.log'), cli.subprocess.STDOUT, True))
            self.assertEqual(env['SC2PATH'], cli.SC2_PATH)
            self.assertEqual(env['SMAC_DAMAGE_MAP_DIR'], str(cli.MAP_DIR))
            self.assertEqual(env['ZERG_SHADOW_DIR'], str(work / 'shadow'))
            self.assertEqual(env['ZERG_SHADOW_ROSTER'], self.pair['roster'])
            self.assertNotIn('PYTHONPATH', env)
            barrier.wait()

        with patch.object(cli.subprocess, 'run', side_effect=fake_run) as run, \
                patch.object(cli, 'validate_arm') as validate:
            self.invoke('--workers', '48')
        self.assertEqual(run.call_count, 2)
        for arm in ('A', 'B'):
            validate.assert_any_call(self.out / 'pair-002' / arm, arm, self.pair, self.plan)

    def test_zero_exit_missing_output_fails_both_arms(self):
        with patch.object(cli.subprocess, 'run') as run, self.assertRaisesRegex(SystemExit, '2/2 arms failed'):
            self.invoke('--workers', '1')
        self.assertEqual(run.call_count, 2)

    def test_receipt_is_bound_to_arm_anchor_initial_state_and_protocol(self):
        original = self.root / 'original'
        work = self.root / 'pair-002/B'
        for directory in (original / 'runner', work / 'runner', work / 'shadow'):
            directory.mkdir(parents=True)
        identity = dict(initial_state_sha256='initial', scenario_sha256='scenario')
        run = dict(**cli.PROTOCOL, macro_loops=8, engine_step_loops=1, versions={}, env_args={},
                   map_file=str(cli.MAP_FILE), damage_by_type_enabled=True, skill_parameters={},
                   native_teacher_control='off', skill_source_hashes={}, manifest_sha256='manifest',
                   source_hashes={cli.MAP_FILE.relative_to(cli.ROOT).as_posix(): 'maphash'})
        for directory in (original, work):
            cli.write_json(directory / 'runner/run.json', run)
            cli.write_json(directory / 'runner/episodes.json', [dict(**identity, steps=1, zerg_shadow_decisions=1,
                outcome='win', reward=20)])
        cli.write_json(original / 'manifest.json', dict(sha256='manifest'))
        cli.write_json(work / 'runner/summary.json', dict(**cli.PROTOCOL, status='complete', completed_episodes=1,
            expected_episodes=1, manifest_sha256='manifest', zerg_shadow=dict(source_fingerprint='source',
            run_id='new', scene_config_id='roster', execution_context_hash='ctx', stats=dict(decisions_seen=1,
            decisions_recorded=1, model_errors=0))))
        row = dict(run_id='new', episode_id='1', scene_config_id='roster', source_hash='source', seed=7,
            snapshot_id=4, game_loop=26, execution_context_hash='ctx', outcome='win', **{'return': 20})
        (work / 'shadow/shadow.jsonl').write_text(json.dumps(row) + '\n')
        receipt = dict(kind='zerg_one_shot_fork_receipt_v1', status='complete', error=None, arm='B',
            episode_group='run:1', roster='roster', seed=7, snapshot_id=4, game_loop=26, source_hash='source',
            collection_context='ctx', candidate_id='b', source_plan_id='p1', final_actions={'u0': 1},
            anchor_commits=1, prefix_verified=True, engine_receipt='unknown', **identity)
        path = work / 'runner/fork_receipt.json'
        with self.assertRaises(FileNotFoundError):
            cli.validate_arm(work, 'B', self.pair, self.plan)
        for field, value in [('status', 'pending'), ('arm', 'A'), ('prefix_verified', 1), ('anchor_commits', 2),
                             ('source_hash', 'other'), ('scenario_sha256', ''), ('initial_state_sha256', 'other'),
                             ('final_actions', {'u0': True}), ('engine_receipt', 'rejected')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                path.write_text(json.dumps({**receipt, field: value}))
                cli.validate_arm(work, 'B', self.pair, self.plan)
        path.write_text(json.dumps(receipt))
        self.assertEqual(cli.validate_arm(work, 'B', self.pair, self.plan), receipt)
        (work / 'runner/run.json').write_text(json.dumps({**run, 'damage_by_type_enabled': False}))
        with self.assertRaisesRegex(ValueError, 'damage_by_type_enabled'):
            cli.validate_arm(work, 'B', self.pair, self.plan)


class ArchivedPlanTest(unittest.TestCase):
    def test_original_preflight_and_tampering(self):
        archive = cli.ROOT / 'runs/zerg_joint_imitation_480_20261123_194909/fork_plan_24.json'
        if not archive.is_file():
            self.skipTest('requires original archived fork plan')
        try:
            config, versions = cli.historical_config()
        except ValueError as exc:
            if 'use ' + cli.RSI_PYTHON in str(exc):
                self.skipTest(str(exc))
            raise
        plan = cli.read_json(archive)
        cli.validate_plan(plan, config, versions)
        for changes in [dict(split_hash='other'), dict(model_id='other'), dict(collection_context='other'),
                        dict(horizon_steps=2)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                cli.validate_plan({**plan, **changes}, config, versions)
        p = plan['pairs'][0]
        for changes in [dict(seed=p['seed'] + 1), dict(manifest=plan['pairs'][1]['manifest']),
                        dict(game_loop=-1), dict(B={**p['B'], 'imitation_score': -1}), dict(changed={})]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                cli.validate_plan({**plan, 'pairs': [{**p, **changes}, *plan['pairs'][1:]]}, config, versions)


if __name__ == '__main__':
    unittest.main()
