"""Exercise the real resolved manager, recorder and runner with no SC2 process."""
import asyncio
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from bridge import runner, zerg_shadow
from bridge.config import load_config
from decision_tree.zerg.dataset import DecisionRecordWriter, read_records
from skills.tactical_v1.tests import fixtures as fx
from test_micro_trigger import WindowEnv


def manifest():
    team = {'ally_team': ['hydralisk'], 'enemy_team': ['zergling', 'zergling']}
    return {'env_args': load_config(5, 'zerg'), 'sha256': 'mock', 'episodes': [
        {'sc2_seed': seed, 'sha256': str(seed), 'episode_config': {'team_gen': team}} for seed in (31, 32)]}


class OfflineEnv(WindowEnv):
    _step_mul = 8

    def __init__(self, config):
        super().__init__()
        self.env = self
        self._controller = NS(data=lambda: NS(), ping=lambda: NS(game_version='offline'), save_replay=lambda: b'')

    def reset_record(self, scenario):
        pass

    def step_micro_window(self, actions, window, macro_loops):
        self._obs.observation.game_loop += macro_loops
        return 0.5, self._obs.observation.game_loop == 36, {'dead_allies': 0, 'dead_enemies': 2, 'battle_won': True}

    def close(self):
        pass


def payload(env, data, model, episode, *unused):
    return fx.with_questions(fx.scene([((12., 12.), 'hydralisk')], {'e0': fx.enemy((13., 12.))},
                                      episode=episode, loop=env._obs.observation.game_loop, map_name='10gen_zerg'))


class ShadowRunnerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.out, self.shadow = root / 'run', root / 'shadow'
        self.args = runner.parser().parse_args(['run', '--method', 'adaptive-skill', '--skill-version', 'tactical_v1',
                                                '--execution-mode', 'micro8-all', '--manifest', 'unused',
                                                '--output-dir', str(self.out)])
        self.flags_at_step = []
        start = zerg_shadow.start_shadow

        def start_checked(*args):
            runtime, controller = start(*args)
            step, reset = controller.manager.step, controller.manager.reset

            def checked(*args):
                self.flags_at_step.append(controller.manager.micro_all_attackers)
                return step(*args)

            def clear_flag():
                reset()
                controller.manager.micro_all_attackers = False

            controller.manager.step, controller.manager.reset = checked, clear_flag
            return runtime, controller

        for mocked in [patch.dict(os.environ, {'ZERG_SHADOW_DIR': str(self.shadow), 'ZERG_SHADOW_ROSTER': 'test-roster',
                                               'SMAC_DAMAGE_MAP_DIR': ''}),
                       patch('bridge.environment.ReplayableEnv', OfflineEnv),
                       patch('bridge.policy.build_request', payload), patch('bridge.feedback.action_feedback', return_value={}),
                       patch.object(runner, 'score_snapshot', side_effect=lambda e: {'score_details': {
                           k: {'life': e._obs.observation.game_loop} for k in ('total_damage_dealt', 'total_damage_taken')}}),
                       patch.object(zerg_shadow, 'start_shadow', side_effect=start_checked)]:
            mocked.start()
            self.addCleanup(mocked.stop)

    def run_worker(self):
        return asyncio.run(runner.evaluate(self.args, manifest(), 'adaptive-skill', self.out))

    def test_real_collection_and_micro_flag_after_reset(self):
        summary = self.run_worker()
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(self.flags_at_step, [True] * 4)
        rows = read_records(str(self.shadow / 'shadow.jsonl'))
        self.assertEqual([(r['episode_id'], r['seed'], r['game_loop']) for r in rows],
                         [('1', 31, 20), ('1', 31, 28), ('2', 32, 20), ('2', 32, 28)])
        meta = summary['zerg_shadow']
        self.assertEqual(meta['stats']['decisions_recorded'], 4)
        for row in rows:
            self.assertEqual((row['run_id'], row['scene_config_id'], row['source_hash'], row['outcome'], row['return']),
                             (meta['run_id'], 'test-roster', meta['source_fingerprint'], 'win', 1.0))
        events = [json.loads(line) for line in (self.shadow / 'events.jsonl').read_text().splitlines()]
        terminals = [r for r in events if r['event'] == 'episode_finished']
        self.assertEqual([(r['episode_id'], r['records_written'], r['damage_dealt']) for r in terminals],
                         [('1', 2, 16), ('2', 2, 16)])
        self.assertTrue(all(r['ally_losses'] == 0 and r['enemy_losses'] == 2 for r in terminals))

    def test_recorder_error_fails_worker(self):
        with patch.object(zerg_shadow.ShadowRecorder, 'record_decision', side_effect=ValueError('bad snapshot')):
            with self.assertRaisesRegex(RuntimeError, 'bad snapshot'):
                self.run_worker()
        summary = json.loads((self.out / 'summary.json').read_text())
        self.assertEqual((summary['status'], summary['completed_episodes']), ('failed', 0))

    def test_short_write_fails_worker(self):
        with patch.object(DecisionRecordWriter, 'write', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'wrote 0 decisions for 2 steps'):
                self.run_worker()

    def test_refuse_shared_directory(self):
        self.shadow.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_worker()

    def test_refuse_mixed_rosters(self):
        mixed = manifest()
        mixed['episodes'][1]['episode_config']['team_gen'] = {'ally_team': ['baneling'], 'enemy_team': ['zergling']}
        with self.assertRaisesRegex(ValueError, 'fixed composition'):
            asyncio.run(runner.evaluate(self.args, mixed, 'adaptive-skill', self.out))
        self.assertFalse(self.shadow.exists())
