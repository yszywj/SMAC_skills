"""Exercise mode selection, policy resets and micro receipts without launching SC2."""
import asyncio
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from bridge import runner
from bridge.config import load_config
from test_micro_trigger import INTENT, WindowEnv


class RunnerMicroTests(unittest.TestCase):
    def test_modes_across_episode_resets(self):
        config = load_config(5, 'zerg')
        manifest = {'env_args': config, 'sha256': 'offline', 'episodes': [
            {'sc2_seed': seed, 'sha256': str(seed)} for seed in (1, 2)]}

        for mode in runner.EXECUTION_MODES:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                flags_at_reset, flags_at_step, receipts = [], [], []

                class Manager:
                    active, stats = True, {}

                    def __init__(self, **kwargs):
                        pass

                    def reset(self):
                        flags_at_reset.append(self.micro_all_attackers)
                        self.micro_all_attackers = None  # runner must reapply after every reset

                    def step(self, payload, count):
                        flags_at_step.append(self.micro_all_attackers)
                        return [4], {}

                    def commit_final_actions(self, actions):
                        # Deliberately give all modes the same, valid ling intent;
                        # only micro8-all may register/execute it.
                        self.last_micro_intents = [dict(INTENT, macro_action_id=actions[0])]

                    def record_micro_action(self, actor, cmd):
                        receipts.append((actor, cmd))

                class Env(WindowEnv):
                    _step_mul, zergling_id = 8, 105

                    def __init__(self, config):
                        super().__init__()
                        self.env = self
                        self.agents[0].unit_type = self.zergling_id
                        self.agents[0].radius = self.enemies[1].radius = .375
                        self.enemies[1].pos.x = 2
                        data = NS(unit_stats={105: NS(weapons=[NS(type=1, range=.125)])})
                        self._controller = NS(data=lambda: data, ping=lambda: NS(game_version='offline'),
                                              save_replay=lambda: b'offline')

                    def reset_record(self, scenario):
                        pass

                    def step_micro_window(self, actions, window, macro_loops):
                        for delta in range(1, macro_loops):
                            if delta == 3:
                                self.enemies[1].pos.x = .875
                            self.check(window, delta)
                        return self.step(actions)

                    def step(self, actions):
                        self._obs.observation.game_loop = 28
                        return 1, True, {'battle_won': True, 'dead_enemies': 2, 'dead_allies': 0}

                    def close(self):
                        pass

                opts = ['--manifest', 'unused', '--output-dir', tmp, '--skill-version', 'tactical_v1']
                if mode != 'legacy8':
                    opts += ['--execution-mode', mode]
                args = runner.parser().parse_args(['run', '--method', 'adaptive-skill', *opts])
                self.assertEqual(runner.parser().parse_args(['suite', *opts]).execution_mode, mode)
                self.assertEqual(args.execution_mode, mode)
                out = Path(tmp) / mode
                with patch('bridge.skill_runtime.load_manager', return_value=Manager), \
                        patch('bridge.environment.ReplayableEnv', Env), \
                        patch('bridge.policy.build_request', return_value={}), \
                        patch('bridge.feedback.action_feedback', side_effect=lambda *_: {}), \
                        patch.object(runner, 'score_snapshot', return_value={}), \
                        patch.dict(os.environ, {'SMAC_DAMAGE_MAP_DIR': ''}):
                    summary = asyncio.run(runner.evaluate(args, manifest, 'adaptive-skill', out))
                enabled = mode == 'micro8-all'
                self.assertEqual(flags_at_reset, [enabled]*2)
                self.assertEqual(flags_at_step, [enabled]*2)
                self.assertEqual(receipts, [('u0', 7)]*2 if enabled else [])
                totals = summary['micro_trigger_totals']
                if mode == 'legacy8':
                    self.assertIsNone(totals)
                    self.assertFalse((out / 'micro_triggers').exists())
                else:
                    self.assertEqual((totals['windows'], totals['authorizations']), (2, 2*enabled))
                    for ep in (1, 2):
                        rows = json.loads((out / 'micro_triggers' / f'episode-{ep:04}.json').read_text())
                        self.assertEqual(rows['mode'], mode)
                        if enabled:
                            row, = rows['authorizations']
                            self.assertEqual((row['actor_type'], row['first_legal_loop'],
                                              row['submitted_loop'], row['sc2_response']),
                                             ('zergling', 23, 23, [1]))
                        else:
                            self.assertEqual(rows['authorizations'], [])
