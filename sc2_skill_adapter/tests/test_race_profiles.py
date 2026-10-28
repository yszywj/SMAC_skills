"""Opt-in race controllers through the actual adapter and manager lifecycle."""
import asyncio
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from bridge import runner, skill_runtime
from bridge.config import load_config, public_brief
from bridge.policy import build_request
from test_bridge import FakeEnv


def snapshot(race):
    env = FakeEnv()
    env.map_name = '10gen_' + race
    env.scenario_brief = public_brief(load_config(5, race))
    ids = [48, 51, 54] if race == 'terran' else [105, 107, 9]
    names = dict(zip(ids, ['Marine', 'Marauder', 'Medivac'] if race == 'terran'
                     else ['Zergling', 'Hydralisk', 'Baneling']))
    for team in (env.agents, env.enemies):
        for i, unit in team.items():
            unit.unit_type = ids[i % 3]
    ranges = {48: 5, 51: 6, 54: 4, 105: 2, 107: 5, 9: 2}
    env.unit_shoot_range = lambda i: ranges[env.agents[i].unit_type]
    return build_request(env, NS(units=names, unit_stats={}), 'unused', 1)


class RaceProfileTests(unittest.TestCase):
    def arguments(self, flags=()):
        return runner.parser().parse_args(['run', '--method', 'adaptive-skill',
            '--manifest', 'unused.json', '--output-dir', 'unused', *flags])

    def test_default_constructor_and_metadata_stay_legacy(self):
        config = skill_runtime.manager_configuration(self.arguments())
        self.assertNotIn('race_profile', config['manager_kwargs'])
        payload = snapshot('terran')
        outputs = []
        for kwargs in ({}, {'race_profile': 'off'}):
            manager = skill_runtime.load_manager()(**kwargs)
            manager.select('adaptive_focus_fire')
            outputs.append(manager.step(copy.deepcopy(payload), 5))
        self.assertEqual(outputs[0], outputs[1])
        self.assertNotIn('race_profile', outputs[0][1])

    def test_explicit_profile_is_loaded_only_for_current(self):
        for race in ('terran', 'zerg'):
            cfg = skill_runtime.manager_configuration(self.arguments(['--race-profile', race]))
            self.assertEqual(cfg['manager_kwargs']['race_profile'], race)
            for version in ('baseline_v2_70pct', 'baseline_v6_20261024', 'baseline_v7_20261024'):
                with self.assertRaisesRegex(ValueError, 'current'):
                    skill_runtime.manager_configuration(self.arguments(
                        ['--race-profile', race, '--skill-version', version]))

    def test_actual_profiles_emit_legal_actions_and_reset_episode_memory(self):
        for race in ('terran', 'zerg'):
            with self.subTest(race=race):
                payload = snapshot(race)
                before = copy.deepcopy(payload)
                manager = skill_runtime.load_manager()(race_profile=race)
                manager.select('adaptive_focus_fire')
                first = manager.step(payload, 5)
                self.assertEqual(payload, before)
                self.assertEqual(first[1]['race_profile'], race)
                for i, action in enumerate(first[0]):
                    self.assertIn(str(action), payload['questions'][f'u{i}']['criteria'])
                memory = copy.deepcopy(manager._race_memory)
                elapsed = manager.elapsed_steps
                self.assertEqual(manager.step(payload, 5), first)
                self.assertEqual(manager.elapsed_steps, elapsed)
                self.assertEqual(manager._race_memory, memory)
                manager._race_memory['sentinel'] = True
                manager.target_assignments = {'u0': 'old_enemy'}
                next_episode = copy.deepcopy(payload)
                next_episode['state']['episode'] = 2
                manager.step(next_episode, 5)
                self.assertNotIn('sentinel', manager._race_memory)
                self.assertNotIn('old_enemy', manager.target_assignments.values())
                manager.reset()
                self.assertEqual(manager._race_memory, {})
                self.assertIsNone(manager._engagement_cached_step)

    def test_posture_change_invalidates_same_frame_cache(self):
        manager = skill_runtime.load_manager()(race_profile='terran')
        manager.select('adaptive_focus_fire', posture='balanced')
        manager.step(snapshot('terran'), 5)
        manager.select('adaptive_focus_fire', posture='defensive')
        self.assertIsNone(manager._engagement_cached_step)
        self.assertEqual(manager.step(snapshot('terran'), 5)[1]['posture'], 'defensive')

    def test_clock_rollback_clears_race_and_target_memory(self):
        manager = skill_runtime.load_manager()(race_profile='zerg')
        payload = snapshot('zerg')
        manager.select('adaptive_focus_fire')
        manager.step(payload, 5)
        manager._race_memory['sentinel'] = True
        manager.target_assignments = {'u0': 'old_enemy'}
        payload['state']['game_loop'] -= 1
        manager.step(payload, 5)
        self.assertNotIn('sentinel', manager._race_memory)
        self.assertNotIn('old_enemy', manager.target_assignments.values())

    def test_wrong_map_or_method_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'must-not-exist'
            args = self.arguments(['--race-profile', 'terran'])
            manifest = {'env_args': load_config(5, 'zerg')}
            for method in ('adaptive-skill', 'nearest'):
                with self.assertRaisesRegex(ValueError, 'race-profile'):
                    asyncio.run(runner.evaluate(args, manifest, method, out))
                self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
