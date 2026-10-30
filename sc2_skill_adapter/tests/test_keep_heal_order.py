"""Exercise the opt-in action through upstream masks/step and public feedback."""
import copy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from bridge.config import load_config
from bridge.environment import RecordedEnv, RecordingController, ReplayableEnv
from bridge.feedback import action_feedback
from bridge.keep_order import (raw_keep_heal_target, semantic_keep_heal_target,
    extend_keep_heal_order_request, validate_keep_heal_order)
from bridge.policy import build_request
from bridge.runner import parser
from test_bridge import FakeEnv, unit


def snapshot():
    env = FakeEnv()
    env.map_name = '10gen_terran'
    for roster in (env.agents, env.enemies):
        for u in roster.values():
            u.unit_type = 48
    healer = env.agents[0]
    healer.unit_type = 54; healer.energy = 50
    healer.orders.add(ability_id=386, target_unit_tag=env.agents[1].tag)
    env.agents[1].health = 30
    env.unit_shoot_range = lambda i: 4 if i == 0 else 5
    return env, NS(units={48: 'Marine', 54: 'Medivac'}, unit_stats={})


def bare_native(enabled=True, obs_last_action=False):
    config = load_config(5, 'terran'); config['unit_balance'] = 'native'
    config['obs_last_action'] = obs_last_action
    wrapper = ReplayableEnv(config, terran_keep_heal_order=enabled)
    env = wrapper.env
    snap, _ = snapshot()
    env.agents, env.enemies = snap.agents, snap.enemies
    env.medivac_id = 54; env.marine_id = 48; env.marauder_id = 51
    env.can_move = lambda *args: True
    env.unit_shoot_range = snap.unit_shoot_range
    return env


class KeepHealOrderTests(unittest.TestCase):
    def test_defaults_unchanged_and_explicit_cli(self):
        args = parser().parse_args(['run', '--manifest', 'x.json', '--output-dir', 'x'])
        self.assertFalse(args.terran_keep_heal_order)
        self.assertIsNone(validate_keep_heal_order(args, 'nearest', load_config(5)))
        env, data = snapshot()
        off = build_request(env, data, 'unused', 1)
        self.assertNotIn('keep_heal_order', off['state']['action_contract'])
        self.assertNotIn('keep_heal_order_candidates', off['state'])
        self.assertNotIn('11', off['questions']['u0']['criteria'])

    def test_only_native_current_terran_skill_can_enable(self):
        args = NS(terran_keep_heal_order=True, skill_version='current', race_profile='terran')
        config = load_config(10, 'terran'); config['unit_balance'] = 'native'
        before = copy.deepcopy(config)
        result = validate_keep_heal_order(args, 'adaptive-skill', config)
        self.assertEqual(result['terran_keep_heal_order']['action_id'], 16)
        self.assertEqual(config, before)
        for field, value in [('skill_version', 'baseline_v7_20261024'), ('race_profile', 'off')]:
            changed = copy.copy(args); setattr(changed, field, value)
            with self.assertRaises(ValueError): validate_keep_heal_order(changed, 'adaptive-skill', config)
        with self.assertRaises(ValueError): validate_keep_heal_order(args, 'random', config)
        for field, value in [('unit_balance', 'original'), ('map_name', '10gen_protoss')]:
            changed = copy.deepcopy(config); changed[field] = value
            with self.assertRaises(ValueError): validate_keep_heal_order(args, 'adaptive-skill', changed)

    def test_mask_requires_actual_first_heal_order_and_valid_patient(self):
        base, _ = snapshot()
        self.assertEqual(raw_keep_heal_target(base.agents, 0), 1)
        alterations = [
            lambda e: setattr(e.agents[0], 'health', 0),
            lambda e: setattr(e.agents[0], 'energy', .99),
            lambda e: setattr(e.agents[0], 'unit_type', 48),
            lambda e: setattr(e.agents[0].orders[0], 'ability_id', 16),
            lambda e: setattr(e.agents[0].orders[0], 'target_unit_tag', 9000),
            lambda e: setattr(e.agents[1], 'health', 0),
            lambda e: setattr(e.agents[1], 'health', e.agents[1].health_max),
            lambda e: setattr(e.agents[1], 'unit_type', 54),
            lambda e: e.agents[0].ClearField('orders'),
        ]
        for change in alterations:
            env = copy.deepcopy(base); change(env)
            self.assertIsNone(raw_keep_heal_target(env.agents, 0))
        base.agents[0].energy = 1
        self.assertEqual(raw_keep_heal_target(base.agents, 0), 1)

    def test_retention_outside_new_heal_range_has_distinct_candidate(self):
        env, data = snapshot()
        env.agents[1].pos.x = 20
        env.terran_keep_heal_order = True; env.keep_heal_order_action_id = 11
        env.keep_heal_order_target = lambda i: raw_keep_heal_target(env.agents, i)
        env.get_avail_agent_actions = lambda i: [0,1,1,1,1,1,0,0,0,0,0,int(i == 0)]
        payload = build_request(env, data, 'unused', 1)
        self.assertEqual(payload['state']['action_contract']['keep_heal_order'], {'enabled': True, 'action_id': 11})
        self.assertEqual(payload['state']['keep_heal_order_candidates'], {'u0': {'action_id': 11, 'target': 'u1'}})
        self.assertEqual(payload['state']['heal_candidates']['u0'], {})
        self.assertIn('11', payload['questions']['u0']['criteria'])
        self.assertNotIn('11', payload['questions']['u1']['criteria'])

    def test_offline_extension_is_pure_and_raw_mask_handles_energy_rounding(self):
        env, data = snapshot(); original = build_request(env, data, 'unused', 1)
        before = copy.deepcopy(original)
        expanded = extend_keep_heal_order_request(original, 11)
        self.assertEqual(original, before)
        self.assertEqual(semantic_keep_heal_target(original['state'], 'u0'), 'u1')
        invalid = copy.deepcopy(original['state']); invalid['allies']['u0']['energy'] = True
        self.assertIsNone(semantic_keep_heal_target(invalid, 'u0'))
        self.assertIn('11', expanded['questions']['u0']['criteria'])
        # A rounded semantic energy of 1.0 must not override raw .999 masking.
        env.agents[0].energy = .999
        env.terran_keep_heal_order = True; env.keep_heal_order_action_id = 11
        env.keep_heal_order_target = lambda i: raw_keep_heal_target(env.agents, i)
        payload = build_request(env, data, 'unused', 1)
        self.assertEqual(payload['state']['allies']['u0']['energy'], 1.0)
        self.assertEqual(payload['state']['keep_heal_order_candidates'], {})

    def test_upstream_action_and_observation_dimensions_append_exactly_one(self):
        for obs_last in (False, True):
            old, new = bare_native(False, obs_last), bare_native(True, obs_last)
            try:
                self.assertEqual((old.n_actions, new.n_actions, new.keep_heal_order_action_id), (11,12,11))
                self.assertEqual(new.last_action.shape, (5,12))
                self.assertEqual(new.get_obs_size() - old.get_obs_size(), 4 if obs_last else 0)
                self.assertEqual(new.get_state_size() - old.get_state_size(), 5)
                self.assertEqual(len(new.get_obs_feature_names()), new.get_obs_size())
                self.assertEqual(len(new.get_state_feature_names()), new.get_state_size())
                self.assertEqual(len(new.get_avail_agent_actions(0)), 12)
                self.assertEqual(new.get_avail_agent_actions(0)[11], 1)
                self.assertEqual(new.get_avail_agent_actions(1)[11], 0)
                new.agents[0].health = 0
                self.assertEqual(new.get_avail_agent_actions(0), [1] + [0]*11)
            finally:
                old.close(); new.close()

    def test_native_step_retains_without_command_and_feedback_has_no_success(self):
        env = bare_native(); requests = []
        obs = NS(observation=NS(game_loop=16), action_errors=[])
        def send(request):
            requests.append(request)
            return NS(result=[1] * len(request.actions))
        env._controller = RecordingController(NS(actions=send, step=lambda n: None, observe=lambda: obs), env)
        env.update_units = lambda: None; env.reward_battle = lambda: 0; env.reward_scale = False
        try:
            env.step([11, 1, 1, 1, 1])
            self.assertEqual(len(requests[0].actions), 4)
            self.assertEqual(env.last_action.shape, (5,12))
            self.assertEqual(env.last_action[0,11], 1)
            self.assertTrue(np.array_equal(env.last_action[1], np.eye(12)[1]))
            with patch('bridge.feedback.visible_enemies', return_value={}):
                feedback = action_feedback(env, 8)
            keep = feedback['actors']['u0']; stop = feedback['actors']['u1']
            self.assertEqual(keep['submission'], 'retained_order')
            self.assertIsNone(keep['command']); self.assertIsNone(keep['response'])
            self.assertNotIn('command_index', keep)
            self.assertEqual(stop['command']['ability_id'], 4)
            self.assertEqual(stop['command_index'], 0)
            self.assertEqual(feedback['response_mapping'], 'exact')
        finally:
            env._controller = None; env.close()

    def test_invalid_retention_is_rejected_and_dead_noop_is_unchanged(self):
        env = bare_native(); env.collecting = True
        try:
            with self.assertRaises(ValueError): env.get_agent_action(1, 11)
            env.agents[0].health = 0
            self.assertIsNone(env.get_agent_action(0, 0))
            self.assertEqual(env.last_submissions[-1]['submission'], 'dead_noop')
            with self.assertRaises(ValueError): env.get_agent_action(0, 11)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
