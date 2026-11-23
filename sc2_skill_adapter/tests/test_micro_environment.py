"""Native SMAC step vs. sliced engine clock; scripted protobuf frames, no SC2."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from s2clientprotocol import sc2api_pb2 as sc_pb

from bridge.environment import RecordedEnv, RecordingController, visible_enemies


class Clock:
    def __init__(self, outcome):
        self.outcome = outcome
        self.loop = self.last_seen = 0
        self.steps, self.requests = [], []

    def step(self, n):
        self.steps.append(n)
        self.loop += n

    def actions(self, request):
        self.requests.append(request)
        return sc_pb.ResponseAction(result=[1] * len(request.actions))

    def observe(self):
        obs = sc_pb.ResponseObservation()
        obs.observation.game_loop = self.loop
        for tag in (101, 102, 201):
            is_ally = tag < 200
            dies = (is_ally and self.outcome in ('loss', 'draw') and self.loop >= 7
                    or not is_ally and self.outcome in ('win', 'draw') and self.loop >= 4)
            if not dies:
                u = obs.observation.raw_data.units.add(tag=tag, owner=1 if is_ally else 2,
                                                       unit_type=107, health=90-self.loop,
                                                       health_max=90, shield=max(0, 8-self.loop))
                u.pos.x = 8 if is_ally else 11
                u.weapon_cooldown = self.loop
        for i in (1, 8):
            if self.last_seen < i <= self.loop:
                obs.action_errors.add(unit_tag=101, ability_id=16, result=3)
        self.last_seen = self.loop
        return obs


def make_env(outcome):
    # The constructor only initializes SMAC state; _launch/reset are never called.
    team = {'n_units': 2, 'n_enemies': 1, 'observe': True}
    env = RecordedEnv(map_name='10gen_zerg', capability_config={'team_gen': team},
                      step_mul=8, reward_only_positive=False, continuing_episode=True)
    clock = Clock(outcome)
    env._controller = RecordingController(clock, env)
    env._obs = clock.observe()
    a, b, e = env._obs.observation.raw_data.units
    env.agents, env.enemies = {0: a, 1: b}, {0: e}
    env.can_move = lambda *args: True  # flat pathing
    env.unit_type_bits = 0
    env.enemy_tags, env.obs_enemies = [None], np.zeros((1, 2))
    env.defeat_counted = env.win_counted = False
    if outcome == 'timeout':
        env.episode_limit = 1
    return env, clock


class MicroEnvironmentTests(unittest.TestCase):
    def test_native_settlement_and_isolation(self):
        for outcome in ('ongoing', 'win', 'loss', 'draw', 'timeout'):
            for sparse in (False, True):
                with self.subTest(outcome=outcome, sparse=sparse):
                    env, clock = make_env(outcome)
                    legacy, native_clock = make_env(outcome)
                    env.reward_sparse = legacy.reward_sparse = sparse
                    actions = [4, 6]  # native move and attack protobuf builders
                    expected = legacy.step(actions)
                    seen = []

                    def check(offset):
                        seen.append(env._obs.observation.game_loop)
                        self.assertEqual(seen[-1], offset)
                        self.assertEqual((env._episode_steps, env._total_steps, env.reward), (0, 0, 0))
                        self.assertIsNone(env.previous_ally_units)
                        self.assertFalse(env.death_tracker_enemy.any())
                        if env.agents[0].health > 0:
                            self.assertEqual(env.agents[0].weapon_cooldown, offset)
                        visible = visible_enemies(env)  # real get_obs_agent changes cache + NumPy RNG
                        if offset == 1:
                            self.assertEqual(set(visible), {0})
                            self.assertEqual(env.enemy_tags, [0])
                            cmd = env.get_agent_action(0, 6)
                            env.submit_micro_commands(sc_pb.RequestAction(actions=[cmd]))
                        if outcome == 'draw' and offset == 4:
                            self.assertEqual(visible, {})
                            self.assertGreater(env.agents[0].health, 0)  # not the final result

                    rng = np.random.get_state()
                    with patch.object(env, 'update_units', wraps=env.update_units) as update:
                        reward, terminal, info = env.step_micro_window(
                            actions, SimpleNamespace(on_interior_loop=check), 8)
                    update.assert_called_once_with()
                    self.assertEqual(seen, list(range(1, 8)))
                    self.assertEqual(info.pop('micro_advanced_loops'), 8)
                    self.assertEqual((reward, terminal, info), expected)
                    self.assertEqual(env._obs, legacy._obs)
                    self.assertEqual(clock.steps, [1]*8)
                    self.assertEqual(native_clock.steps, [8])
                    self.assertEqual(clock.requests[:1], native_clock.requests)
                    self.assertEqual(len(clock.requests), 2)
                    self.assertEqual(env.last_action_results, legacy.last_action_results)
                    self.assertEqual(env.last_submissions, legacy.last_submissions)
                    for attr in ('agents', 'enemies', 'enemy_tags', 'previous_ally_units',
                                 'previous_enemy_units', '_episode_steps', '_total_steps', '_episode_count',
                                 'win_counted', 'defeat_counted', 'battles_won', 'battles_game', 'timeouts'):
                        self.assertEqual(getattr(env, attr), getattr(legacy, attr), attr)
                    for attr in ('death_tracker_ally', 'death_tracker_enemy', 'last_action',
                                 'new_unit_positions', 'obs_enemies'):
                        np.testing.assert_array_equal(getattr(env, attr), getattr(legacy, attr))
                    for v, w in zip(rng, np.random.get_state()):
                        np.testing.assert_equal(v, w)

    def test_bad_clock_does_not_commit(self):
        env, clock = make_env('ongoing')
        original = env._obs
        with patch.object(clock, 'step', lambda n: None):
            with self.assertRaisesRegex(RuntimeError, 'Expected SC2 loop'):
                env.step_micro_window([4, 6], None, 8)
        self.assertIs(env._obs, original)
        self.assertFalse(env.collecting)
        self.assertEqual(env._episode_steps, 0)
        self.assertEqual(env._controller.step, clock.step)

    def test_next_legacy_step_is_not_intercepted(self):
        env, clock = make_env('ongoing')
        legacy, _ = make_env('ongoing')
        env.step_micro_window([4, 6], SimpleNamespace(on_interior_loop=lambda offset: None), 8)
        legacy.step([4, 6])
        self.assertEqual(env.step([1, 6]), legacy.step([1, 6]))
        self.assertEqual(clock.steps, [1]*8 + [8])
        self.assertEqual(env.previous_enemy_units, legacy.previous_enemy_units)
        self.assertEqual(env.last_submissions, legacy.last_submissions)
        self.assertEqual(env._episode_steps, 2)
