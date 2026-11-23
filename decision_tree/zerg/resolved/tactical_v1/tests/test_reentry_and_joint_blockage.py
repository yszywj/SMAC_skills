"""Regressions for effective re-entry range and simultaneous friendly yielding."""
import copy
import json
from pathlib import Path
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.contracts import Commitment, ResourceClaims
from skills.tactical_v1.evaluator import Evaluator
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.common import make_plan
from skills.tactical_v1.skills.disengage_reenter import DisengageReenterSkill
from .fixtures import scene, enemy, with_questions


class ReentryAndJointBlockageTests(unittest.TestCase):
    def context(self, state):
        payload = state if 'state' in state else with_questions(state)
        return Context(build_snapshot(payload, payload['state']['game_loop']))

    def blocked(self, enemy_body=False):
        memory = TacticalMemory()
        s = scene([((0., 0.), 'marine'), ((1.5, .5), 'marine')], distance=2.,
                  enemies={'e0': enemy((1., .5))} if enemy_body else {})
        c = self.context(s)
        memory._commit_movement(c.snapshot, c, {'u0': 4, 'u1': 1}, [])
        s['game_loop'] += 8
        s['allies']['u0']['position'][0] = .05
        # Refresh candidate destinations at the actual observed location.
        for value in s['movement_candidates']['u0'].values():
            value['destination'][0] += .05
        c = self.context(s)
        memory._observe_movement(c.snapshot, c)
        return memory, c, s

    def joint_cost(self, memory, c, actions):
        return memory.movement_blockage(c, 'u0', 4, joint_actions=actions,
            joint_positions={a: c.predict_position(a, x).position for a, x in actions.items()})

    def test_joint_yield_discount_is_pure_bounded_and_preserves_ttl(self):
        memory, c, _ = self.blocked()
        before = copy.deepcopy(memory.__dict__)
        cost = memory.movement_blockage(c, 'u0', 4)
        self.assertGreater(cost, .9)
        self.assertEqual(self.joint_cost(memory, c, {'u0': 4, 'u1': 1}), cost)
        self.assertAlmostEqual(self.joint_cost(memory, c, {'u0': 4, 'u1': 2}), cost * .5)
        self.assertEqual(memory.__dict__, before)
        # Final scoring receives only actions, not a skill label loophole.
        evaluator = Evaluator()
        costs = []
        for skill in ('baseline', 'regroup'):
            plans = [] if skill == 'baseline' else [make_plan(
                skill, ('u0',), 'safe', {'u0': 4}, [], ResourceClaims(controls=('u0',)),
                Commitment('moving'), {}, c.snapshot.snapshot_id, 'test')]
            _, breakdown = evaluator.score_joint(c, plans, {'u0': 4, 'u1': 2}, memory)
            costs.append(breakdown['blocked_move'])
        self.assertAlmostEqual(costs[0], -.6 * cost * .5)
        self.assertEqual(costs[0], costs[1])

    def test_enemy_or_unexplained_obstruction_keeps_full_cost(self):
        for enemy_body in (False, True):
            memory, c, _ = self.blocked(enemy_body)
            if not enemy_body:
                memory.blocked_directions['u0', 4]['friendly_blockers'] = {}
            self.assertEqual(self.joint_cost(memory, c, {'u0': 4, 'u1': 2}),
                             memory.movement_blockage(c, 'u0', 4))

    def test_friend_crossing_route_is_not_yielding(self):
        memory, c, _ = self.blocked()
        # South ends off the corridor, but crosses the actor's moving path.
        self.assertEqual(self.joint_cost(memory, c, {'u0': 4, 'u1': 3}),
                         memory.movement_blockage(c, 'u0', 4))

    def test_unknown_speed_or_recycled_blocker_identity_keeps_full_cost(self):
        for variant in ('speed', 'identity'):
            memory, c, s = self.blocked()
            if variant == 'speed':
                s['unit_type_profiles']['48'].pop('movement_speed_native')
            else:
                s['allies']['u1']['tag'] = 'replacement'
            c = self.context(s)
            self.assertEqual(self.joint_cost(memory, c, {'u0': 4, 'u1': 2}),
                             memory.movement_blockage(c, 'u0', 4))

    def test_expired_blockage_cost_stays_zero(self):
        memory, _, s = self.blocked()
        s['game_loop'] = memory.blocked_directions['u0', 4]['until_loop']
        c = self.context(s)
        self.assertEqual(self.joint_cost(memory, c, {'u0': 4, 'u1': 2}), 0.)

    def test_observation_without_context_retains_failure_without_body_inference(self):
        memory, c, s = self.blocked()
        memory._commit_movement(c.snapshot, c, {'u0': 4}, [])
        s['game_loop'] += 8
        c = self.context(s)
        memory._observe_movement(c.snapshot, None)
        self.assertEqual(memory.blocked_directions['u0', 4]['friendly_blockers'], {})
        self.assertGreater(memory.movement_blockage(c, 'u0', 4), .99)

    def test_new_crossing_body_prevents_route_discount(self):
        memory, c, s = self.blocked()
        s['allies']['u2'] = dict(s['allies']['u1'], position=[1.1, -1.2], tag='u2')
        s['movement_candidates']['u2'] = {2: {'destination': [1.1, .8]}}
        s['relations']['u2'] = {}
        c = self.context(s)
        self.assertEqual(self.joint_cost(memory, c, {'u0': 4, 'u1': 2, 'u2': 2}),
                         memory.movement_blockage(c, 'u0', 4))

    def test_reentry_uses_native_radii_and_command_limit(self):
        s = scene([((0., 0.), 'zealot')], enemies={'e0': enemy((4., 0.))},
                  attack_range=.1, distance=2.)
        s['relations']['u0']['e0']['smac_attack_distance_limit'] = 2.
        skill = DisengageReenterSkill({})
        c = self.context(s)
        objective = skill._reentry_objective(c, 'u0', 'e0')
        self.assertAlmostEqual(objective['r_max'], 1.1)
        plan = make_plan('disengage_reenter', ('u0',), 'e0', {'u0': 4}, [],
                         ResourceClaims(controls=('u0',)), Commitment('REENTER'), {},
                         c.snapshot.snapshot_id, 'test', goal_data={'objective': objective})
        evaluator = Evaluator()
        report = evaluator.effects(c, {'u0': 4})
        self.assertGreater(evaluator._range_band_gain(c, report, plan, objective), .9)
        self.assertEqual(evaluator._range_band_gain(c, report, plan,
                            dict(objective, r_max=5.)), 0.)
        s['relations']['u0']['e0']['native_weapon_range'] = 7.
        s['relations']['u0']['e0']['smac_attack_distance_limit'] = 3.
        self.assertEqual(skill._reentry_objective(self.context(s), 'u0', 'e0')['r_max'], 3.)

    def test_unknown_or_incompatible_reentry_range_is_not_fabricated(self):
        s = scene([((0., 0.), 'zealot')], enemies={'e0': enemy((4., 0.))})
        skill = DisengageReenterSkill({})
        s['relations']['u0']['e0']['native_weapon_range'] = None
        self.assertIsNone(skill._reentry_objective(self.context(s), 'u0', 'e0'))

    def test_unknown_future_command_range_requires_current_legal_evidence(self):
        s = scene([((0., 0.), 'zealot')], enemies={'e0': enemy((4., 0.))}, attack_range=.1)
        s['relations']['u0']['e0'].pop('smac_attack_distance_limit')
        s['action_contract']['candidate_center_distance_by_type'].pop('zealot')
        skill = DisengageReenterSkill({})
        self.assertIsNone(skill._reentry_objective(self.context(s), 'u0', 'e0'))
        s['relations']['u0']['e0']['smac_attack_available'] = True
        self.assertAlmostEqual(skill._reentry_objective(self.context(s), 'u0', 'e0')['r_max'], 1.1)
        s['relations']['u0']['e0']['native_weapon_range'] = .1
        s['relations']['u0']['e0']['weapon_range_status'] = 'no_compatible_weapon'
        self.assertIsNone(skill._reentry_objective(self.context(s), 'u0', 'e0'))

    def recorded(self, roster, episode, steps):
        path = Path(__file__).resolve().parents[3] / (
            'runs/tactical_protoss_10v10_3x40_support_20261113/' + roster + '/events.jsonl')
        if not path.exists():
            self.skipTest('optional recorded regression observations unavailable')
        observations, decisions = {}, {}
        with path.open(encoding='utf8') as stream:
            for line in stream:
                if '"event": "skill_observation"' not in line and '"event": "skill_step"' not in line:
                    continue
                row = json.loads(line)
                if row['episode'] == episode and row['step'] in steps:
                    (observations if row['event'] == 'skill_observation' else decisions)[row['step']] = row
                if len(observations) == len(steps) and len(decisions) == len(steps):
                    break
        self.assertEqual(set(observations), set(steps))
        self.assertEqual(set(decisions), set(steps))
        return observations, decisions

    def test_seed174_real_melee_reentry_gains_progress_inside_five(self):
        observations, _ = self.recorded('4z5s1c', 5, {49})
        c = self.context(observations[49]['payload'])
        skill, evaluator = DisengageReenterSkill({}), Evaluator()
        objective = skill._reentry_objective(c, 'u1', 'e2')
        self.assertAlmostEqual(objective['r_max'], 1.22509765625)
        plan = make_plan('disengage_reenter', ('u1',), 'e2', {'u1': 2}, [],
                         ResourceClaims(controls=('u1',)), Commitment('REENTER'), {},
                         c.snapshot.snapshot_id, 'test', goal_data={'objective': objective})
        report = evaluator.effects(c, plan.current_actions)
        self.assertGreater(.6 * evaluator._range_band_gain(c, report, plan, objective), .4)
        self.assertEqual(evaluator._range_band_gain(c, report, plan, dict(objective, r_max=5.)), 0.)

    def test_seed180_real_blockage_retains_evidence_but_sees_joint_yield(self):
        observations, decisions = self.recorded('6z4s', 11, {53, 54, 55, 56})
        memory = TacticalMemory()
        for step in (53, 54, 55, 56):
            c = self.context(observations[step]['payload'])
            memory._observe_movement(c.snapshot, c)
            if step < 56:
                memory._commit_movement(c.snapshot, c, decisions[step]['actions'], [])
        cost = memory.movement_blockage(c, 'u7', 4)
        self.assertGreater(cost, .8)
        actions = dict(next(a['actions'] for a in decisions[56]['top_alternates']
                            if a['actions']['u7'] == 4))
        self.assertEqual(actions['u6'], 4)
        self.assertIn('u6', memory.blocked_directions['u7', 4]['friendly_blockers'])
        positions = {a: c.predict_position(a, x).position for a, x in actions.items()}
        self.assertAlmostEqual(memory.movement_blockage(c, 'u7', 4,
            joint_actions=actions, joint_positions=positions), cost * .5)
        actions['u6'] = 1
        positions['u6'] = c.snapshot.allies['u6'].position
        self.assertEqual(memory.movement_blockage(c, 'u7', 4,
            joint_actions=actions, joint_positions=positions), cost)


if __name__ == '__main__':
    unittest.main()
