"""A supporting group must reach usable weapons, not merely a distance ring."""
import copy
import json
import math
from pathlib import Path
import unittest

from skills.tactical_v1.combat_groups import effective_support, task_interrupt
from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.combat_rejoin import CombatRejoinSkill
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import enemy, scene, with_questions


class EffectiveRejoinTests(unittest.TestCase):
    def context(self, state):
        return Context(build_snapshot(with_questions(state), state['game_loop']))

    def frontline(self):
        state = scene([((0., 0.), 'zealot'), ((3.5, 0.), 'zealot'),
                       ((6.5, 0.), 'stalker')], enemies={'e0': enemy((8., 0.), 'stalker')})
        for actor, unit in state['allies'].items():
            d = math.dist(unit['position'], (8., 0.))
            reach = .1 if unit['type'] == 'zealot' else 6.
            state['relations'][actor]['e0'].update(
                native_weapon_range=reach, in_weapon_range=d <= reach + 1.,
                weapon_range_status='in_range' if d <= reach + 1. else 'out_of_range',
                smac_attack_distance_limit=2. if unit['type'] == 'zealot' else 6.,
                smac_attack_available=d <= (2. if unit['type'] == 'zealot' else 6.))
        return state

    def test_melee_inside_support_radius_keeps_closing_to_real_reach(self):
        context = self.context(self.frontline())
        data = {'joiners': ['u0'], 'supporters': ['u1', 'u2'], 'anchor': [6.5, 0.],
                'support_radius': 6., 'roles': {'u0': 'join', 'u1': 'cover', 'u2': 'cover'}}
        actions, _ = CombatRejoinSkill({})._actions(context, data)
        self.assertEqual(actions['u1'], 4)
        self.assertEqual(actions['u2'], 6)
        self.assertEqual(actions['u0'], 4)
        before = effective_support(context, ['u0'], ['u1', 'u2'])
        after = effective_support(context, ['u0'], ['u1', 'u2'], positions={
            a: context.predict_position(a, move).position for a, move in actions.items()})
        self.assertGreater(before['eta_by_actor']['u1'], 2.)
        self.assertLess(after['gap_by_actor']['u1'], before['gap_by_actor']['u1'])

    def test_wounded_receiver_is_unsafe_between_hits(self):
        state = self.frontline()
        state['allies']['u2'].update(health=8., max_health=80., health_lost_since_previous_decision=0.)
        context = self.context(state)
        memory = TacticalMemory()
        instance = memory.create_instance('combat_rejoin', ['u0', 'u1', 'u2'], 'join',
            'rejoining', 80, 64, goal_kind='area', extra={'joiners': ['u0', 'u1'],
             'supporters': ['u2'], 'initial_threat_counts': {'u2': 1}})
        self.assertEqual(task_interrupt(context, instance), 'cover_emergency')
        self.assertEqual(effective_support(context, ['u0', 'u1'], ['u2'])['cover_viability'], 0.)

    def test_short_survival_cannot_promise_long_waiting_cover(self):
        state = scene([((0., 0.), 'stalker'), ((14., 0.), 'stalker')],
                      enemies={'e0': enemy((16., 0.), 'colossus')})
        context = self.context(state)
        facts = effective_support(context, ['u0'], ['u1'])
        self.assertGreater(facts['arrival_windows'], 5.)
        self.assertLess(facts['cover_viability'], .5)
        self.assertTrue(all(p.goal_data['supporters'] != ['u1'] for p in
            CombatRejoinSkill({}).propose(context, TacticalMemory(), Telemetry())))

    def test_geometric_join_is_not_completion_without_combat_readiness(self):
        state = self.frontline()
        context = self.context(state)
        memory = TacticalMemory()
        instance = memory.create_instance('combat_rejoin', ['u1', 'u2'], 'join',
            'rejoining', 72, 64, goal_kind='area', extra={'joiners': ['u1'],
             'supporters': ['u2'], 'support_radius': 6., 'readiness_fraction': 1.,
             'roles': {'u1': 'join', 'u2': 'cover'}, 'no_progress_loops': 24,
             'last_support_progress_loop': 72, 'best_support_gap': 50.})
        from skills.tactical_v1.combat_groups import task_outcome
        self.assertIsNone(task_outcome(context, instance))

    def test_effective_support_cache_respects_changed_final_positions(self):
        context = self.context(self.frontline())
        first = copy.deepcopy(effective_support(context, ['u0'], ['u1', 'u2']))
        second = effective_support(context, ['u0'], ['u1', 'u2'], positions={'u1': (7., 0.)})
        self.assertLess(second['eta_by_actor']['u1'], first['eta_by_actor']['u1'])
        self.assertEqual(first, effective_support(context, ['u0'], ['u1', 'u2']))

    def test_seed181_healthy_receivers_move_and_critical_cover_is_rejected(self):
        path = Path(__file__).resolve().parents[3] / 'runs/tactical_6z4s_20_rejoin_20261113/6z4s/events.jsonl'
        if not path.exists():
            self.skipTest('optional local real-run evidence')
        snapshots = {}
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if '"event": "skill_observation"' not in line:
                    continue
                row = json.loads(line)
                if row['episode'] == 12 and row['step'] in (17, 19, 23):
                    snapshots[row['step']] = row['payload']
                    if len(snapshots) == 3:
                        break
        self.assertEqual(len(snapshots), 3)
        skill = CombatRejoinSkill({})
        for step in (19, 23):
            context = Context(build_snapshot(snapshots[step], step + 1))
            data = {'joiners': ['u4'], 'supporters': ['u0', 'u1', 'u2', 'u3', 'u6', 'u7', 'u8'],
                    'support_radius': 6., 'anchor': list(context.snapshot.allies['u0'].position),
                    'roles': {'u4': 'join', **{a: 'cover' for a in ('u0','u1','u2','u3','u6','u7','u8')}}}
            result = skill._actions(context, data, active=True)
            self.assertIsNotNone(result)
            for actor in ('u1', 'u2', 'u3', 'u6'):
                self.assertIn(result[0][actor], (2, 3, 4, 5))
        context = Context(build_snapshot(snapshots[17], 18))
        self.assertEqual(context.snapshot.allies['u5'].health_lost, 0.)
        self.assertLess(context.snapshot.allies['u5'].health_fraction, .25)
        plans = skill.propose(context, TacticalMemory(), Telemetry())
        self.assertFalse(any('u5' in p.goal_data['supporters'] for p in plans))


if __name__ == '__main__':
    unittest.main()
