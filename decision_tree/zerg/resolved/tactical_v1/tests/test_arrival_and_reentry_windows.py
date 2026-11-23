"""Capability-driven arrival reserve and observed, bounded re-entry memory."""
import copy
import json
from pathlib import Path
import unittest

from skills.tactical_v1.combat_exchange import (
    movement_exchange, ReentryWindowMemory, safe_reentry_window)
from skills.tactical_v1.combat_groups import viable_rejoin_data
from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.combat_rejoin import CombatRejoinSkill
from skills.tactical_v1.skills.disengage_reenter import DisengageReenterSkill
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import scene, enemy, with_questions


def context(state):
    payload = state if 'state' in state else with_questions(state)
    return Context(build_snapshot(payload, payload['state']['game_loop']))


def boundary(x=0., loop=80, health=20., count=3):
    state = scene([((x, 0.), 'marine')],
        enemies={f'e{i}': enemy((6.3, i * .1), 'hydralisk', tag=100+i)
                 for i in range(count)}, loop=loop)
    state['allies']['u0'].update(health=health, max_health=health, effective_health=health)
    return state


class ArrivalAndReentryWindowTests(unittest.TestCase):
    def test_same_threats_use_own_ehp_not_full_health_percentage(self):
        fragile = context(boundary(health=20.))
        robust = context(boundary(health=125.))
        self.assertFalse(movement_exchange(fragile, 'u0', (1., 0.))['admissible'])
        facts = movement_exchange(robust, 'u0', (1., 0.))
        self.assertTrue(facts['admissible'])
        self.assertGreater(facts['after']['amount'], 0.)

    def test_unknown_speed_does_not_fabricate_arrival(self):
        state = boundary(health=125.)
        state['unit_type_profiles']['48'].pop('movement_speed_native')
        self.assertFalse(movement_exchange(context(state), 'u0', (1., 0.))['admissible'])

    def test_observed_loss_and_cooldown_share_the_same_window(self):
        state = boundary(x=1., health=100., count=1)
        state['allies']['u0'].update(health_lost_since_previous_decision=40.,
                                   weapon_cooldown_loops=24.)
        facts = movement_exchange(context(state), 'u0', (1.2, 0.))
        self.assertEqual(facts['after']['horizon_h'], 3.)
        self.assertGreater(facts['after']['observed_trend'], 100.)
        self.assertFalse(facts['admissible'])

    def escaped(self):
        memory = ReentryWindowMemory()
        before, after = context(boundary()), context(boundary(-1., 88))
        memory.observe(before)
        memory.commit(before, {'u0': 5})
        memory.observe(after)
        self.assertIn('u0', memory.records)
        return memory, after

    def test_guard_requires_observed_relief_and_is_query_pure(self):
        memory, after = self.escaped()
        original = copy.deepcopy(memory.__dict__)
        self.assertFalse(memory.allowed(after, 'u0', (0., 0.)))
        self.assertTrue(memory.allowed(after, 'u0', (-2., 0.)))
        self.assertEqual(memory.__dict__, original)
        blocked = ReentryWindowMemory()
        blocked.commit(context(boundary()), {'u0': 5})
        blocked.observe(context(boundary(loop=88)))
        self.assertFalse(blocked.records)

    def test_guard_expires_without_renewal_and_real_fire_clears_it(self):
        memory, after = self.escaped()
        deadline = memory.records['u0']['until_loop']
        memory.commit(after, {'u0': 5})
        memory.observe(context(boundary(-2., 96)))
        self.assertEqual(memory.records['u0']['until_loop'], deadline)
        self.assertTrue(memory.allowed(context(boundary(-1., deadline)), 'u0', (0., 0.)))
        memory.commit(context(boundary(-1., 96)), {'u0': 1})
        fired = boundary(-1., 100)
        fired['allies']['u0']['weapon_cooldown_loops'] = 12.
        memory.observe(context(fired))
        self.assertFalse(memory.records)

    def test_identity_reset_and_disabled_guard(self):
        memory, _ = self.escaped()
        replaced = boundary(-1., 96)
        replaced['allies']['u0']['tag'] = 'replacement'
        memory.observe(context(replaced))
        self.assertFalse(memory.records)
        memory = TacticalMemory()
        memory.reentry_window.records['u0'] = {'test': True}
        memory.reset()
        self.assertFalse(memory.reentry_window.records)
        disabled = ReentryWindowMemory({'common_combat': {'reentry_window_enabled': False}})
        disabled.commit(context(boundary()), {'u0': 5})
        disabled.observe(context(boundary(-1., 88)))
        self.assertFalse(disabled.records)

    def test_legal_next_shot_with_reserve_is_not_blocked(self):
        memory, _ = self.escaped()
        state = boundary(-1., 96, count=1)
        state['visible_enemies']['e0']['position'] = [5., 0.]
        c = context(state)
        facts = movement_exchange(c, 'u0', (0., 0.))
        self.assertTrue(facts['ready_after_move'])
        self.assertTrue(facts['admissible'])
        self.assertTrue(memory.allowed(c, 'u0', (0., 0.)))

    def test_g4_safe_phase_uses_forecast_not_empty_physical_threats(self):
        c = context(boundary())
        self.assertFalse(c.risk_at('u0', (0., 0.)).threats)
        self.assertFalse(safe_reentry_window(c, 'u0'))
        memory = TacticalMemory()
        task = memory.create_instance('disengage_reenter', ['u0'], 'safe', 'ESCAPE',
                                      c.snapshot.game_loop, 96, goal_kind='internal')
        self.assertEqual(DisengageReenterSkill({})._phase(c, 'u0', task), 'ESCAPE')
        old = DisengageReenterSkill({'common_combat': {'reentry_window_enabled': False}})
        self.assertEqual(old._phase(c, 'u0', task), 'REENTER')
        self.assertEqual(task.start_loop, c.snapshot.game_loop)
        self.assertEqual(task.max_loops, 96)


class RecordedArrivalAndReentryTests(unittest.TestCase):
    @staticmethod
    def recorded(roster, episode):
        path = Path(__file__).resolve().parents[3] / (
            'runs/tactical_all_races_10v10_9x20_firecycle_20261114/current/' + roster + '/events.jsonl')
        if not path.exists():
            raise unittest.SkipTest('optional immutable completed-run observations unavailable')
        rows, payload = [], None
        with path.open(encoding='utf8') as stream:
            for line in stream:
                if '"skill_observation"' not in line and '"skill_step"' not in line:
                    continue
                row = json.loads(line)
                if row.get('episode') != episode:
                    continue
                if row['event'] == 'skill_observation':
                    payload = row['payload']
                elif row['event'] == 'skill_step':
                    rows.append((row['step'], context(payload), row))
        return rows

    def test_t173_observed_escape_guard_blocks_three_unproductive_returns(self):
        memory = ReentryWindowMemory()
        blocked = []
        for step, c, row in self.recorded('t_6m4r', 4):
            memory.observe(c)
            if step in (39, 42, 45):
                point = c.predict_position('u8', 5).position
                self.assertFalse(memory.allowed(c, 'u8', point))
                blocked.append(step)
            if step == 30:
                self.assertTrue(memory.allowed(c, 'u8', c.predict_position('u8', 4).position))
            memory.commit(c, {a: int(action) for a, action in row['actions'].items()})
        self.assertEqual(blocked, [39, 42, 45])

    def test_z171_joiners_cannot_borrow_healthy_cover_reserve(self):
        _, c, _ = next(r for r in self.recorded('z_6l4h', 2) if r[0] == 4)
        data = {'joiners': ['u0', 'u2', 'u3', 'u5'], 'supporters': ['u7', 'u8'],
            'roles': {**{a: 'join' for a in ('u0', 'u2', 'u3', 'u5')}, 'u7': 'cover', 'u8': 'cover'},
            'support_radius': 6., 'local_cover_enabled': True,
            'arrival_exposure_enabled': True}
        before = copy.deepcopy(data)
        retained = viable_rejoin_data(c, data, 8.)
        self.assertIsNotNone(retained)
        self.assertNotIn('u5', retained['joiners'])
        self.assertEqual(data, before)
        self.assertFalse(movement_exchange(c, 'u2', c.predict_position('u2', 4).position)['admissible'])
        skill = CombatRejoinSkill({})
        self.assertNotEqual(skill._engage_move(c, 'u2', data), 4)
        legacy = dict(data, arrival_exposure_enabled=False)
        self.assertIn('u5', viable_rejoin_data(c, legacy, 8.)['joiners'])


if __name__ == '__main__':
    unittest.main()
