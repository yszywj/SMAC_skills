"""Battle participation rules across reach, health and formation geometries."""
import unittest

from skills.tactical_v1.combat_groups import connected_groups
from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.coordinated_engage import CoordinatedEngageSkill
from skills.tactical_v1.skills.disengage_reenter import DisengageReenterSkill, ESCAPE, REENTER
from skills.tactical_v1.skills.combat_rejoin import CombatRejoinSkill
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import enemy, scene, with_questions


class SustainedSkillsTests(unittest.TestCase):
    def context(self, state):
        return Context(build_snapshot(with_questions(state), state['game_loop']))

    def plans(self, skill, state, memory=None):
        return skill.propose(self.context(state), memory or TacticalMemory(), Telemetry())

    def test_healthy_rear_never_waits_for_already_fighting_teammate(self):
        for unit in ('marine', 'stalker', 'hydralisk', 'zealot'):
            with self.subTest(unit=unit):
                state = scene([((0., 0.), unit), ((6., 0.), unit)],
                              enemies={'e0': enemy((8., 0.))})
                rear = [p for p in self.plans(CoordinatedEngageSkill({}), state)
                        if p.controlled_actors == ('u0',)]
                self.assertTrue(rear)
                self.assertFalse(any(p.commitment.phase == 'waiting' for p in rear))

    def test_wait_requires_cooldown_and_next_window_partner_arrival(self):
        skill = CoordinatedEngageSkill({})
        state = scene([((2.7, 0.), 'marine'), ((2.1, 0.), 'marine')],
                      enemies={'e0': enemy((8., 0.))})
        state['allies']['u0'].update(weapon_cooldown_loops=12., weapon_ready=False)
        waits = [p for p in self.plans(skill, state)
                 if p.controlled_actors == ('u0',) and p.commitment.phase == 'waiting']
        self.assertTrue(waits)
        self.assertEqual(waits[0].features.get('hold_value', 0.), 0.)
        state['allies']['u0']['weapon_cooldown_loops'] = 0.
        self.assertFalse(any(p.commitment.phase == 'waiting' for p in self.plans(skill, state)))
        state['allies']['u0']['weapon_cooldown_loops'] = 12.
        state['allies']['u1']['health_lost_since_previous_decision'] = 1.
        self.assertFalse(any(p.commitment.phase == 'waiting' for p in self.plans(skill, state)))

    def test_connected_chain_gets_functional_join_without_moving_current_shooter(self):
        for unit in ('marine', 'stalker', 'hydralisk'):
            with self.subTest(unit=unit):
                state = scene([((8., 0.), unit), ((12., 0.), unit), ((16., 0.), unit)],
                              enemies={'e0': enemy((20., 0.))})
                context = self.context(state)
                self.assertEqual(len(connected_groups(context, 6.)), 1)
                plans = CombatRejoinSkill({}).propose(context, TacticalMemory(), Telemetry())
                self.assertTrue(plans)
                self.assertTrue(all(p.current_actions['u2'] == 6 for p in plans))
                self.assertTrue(any('u1' in p.goal_data['joiners'] for p in plans))
                self.assertTrue(all(context.decode(a, p.current_actions[a]).kind == 'move'
                    for p in plans for a in p.goal_data['joiners']))

    def test_connected_shooters_are_not_reclassified_during_cooldown(self):
        state = scene([((14., 0.), 'marine'), ((15., 0.), 'marine')],
                      enemies={'e0': enemy((18., 0.))})
        for unit in state['allies'].values():
            unit.update(weapon_cooldown_loops=30., weapon_ready=False)
        self.assertEqual(self.plans(CombatRejoinSkill({}), state), [])

    def test_functional_join_lives_until_weapon_readiness_not_geometric_connection(self):
        skill, memory = CombatRejoinSkill({}), TacticalMemory()

        def observe(xs, loop):
            state = scene([((x, 0.), 'marine') for x in xs], loop=loop,
                          enemies={'e0': enemy((20., 0.))})
            context = self.context(state)
            memory.begin_step(context.snapshot)
            memory.observe(context.snapshot)
            memory.update_task_facts(context.snapshot, context)
            return context

        first = observe((10., 12., 16.), 80)
        self.assertEqual(len(connected_groups(first, 6.)), 1)
        plan = skill.propose(first, memory, Telemetry())[0]
        memory.commit(first.snapshot, plan.current_actions, [plan], context=first)
        identity = plan.instance_id
        original = memory.instances[identity]
        second = observe((11., 13., 16.), 88)
        self.assertIn(identity, memory.instances)
        continued = skill.propose(second, memory, Telemetry())[0]
        self.assertEqual(continued.instance_id, identity)
        self.assertEqual(continued.proposed_memory_delta['kind'], 'continue')
        self.assertEqual(memory.instances[identity].start_loop, original.start_loop)
        memory.commit(second.snapshot, continued.current_actions, [continued], context=second)
        observe((15., 15.5, 16.), 96)
        self.assertNotIn(identity, memory.instances)
        self.assertFalse(memory.retry_records)

    def test_exhausted_retreat_returns_to_reachable_battle_target(self):
        state = scene([((0., 0.), 'marine')], enemies={
            'e0': enemy((8., 0.)), 'e1': enemy((0., 9.))})
        memory = TacticalMemory()
        memory.retreat_exhausted = lambda *_: True
        memory.battle_target = lambda *_: 'e1'
        plans = self.plans(DisengageReenterSkill({}), state, memory)
        self.assertTrue(plans)
        self.assertTrue(all(p.commitment.phase == REENTER for p in plans))
        self.assertTrue(all(p.goal == 'e1' for p in plans))
        self.assertEqual(plans[0].proposed_memory_delta['goal_kind'], 'enemy')

    def test_exhausted_retreat_can_compare_improving_escape_without_forcing_reentry(self):
        state = scene([((0., 0.), 'marine')], enemies={'e0': enemy((4., 0.))})
        state['allies']['u0'].update(weapon_cooldown_loops=24., weapon_ready=False)
        memory = TacticalMemory()
        memory.retreat_exhausted = lambda *_: True
        context = self.context(state)
        skill = DisengageReenterSkill({})
        plans = skill.propose(context, memory, Telemetry())
        self.assertTrue(plans)
        for plan in plans:
            self.assertEqual(plan.commitment.phase, ESCAPE)
            position = context.predict_position('u0', plan.current_actions['u0']).position
            self.assertLess(skill._forecast_pressure(context, 'u0', position),
                            skill._forecast_pressure(context, 'u0', context.snapshot.allies['u0'].position))

    def test_exhausted_retreat_never_blocks_critical_escape(self):
        state = scene([((0., 0.), 'marine')], distance=2., attack_range=6.,
                      enemies={'e0': enemy((5., 0.), 'marine'),
                               'e1': enemy((5., .5), 'marine')})
        state['allies']['u0'].update(health=5., health_lost_since_previous_decision=10.,
                                    weapon_cooldown_loops=24., weapon_ready=False)
        memory = TacticalMemory()
        memory.retreat_exhausted = lambda *_: True
        plans = self.plans(DisengageReenterSkill({}), state, memory)
        self.assertTrue(plans)
        self.assertTrue(all(p.commitment.phase == ESCAPE for p in plans))


if __name__ == '__main__':
    unittest.main()
