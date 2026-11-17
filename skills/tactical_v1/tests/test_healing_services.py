"""Current orders, precise energy and final service/alternative rescue facts."""
import copy
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.effects import physical_effects
from skills.tactical_v1.healing import (approach_assignments, approach_options,
    energy_ready, healing_energy, healing_approach_gain, observed_healing_services,
    planned_healing_services, potential_healing_services, rescue_deadline)
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.heal_support import HealSupportSkill
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import heal_candidates, scene, with_questions


class HealingServiceTests(unittest.TestCase):
    def state(self, two=False):
        state = scene([((0., 0.), 'medivac')] +
                      ([((1., 0.), 'medivac')] if two else []) +
                      [((3., 0.), 'marine'), ((0., 3.), 'marauder')])
        for actor, unit in state['allies'].items():
            if unit['type'] != 'medivac':
                unit['health'] = 20.
        for actor, unit in state['allies'].items():
            if unit['type'] == 'medivac':
                heal_candidates(state, actor)
        return state

    def context(self, state):
        return Context(build_snapshot(with_questions(state), 1))

    def plans(self, context, memory=None):
        return HealSupportSkill({}).propose(context, memory or TacticalMemory(), Telemetry())

    def test_exact_energy_precedes_rounded_display_and_invalid_raw_never_falls_back(self):
        for raw, expected in ((.996, False), (1., True), (1.2, True),
                              (None, False), (True, False), (float('nan'), False)):
            with self.subTest(raw=raw):
                state = self.state()
                state['allies']['u0'].update(energy=1., energy_raw=raw)
                context = self.context(state)
                self.assertEqual(energy_ready(context, 'u0'), expected)
                self.assertEqual(bool(physical_effects(context, {'u0': 7, 'u1': 1, 'u2': 1}).heal), expected)
                self.assertEqual(bool(planned_healing_services(context, {'u0': 7, 'u1': 1})), expected)

    def test_legacy_quantization_boundary_is_unknown_not_a_fake_native_threshold(self):
        for displayed, status in ((.99, 'insufficient'), (1., 'uncertain'),
                                  (1.01, 'sufficient'), (50., 'sufficient')):
            state = self.state()
            state['allies']['u0']['energy'] = displayed
            self.assertEqual(healing_energy(self.context(state), 'u0')['status'], status)

    def test_queued_heal_is_not_a_current_owner(self):
        state = self.state(two=True)
        state['allies']['u0']['orders'] = [
            {'kind': 'move'}, {'kind': 'heal', 'target_unit': 'u2'}]
        context = self.context(state)
        self.assertEqual(context.heal_order_owners(), {})
        self.assertEqual(context.queued_heal_owners(), {'u2': frozenset(('u0',))})
        self.assertEqual(context.heal_occupancy(), {})
        self.assertFalse(observed_healing_services(context))
        self.assertTrue(any(p.goal == 'u2' and p.controlled_actors == ('u1',)
                            for p in self.plans(context)))

    def test_first_and_later_heals_remain_separate(self):
        state = self.state()
        state['allies']['u0']['orders'] = [
            {'kind': 'heal', 'target_unit': 'u1'}, {'kind': 'heal', 'target_unit': 'u2'}]
        context = self.context(state)
        self.assertEqual(context.heal_order_owners(), {'u1': frozenset(('u0',))})
        self.assertEqual(context.queued_heal_owners(), {'u2': frozenset(('u0',))})
        self.assertEqual(set(observed_healing_services(context)), {'u1'})

    def test_multiple_owners_are_preserved_and_canonical_service_is_not_iteration_order(self):
        state = self.state(two=True)
        for actor in ('u0', 'u1'):
            state['allies'][actor]['orders'] = [{'kind': 'heal', 'target_unit': 'u2'}]
        state['allies']['u0']['energy'] = 0.
        for reverse in (False, True):
            candidate = copy.deepcopy(state)
            if reverse:
                candidate['allies'] = dict(reversed(list(candidate['allies'].items())))
            context = self.context(candidate)
            self.assertEqual(context.heal_order_owners(), {'u2': frozenset(('u0', 'u1'))})
            self.assertEqual(context.heal_occupancy(), {'u2': 'u1'})
            services = observed_healing_services(context)
            self.assertEqual(set(services), {'u2'})
            self.assertEqual(services['u2']['healer'], 'u1')
            self.assertEqual(services['u2']['observed_owners'], ('u0', 'u1'))

    def test_sole_zero_energy_owner_is_reserved_but_cannot_promise_service(self):
        state = self.state(two=True)
        state['allies']['u0'].update(energy=0., orders=[{'kind': 'heal', 'target_unit': 'u2'}])
        context = self.context(state)
        self.assertEqual(context.heal_occupancy(), {'u2': 'u0'})
        self.assertNotIn('u2', potential_healing_services(context))
        self.assertNotIn('u2', observed_healing_services(context))

    def test_dead_patient_and_dead_healer_do_not_reserve_current_or_queued_orders(self):
        state = self.state(two=True)
        state['allies']['u0'].update(alive=False, health=0., orders=[{'kind': 'heal', 'target_unit': 'u3'}])
        state['allies']['u2'].update(alive=False, health=0.)
        state['allies']['u1']['orders'] = [{'kind': 'heal', 'target_unit': 'u2'}, {'kind': 'heal', 'target_unit': 'missing'}]
        context = self.context(state)
        self.assertEqual(context.heal_order_owners(), {})
        self.assertEqual(context.queued_heal_owners(), {})

    def test_nearby_possible_healer_is_not_an_assigned_service(self):
        context = self.context(self.state())
        self.assertIn('u1', potential_healing_services(context))
        self.assertFalse(observed_healing_services(context))
        self.assertFalse(planned_healing_services(context, {'u0': 1, 'u1': 1, 'u2': 1}))
        self.assertFalse(planned_healing_services(context, {'u1': 1}))
        self.assertFalse(planned_healing_services(context, {'u0': 7}))

    def test_stop_retains_only_observed_service_with_discount_not_a_tick_claim(self):
        state = self.state()
        state['allies']['u0']['orders'] = [{'kind': 'heal', 'target_unit': 'u1'}]
        context = self.context(state)
        stopped = planned_healing_services(context, {'u0': 1, 'u1': 1})['u1']
        submitted = planned_healing_services(context, {'u0': 7, 'u1': 1})['u1']
        self.assertEqual(stopped['kind'], 'observed_retained')
        self.assertEqual(submitted['kind'], 'submitted_heal')
        self.assertLess(stopped['opportunity'], submitted['opportunity'])
        self.assertNotIn('healed_hp', stopped)

    def test_final_healer_move_or_patient_move_does_not_promise_stationary_service(self):
        state = self.state()
        state['allies']['u0']['orders'] = [{'kind': 'heal', 'target_unit': 'u1'}]
        context = self.context(state)
        self.assertFalse(planned_healing_services(context, {'u0': 4, 'u1': 1}))
        self.assertFalse(planned_healing_services(context, {'u0': 7, 'u1': 4}))

    def test_final_target_switch_does_not_credit_both_old_and_new_service(self):
        state = self.state()
        state['allies']['u0']['orders'] = [{'kind': 'heal', 'target_unit': 'u1'}]
        services = planned_healing_services(self.context(state), {'u0': 8, 'u1': 1, 'u2': 1})
        self.assertEqual(set(services), {'u2'})

    def test_retained_second_observed_owner_can_serve_after_preferred_owner_retargets(self):
        state = self.state(two=True)
        for actor in ('u0', 'u1'):
            state['allies'][actor]['orders'] = [{'kind': 'heal', 'target_unit': 'u2'}]
        # u1 is nearer and therefore the preferred allocation before choices.
        context = self.context(state)
        self.assertEqual(context.heal_occupancy()['u2'], 'u1')
        services = planned_healing_services(context, {'u0': 1, 'u1': 9, 'u2': 1, 'u3': 1})
        self.assertEqual(set(services), {'u2', 'u3'})
        self.assertEqual(services['u2']['healer'], 'u0')
        self.assertEqual(services['u2']['kind'], 'observed_retained')
        self.assertEqual(services['u3']['healer'], 'u1')

    def test_two_explicit_commands_do_not_supply_two_or_even_one_valid_service(self):
        context = self.context(self.state(two=True))
        self.assertFalse(planned_healing_services(context, {'u0': 8, 'u1': 8, 'u2': 1, 'u3': 1}))

    def rescue_state(self):
        state = scene([((0., 0.), 'medivac'), ((8., 0.), 'marine'),
                       ((0., 8.), 'marine'), ((-8., 0.), 'marine')])
        for actor, health in (('u1', 44.), ('u2', 15.), ('u3', 25.)):
            state['allies'][actor]['health'] = health
        return state

    def test_continuation_is_bounded_and_a_better_patient_remains_a_candidate(self):
        context, memory = self.context(self.rescue_state()), TacticalMemory()
        memory.create_instance('heal_support', ('u0',), 'u1', 'meet_patient', 80, 48,
                               goal_kind='meeting', extra={'healer': 'u0'})
        options = approach_options(context, memory, ('u0',))
        self.assertEqual(options['u0'], ('u2', 'u1'))
        self.assertEqual(approach_assignments(context, memory, ('u0',)), {'u0': 'u2'})
        candidates = [p for p in self.plans(context, memory) if p.commitment.phase == 'meet_patient']
        self.assertEqual({p.goal for p in candidates}, {'u1', 'u2'})
        self.assertTrue(any(p.controlled_actors == ('u0',) for p in candidates))

    def test_future_options_are_bounded_disjoint_and_independent_of_healer_order(self):
        state = scene([((0., 0.), 'medivac'), ((0., 1.), 'medivac'),
                       ((8., 0.), 'marine'), ((8., 1.), 'marauder'),
                       ((-8., 0.), 'marine'), ((-8., 1.), 'marauder')])
        for actor in ('u2', 'u3', 'u4', 'u5'):
            state['allies'][actor]['health'] = 20.
        context, memory = self.context(state), TacticalMemory()
        options = approach_options(context, memory, ('u0', 'u1'))
        self.assertEqual(options, approach_options(context, memory, ('u1', 'u0')))
        self.assertEqual({len(values) for values in options.values()}, {2})
        self.assertFalse(set(options['u0']) & set(options['u1']))

    def test_remaining_deadline_rejects_new_distance_progress_without_refresh(self):
        state = self.rescue_state()
        state['game_loop'] = 120
        context, memory = self.context(state), TacticalMemory()
        instance = memory.create_instance('heal_support', ('u0',), 'u2', 'meet_patient', 80, 48,
                                           goal_kind='meeting', extra={'healer': 'u0'})
        self.assertEqual(rescue_deadline(context, memory, 'u0', 'u2'), 128)
        self.assertFalse(any(p.goal == 'u2' and p.commitment.phase == 'meet_patient'
                             for p in self.plans(context, memory)))
        self.assertEqual((instance.start_loop, instance.max_loops), (80, 48))
        fresh_context = self.context(self.rescue_state())
        plan = next(p for p in self.plans(fresh_context) if p.goal == 'u2' and p.commitment.phase == 'meet_patient')
        plan.goal_data['rescue_deadline_loop'] = 128
        actions = {'u0': plan.current_actions['u0'], 'u1': 1, 'u2': 1, 'u3': 1}
        self.assertEqual(healing_approach_gain(context, actions, plan, physical_effects(context, actions)), 0.)


if __name__ == '__main__':
    unittest.main()
