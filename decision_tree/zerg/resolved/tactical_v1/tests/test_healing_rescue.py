"""Rescue mechanisms: final actions, bounded lifetime, and native failure frames."""
import copy
import json
from pathlib import Path
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.effects import physical_effects
from skills.tactical_v1.healing import (approach_assignments, approach_facts,
    capability_retention, healing_approach_gain, healing_opportunity)
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.heal_support import HealSupportSkill
from skills.tactical_v1.skill_manager import SkillManager
from skills.tactical_v1.telemetry import Telemetry
from skills.tactical_v1.validator import validate_actions
from .fixtures import enemy, heal_candidates, scene, with_questions


class HealingRescueTests(unittest.TestCase):
    def context(self, state, sid=1):
        return Context(build_snapshot(with_questions(state), sid))

    def state(self, gap=9., energy=50.):
        state = scene([((0., 0.), "medivac"), ((gap, 0.), "marine")])
        state['allies']['u0']['energy'] = energy
        state['allies']['u1']['health'] = 20.
        heal_candidates(state, 'u0')
        return state

    def plans(self, context, memory=None):
        return HealSupportSkill({}).propose(context, memory or TacticalMemory(), Telemetry())

    def test_energy_contract_blocks_candidates_effects_baseline_and_validator(self):
        for energy in (None, 0., .99, True):
            with self.subTest(energy=energy):
                state = self.state(2., energy)
                context = self.context(state)
                self.assertFalse(any(p.commitment.phase in ('sustained_heal', 'meet_patient')
                                     for p in self.plans(context)))
                self.assertFalse(physical_effects(context, {'u0': 7, 'u1': 1}).heal)
                self.assertTrue(any(p['type'] == 'heal_not_executable'
                                    for p in validate_actions([7, 1], context, 2)))
                actions, _ = SkillManager().step(with_questions(state), 2)
                self.assertLess(actions[0], 6)

    def test_one_energy_allows_opportunity_without_claiming_hp(self):
        state = self.state(2., 1.)
        state['allies']['u0']['energy_raw'] = 1.  # exact native value, not rounded 1.00
        context = self.context(state)
        report = physical_effects(context, {'u0': 7, 'u1': 1})
        self.assertGreater(report.heal['u1'].opportunity, 0.)
        self.assertFalse(report.damage_known)

    def test_mechanical_patient_has_no_rescue_credit_even_with_forged_menu(self):
        state = scene([((0., 0.), 'medivac'), ((2., 0.), 'stalker')])
        state['allies']['u1']['health'] = 20.
        heal_candidates(state, 'u0')
        context = self.context(state)
        self.assertFalse(physical_effects(context, {'u0': 7, 'u1': 1}).heal)
        self.assertFalse(any(p.commitment.phase in ('sustained_heal', 'meet_patient')
                             for p in self.plans(context)))

    def test_legal_patient_never_receives_distance_only_meeting(self):
        context = self.context(self.state(3.))
        self.assertTrue(any(p.commitment.phase == 'sustained_heal' for p in self.plans(context)))
        self.assertFalse(any(p.commitment.phase == 'meet_patient' for p in self.plans(context)))
        self.assertIsNone(approach_facts(context, 'u0', 'u1', {'u0': 4, 'u1': 5}))

    def test_healer_only_approach_leaves_patient_free_and_has_future_not_present_credit(self):
        context = self.context(self.state())
        plan = next(p for p in self.plans(context) if p.goal_data.get('approach_mode') == 'healer_only')
        self.assertEqual(plan.controlled_actors, ('u0',))
        actions = {'u0': plan.current_actions['u0'], 'u1': 1}
        report = physical_effects(context, actions)
        self.assertFalse(report.heal)
        self.assertGreater(healing_approach_gain(context, actions, plan, report), 0.)

    def test_patient_move_requires_eta_advantage_and_no_ready_shot(self):
        state = self.state(5.)
        self.assertIsNone(approach_facts(self.context(state), 'u0', 'u1', {'u0': 4, 'u1': 5}),
                          'one healer move reaches the range; moving the patient adds no rescue')
        state = self.state(9.)
        state['visible_enemies'] = {'e0': enemy((12., 0.), 'marine')}
        state['relations']['u0']['e0'] = {'distance': 12., 'smac_attack_available': False,
                                        'in_weapon_range': False, 'weapon_range_status': 'no_compatible_weapon',
                                        'native_weapon_range': None, 'in_enemy_weapon_range': False,
                                        'enemy_weapon_range_status': 'no_compatible_weapon'}
        state['relations']['u1']['e0'] = {'distance': 3., 'smac_attack_available': True,
                                        'in_weapon_range': True, 'weapon_range_status': 'in_range',
                                        'native_weapon_range': 5., 'in_enemy_weapon_range': False,
                                        'enemy_weapon_range_status': 'no_compatible_weapon'}
        context = self.context(state)
        self.assertIsNone(approach_facts(context, 'u0', 'u1', {'u0': 4, 'u1': 5}))
        self.assertIsNotNone(approach_facts(context, 'u0', 'u1', {'u0': 4}))

    def test_lethal_before_arrival_is_not_promised_a_meeting(self):
        state = scene([((0., 0.), 'medivac'), ((9., 0.), 'marine')],
                      {'e0': enemy((9., 1.), 'marauder')})
        state['allies']['u1'].update(health=5., health_lost_since_previous_decision=20.)
        context = self.context(state)
        self.assertFalse(any(p.commitment.phase == 'meet_patient' for p in self.plans(context)))

    def test_unknown_speed_does_not_promise_rescue(self):
        state = self.state()
        del state['unit_type_profiles']['54']['movement_speed_native']
        context = self.context(state)
        self.assertFalse(any(p.commitment.phase == 'meet_patient' for p in self.plans(context)))

    def test_doomed_patient_does_not_reserve_only_rescue_allocation(self):
        state = scene([((0., 0.), 'medivac'), ((9., 0.), 'marine'), ((0., 8.), 'marine')],
                      {'e0': enemy((9., 1.), 'marauder')})
        state['allies']['u1'].update(health=5., health_lost_since_previous_decision=20.)
        state['allies']['u2']['health'] = 25.
        context = self.context(state)
        allocation = approach_assignments(context, TacticalMemory(), ('u0',))
        self.assertEqual(allocation, {'u0': 'u2'})

    def test_two_healers_do_not_allocate_same_future_patient(self):
        state = scene([((0., 0.), 'medivac'), ((0., 1.), 'medivac'),
                       ((8., 0.), 'marine'), ((8., 1.), 'marauder')])
        for actor in ('u2', 'u3'):
            state['allies'][actor]['health'] = 20.
        context = self.context(state)
        allocation = approach_assignments(context, TacticalMemory(), ('u0', 'u1'))
        self.assertEqual(len(allocation), 2)
        self.assertEqual(len(set(allocation.values())), 2)
        self.assertEqual(allocation, approach_assignments(context, TacticalMemory(), ('u1', 'u0')))

    def test_observed_owner_is_not_replaced_by_second_healer_meeting(self):
        state = scene([((0., 0.), 'medivac'), ((5., 0.), 'medivac'), ((8., 0.), 'marine')])
        state['allies']['u2']['health'] = 20.
        state['allies']['u0']['orders'] = [{'kind': 'heal', 'target_unit': 'u2'}]
        heal_candidates(state, 'u1')
        context = self.context(state)
        self.assertFalse(any('u1' in p.controlled_actors and p.goal == 'u2' for p in self.plans(context)))
        self.assertIsNone(approach_facts(context, 'u1', 'u2', {'u1': 4, 'u2': 5}))

    def test_observed_death_releases_owner_but_low_energy_does_not_fake_release(self):
        state = scene([((0., 0.), 'medivac'), ((1., 0.), 'medivac'), ((3., 0.), 'marine')])
        state['allies']['u2']['health'] = 20.
        state['allies']['u0']['orders'] = [{'kind': 'heal', 'target_unit': 'u2'}]
        state['allies']['u0']['energy'] = 0.
        heal_candidates(state, 'u1')
        context = self.context(state)
        self.assertEqual(context.heal_occupancy(), {'u2': 'u0'})
        self.assertFalse(any(p.goal == 'u2' and 'u1' in p.controlled_actors for p in self.plans(context)))
        state['allies']['u0'].update(alive=False, health=0.)
        context = self.context(state, 2)
        self.assertEqual(context.heal_occupancy(), {})
        self.assertTrue(any(p.goal == 'u2' and 'u1' in p.controlled_actors for p in self.plans(context)))

    def test_repeated_order_feedback_is_not_interpreted_as_channel_failure(self):
        state = self.state(2.)
        state['allies']['u0']['orders'] = [{'kind': 'heal', 'target_unit': 'u1'}]
        original = self.context(state)
        expected = healing_opportunity(original, 'u0', 'u1')
        for response in ('AlreadyTargeted', 'Success'):
            updated = copy.deepcopy(state)
            updated['previous_action_ids'] = {'u0': 1}
            updated['last_action_result'] = {'actors': {'u0': {'submission': 'rejected',
                'response': {'name': response}}}}
            context = self.context(updated)
            self.assertEqual(context.heal_occupancy(), {'u1': 'u0'})
            self.assertEqual(healing_opportunity(context, 'u0', 'u1'), expected)

    def scarce_air_state(self, target_x=6.):
        state = scene([((0., 0.), 'medivac'), ((2., 0.), 'marine'), ((2., 1.), 'marauder')],
                      {'e0': enemy((target_x, 0.), 'medivac', is_flying=True)})
        state['unit_type_profiles']['48']['weapons'][0]['target_type'] = 'any'
        state['relations']['u2']['e0'].update(native_weapon_range=None,
            weapon_range_status='no_compatible_weapon', smac_attack_available=False)
        for actor in ('u1', 'u2'):
            state['allies'][actor]['health'] = 20.
        return state

    def test_scarce_visible_attack_domain_changes_rescue_value_without_named_priority(self):
        state = self.scarce_air_state()
        context = self.context(state)
        self.assertGreater(capability_retention(context, 'u1'), capability_retention(context, 'u2'))
        self.assertGreater(healing_opportunity(context, 'u0', 'u1')[3],
                           healing_opportunity(context, 'u0', 'u2')[3])

    def test_distant_air_target_does_not_inflate_unique_patient_rescue(self):
        near = self.context(self.scarce_air_state())
        distant = self.context(self.scarce_air_state(30.))
        self.assertEqual(capability_retention(near, 'u1'), 1.)
        self.assertEqual(capability_retention(distant, 'u1'), 0.)
        self.assertEqual(healing_opportunity(distant, 'u0', 'u1')[3],
                         healing_opportunity(distant, 'u0', 'u2')[3])
        self.assertGreater(healing_opportunity(near, 'u0', 'u1')[3],
                           healing_opportunity(distant, 'u0', 'u1')[3])

    def test_near_term_air_opportunity_uses_known_speed_and_native_command_reach(self):
        state = self.scarce_air_state(8.)  # outside mask, within one move window
        value = capability_retention(self.context(state), 'u1')
        self.assertGreater(value, 0.)
        self.assertLess(value, 1.)
        del state['unit_type_profiles']['48']['movement_speed_native']
        self.assertEqual(capability_retention(self.context(state), 'u1'), 0.)
        # Native range alone cannot promote an unknown command boundary.
        state = self.scarce_air_state(8.)
        del state['relations']['u1']['e0']['smac_attack_distance_limit']
        del state['action_contract']['candidate_center_distance_by_type']['marine']
        self.assertEqual(capability_retention(self.context(state), 'u1'), 0.)

    def test_air_scarcity_respects_current_cooldown_and_missing_evidence(self):
        state = self.scarce_air_state()
        state['allies']['u1']['weapon_cooldown_loops'] = 8.
        value = capability_retention(self.context(state), 'u1')
        self.assertGreater(value, 0.)
        self.assertLess(value, 1.)
        for cooldown in (16., 40., None, True):
            with self.subTest(cooldown=cooldown):
                state['allies']['u1']['weapon_cooldown_loops'] = cooldown
                self.assertEqual(capability_retention(self.context(state), 'u1'), 0.)
        del state['allies']['u1']['weapon_cooldown_loops']
        self.assertEqual(capability_retention(self.context(state), 'u1'), 0.)

    def test_healer_only_lifetime_and_mode_change_cannot_refresh_deadline(self):
        memory = TacticalMemory()
        state = self.state()
        context = self.context(state)
        plan = next(p for p in self.plans(context, memory) if p.goal_data.get('approach_mode') == 'healer_only')
        memory.commit(context.snapshot, {'u0': 4, 'u1': 1}, [plan], context=context)
        state['game_loop'] += 8
        context = self.context(state, 2)
        plans = self.plans(context, memory)
        meetings = [p for p in plans if p.commitment.phase == 'meet_patient']
        self.assertTrue(meetings)
        self.assertTrue(all(p.controlled_actors == ('u0',) for p in meetings))
        self.assertTrue(all(p.proposed_memory_delta['kind'] == 'continue' for p in meetings))
        memory.commit(context.snapshot, {'u0': 4, 'u1': 1}, [meetings[0]], context=context)
        self.assertEqual(memory.instance_for('heal_support', 'u0').start_loop, 80)

    def test_native_seed173_already_legal_and_late_rescue_counterexamples(self):
        root = Path(__file__).resolve().parents[3]
        batch = root / 'runs/tactical_all_races_10v10_9x20_firecycle_20261114/current'
        for roster, step in (('t_4m5r1h', 6), ('t_2m6r2h', 5)):
            path = batch / roster / 'events.jsonl'
            if not path.exists():
                self.skipTest('historical native batch unavailable')
            with path.open() as stream:
                row = next(json.loads(line) for line in stream
                           if '"skill_observation"' in line
                           and (lambda e: e.get('episode') == 4 and e.get('step') == step)(json.loads(line)))
            context = Context(build_snapshot(row['payload'], 1))
            plans = self.plans(context)
            if roster == 't_4m5r1h':
                self.assertFalse(any(p.goal == 'u5' and p.atomic for p in plans),
                                 '25.05 HP patient died before this historical meeting could pay off')
            else:
                self.assertFalse(any('u9' in p.controlled_actors and p.goal == 'u2'
                                     and p.commitment.phase == 'meet_patient' for p in plans))


if __name__ == '__main__':
    unittest.main()
