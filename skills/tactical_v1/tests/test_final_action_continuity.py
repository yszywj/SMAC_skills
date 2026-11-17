"""Integration boundaries: final target budget, timing, and safety gates."""
import unittest
from unittest.mock import patch

from skills.tactical_v1.combat_exchange import escape_urgency, next_fire_timing
from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.evaluator import DEFAULT_WEIGHTS, Evaluator
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.joint_combat import assess_joint
from skills.tactical_v1.local_entry import assess_local_entry
from .fixtures import enemy, scene, with_questions


def context(raw):
    return Context(build_snapshot(with_questions(raw), raw['game_loop']))


def departure(health=5., cooldown=9.):
    raw = scene([((.3, 1.), 'marine'), ((0., 0.), 'marine')],
                {'e0': enemy((4.8, 0.), 'hydralisk', health=health)},
                enemy_can_reach=False, distance=2.)
    raw['allies']['u1']['weapon_cooldown_loops'] = cooldown
    raw['allies']['u1']['orders'] = [{'kind': 'attack', 'target_unit': 'e0'}]
    return raw


def costs(c, actions, memory=None, evaluator=None):
    e = evaluator or Evaluator()
    return e._coordination_costs(c, actions, e.effects(c, actions), memory, [])


def shot_cost(terms):
    return -sum(terms[k] for k in ('attack_continuity', 'next_fire_delay', 'return_fire_continuity'))


class FinalActionContinuityTests(unittest.TestCase):
    def test_depleted_backup_target_does_not_create_delay_with_or_without_memory(self):
        c = context(departure())
        self.assertGreater(next_fire_timing(c, 'u1', 5)['cost'], 0.)
        for memory in (None, TacticalMemory()):
            with self.subTest(memory=memory is not None):
                self.assertEqual(shot_cost(costs(c, {'u0': 6, 'u1': 5}, memory)), 0.)

    def test_same_actor_move_reprices_when_peer_attack_becomes_stop(self):
        c, memory = context(departure()), TacticalMemory()
        kill = {'u0': 6, 'u1': 5}
        idle = {'u0': 1, 'u1': 5}
        self.assertEqual(shot_cost(costs(c, kill, memory)), 0.)
        self.assertGreater(shot_cost(costs(c, idle, memory)), 0.)
        self.assertEqual(shot_cost(costs(c, kill, memory)), 0., 'no branch cache leakage')

    def test_budget_passes_explicit_empty_primary_not_none_fallback(self):
        c, memory = context(departure()), TacticalMemory()
        with patch.object(memory, 'next_fire_cost', wraps=memory.next_fire_cost) as spy:
            costs(c, {'u0': 6, 'u1': 5}, memory)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.call_args.kwargs['eligible_targets'], ())

    def test_backup_near_target_cannot_hide_delay_to_the_remaining_front(self):
        raw = scene([((.3, 1.), 'marine'), ((0., 0.), 'marine')], {
            'e0': enemy((4.8, 0.), 'hydralisk', health=5.),
            'e1': enemy((8., 0.), 'hydralisk', health=90.)},
            enemy_can_reach=False, distance=2.)
        raw['allies']['u1']['weapon_cooldown_loops'] = 16.
        c, memory = context(raw), TacticalMemory()
        self.assertEqual(next_fire_timing(c, 'u1', 5)['cost'], 0., 'near target masks transfer time')
        with patch.object(memory, 'next_fire_cost', wraps=memory.next_fire_cost) as spy:
            terms = costs(c, {'u0': 6, 'u1': 5}, memory)
        self.assertEqual(spy.call_args.kwargs['eligible_targets'], ('e1',))
        self.assertGreater(shot_cost(terms), 0.)

    def test_current_window_legal_shot_is_kept_even_if_peer_budgets_the_kill(self):
        c = context(departure(cooldown=0.))
        for memory in (None, TacticalMemory()):
            self.assertGreater(shot_cost(costs(c, {'u0': 6, 'u1': 5}, memory)), 0.)
        self.assertIsNotNone(c.attack_action('u1', 'e0'))

    def test_safe_low_health_tiny_chip_does_not_bypass_joint_cost(self):
        raw = departure(health=90.)
        raw['allies']['u1'].update(health=10., effective_health=10., health_lost_since_previous_decision=.01)
        c, memory = context(raw), TacticalMemory()
        self.assertFalse(escape_urgency(c, 'u1')['immediate'])
        self.assertFalse(memory.combat_progress._safety_override(c, 'u1', 5, memory))
        self.assertGreater(shot_cost(costs(c, {'u0': 1, 'u1': 5}, memory)), 0.)

    def test_real_immediate_survival_exemption_still_wins(self):
        raw = departure(health=90.)
        raw['allies']['u1'].update(health=1., effective_health=1., health_lost_since_previous_decision=12.)
        raw['relations']['u1']['e0'].update(in_enemy_weapon_range=True, enemy_weapon_range_status='in_range')
        c = context(raw)
        self.assertTrue(escape_urgency(c, 'u1')['immediate'])
        for memory in (None, TacticalMemory()):
            self.assertEqual(shot_cost(costs(c, {'u0': 1, 'u1': 5}, memory)), 0.)

    def test_no_contact_and_no_observed_output_duty_has_no_delay_charge(self):
        raw = departure(health=90.)
        raw['visible_enemies']['e0']['position'] = [20., 0.]
        for relations in raw['relations'].values():
            relations['e0'].update(in_weapon_range=False, weapon_range_status='out_of_range',
                                   smac_attack_available=False, distance=20.)
        c = context(raw)
        for memory in (None, TacticalMemory()):
            self.assertEqual(shot_cost(costs(c, {'u0': 1, 'u1': 5}, memory)), 0.)

    def test_unbilled_outside_delay_cannot_erase_negative_participation_gain(self):
        raw = scene([((0., 0.), 'marine')],
                    {'e0': enemy((8., 0.), 'hydralisk', health=90.)},
                    distance=2., enemy_can_reach=False)
        c, e, memory, actions = context(raw), Evaluator(), TacticalMemory(), {'u0': 5}
        report = e.effects(c, actions)
        battle = {'engaged': True, 'phase': 'engage', 'targets': ('e0',)}
        with patch.object(e, '_engagement_state', return_value=battle):
            self.assertIsNone(memory.combat_progress.output_duty(c, 'u0'))
            self.assertEqual(shot_cost(e._coordination_costs(c, actions, report, memory, [])), 0.)
            self.assertEqual(e._final_next_fire_cost(c, 'u0', 5, actions, report, memory), 0.)
            value = e._sustained_contact_value(c, actions, report, memory)
            self.assertLess(value, 0.)
            self.assertEqual(value, e._sustained_contact_value(c, actions, report, None))

    def test_specialists_are_not_given_ordinary_output_obligations(self):
        raw = scene([((0., 0.), 'medivac'), ((0., 1.), 'baneling')],
                    {'e0': enemy((3., 0.), 'hydralisk')}, enemy_can_reach=False)
        c, memory = context(raw), TacticalMemory()
        with patch.object(memory, 'next_fire_cost', side_effect=AssertionError('specialist queried')):
            self.assertEqual(shot_cost(costs(c, {'u0': 5, 'u1': 5}, memory)), 0.)

    def test_legacy_low_health_chip_gate_is_preserved(self):
        raw = departure(health=90.)
        raw['allies']['u1'].update(health=10., effective_health=10., health_lost_since_previous_decision=.01)
        c = context(raw)
        e = Evaluator(common_combat={'joint_assessment_enabled': False})
        memory = TacticalMemory(encounter_config={'common_combat': {'joint_assessment_enabled': False}})
        self.assertEqual(shot_cost(costs(c, {'u0': 1, 'u1': 5}, memory, e)), 0.)

    def test_legacy_fallback_does_not_silently_enable_native_cycle_penalty(self):
        raw = departure(health=90., cooldown=8.)
        raw['unit_type_profiles']['48']['weapons'][0]['attack_period_loops'] = 8.
        c = context(raw)
        numeric = next_fire_timing(c, 'u1', 5, use_native_period=False)['cost']
        cycles = next_fire_timing(c, 'u1', 5, use_native_period=True)['cost']
        self.assertGreater(cycles, numeric)
        e = Evaluator(common_combat={'joint_assessment_enabled': False})
        memory = TacticalMemory(encounter_config={'common_combat': {'joint_assessment_enabled': False}})
        self.assertAlmostEqual(shot_cost(costs(c, {'u0': 1, 'u1': 5}, None, e)), .4 * numeric)
        self.assertAlmostEqual(shot_cost(costs(c, {'u0': 1, 'u1': 5}, memory, e)), .4 * numeric)

    def test_zero_survival_cover_cannot_approve_another_units_new_entry(self):
        own_costs = []
        for peer_health in (45., 1.):
            raw = scene([((0., 0.), 'marine'), ((3.2, 0.), 'marine')],
                        {'e0': enemy((8., 0.), 'hydralisk', health=90.)}, distance=2.)
            raw['allies']['u1'].update(health=peer_health, effective_health=peer_health)
            c, actions = context(raw), {'u0': 4, 'u1': 6}
            joint = assess_joint(c, actions)
            if peer_health == 1.:
                self.assertEqual(joint['actors']['u1']['cover_after_viability'], 0.)
            local = assess_local_entry(c, actions, joint)
            own_costs.append(local['fronts'][0]['actor_costs']['u0'])
        self.assertEqual(own_costs[0], 0.)
        self.assertGreater(own_costs[1], own_costs[0])

    def test_evaluator_prices_invalid_wait_once_and_exempts_verified_short_wait(self):
        raw = scene([((0., 0.), 'marine'), ((1.5, 0.), 'marine')],
                    {'e0': enemy((7.5, 0.), 'hydralisk')}, distance=2.)
        c, memory = context(raw), TacticalMemory()
        e = Evaluator(weights=dict.fromkeys(DEFAULT_WEIGHTS, 0.) |
                      {'support_break': .6, 'combat_idle': .35})
        valid, invalid = {'u0': 1, 'u1': 4}, {'u0': 1, 'u1': 5}
        good = assess_local_entry(c, valid, e.joint_facts(c, valid), memory=memory)
        bad = assess_local_entry(c, invalid, e.joint_facts(c, invalid), memory=memory)
        self.assertIn('u0', good['valid_wait_actors'])
        self.assertEqual(bad['invalid_wait_actors'], ('u0',))
        for actions, entry in ((valid, good), (invalid, bad)):
            score, terms = e.score_joint(c, [], actions, memory)
            self.assertAlmostEqual(terms['local_entry'], -.6 * entry['cost'])
            self.assertAlmostEqual(terms['combat_idle'], -.35 * len(entry['invalid_wait_actors']))
            self.assertAlmostEqual(score, sum(terms[k] for k in ('local_entry', 'combat_idle', 'support_break')))
        self.assertEqual(good['cost'], bad['cost'], 'invalid waiting is not charged again as entry time')

    def test_local_entry_is_disabled_by_explicit_legacy_mode(self):
        raw = scene([((0., 0.), 'marine'), ((1.5, 0.), 'marine')],
                    {'e0': enemy((7.5, 0.), 'hydralisk')}, distance=2.)
        c = context(raw)
        e = Evaluator(common_combat={'joint_assessment_enabled': False})
        self.assertEqual(e.score_joint(c, [], {'u0': 4, 'u1': 1})[1]['local_entry'], 0.)


if __name__ == '__main__':
    unittest.main()
