"""Executable-region and timing counterexamples; no native game launch."""
import copy
import unittest

from skills.tactical_v1.baseline import _Resources, _baneling_baseline
from skills.tactical_v1.burst import (action_region, approach_value, candidate_actions,
                                      continuation_state, outcome)
from skills.tactical_v1.effects import physical_effects
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.burst_engage import BurstEngageSkill
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import enemy
from .test_burst_mechanism import context, native_scene


class BurstExecutionTests(unittest.TestCase):
    def test_second_source_does_not_lose_candidates_to_unselected_first_proposal(self):
        state = native_scene([((0., 0.), 'baneling'), ((0., 0.), 'baneling')], {
            'e0': enemy((5., 0.), 'zergling'), 'e1': enemy((0., 6.), 'zergling')},
            enemy_can_reach=False)
        c = context(state)
        before = copy.deepcopy(state)
        skill = BurstEngageSkill({})
        plans = skill.propose(c, TacticalMemory(), Telemetry())
        choices = lambda actor: {(p.goal, p.current_actions[actor]) for p in plans
                                 if actor in p.current_actions}
        self.assertEqual(choices('u0'), choices('u1'))
        self.assertTrue(choices('u0'))
        self.assertEqual(state, before, 'proposal queries do not reserve source effects')

    def test_immediate_contact_keeps_low_health_productive_trigger(self):
        state = native_scene(enemies={'e0': enemy((1., 0.), 'hydralisk', health=90., radius=.625)})
        state['allies']['u0']['health'] = 1.
        c = context(state)
        result = outcome(c, 'u0', 6)
        self.assertEqual(result.trigger, 'active_contact_opportunity')
        self.assertEqual(result.opportunity, 1.)
        self.assertEqual(result.position, c.snapshot.allies['u0'].position)
        self.assertEqual(result.damage_potential, {'e0': 35.})
        self.assertIn(6, candidate_actions(c, 'u0'))
        self.assertNotIn(1, candidate_actions(c, 'u0'))

    def test_origin_death_and_later_active_arrival_are_not_added(self):
        state = native_scene()
        state['allies']['u0']['health'] = 1.
        state['visible_enemies']['e0']['weapon_cooldown_loops'] = 0.
        c = context(state)
        report = physical_effects(c, {'u0': 6})
        result = report.burst_outcomes['u0']
        self.assertEqual(result.trigger, 'death_at_origin_opportunity')
        self.assertGreater(result.opportunity, 0.)
        self.assertLess(result.opportunity, 1.)
        self.assertTrue(result.uncertain)
        self.assertEqual(report.planned_loss, {'u0': result.opportunity})
        self.assertAlmostEqual(report.burst_damage_potential['e0'], 35. * result.opportunity)
        self.assertEqual(approach_value(c, {'u0': 6}, report), 0.)

    def test_path_death_does_not_receive_distant_arrival_area(self):
        state = native_scene(enemies={'e0': enemy((6.6, 0.), 'hydralisk', health=90., radius=.625)})
        state['allies']['u0']['health'] = 5.
        state['visible_enemies']['e0']['weapon_cooldown_loops'] = 0.
        result = outcome(context(state), 'u0', 4)
        self.assertEqual(result.trigger, 'death_on_path_opportunity')
        self.assertLessEqual(result.position[0], 1.)
        self.assertEqual(result.damage_potential, {})
        self.assertLess(result.opportunity, 1.)

    def test_enemy_cooldown_prevents_impossible_lethal_volley_branch(self):
        state = native_scene()
        state['allies']['u0']['health'] = 1.
        state['visible_enemies']['e0']['weapon_cooldown_loops'] = 100.
        result = outcome(context(state), 'u0', 6)
        self.assertEqual(result.trigger, 'active_contact_opportunity')
        self.assertEqual(result.opportunity, 1.)
        self.assertTrue(result.uncertain, 'moving attack still has unmodelled path/target motion')

    def test_proposals_and_final_scoring_share_the_same_action_reference(self):
        state = native_scene(enemies={'e0': enemy((5., 0.), 'hydralisk', health=90., radius=.625),
                                    'e1': enemy((0., 5.), 'hydralisk', health=90., radius=.625)},
                             enemy_can_reach=False)
        c = context(state)
        plans = BurstEngageSkill({}).propose(c, TacticalMemory(), Telemetry())
        for plan in plans:
            action = plan.current_actions['u0']
            region = action_region(c, 'u0', action)
            self.assertEqual(plan.goal, region.target_id)
            self.assertEqual(plan.goal_data['burst_region'], list(region.position))
        reference = action_region(c, 'u0', 4)
        report = physical_effects(c, {'u0': 4})
        positive = approach_value(c, {'u0': 4}, report)
        report.damage_known = {e: u.effective_health for e, u in c.snapshot.enemies.items()}
        self.assertEqual(approach_value(c, {'u0': 4}, report), 0.)
        self.assertGreater(positive, 0.)
        self.assertIs(action_region(c, 'u0', 4), reference)

    def test_baseline_uses_actual_blast_area_instead_of_first_enemy_id(self):
        state = native_scene(enemies={
            'e0': enemy((-1., 0.), 'zergling', health=35.),
            'e1': enemy((1.5, 0.), 'hydralisk', health=90., radius=.625),
            'e2': enemy((3., 0.), 'hydralisk', health=90., radius=.625)}, enemy_can_reach=False)
        c = context(state)
        action = _baneling_baseline(c, 'u0', TacticalMemory(),
                                   {'burst_engage': BurstEngageSkill({})}, _Resources(c))
        self.assertEqual(action, 7)
        self.assertGreater(len(outcome(c, 'u0', 7).damage_potential),
                           len(outcome(c, 'u0', 6).damage_potential))

    def test_staging_has_fixed_two_h_budget_without_restarting_total_task(self):
        state = native_scene(enemies={'e0': enemy((5., 0.), 'hydralisk', health=90., radius=.625)},
                             enemy_can_reach=False)
        c = context(state)
        skill = BurstEngageSkill({})
        plans = skill.propose(c, TacticalMemory(), Telemetry())
        staging = next(p for p in plans if p.current_actions['u0'] == 1)
        self.assertEqual(staging.commitment.phase, 'staging')
        memory = TacticalMemory()
        task = memory.create_instance('burst_engage', ['u0'], 'e0', 'staging', 80, 48,
                                      goal_kind='enemy', extra={'burst_wait_started': 80})
        state['game_loop'] = 96
        plans = skill.propose(context(state), memory, Telemetry())
        self.assertFalse(any(p.current_actions['u0'] == 1 for p in plans))
        self.assertEqual(task.start_loop, 80)
        self.assertEqual(task.max_loops, 48)

    def test_real_lower_exposure_side_option_is_not_discarded_for_no_immediate_progress(self):
        state = native_scene(enemies={'e0': enemy((6.2, 0.), 'hydralisk', health=90., radius=.625)})
        state['visible_enemies']['e0']['weapon_cooldown_loops'] = 0.
        c = context(state)
        actions = candidate_actions(c, 'u0')
        self.assertIn(4, actions)
        self.assertIn(5, actions)
        self.assertEqual(action_region(c, 'u0', 5).progress, 0.)

    def test_continuation_is_action_based_and_cannot_renew_wait_or_changed_target(self):
        c = context(native_scene(enemies={'e0': enemy((5., 0.), 'hydralisk', health=90., radius=.625)},
                                 enemy_can_reach=False))
        state = continuation_state(c, 'u0', 4, 'e0')
        self.assertEqual(state['phase'], 'approaching')
        self.assertIsNone(continuation_state(c, 'u0', 1, 'e0'))
        self.assertIsNone(continuation_state(c, 'u0', 4, 'e1'))


if __name__ == '__main__':
    unittest.main()
