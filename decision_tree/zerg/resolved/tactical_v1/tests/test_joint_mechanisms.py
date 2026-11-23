import copy
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.joint_combat import assess_joint, action_position
from skills.tactical_v1.recovery import RecoveryMemory
from skills.tactical_v1.skills.disengage_reenter import DisengageReenterSkill
from .fixtures import scene, enemy, with_questions


def ctx(state):
    return Context(build_snapshot(with_questions(state), 1))


class JointMechanismTests(unittest.TestCase):
    def test_withdrawing_peer_cannot_keep_promised_cover(self):
        state = scene([((0, 0), 'marine'), ((0, 1), 'marine')],
                      {'e0': enemy((4.75, 1), 'hydralisk')})
        context = ctx(state)
        firing = assess_joint(context, {'u0': 6, 'u1': 6})
        leaving = assess_joint(context, {'u0': 6, 'u1': 5})
        self.assertGreater(firing['actors']['u0']['support_after'],
                           leaving['actors']['u0']['support_after'])
        self.assertGreater(leaving['actors']['u0']['support_loss'],
                           firing['actors']['u0']['support_loss'])

    def test_healer_is_not_counted_as_gun_cover(self):
        state = scene([((0, 0), 'marine'), ((0, 1), 'medivac')],
                      {'e0': enemy((4, 0), 'hydralisk')})
        row = assess_joint(ctx(state), {'u0': 6, 'u1': 1})['actors']['u0']
        self.assertEqual(row['support_after'], 0.)

    def test_attack_auto_approach_is_included(self):
        state = scene([((0, 0), 'zergling')], {'e0': enemy((4, 0), 'hydralisk')})
        state['relations']['u0']['e0'].update(native_weapon_range=.1,
            in_weapon_range=False, weapon_range_status='out_of_range',
            smac_attack_available=True, smac_attack_distance_limit=5.)
        context = ctx(state)
        point, certain = action_position(context, 'u0', 6)
        self.assertGreater(point[0], 0.)
        self.assertLessEqual(point[0], context.speed('u0') * context.H / 16.)
        self.assertFalse(certain)
        self.assertTrue(assess_joint(context, {'u0': 6})['actors']['u0']['moving'])

    def test_facts_depend_on_commands_not_candidate_labels(self):
        context = ctx(scene([((0, 0), 'marine')], {'e0': enemy((4, 0))}))
        a = assess_joint(context, {'u0': 4})
        b = assess_joint(context, dict(u0=4))
        self.assertEqual(a, b)
        self.assertNotEqual(a['actors']['u0']['position'],
                            assess_joint(context, {'u0': 5})['actors']['u0']['position'])

    def test_unknown_recovery_does_not_reward_waiting(self):
        state = scene([((0, 0), 'stalker')])
        state['allies']['u0'].update(shield=10., max_shield=80.)
        context = ctx(state)
        recovery = RecoveryMemory()
        recovery.observe(context)
        self.assertEqual(recovery.opportunity(context, 'u0')['gain'], 0.)

    def test_recovery_honors_delay_and_remaining_budget(self):
        state = scene([((0, 0), 'stalker')], loop=80)
        state['allies']['u0'].update(shield=10., max_shield=80., shield_lost_since_previous_decision=1.)
        state['unit_type_profiles']['74']['passive_recovery'] = {
            'shield': {'verified': True, 'rate_per_loop': .125, 'delay_loops': 24}}
        recovery = RecoveryMemory()
        recovery.observe(ctx(state))
        state['allies']['u0']['shield_lost_since_previous_decision'] = 0.
        state['game_loop'] = 96
        context = ctx(state)
        recovery.observe(context)
        self.assertEqual(recovery.opportunity(context, 'u0', horizon_loops=8)['gain'], 0.)
        self.assertEqual(recovery.opportunity(context, 'u0', horizon_loops=16)['gain'], 1.)

    def test_clean_regeneration_is_learned_without_heal_contamination(self):
        state = scene([((0, 0), 'zergling'), ((0, 1), 'medivac')], loop=80)
        state['allies']['u0'].update(health=20., health_lost_since_previous_decision=1.)
        recovery = RecoveryMemory()
        recovery.observe(ctx(state))
        for loop in (88, 96):
            state['game_loop'] = loop
            state['allies']['u0']['health'] += 2.
            state['allies']['u0']['health_lost_since_previous_decision'] = 0.
            recovery.observe(ctx(state))
        self.assertNotIn((105, 'health'), recovery.models)

    def test_shields_are_reserve_not_irrecoverable_hp_loss(self):
        state = scene([((0, 0), 'stalker')], {'e0': enemy((4, 0), 'hydralisk')})
        state['allies']['u0'].update(health=10., max_health=80., shield=80., max_shield=80.,
                                   health_lost_since_previous_decision=1.)
        skill = DisengageReenterSkill({})
        self.assertFalse(skill._severe(ctx(state), 'u0'))
        state['allies']['u0']['shield'] = 0.
        self.assertTrue(skill._severe(ctx(state), 'u0'))

    def test_weapon_cycle_is_domain_specific_and_unknown_stays_unknown(self):
        state = scene([((0, 0), 'marine')], {'e0': enemy((4, 0))})
        self.assertIsNone(ctx(state).weapon_period('u0', 'e0'))
        state['unit_type_profiles']['48']['weapons'][0]['attack_period_loops'] = 12.
        self.assertEqual(ctx(state).weapon_period('u0', 'e0'), 12.)
        state['visible_enemies']['e0']['is_flying'] = True
        self.assertIsNone(ctx(state).weapon_period('u0', 'e0'))


if __name__ == '__main__':
    unittest.main()
