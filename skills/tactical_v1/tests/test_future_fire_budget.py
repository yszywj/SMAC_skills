"""Final-action future-fire budgets; no native games or run dependencies."""
import math
import unittest
from dataclasses import replace
from unittest.mock import patch

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.effects import physical_effects
from skills.tactical_v1.evaluator import Evaluator
from skills.tactical_v1.fire_opportunities import CONTINGENCY_CREDIT, final_action_budget
from skills.tactical_v1.joint_combat import assess_joint
from .fixtures import enemy, scene, with_questions


def context(raw):
    return Context(build_snapshot(with_questions(raw), raw['game_loop']))


def pair(health=5., count=1):
    raw = scene([((0., 0.), 'marine')] + [((0., 1.), 'marine')] * count,
                {'e0': enemy((4., 0.), 'hydralisk', health=health)})
    for actor, unit in raw['allies'].items():
        unit['weapon_cooldown_loops'] = 0. if actor == 'u0' else 16.
        unit['orders'] = [{'kind': 'attack', 'target_unit': 'e0'}]
    return raw


class FutureFireBudgetTests(unittest.TestCase):
    def test_depleted_target_is_backup_not_a_primary_next_shot(self):
        c = context(pair())
        b = final_action_budget(c, {'u0': 6, 'u1': 6})
        selected = b.future_targets('u1', {'e0': 2.})
        self.assertEqual(selected['primary'], {})
        self.assertEqual(selected['contingency'], {'e0': 2.})
        self.assertEqual(selected['credits']['e0'], CONTINGENCY_CREDIT)
        self.assertTrue(math.isinf(selected['eta_h']))
        self.assertIsNotNone(c.attack_action('u1', 'e0'), 'backup attack stays legal')

    def test_current_and_same_window_shots_are_not_deleted(self):
        c = context(pair())
        b = final_action_budget(c, {'u0': 6, 'u1': 6})
        for eta in (0., .5, 1.):
            self.assertEqual(b.credit('u1', 'e0', eta), 1.)
        self.assertEqual(b.credit('u0', 'e0', 2.), 1., 'a source cannot consume itself')
        self.assertEqual(physical_effects(c, {'u0': 6, 'u1': 6}).damage_known, {'e0': 6.})

    def test_reservation_uses_lower_bound_and_only_current_window_sources(self):
        c = context(pair(20.))
        b = final_action_budget(c, {'u0': 6, 'u1': 6})
        self.assertAlmostEqual(b.remaining_fraction('u1', 'e0', 2.), .7)
        raw = pair(); raw['allies']['u0']['weapon_cooldown_loops'] = 9.
        b = final_action_budget(context(raw), {'u0': 6, 'u1': 6})
        self.assertEqual(b.shots, ())
        self.assertEqual(b.credit('u1', 'e0', 2.), 1.)

    def test_unknown_magnitude_does_not_exhaust_target(self):
        c = context(pair())
        estimate = c.fire_estimate('u0', 'e0')
        with patch.object(c, 'fire_estimate', return_value=replace(estimate, damage_lo=None, damage_est=None)):
            b = final_action_budget(c, {'u0': 6, 'u1': 6})
        self.assertEqual(b.shots, ())
        self.assertEqual(b.future_targets('u1', {'e0': 2.})['primary'], {'e0': 2.})

    def test_unknown_projectile_time_never_claims_a_subwindow_arrival(self):
        c = context(pair())
        b = final_action_budget(c, {'u0': 6, 'u1': 6})
        self.assertFalse(b.shots[0].arrival_known)
        self.assertEqual(b.shots[0].after_h, 1.)
        self.assertEqual(b.credit('u1', 'e0', .9), 1.)

    def test_illegal_attack_or_nonattack_cannot_reserve_damage(self):
        raw = pair();raw['relations']['u0']['e0']['smac_attack_available'] = False
        self.assertEqual(final_action_budget(context(raw), {'u0': 6, 'u1': 6}).shots, ())
        c = context(pair())
        for action in (1, 2, 3, 4, 5):
            self.assertEqual(final_action_budget(c, {'u0': action, 'u1': 6}).shots, ())

    def test_peer_action_cache_key_and_joint_actor_cache_cannot_leak_budget(self):
        c = context(pair())
        firing = assess_joint(c, {'u0': 6, 'u1': 6})
        row = firing['actors']['u1']; saved = dict(row)
        self.assertEqual(row['target_etas'], {})
        stopped = assess_joint(c, {'u0': 1, 'u1': 6})
        self.assertEqual(stopped['actors']['u1']['target_etas'], {'e0': 2.})
        self.assertEqual(row, saved)
        self.assertIs(firing, assess_joint(c, {'u1': 6, 'u0': 6}))
        self.assertIsNot(final_action_budget(c, {'u0': 1, 'u1': 6}),
                         final_action_budget(c, {'u0': 6, 'u1': 6}))

    def test_projected_kill_never_erases_enemy_exposure(self):
        c = context(pair())
        attack = assess_joint(c, {'u0': 6, 'u1': 6})['actors']['u1']
        stop = assess_joint(c, {'u0': 1, 'u1': 6})['actors']['u1']
        self.assertGreater(attack['exposure']['amount'], 0.)
        self.assertEqual(attack['exposure'], stop['exposure'])

    def test_non_depleted_target_defines_primary_eta(self):
        raw = pair()
        raw['visible_enemies']['e1'] = enemy((4., 1.), 'hydralisk', health=90.)
        # Rebuild geometry/menu for the added target.
        rebuilt = scene([((0., 0.), 'marine'), ((0., 1.), 'marine')], raw['visible_enemies'])
        rebuilt['allies']['u1']['weapon_cooldown_loops'] = 16.
        c = context(rebuilt)
        selected = final_action_budget(c, {'u0': 6, 'u1': 2}).future_targets('u1', {'e0': 2., 'e1': 3.})
        self.assertEqual(selected['primary'], {'e1': 3.})
        self.assertEqual(selected['contingency'], {'e0': 2.})
        self.assertEqual(selected['eta_h'], 3.)

    def test_many_backup_actors_cannot_restore_full_future_reward(self):
        values = []
        for count in (1, 4, 8):
            c = context(pair(count=count)); e = Evaluator()
            actions = {a: 6 for a in c.snapshot.allies}
            values.append(e._future_attack_value(c, actions, e.effects(c, actions), []))
        self.assertAlmostEqual(values[0], .35 * CONTINGENCY_CREDIT)
        for value in values[1:]:
            self.assertAlmostEqual(value, values[0])

    def test_many_backup_actors_cannot_restore_full_support(self):
        values = []
        for count in (1, 4, 8):
            c = context(pair(count=count))
            row = assess_joint(c, {a: 6 for a in c.snapshot.allies})['actors']['u0']
            values.append(row['support_after'])
            self.assertAlmostEqual(row['support_before'], row['support_after'])
        self.assertGreater(values[0], 0.)
        for value in values[1:]:
            self.assertAlmostEqual(value, values[0])

    def test_legacy_ablation_retains_original_future_value(self):
        c = context(pair()); e = Evaluator(common_combat={'joint_assessment_enabled': False})
        actions = {'u0': 6, 'u1': 6}
        self.assertAlmostEqual(e._future_attack_value(c, actions, e.effects(c, actions), []), .35)

    def test_colossus_only_future_branch_retains_original_value(self):
        raw = scene([((0., 0.), 'colossus'), ((0., 1.), 'colossus')],
                    {'e0': enemy((4., 0.), 'hydralisk', health=5.)})
        raw['allies']['u1']['weapon_cooldown_loops'] = 16.
        c = context(raw); actions = {'u0': 6, 'u1': 6}
        current, legacy = Evaluator(), Evaluator(common_combat={'joint_assessment_enabled': False})
        self.assertEqual(current._future_attack_value(c, actions, current.effects(c, actions), []),
                         legacy._future_attack_value(c, actions, legacy.effects(c, actions), []))


if __name__ == '__main__':
    unittest.main()
