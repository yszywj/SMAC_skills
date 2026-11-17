"""Cross-skill output duties and bounded native-cycle opportunity regressions.

No SC2. The scenes declare observed cooldowns and native timing explicitly;
commands, contact and geometric approach are not proof of an actual shot.
"""
import copy
import math
import unittest

from skills.tactical_v1.combat_exchange import next_fire_delay
from skills.tactical_v1.combat_progress import CombatProgress
from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.tests.fixtures import scene, enemy, with_questions


class QuietMemory:
    def combat_emergency(self, *_):
        return False

    def movement_blockage(self, *_):
        return 0.


def observed(loop=80, position=(5., 0.), cooldown=0., period=16., targets=None,
             speed=2.25, native=5., command=5.5):
    raw = scene([(position, 'marine')], targets or {'e0': enemy((10., 0.), tag='first')},
                loop=loop, distance=2., attack_range=native, enemy_can_reach=False)
    raw['allies']['u0']['weapon_cooldown_loops'] = cooldown
    profile = raw['unit_type_profiles']['48']
    profile['movement_speed_native'] = speed
    if period is not None:
        profile['weapons'][0]['attack_period_loops'] = period
    for target, other in raw['visible_enemies'].items():
        gap = math.dist(position, other['position'])
        raw['relations']['u0'][target].update(
            in_weapon_range=gap <= native + 1.,
            weapon_range_status='in_range' if gap <= native + 1. else 'out_of_range',
            smac_attack_available=gap <= command,
            smac_attack_distance_limit=command)
    return Context(build_snapshot(with_questions(raw), loop))


class OutputCycleContractTests(unittest.TestCase):
    def test_same_avoidable_delay_preserves_floor_and_counts_native_cycle_loss(self):
        # North remains inside contact. CD2 -> H8 is six avoidable loops.
        # A 12-loop weapon loses its second opportunity before fixed2H;
        # a 24-loop weapon retains the existing 6/8 delay cost, never less.
        fast, slow = observed(period=12., cooldown=2.), observed(period=24., cooldown=2.)
        self.assertAlmostEqual(next_fire_delay(fast, 'u0', 2), 1.)
        self.assertAlmostEqual(next_fire_delay(slow, 'u0', 2), .75)

    def test_native_period_never_reduces_existing_next_fire_delay_cost(self):
        for cooldown in (0., 2., 8., 10., 20., 40.):
            for action in (2, 4, 5):
                legacy = next_fire_delay(observed(period=None, cooldown=cooldown), 'u0', action)
                for period in (8., 12., 24., 32.):
                    cost = next_fire_delay(observed(period=period, cooldown=cooldown), 'u0', action)
                    self.assertGreaterEqual(cost, legacy)
                    self.assertLessEqual(cost, 1.)
                    if legacy == 0.:
                        self.assertEqual(cost, 0.)

    def test_real_cooldown_absorbs_move_and_return_for_any_unit_period(self):
        for period in (8., 24.):
            c = observed(cooldown=24., period=period)
            self.assertEqual(next_fire_delay(c, 'u0', 5), 0.)
        unknown = observed(period=None)
        self.assertEqual(next_fire_delay(unknown, 'u0', 2), 1.)  # explicit H fallback

    def test_native_only_contact_does_not_create_a_current_legal_shot(self):
        c = observed(position=(4.2, 0.))
        self.assertEqual(c.contact_status('u0', 'e0'), 'in_range')
        self.assertFalse(c.can_issue_attack('u0', 'e0'))
        self.assertEqual(next_fire_delay(c, 'u0', 5), 0.)
        self.assertGreater(next_fire_delay(c, 'u0', 5, eligible_targets=('e0',)), 0.)

    def test_first_observed_arrival_creates_fixed_two_cycle_duty(self):
        p = CombatProgress()
        before = observed(position=(3.875, 0.), cooldown=8.)
        p.observe(before)
        self.assertIsNone(p.output_duty(before, 'u0'))
        p.commit(before, {'u0': 4})
        arrived = observed(88)
        p.observe(arrived)
        duty = p.output_duty(arrived, 'u0')
        self.assertIsNotNone(duty)
        self.assertEqual(duty['cycles'], 0)
        until = duty['until_loop']
        for loop, action in ((96, 6), (104, 1), (112, 2)):
            p.commit(arrived, {'u0': action})
            arrived = observed(loop)
            p.observe(arrived)
            current = p.output_duty(arrived, 'u0')
            self.assertEqual(current['until_loop'], until)
            self.assertEqual(current['cycles'], 0)

    def test_geometry_and_provider_switch_do_not_refresh_output_progress(self):
        p = CombatProgress()
        first = observed()
        p.observe(first)
        p.commit(first, {'u0': 5})
        away = observed(88, position=(3.875, 0.))
        p.observe(away)
        duty = p.output_duty(away, 'u0')
        until, stalled = duty['until_loop'], duty['stalled_h']
        # Moving back is approach progress, but no observed cooldown restart.
        p.commit(away, {'u0': 4})
        back = observed(96)
        p.observe(back)
        duty = p.output_duty(back, 'u0')
        self.assertEqual(duty['until_loop'], until)
        self.assertGreater(duty['stalled_h'], stalled)
        self.assertEqual(duty['cycles'], 0)

    def test_dead_target_legal_replacement_keeps_same_duty_and_deadline(self):
        p = CombatProgress()
        first = observed()
        p.observe(first)
        p.commit(first, {'u0': 5})
        away = observed(88, position=(3.875, 0.))
        p.observe(away)
        original = p.output_duty(away, 'u0')
        p.commit(away, {'u0': 1})
        changed = observed(96, position=(3.875, 0.),
                           targets={'e1': enemy((0., 0.), tag='replacement')})
        p.observe(changed)
        current = p.output_duty(changed, 'u0')
        self.assertIsNotNone(current)
        self.assertEqual((current['start_loop'], current['until_loop']),
                         (original['start_loop'], original['until_loop']))
        self.assertIn('e1', current['targets'])
        self.assertEqual(current['cycles'], 0)

    def test_remote_replacement_does_not_extend_an_abandoned_encounter(self):
        p = CombatProgress()
        first = observed()
        p.observe(first)
        p.commit(first, {'u0': 5})
        away = observed(88, position=(3.875, 0.))
        p.observe(away)
        p.commit(away, {'u0': 1})
        far = observed(96, position=(3.875, 0.),
                       targets={'e1': enemy((30., 0.), tag='other-encounter')})
        p.observe(far)
        self.assertIsNone(p.output_duty(far, 'u0'))

    def test_primary_target_filter_is_shared_by_before_and_after_positions(self):
        p, memory = CombatProgress(), QuietMemory()
        c = observed(cooldown=8., targets={'e0': enemy((10., 0.), tag='front'),
                              'e1': enemy((0., 0.), tag='exhausted')})
        # West keeps e1 contact. If final known fire has exhausted its primary
        # budget, it must not erase the real departure from remaining e0.
        self.assertEqual(p.next_fire_cost(c, 'u0', 5, memory,
                                          eligible_targets=()), 0.)
        self.assertGreater(p.next_fire_cost(c, 'u0', 5, memory,
                                            eligible_targets=('e0',)), 0.)
        self.assertEqual(p.next_fire_cost(observed(cooldown=8., targets={
            'e0': enemy((10., 0.), tag='front'), 'e1': enemy((0., 0.), tag='alternative')}),
            'u0', 5, memory, eligible_targets=('e0','e1')), 0.)

    def test_primary_replacement_outside_contact_prices_avoidable_detours(self):
        p, memory = CombatProgress(), QuietMemory()
        c = observed(position=(0., 0.), targets={
            'e0': enemy((4., 0.), tag='expected-exhausted'),
            'e1': enemy((7., 0.), tag='nearby-next-target')})
        self.assertNotEqual(c.contact_status('u0', 'e1'), 'in_range')
        self.assertEqual(p.next_fire_cost(c, 'u0', 4, memory, eligible_targets=('e1',)), 0.)
        self.assertGreater(p.next_fire_cost(c, 'u0', 2, memory, eligible_targets=('e1',)), 0.)
        self.assertEqual(p.next_fire_cost(c, 'u0', 2, memory, eligible_targets=()), 0.)

    def test_late_second_shot_cannot_retroactively_fulfill_expired_duty(self):
        p = CombatProgress()
        c = observed(period=8.)
        p.observe(c)
        deadline = p.output_duty(c, 'u0')['until_loop']
        for loop in range(88, deadline + 1, 8):
            p.commit(c, {'u0': 6})
            c = observed(loop, cooldown=8. if loop in (88, deadline) else 0., period=8.)
            p.observe(c)
        self.assertIsNone(p.output_duty(c, 'u0'))
        self.assertEqual(p.output_outcomes[-1]['reason'], 'deadline')
        self.assertEqual(p.output_outcomes[-1]['cycles'], 1)

    def test_unknown_period_does_not_fabricate_a_second_observed_shot(self):
        p = CombatProgress()
        c = observed(period=None)
        p.observe(c)
        p.commit(c, {'u0': 6})
        same_cd = observed(88, period=None)
        p.observe(same_cd)
        self.assertEqual(p.output_duty(same_cd, 'u0')['cycles'], 0)
        self.assertFalse(p.output_duty(same_cd, 'u0')['period_known'])
        frozen = copy.deepcopy(p.__dict__)
        p.next_fire_cost(same_cd, 'u0', 5, QuietMemory(), eligible_targets=('e0',))
        self.assertEqual(p.__dict__, frozen)


if __name__ == '__main__':
    unittest.main()
