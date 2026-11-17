"""Observed departure through a second cycle, independent of skill ownership."""
import copy
import unittest

from skills.tactical_v1.combat_progress import CombatProgress
from skills.tactical_v1.context import Context, build_snapshot
from .fixtures import enemy, scene, with_questions


class QuietMemory:
    def combat_emergency(self, *_):
        return False

    def movement_blockage(self, *_):
        return 0.


def state(loop, x, cooldown=0., period=16., kind='marine'):
    raw = scene([((x, 0.), kind)], {'e0': enemy((10., 0.), tag='enemy')},
                loop=loop, distance=2., enemy_can_reach=False)
    raw['allies']['u0']['weapon_cooldown_loops'] = cooldown
    unit_type = str(raw['allies']['u0']['unit_type_id'])
    if raw['unit_type_profiles'][unit_type]['weapons'] and period is not None:
        raw['unit_type_profiles'][unit_type]['weapons'][0]['attack_period_loops'] = period
    context = Context(build_snapshot(with_questions(raw), loop))
    # Isolate the observed-duty contract while the public weapon-period query
    # is tested independently by the Context/bridge owner.
    context.weapon_period = lambda *args: period
    return context


class OutputDutyTests(unittest.TestCase):
    def departed(self, progress=None, period=16.):
        progress = progress or CombatProgress()
        before = state(80, 5., period=period)
        progress.observe(before)
        progress.commit(before, {'u0': 5})
        after = state(88, 3.875, period=period)
        progress.observe(after)
        return progress, after

    def test_departure_starts_before_arrival_and_second_cycle_completes(self):
        progress, c = self.departed()
        duty = progress.output_duty(c, 'u0')
        self.assertEqual(duty['phase'], 'return')
        self.assertEqual(duty['start_loop'], 80)
        until = duty['until_loop']
        self.assertFalse(progress.return_duties)
        for loop, x, cooldown, action in ((96, 5., 0., 6), (104, 5., 16., 2),
                                           (112, 5., 8., 6)):
            progress.commit(c, {'u0': 4 if loop == 96 else action})
            c = state(loop, x, cooldown)
            progress.observe(c)
            self.assertEqual(progress.output_duty(c, 'u0')['until_loop'], until)
        self.assertEqual(progress.output_duty(c, 'u0')['cycles'], 1)
        self.assertEqual(progress.output_duty(c, 'u0')['phase'], 'second_cycle')
        progress.commit(c, {'u0': 6})
        c = state(120, 5., 16.)
        progress.observe(c)
        self.assertIsNone(progress.output_duty(c, 'u0'))
        self.assertEqual(progress.output_outcomes[-1]['reason'], 'second_cycle')

    def test_deadline_does_not_refresh_during_continued_departure(self):
        progress, c = self.departed()
        until = progress.output_duty(c, 'u0')['until_loop']
        loop, x = 96, 2.75
        while loop <= until + 24:
            progress.commit(c, {'u0': 5})
            c = state(loop, x)
            progress.observe(c)
            duty = progress.output_duty(c, 'u0')
            if loop < until:
                self.assertEqual(duty['until_loop'], until)
            else:
                self.assertIsNone(duty)
            loop, x = loop + 8, x - 1.125
        self.assertEqual(len(progress.output_outcomes), 1)
        self.assertEqual(progress.output_outcomes[0]['reason'], 'deadline')

    def test_command_and_contact_are_not_firing_query_is_pure(self):
        progress, c = self.departed()
        progress.commit(c, {'u0': 4})
        c = state(96, 5.)
        progress.observe(c)
        progress.commit(c, {'u0': 6})
        c = state(104, 5.)
        progress.observe(c)
        frozen = copy.deepcopy(progress.__dict__)
        self.assertEqual(progress.output_duty(c, 'u0')['cycles'], 0)
        self.assertEqual(progress.output_duty(c, 'u0')['phase'], 'first_cycle')
        progress.next_fire_cost(c, 'u0', 5, QuietMemory())
        self.assertEqual(progress.__dict__, frozen)

    def test_outside_contact_delay_cost_has_closing_and_cooldown_exceptions(self):
        progress, c = self.departed()
        self.assertEqual(progress.next_fire_cost(c, 'u0', 4, QuietMemory()), 0.)
        self.assertGreater(progress.next_fire_cost(c, 'u0', 5, QuietMemory()), 0.)
        cooling = state(88, 3.875, cooldown=40.)
        self.assertEqual(progress.next_fire_cost(cooling, 'u0', 5, QuietMemory()), 0.)
        self.assertEqual(progress.next_fire_cost(c, 'u0', 6, QuietMemory()), 0.)

    def test_period_sizes_fixed_budget_not_weapon_damage(self):
        fast, c_fast = self.departed(period=8.)
        slow, c_slow = self.departed(period=32.)
        unknown, c_unknown = self.departed(period=None)
        self.assertLess(fast.output_duty(c_fast, 'u0')['until_loop'],
                        slow.output_duty(c_slow, 'u0')['until_loop'])
        self.assertFalse(unknown.output_duty(c_unknown, 'u0')['period_known'])
        self.assertLessEqual(slow.output_duty(c_slow, 'u0')['until_loop'], 80 + 12 * 8)

    def test_new_legal_target_can_discharge_the_same_duty_without_old_target_lock(self):
        progress, _ = self.departed()
        raw = scene([((3.875, 0.), 'marine')], {
            'e0': enemy((10., 0.), tag='enemy'), 'e1': enemy((-1., 0.), tag='new')},
            loop=88, distance=2., enemy_can_reach=False)
        raw['allies']['u0']['weapon_cooldown_loops'] = 8.
        c = Context(build_snapshot(with_questions(raw), 88))
        self.assertIsNotNone(progress.output_duty(c, 'u0'))
        self.assertEqual(progress.next_fire_cost(c, 'u0', 5, QuietMemory()), 0.)

    def test_identity_target_loss_and_disabled_mode_leave_no_obligation(self):
        for reason in ('identity', 'target'):
            progress, c = self.departed()
            raw = copy.deepcopy(c.snapshot.raw_state)
            raw['game_loop'] = 96
            if reason == 'identity':
                raw['allies']['u0']['tag'] = 'replacement'
            else:
                raw['visible_enemies'] = {}
                raw['relations']['u0'] = {}
            changed = Context(build_snapshot(with_questions(raw), 96))
            progress.observe(changed)
            self.assertIsNone(progress.output_duty(changed, 'u0'))
        progress, c = self.departed(CombatProgress({'common_combat':
            {'joint_assessment_enabled': False}}))
        self.assertIsNone(progress.output_duty(c, 'u0'))

    def test_real_emergency_and_blockage_release_cost_without_renewing_budget(self):
        progress, c = self.departed()
        until = progress.output_duty(c, 'u0')['until_loop']
        memory = QuietMemory()
        # A generic task interrupt (including newly visible threats) is not
        # proof that this otherwise safe unit must abandon its next shot.
        memory.combat_emergency = lambda *_: True
        self.assertGreater(progress.next_fire_cost(c, 'u0', 5, memory), 0.)
        self.assertEqual(progress.output_duty(c, 'u0')['until_loop'], until)
        memory.combat_emergency = lambda *_: False
        memory.movement_blockage = lambda *_: 1.
        self.assertEqual(progress.next_fire_cost(c, 'u0', 5, memory), 0.)


if __name__ == '__main__':
    unittest.main()
