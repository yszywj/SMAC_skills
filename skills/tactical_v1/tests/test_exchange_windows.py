"""Visible exposure, isolated frontline reserves and conservative coverage."""
import unittest

from skills.tactical_v1.combat_exchange import exposure_window, imminent_exchange_danger
from skills.tactical_v1.combat_groups import effective_support
from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.evaluator import Evaluator
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.disengage_reenter import DisengageReenterSkill
from .fixtures import scene, enemy, with_questions


def context(state):
    return Context(build_snapshot(with_questions(state), 1))


class ExchangeWindowTests(unittest.TestCase):
    def under_fire(self, loss=0.):
        state = scene([((0., 0.), 'stalker')],
                      {'e0': enemy((4., 0.), 'colossus', health=300., max_health=300.)},
                      attack_range=6.)
        state['allies']['u0'].update(health=100., max_health=100., shield=0.,
            max_shield=0., effective_health=100., health_lost_since_previous_decision=loss)
        return state

    def test_long_wait_uses_observed_loss_without_inventing_enemy_shots(self):
        calm = context(self.under_fire())
        self.assertEqual(exposure_window(calm, 'u0', horizon_h=1.)['amount'],
                         exposure_window(calm, 'u0', horizon_h=3.)['amount'])
        hurt = context(self.under_fire(40.))
        self.assertGreater(exposure_window(hurt, 'u0', horizon_h=3.)['amount'],
                           exposure_window(hurt, 'u0', horizon_h=1.)['amount'])
        self.assertEqual(exposure_window(hurt, 'u0', horizon_h=30.)['horizon_h'], 3.)

    def test_safe_move_does_not_keep_previous_damage_as_current_pressure(self):
        c = context(self.under_fire(40.))
        safe = exposure_window(c, 'u0', (-20., 0.), horizon_h=3.)
        self.assertEqual(safe['amount'], 0.)
        self.assertGreater(safe['viability'], 0.)

    def test_restored_shields_count_as_reserve_without_hiding_low_health(self):
        bare = self.under_fire(20.)
        shielded = self.under_fire(20.)
        shielded['allies']['u0'].update(shield=100., max_shield=100., effective_health=200.,
                                       shield_upgrade_level=0)
        shielded['unit_type_profiles']['74']['shield_armor_base'] = 0.
        self.assertGreater(exposure_window(context(shielded), 'u0')['viability'],
                           exposure_window(context(bare), 'u0')['viability'])

    def test_collapse_before_next_shot_can_interrupt_high_raw_hp(self):
        state = self.under_fire(40.)
        state['allies']['u0']['weapon_cooldown_loops'] = 20.
        self.assertTrue(imminent_exchange_danger(context(state), 'u0'))

    def test_shield_loss_and_ready_attack_do_not_mask_collapse(self):
        state = self.under_fire(60.)
        state['allies']['u0'].update(shield=20., max_shield=100., shield_upgrade_level=0,
            shield_lost_since_previous_decision=20., weapon_cooldown_loops=6.)
        state['unit_type_profiles']['74']['shield_armor_base'] = 0.
        c = context(state)
        skill = DisengageReenterSkill({})
        self.assertTrue(skill._severe(c, 'u0'))
        self.assertTrue(skill._entry(c, 'u0'))
        memory = TacticalMemory()
        memory.last_committed_actions = {'u0': 6}
        costs = Evaluator().score_joint(c, [], {'u0': 5}, memory)[1]
        self.assertEqual(costs['attack_continuity'], 0.)

    def test_rear_reserves_do_not_hide_fragile_receiver(self):
        state = scene([((0., 0.), 'stalker'), ((8., 0.), 'stalker'),
                       ((12., 0.), 'stalker')],
                      {'e0': enemy((16., 0.), 'colossus')}, attack_range=6.)
        for unit in state['allies'].values():
            unit.update(health=300., max_health=300., effective_health=300.)
        state['allies']['u2'].update(health=50., max_health=50., effective_health=50.)
        c = context(state)
        facts = effective_support(c, ['u0'], ['u1', 'u2'])
        self.assertLess(facts['cover_viability'], .5)

    def test_splash_discount_changes_only_proxy_not_primary_damage(self):
        state = scene([((0., 0.), 'colossus')], {
            'e0': enemy((4., 0.), health=300., max_health=300.),
            'e1': enemy((4., 1.), health=300., max_health=300.)}, attack_range=7.)
        c = context(state)
        full = Evaluator(splash_reliability=1.).score_joint(c, [], {'u0': 6})[1]
        half = Evaluator(splash_reliability=.5).score_joint(c, [], {'u0': 6})[1]
        off = Evaluator(splash_reliability=0.).score_joint(c, [], {'u0': 6})[1]
        self.assertEqual(full['useful_damage'], half['useful_damage'])
        self.assertEqual(full['expected_kills'], off['expected_kills'])
        self.assertEqual(full['coverage_raw'], half['coverage_raw'])
        self.assertEqual(full['coverage'], 2 * half['coverage'])
        self.assertEqual(off['coverage'], 0.)


if __name__ == '__main__':
    unittest.main()
