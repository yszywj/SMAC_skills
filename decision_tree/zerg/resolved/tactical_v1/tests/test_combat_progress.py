"""Real paths, firing evidence and bounded duties across skill ownership."""
import copy
import json
from pathlib import Path
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.combat_exchange import imminent_exchange_danger
from skills.tactical_v1.contracts import Commitment, ResourceClaims
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.common import make_plan
from .fixtures import enemy, scene, with_questions


class CombatProgressTests(unittest.TestCase):
    def setUp(self):
        self.memory = TacticalMemory()

    def state(self, loop=80, x=0., cooldown=0., enemy_x=10.):
        s = scene([((x, 0.), 'marine'), ((enemy_x-2., 0.), 'marine')],
                  {'e0': enemy((enemy_x, 0.))}, loop=loop, distance=2.)
        s['allies']['u0']['weapon_cooldown_loops'] = cooldown
        return s

    def observe(self, s):
        c = Context(build_snapshot(with_questions(s), s['game_loop']))
        self.memory.begin_step(c.snapshot)
        self.memory.observe(c.snapshot)
        self.memory.update_task_facts(c.snapshot, c)
        return c

    def commit(self, c, action, skill=None):
        plans = []
        if skill:
            plans = [make_plan(skill, ('u0',), 'e0', {'u0':action}, [],
                ResourceClaims(controls=('u0',)), Commitment('approaching'), {},
                c.snapshot.snapshot_id, 'test', goal_kind='enemy')]
        self.memory.commit(c.snapshot, {'u0':action, 'u1':1}, plans, context=c)

    def arrived(self, cooldown=0.):
        c = self.observe(self.state(x=4., cooldown=cooldown+8))
        self.commit(c, 4, 'coordinated_engage')
        return self.observe(self.state(88, x=5.125, cooldown=cooldown))

    def test_real_return_cost_survives_missing_task_and_does_not_repay_advance(self):
        c = self.observe(self.state())
        self.commit(c, 4)
        c = self.observe(self.state(88, x=1.125))
        self.assertFalse(self.memory.movement_intents)
        before = copy.deepcopy(self.memory.__dict__)
        self.assertEqual(self.memory.movement_cycle_cost(c, 'u0', 5), 1.)
        self.assertEqual(self.memory.movement_conflict(c, 'u0', 5), 1.)
        self.assertEqual(self.memory.advance_credit_scale(c, 'u0', 5), 0.)
        self.assertEqual(self.memory.movement_cycle_cost(c, 'u0', 4), 0.)
        self.assertEqual(self.memory.__dict__, before)

    def test_observed_shot_preserves_legitimate_shoot_and_move_kiting(self):
        c = self.observe(self.state())
        self.commit(c, 4)
        c = self.observe(self.state(88, x=1.125, cooldown=16.))
        self.assertEqual(self.memory.movement_cycle_cost(c, 'u0', 5), 0.)
        self.assertIn('u0', self.memory.battle_state(c)['observed_weapon_cycles'])

    def test_real_pressure_reduction_is_productive_when_return_keeps_the_gain(self):
        def state(loop, x, enemy_x=0.):
            s = self.state(loop, x, enemy_x=enemy_x)
            s['visible_enemies']['e0'] = enemy((enemy_x,0.), 'stalker')
            return s
        c = self.observe(state(80, 7.))
        self.commit(c, 4)
        c = self.observe(state(88, 8.125, enemy_x=-2.))
        self.assertEqual(self.memory.movement_cycle_cost(c, 'u0', 5), 0.)

    def test_observed_blockage_and_critical_damage_release_path_cost(self):
        c = self.observe(self.state())
        self.commit(c, 4)
        c = self.observe(self.state(88, x=1.125))
        self.commit(c, 4)
        c = self.observe(self.state(96, x=1.125))
        self.assertGreater(self.memory.movement_blockage(c, 'u0', 4), .5)
        self.assertEqual(self.memory.movement_cycle_cost(c, 'u0', 5), 0.)
        self.memory.blocked_directions.clear()
        self.commit(c, 1)
        s = self.state(104, x=1.125)
        s['allies']['u0'].update(health=5., health_lost_since_previous_decision=30.)
        c = self.observe(s)
        self.assertEqual(self.memory.movement_cycle_cost(c, 'u0', 5), 0.)

    def test_history_expires_and_identity_change_is_not_the_same_path(self):
        c = self.observe(self.state())
        self.commit(c, 4)
        c = self.observe(self.state(88, x=1.125))
        self.assertGreater(self.memory.movement_cycle_cost(c, 'u0', 5), 0.)
        c = self.observe(self.state(128, x=1.125))
        self.assertEqual(self.memory.movement_cycle_cost(c, 'u0', 5), 0.)
        self.commit(c, 4)
        s = self.state(136, x=2.25)
        s['allies']['u0']['tag'] = 'replacement'
        c = self.observe(s)
        self.assertEqual(self.memory.movement_cycle_cost(c, 'u0', 5), 0.)

    def test_attack_command_and_contact_do_not_clear_return_duty_or_movement(self):
        c = self.arrived(cooldown=12.)
        until = self.memory.combat_progress.return_duties['u0']['until_loop']
        self.assertTrue(self.memory.return_fire_pending(c, 'u0'))
        self.commit(c, 6, 'focus_fire')
        self.assertIn('u0', self.memory.movement_intents)
        self.assertIn('u0', self.memory.combat_cycles)
        c = self.observe(self.state(96, x=5.125, cooldown=4.))
        self.assertTrue(self.memory.return_fire_pending(c, 'u0'))
        self.assertEqual(self.memory.combat_progress.return_duties['u0']['until_loop'], until)
        self.commit(c, 6, 'focus_fire')
        c = self.observe(self.state(104, x=5.125, cooldown=20.))
        self.assertFalse(self.memory.return_fire_pending(c, 'u0'))
        self.assertNotIn('u0', self.memory.movement_intents)
        self.assertNotIn('u0', self.memory.combat_cycles)

    def test_return_duty_prices_lost_next_shot_but_keeps_in_range_movement_free(self):
        c = self.arrived(cooldown=0.)
        before = copy.deepcopy(self.memory.__dict__)
        self.assertGreater(self.memory.return_fire_cost(c, 'u0', 5), 0.)
        self.assertEqual(self.memory.return_fire_cost(c, 'u0', 2), 0.)
        self.assertEqual(self.memory.return_fire_cost(c, 'u0', 6), 0.)
        self.assertEqual(self.memory.__dict__, before)

    def test_cooldown_budget_allows_kiting_with_time_to_return(self):
        c = self.arrived(cooldown=20.)
        self.assertEqual(self.memory.return_fire_cost(c, 'u0', 5), 0.)

    def test_duty_expires_without_observed_fire_and_does_not_restart_in_contact(self):
        c = self.arrived(cooldown=0.)
        self.commit(c, 6)
        c = self.observe(self.state(96, x=5.125))
        self.assertFalse(self.memory.return_fire_pending(c, 'u0'))
        self.commit(c, 6)
        c = self.observe(self.state(104, x=5.125))
        self.assertFalse(self.memory.return_fire_pending(c, 'u0'))

    def test_critical_danger_and_target_loss_interrupt_return_duty(self):
        c = self.arrived(cooldown=20.)
        self.commit(c, 6)
        s = self.state(96, x=5.125, cooldown=12.)
        s['allies']['u0'].update(health=5., health_lost_since_previous_decision=30.)
        c = self.observe(s)
        self.assertFalse(self.memory.return_fire_pending(c, 'u0'))
        self.assertEqual(self.memory.return_fire_cost(c, 'u0', 5), 0.)
        self.commit(c, 6)
        s = self.state(104, x=5.125, cooldown=4.)
        s['visible_enemies'] = {}
        s['relations'] = {a:{} for a in s['allies']}
        c = self.observe(s)
        self.assertFalse(self.memory.return_fire_pending(c, 'u0'))

    def test_shield_collapse_before_next_shot_interrupts_high_hp_return_duty(self):
        def state(loop, x, cooldown, loss=0.):
            s = self.state(loop, x=x, cooldown=cooldown)
            s['visible_enemies']['e0'] = enemy((10., 0.), 'colossus')
            s['allies']['u0'].update(health=100., max_health=100.,
                shield=10., max_shield=100., effective_health=110.,
                shield_upgrade_level=0., shield_lost_since_previous_decision=loss)
            s['unit_type_profiles']['48']['shield_armor_base'] = 0.
            return s
        c = self.observe(state(80, 4., 28.))
        self.commit(c, 4, 'coordinated_engage')
        c = self.observe(state(88, 5.125, 20., loss=60.))
        self.assertIn('u0', self.memory.combat_progress.return_duties)
        self.assertTrue(self.memory.combat_emergency(c, 'u0'))
        self.assertTrue(imminent_exchange_danger(c, 'u0'))
        before = copy.deepcopy(self.memory.__dict__)
        self.assertFalse(self.memory.return_fire_pending(c, 'u0'))
        self.assertEqual(self.memory.return_fire_cost(c, 'u0', 5), 0.)
        self.assertEqual(self.memory.movement_cycle_cost(c, 'u0', 5), 0.)
        self.assertEqual(self.memory.__dict__, before)

    def test_zero_commitment_disables_new_costs_for_ablation(self):
        self.memory = TacticalMemory(encounter_config={'lifecycle':{'combat_cycle_commit_h':0.}})
        c = self.arrived(cooldown=0.)
        self.assertFalse(self.memory.return_fire_pending(c, 'u0'))
        self.assertEqual(self.memory.return_fire_cost(c, 'u0', 5), 0.)
        self.assertEqual(self.memory.movement_cycle_cost(c, 'u0', 5), 0.)
        self.assertEqual(self.memory.advance_credit_scale(c, 'u0', 5), 1.)

    def test_recorded_6z4s_seed183_does_not_repay_immediate_return_to_danger(self):
        records = json.loads(Path(__file__).with_name('combat_progress_183.json').read_text(
            encoding='utf8'))['frames']
        for row in records:
            c = Context(build_snapshot(row['payload'], row['step'] + 1))
            self.memory.begin_step(c.snapshot)
            self.memory.observe(c.snapshot)
            self.memory.update_task_facts(c.snapshot, c)
            if row['step'] == 32:
                # Real31->32 moved west into safety. Reversing east would
                # return to the same danger with no weapon cycle in between.
                self.assertGreater(self.memory.movement_cycle_cost(c, 'u0', 4), .9)
                self.assertLess(self.memory.advance_credit_scale(c, 'u0', 4), .1)
            else:
                self.memory.commit(c.snapshot, row['actions'], [], context=c)


if __name__ == '__main__': unittest.main()
