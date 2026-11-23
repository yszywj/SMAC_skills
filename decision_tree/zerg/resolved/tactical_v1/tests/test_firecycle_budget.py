"""A single, capability-based next-shot budget, before and after re-entry."""
import copy
import unittest

from skills.tactical_v1.combat_exchange import next_fire_delay
from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.evaluator import Evaluator
from skills.tactical_v1.memory import TacticalMemory
from .fixtures import scene, enemy, with_questions


class FireCycleBudgetTests(unittest.TestCase):
    def state(self, loop=80, x=5., cooldown=10., kind='marine'):
        s = scene([((x, 0.), kind)], {'e0': enemy((10., 0.))},
                  loop=loop, distance=2.)
        s['allies']['u0']['weapon_cooldown_loops'] = cooldown
        return s

    def observe(self, memory, state):
        c = Context(build_snapshot(with_questions(state), state['game_loop']))
        memory.begin_step(c.snapshot)
        memory.observe(c.snapshot)
        memory.update_task_facts(c.snapshot, c)
        return c

    def test_first_departure_has_same_budget_as_observed_rearrival(self):
        fresh = TacticalMemory()
        c = self.observe(fresh, self.state(loop=88))
        self.assertFalse(fresh.combat_progress.return_duties)
        first_cost = fresh.next_fire_cost(c, 'u0', 5)
        self.assertGreater(first_cost, 0.)
        arrived = TacticalMemory()
        before = self.observe(arrived, self.state(x=4., cooldown=18.))
        arrived.commit(before.snapshot, {'u0': 4}, [], context=before)
        c = self.observe(arrived, self.state(loop=88))
        self.assertTrue(arrived.return_fire_pending(c, 'u0'))
        frozen = copy.deepcopy(arrived.__dict__)
        self.assertEqual(arrived.next_fire_cost(c, 'u0', 5), first_cost)
        costs = Evaluator().score_joint(c, [], {'u0': 5}, arrived)[1]
        self.assertAlmostEqual(costs['next_fire_delay'], -.4 * first_cost)
        self.assertEqual(costs['contact_reentry'], 0.)
        self.assertEqual(costs['return_fire_continuity'], 0.)
        self.assertEqual(costs['attack_continuity'], 0.)
        self.assertEqual(arrived.__dict__, frozen)
        # The old path remains independently selectable for an exact local
        # ablation; it demonstrates the two old costs on this SAME snapshot.
        old = Evaluator(common_combat={'next_fire_budget_enabled': False})
        legacy = old.score_joint(c, [], {'u0': 5}, arrived)[1]
        self.assertLess(legacy['return_fire_continuity'], 0.)
        self.assertLess(legacy['contact_reentry'], 0.)
        self.assertEqual(legacy['next_fire_delay'], 0.)

    def test_cooldown_absorbs_move_and_return_but_ready_move_delays_shot(self):
        memory = TacticalMemory()
        c = self.observe(memory, self.state(cooldown=20.))
        self.assertEqual(memory.next_fire_cost(c, 'u0', 5), 0.)
        c = self.observe(TacticalMemory(), self.state(cooldown=10.))
        self.assertEqual(next_fire_delay(c, 'u0', 2), 0.)  # stays in legal range
        c = self.observe(TacticalMemory(), self.state(cooldown=0.))
        self.assertEqual(next_fire_delay(c, 'u0', 2), 1.)  # MOVE consumes ready window
        self.assertEqual(next_fire_delay(c, 'u0', 6), 0.)

    def test_native_and_command_reach_use_the_same_gap_not_two_charges(self):
        s = self.state(cooldown=12.)
        s['relations']['u0']['e0'].update(native_weapon_range=10.,
                                        smac_attack_distance_limit=5.)
        c = self.observe(TacticalMemory(), s)
        self.assertAlmostEqual(next_fire_delay(c, 'u0', 5), .5)
        s['relations']['u0']['e0'].update(native_weapon_range=4.,
                                        smac_attack_distance_limit=20.)
        c = self.observe(TacticalMemory(), s)
        self.assertAlmostEqual(next_fire_delay(c, 'u0', 5), .5)

    def test_new_target_in_range_is_not_locked_to_the_old_contact(self):
        s = scene([((5., 0.), 'marine')],
                  {'e0': enemy((10., 0.)), 'e1': enemy((0., 0.))}, distance=2.)
        s['allies']['u0']['weapon_cooldown_loops'] = 10.
        c = self.observe(TacticalMemory(), s)
        self.assertEqual(next_fire_delay(c, 'u0', 5), 0.)

    def test_ordinary_relief_keeps_budget_but_blockage_and_danger_allow_departure(self):
        memory = TacticalMemory()
        s = self.state(x=6., cooldown=0.)
        c = self.observe(memory, s)  # zergling is reachable here
        self.assertGreater(next_fire_delay(c, 'u0', 5), 0.)
        self.assertGreater(memory.next_fire_cost(c, 'u0', 5), 0.)
        c = self.observe(memory, self.state(loop=88, cooldown=0.))
        memory.commit(c.snapshot, {'u0': 4}, [], context=c)
        c = self.observe(memory, self.state(loop=96, cooldown=0.))
        self.assertGreater(memory.movement_blockage(c, 'u0', 4), .5)
        self.assertEqual(memory.next_fire_cost(c, 'u0', 5), 0.)
        s = self.state(loop=104, cooldown=0.)
        s['allies']['u0'].update(health=5., effective_health=5.,
                                health_lost_since_previous_decision=20.)
        c = self.observe(memory, s)
        self.assertEqual(memory.next_fire_cost(c, 'u0', 5), 0.)

    def test_no_invented_cost_for_unknown_speed_reach_or_special_capability(self):
        for kind in ('medivac', 'baneling'):
            with self.subTest(kind=kind):
                c = self.observe(TacticalMemory(), self.state(kind=kind))
                self.assertEqual(next_fire_delay(c, 'u0', 5), 0.)
        s = self.state()
        s['unit_type_profiles']['48'].pop('movement_speed_native')
        self.assertEqual(next_fire_delay(self.observe(TacticalMemory(), s), 'u0', 5), 0.)
        s = self.state()
        s['relations']['u0']['e0']['native_weapon_range'] = None
        self.assertEqual(next_fire_delay(self.observe(TacticalMemory(), s), 'u0', 5), 0.)

    def test_budget_has_its_own_switch_and_does_not_disable_path_memory(self):
        memory = TacticalMemory(encounter_config={
            'common_combat': {'next_fire_budget_enabled': False}})
        c = self.observe(memory, self.state())
        self.assertTrue(memory.combat_progress.enabled)
        self.assertEqual(memory.next_fire_cost(c, 'u0', 5), 0.)
        # Changing the older path-commitment switch does not disable this
        # stateless opportunity budget.
        memory = TacticalMemory(encounter_config={
            'lifecycle': {'combat_cycle_commit_h': 0.}})
        c = self.observe(memory, self.state())
        self.assertGreater(memory.next_fire_cost(c, 'u0', 5), 0.)


if __name__ == '__main__':
    unittest.main()
