"""G2 uses the shared local objective and physically executable contact slots."""
import copy
import math
import unittest

from skills.tactical_v1.config import ConfigError, load, validate
from skills.tactical_v1.context import Context, build_snapshot, distance
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.coordinated_engage import CoordinatedEngageSkill
from skills.tactical_v1.targeting import shared_target_plan
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import enemy, scene, with_questions


def context(state):
    return Context(build_snapshot(with_questions(state), state['game_loop']))


class SharedEngageFrontTests(unittest.TestCase):
    def plans(self, state, config=None, memory=None):
        c = context(state)
        memory = TacticalMemory() if memory is None else memory
        result = CoordinatedEngageSkill(config or {}).propose(c, memory, Telemetry())
        return [p for p in result if p.controlled_actors == ('u0',)]

    def test_native_range_cannot_place_melee_contact_outside_command_mask(self):
        state = scene([((0., 0.), 'zealot')], {'e0': enemy((8., 0.))}, attack_range=2.)
        state['relations']['u0']['e0']['smac_attack_distance_limit'] = 2.2
        c = context(state)
        goal, slot = CoordinatedEngageSkill({})._slot(c, 'u0', 'e0', 0)
        self.assertIsNotNone(slot)
        self.assertLessEqual(distance(goal, c.snapshot.enemies['e0'].position), 2.2)
        self.assertTrue(c.projected_attack_opportunity('u0', 'e0', goal).available)

    def test_ranged_goal_and_objective_share_command_boundary(self):
        state = scene([((0., 0.), 'hydralisk')], {'e0': enemy((8., 0.))}, attack_range=5.)
        state['relations']['u0']['e0']['smac_attack_distance_limit'] = 3.
        plans = self.plans(state)
        self.assertTrue(plans)
        for p in plans:
            self.assertEqual(p.goal_data['objective']['r_max'], 3.)
            self.assertLess(distance(p.goal_data['waypoint'], (8., 0.)), 3.)

    def test_unknown_command_reach_does_not_invent_a_contact_slot(self):
        state = scene([((0., 0.), 'zealot')], {'e0': enemy((8., 0.))}, attack_range=.1)
        state['relations']['u0']['e0'].pop('smac_attack_distance_limit')
        state['action_contract']['candidate_center_distance_by_type'].pop('zealot')
        c = context(state)
        self.assertLess(c.effective_attack_reach('u0', 'e0'), 0.)
        self.assertEqual(CoordinatedEngageSkill({})._slot(c, 'u0', 'e0', 0), (None, None))

    def test_full_melee_ring_uses_another_target_on_the_same_front(self):
        # Four occupied physical contact slots, one additional unengaged
        # melee unit, and a nearby enemy whose contact ring is still open.
        ring = [((8. + 1.05 * math.cos(i * math.pi / 2),
                  1.05 * math.sin(i * math.pi / 2)), 'zealot') for i in range(4)]
        state = scene([((0., 0.), 'zealot'), *ring],
                      {'e0': enemy((8., 0.)), 'e1': enemy((8., 4.))}, attack_range=.1)
        for relations in state['relations'].values():
            for relation in relations.values():
                relation['smac_attack_distance_limit'] = 2.
        c = context(state)
        self.assertEqual(CoordinatedEngageSkill({})._slot(c, 'u0', 'e0', 0), (None, None))
        plans = self.plans(state)
        self.assertTrue(plans)
        self.assertEqual({p.goal for p in plans}, {'e1'})
        self.assertEqual(self.plans(state, {'common_combat': {'shared_front_enabled': False}}), [])

    def test_recent_feasible_allocation_reaches_g2_without_task_query_mutation(self):
        state = scene([((0., 0.), 'stalker')], {
            'e0': enemy((6.7, 0.), 'zealot', health=150., max_health=150.),
            'e1': enemy((6.7, 1.), 'colossus', health=350., max_health=350.)}, attack_range=6.)
        c, memory = context(state), TacticalMemory()
        assignment = shared_target_plan(c, memory).assignments['u0']
        self.assertEqual(assignment.target, 'e1')
        before = copy.deepcopy(memory.__dict__)
        plans = self.plans(state, memory=memory)
        self.assertTrue(plans)
        self.assertEqual({p.goal for p in plans}, {'e1'})
        self.assertEqual(before, memory.__dict__)

    def test_fresh_allocation_does_not_renew_live_approach_deadline(self):
        state = scene([((0., 0.), 'stalker')], {
            'e0': enemy((6.7, 0.), 'zealot', health=150., max_health=150.),
            'e1': enemy((6.7, 1.), 'colossus', health=350., max_health=350.)}, attack_range=6.)
        c, memory = context(state), TacticalMemory()
        memory.begin_step(c.snapshot)
        memory.observe(c.snapshot)
        memory.update_task_facts(c.snapshot, c)
        plan = self.plans(state, memory=memory)[0]
        memory.commit(c.snapshot, dict(plan.current_actions), [plan], context=c)
        original = copy.deepcopy(memory.instance_for('coordinated_engage', 'u0'))
        state['game_loop'] += 8
        state['visible_enemies']['e0']['health'] = 1.
        c = context(state)
        memory.begin_step(c.snapshot)
        memory.observe(c.snapshot)
        memory.update_task_facts(c.snapshot, c)
        plan = self.plans(state, memory=memory)[0]
        self.assertEqual(plan.goal, original.goal)
        self.assertEqual(plan.instance_id, original.instance_id)
        memory.commit(c.snapshot, dict(plan.current_actions), [plan], context=c)
        current = memory.instance_for('coordinated_engage', 'u0')
        self.assertEqual(current.start_loop, original.start_loop)
        self.assertEqual(current.max_loops, original.max_loops)

    def test_local_retry_does_not_redirect_actor_to_a_distant_front(self):
        state = scene([((0., 0.), 'marine'), ((4., 0.), 'marine')], {
            'e0': enemy((8., 0.)), 'e1': enemy((30., 0.))}, attack_range=5.)
        memory = TacticalMemory()
        key = memory.task_retry_key('coordinated_engage', ['u0'], 'enemy', 'e0')
        memory.retry_records[key] = state['game_loop'] + 16
        self.assertEqual(self.plans(state, memory=memory), [])
        state['game_loop'] += 16
        self.assertTrue(self.plans(state, memory=memory))

    def test_common_controls_are_independent_strict_booleans_and_preserve_s1(self):
        config = load()
        controls = config['common_combat']
        self.assertEqual(len(controls), 8)
        self.assertTrue(all(controls.values()))
        splash = copy.deepcopy(config['splash_fire'])
        for key in controls:
            changed = copy.deepcopy(config)
            changed['common_combat'][key] = False
            validate(changed)
            self.assertEqual(changed['splash_fire'], splash)
            changed['common_combat'][key] = 0
            with self.assertRaises(ConfigError):
                validate(changed)


if __name__ == '__main__':
    unittest.main()
