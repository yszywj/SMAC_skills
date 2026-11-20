"""G2 uses the shared local objective and physically executable contact slots."""
import copy
import math
import unittest

from skills.tactical_v1.config import ConfigError, load, validate
from skills.tactical_v1.context import Context, build_snapshot, distance
from skills.tactical_v1.contracts import Commitment, ResourceClaims
from skills.tactical_v1.evaluator import Evaluator
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.common import make_plan
from skills.tactical_v1.skills.coordinated_engage import CoordinatedEngageSkill
from skills.tactical_v1.targeting import shared_target_plan
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import enemy, scene, with_questions


def context(state):
    return Context(build_snapshot(with_questions(state), state['game_loop']))


class SharedEngageFrontTests(unittest.TestCase):
    def test_dispersed_hydras_keep_shots_and_release_screen_on_shared_contact(self):
        allies = [((0., 0.), 'hydralisk'), ((5.5, 0.), 'hydralisk'),
                  ((5.5, -4.), 'zergling'), ((20., 15.), 'zergling'),
                  ((18., 16.), 'zergling')]
        memory = TacticalMemory()
        for loop, enemies in [(2, {}), (10, {'e0': enemy((3., 1.))})]:
            c = context(scene(allies, enemies, loop=loop, enemy_can_reach=False))
            memory.encounters.observe(c)
            plans = CoordinatedEngageSkill({}).propose(c, memory, Telemetry())
            plan = next(p for p in plans if p.goal_data.get('screen') == ['u2'])
            self.assertEqual(plan.goal_data['fire_group'], ['u0', 'u1'])
            self.assertEqual(plan.goal_data['screen_released'], bool(enemies))
            if enemies:
                self.assertEqual(plan.goal_data['shared_target'], 'e0')
                for actor in ('u0', 'u1'):
                    self.assertEqual(plan.current_actions[actor], c.attack_action(actor, 'e0'))
                self.assertEqual(c.decode('u2', plan.current_actions['u2']).kind, 'move')
            else:
                self.assertIsNone(plan.goal_data['shared_target'])
                self.assertEqual(c.decode('u2', plan.current_actions['u2']).kind, 'stop')

    def test_shared_fire_progress_requires_a_second_shooter_to_close(self):
        c = context(scene([((4., 0.), 'hydralisk'), ((0., 0.), 'hydralisk'),
                           ((0., 1.), 'zergling')], {'e0': enemy((8., 0.))}))
        evaluator = Evaluator()
        for rear_action, screen_action, advances in ((1, 4, False), (4, 1, True)):
            with self.subTest(rear_action=rear_action):
                actions = {'u0': 6, 'u1': rear_action, 'u2': screen_action}
                plan = make_plan('coordinated_engage', actions, 'e0', actions, [],
                    ResourceClaims(controls=tuple(actions)), Commitment('opening_fire_group'),
                    {}, c.snapshot.snapshot_id, '', goal_kind='enemy', atomic=True,
                    goal_data={'objective': {'kind': 'shared_fire', 'target': 'e0',
                                             'shooters': ['u0', 'u1']}})
                _, parts = evaluator.task_quality(c, actions, [plan], evaluator.effects(c, actions))
                self.assertEqual(parts['progress'] > 0., advances)

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
        # Boolean mechanism switches plus the bounded fire-delay horizon
        # (a float, validated separately below).
        bool_controls = {k: v for k, v in controls.items()
                         if isinstance(v, bool)}
        self.assertGreaterEqual(len(bool_controls), 8)
        self.assertTrue(all(bool_controls.values()))
        self.assertIn('fire_delay_max_h', controls)
        splash = copy.deepcopy(config['splash_fire'])
        for key in bool_controls:
            changed = copy.deepcopy(config)
            changed['common_combat'][key] = False
            validate(changed)
            self.assertEqual(changed['splash_fire'], splash)
            changed['common_combat'][key] = 0
            with self.assertRaises(ConfigError):
                validate(changed)


if __name__ == '__main__':
    unittest.main()
