"""Local fire fronts and shrinking, bounded G7 duties through real commits."""
import copy
import unittest

from skills.tactical_v1.combat_groups import (effective_support, task_outcome,
                                             active_rejoin_data)
from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.evaluator import Evaluator
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.combat_rejoin import CombatRejoinSkill
from skills.tactical_v1.targeting import local_front_targets, shared_target_plan
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import enemy, scene, with_questions


class LocalRejoinFrontTests(unittest.TestCase):
    def state(self, xs=(7., 8., 16., 16.), loop=80):
        return scene([((x, 0.), 'stalker') for x in xs],
                     enemies={'e0': enemy((18., 0.))}, loop=loop)

    def context(self, state=None):
        state = state or self.state()
        return Context(build_snapshot(with_questions(state), state['game_loop']))

    def observe(self, memory, context):
        memory.begin_step(context.snapshot)
        memory.observe(context.snapshot)
        return memory.update_task_facts(context.snapshot, context)

    def start(self):
        context, memory, skill = self.context(), TacticalMemory(), CombatRejoinSkill({})
        self.observe(memory, context)
        plan = skill.propose(context, memory, Telemetry())[0]
        memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
        return memory, skill, memory.instances[plan.instance_id]

    def test_local_front_does_not_follow_distant_seed_or_enemy_chain(self):
        state = scene([((0., 0.), 'stalker')], enemies={
            'e0': enemy((4., 0.)), 'e1': enemy((9., 0.)),
            'e2': enemy((14., 0.)), 'e3': enemy((30., 0.))})
        context = self.context(state)
        front = local_front_targets(context, TacticalMemory(), ('u0',),
                                    seed_targets=('e3',), radius=6.)
        self.assertEqual(set(front), {'e0', 'e1'})

    def test_eta_cache_and_completion_use_the_same_relevant_front(self):
        state = scene([((0., 0.), 'stalker'), ((2., 0.), 'stalker')],
                      enemies={'e0': enemy((10., 0.)), 'e1': enemy((2., 0.))},
                      enemy_can_reach=False)
        context = self.context(state)
        targets = {'u0': ['e0'], 'u1': ['e0']}
        any_target = effective_support(context, ['u0'], ['u1'])
        local = effective_support(context, ['u0'], ['u1'], targets_by_actor=targets)
        self.assertEqual(any_target['eta_by_actor']['u0'], 0.)
        self.assertGreater(local['eta_by_actor']['u0'], 2.)
        self.assertEqual(any_target, effective_support(context, ['u0'], ['u1']))
        data = {'joiners': ['u0'], 'supporters': ['u1'], 'support_radius': 6.,
                'readiness_fraction': 1., 'roles': {'u0': 'join', 'u1': 'cover'},
                'front_targets': ['e0'], 'targets_by_actor': targets,
                'shared_front_enabled': True, 'no_progress_loops': 24,
                'last_support_progress_loop': 80}
        instance = TacticalMemory().create_instance('combat_rejoin', ['u0', 'u1'],
            'join', 'rejoining', 80, 64, goal_kind='area', extra=data)
        self.assertIsNone(task_outcome(context, instance))
        instance.extra['shared_front_enabled'] = False
        self.assertEqual(task_outcome(context, instance), 'support_connected')

    def test_cover_waits_for_join_fire_but_escape_only_needs_safe_support(self):
        state = scene([((8., 0.), 'stalker'), ((16., 0.), 'stalker')],
                      enemies={'e0': enemy((18., 0.), 'stalker')})
        state['allies']['u0']['weapon_cooldown_loops'] = 64.
        context = self.context(state)
        targets = {'u0': ['e0'], 'u1': ['e0']}
        join = effective_support(context, ['u0'], ['u1'], targets_by_actor=targets,
                                 roles={'u0': 'join', 'u1': 'cover'})
        escape = effective_support(context, ['u0'], ['u1'], targets_by_actor=targets,
                                   roles={'u0': 'escape', 'u1': 'cover'})
        legacy = effective_support(context, ['u0'], ['u1'], targets_by_actor=targets)
        self.assertEqual(join['arrival_windows'], 8.)
        self.assertGreater(join['arrival_windows'], escape['arrival_windows'])
        self.assertLess(join['cover_viability'], .5)
        self.assertGreaterEqual(escape['cover_viability'], .5)
        self.assertEqual(escape, legacy)
        self.assertEqual(join, effective_support(context, ['u0'], ['u1'],
            targets_by_actor=targets, roles={'u0': 'join', 'u1': 'cover'}))

    def test_local_cover_off_retains_old_geometric_wait_budget(self):
        state = scene([((8., 0.), 'stalker'), ((16., 0.), 'stalker')],
                      enemies={'e0': enemy((18., 0.), 'stalker')})
        state['allies']['u0']['weapon_cooldown_loops'] = 64.
        context = self.context(state)
        legacy = {'common_combat': {'joint_assessment_enabled': False}}
        plans = CombatRejoinSkill(legacy).propose(context, TacticalMemory(), Telemetry())
        self.assertTrue(all(p.goal_data['roles'].get('u0') != 'join' for p in plans))
        old_wait = {'common_combat': {'local_cover_enabled': False,
                                    'joint_assessment_enabled': False}}
        old_plans = CombatRejoinSkill(old_wait).propose(context, TacticalMemory(), Telemetry())
        self.assertTrue(any(p.goal_data['roles'].get('u0') == 'join' for p in old_plans))

    def test_escape_away_from_enemy_records_support_progress_without_deadline_renewal(self):
        def context(xs, loop):
            state = scene([((x, 0.), 'stalker') for x in xs],
                          enemies={'e0': enemy((-2., 0.))}, loop=loop)
            for actor in ('u0', 'u1'):
                state['allies'][actor]['weapon_cooldown_loops'] = 40.
            return self.context(state)
        memory, skill = TacticalMemory(), CombatRejoinSkill({})
        current = context((0., 1., 14., 15., 16.), 80)
        self.observe(memory, current)
        plan = next(p for p in skill.propose(current, memory, Telemetry())
                    if p.goal_data['roles'].get('u0') == 'escape')
        memory.commit(current.snapshot, plan.current_actions, [plan], context=current)
        instance = memory.instances[plan.instance_id]
        for loop, x in ((88, 4.), (96, 5.), (104, 6.), (112, 7.)):
            current = context((x, x + 1., 14., 15., 16.), loop)
            events = self.observe(memory, current)
            self.assertIn(instance.instance_id, memory.instances)
            self.assertFalse(any('no_progress' in event for event in events))
            self.assertEqual(instance.extra['last_support_progress_loop'], loop)
            plan = skill.propose(current, memory, Telemetry())[0]
            memory.commit(current.snapshot, plan.current_actions, [plan], context=current)
            self.assertEqual((instance.start_loop, instance.phase_start_loop, instance.max_loops), (80, 80, 64))

    def test_g7_attack_consumes_shared_allocation_without_remote_chase(self):
        state = scene([((0., 0.), 'stalker'), ((-8., 0.), 'stalker')], enemies={
            'e0': enemy((4., 0.), health=100., max_health=100.),
            'e1': enemy((4., 1.), 'colossus', health=350., max_health=350.),
            'e2': enemy((25., 0.), 'colossus')})
        context, memory, skill = self.context(state), TacticalMemory(), CombatRejoinSkill({})
        assigned = shared_target_plan(context, memory).assignments['u0'].target
        data = dict(shared_front_enabled=True,
                    **skill._target_data(context, memory, ('u0', 'u1'), ('u0',)))
        self.assertNotIn('e2', data['front_targets'])
        action = skill._attack(context, 'u0', {}, True, data)
        self.assertEqual(context.decode('u0', action).target, assigned)

    def test_default_continuation_is_read_only_and_never_adopts_new_members(self):
        memory, skill, instance = self.start()
        context = self.context(self.state((8., 9., 16., 16., 8.5), 88))
        self.observe(memory, context)
        before = copy.deepcopy(memory.__dict__)
        plan = skill.propose(context, memory, Telemetry())[0]
        self.assertEqual(memory.__dict__, before)
        self.assertEqual(plan.instance_id, instance.instance_id)
        self.assertNotIn('u4', plan.controlled_actors)
        memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
        self.assertEqual((instance.start_loop, instance.phase_start_loop, instance.max_loops), (80, 80, 64))

    def test_critical_receiver_releases_only_itself_until_retention_commit(self):
        memory, skill, instance = self.start()
        state = self.state((8., 9., 16., 16.), 88)
        state['allies']['u2']['health'] = 8.
        context = self.context(state)
        self.observe(memory, context)
        self.assertIn(instance.instance_id, memory.instances)
        self.assertIn('u2', instance.actors)  # observation cannot silently edit identity
        before = copy.deepcopy(instance.extra)
        plan = skill.propose(context, memory, Telemetry())[0]
        self.assertEqual(plan.proposed_memory_delta['kind'], 'retain_members')
        self.assertEqual(set(plan.controlled_actors), {'u0', 'u1', 'u3'})
        self.assertEqual(instance.extra, before)
        self.assertEqual(context.decode('u3', plan.current_actions['u3']).kind, 'attack')
        memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
        self.assertEqual(instance.actors, ('u0', 'u1', 'u3'))
        self.assertEqual((instance.start_loop, instance.phase_start_loop, instance.max_loops), (80, 80, 64))
        self.assertEqual(instance.extra['retry_key'], before['retry_key'])
        self.assertEqual(instance.extra['last_support_progress_loop'], before['last_support_progress_loop'])

    def test_dead_or_replaced_member_cannot_finish_or_inherit_duty(self):
        for replaced in (False, True):
            with self.subTest(replaced=replaced):
                memory, skill, instance = self.start()
                state = self.state((8., 9., 16., 16.), 88)
                state['allies']['u2']['tag'] = 'replacement' if replaced else 'ally'
                state['allies']['u2']['alive'] = replaced
                context = self.context(state)
                events = self.observe(memory, context)
                self.assertFalse(any('support_connected' in event for event in events))
                plan = skill.propose(context, memory, Telemetry())[0]
                self.assertNotIn('u2', plan.controlled_actors)
                memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
                self.assertEqual(instance.actors, ('u0', 'u1', 'u3'))

    def test_no_remaining_receiver_ends_duty_without_borrowing_health(self):
        memory, _, instance = self.start()
        state = self.state(loop=88)
        for actor in ('u2', 'u3'):
            state['allies'][actor]['health'] = 8.
        context = self.context(state)
        self.assertIsNone(active_rejoin_data(context, instance))
        self.observe(memory, context)
        self.assertNotIn(instance.instance_id, memory.instances)

    def test_shrinking_joiners_does_not_renew_progress_on_the_next_observation(self):
        memory, skill, instance = self.start()
        state = self.state(loop=88)
        state['allies']['u0']['alive'] = False
        context = self.context(state)
        self.observe(memory, context)
        plan = skill.propose(context, memory, Telemetry())[0]
        memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
        self.assertEqual(instance.extra['last_support_progress_loop'], 80)
        state['game_loop'] = 96
        self.observe(memory, self.context(state))
        self.assertEqual(instance.extra['last_support_progress_loop'], 80)
        self.assertFalse(instance.extra['joiner_progress']['u1']['observed_progress'])

    def test_critical_released_member_is_not_charged_old_group_commitment(self):
        memory, _, instance = self.start()
        state = self.state(loop=88)
        state['allies']['u2']['health'] = 8.
        context = self.context(state)
        self.observe(memory, context)
        evaluator = Evaluator()
        actions = {'u0': 1, 'u1': 1, 'u2': 5, 'u3': 6}
        costs = evaluator._coordination_costs(context, actions,
            evaluator.effects(context, actions), memory, [])
        self.assertAlmostEqual(costs['commitment_switch'], -evaluator.weights['switch'] * 3 * .5)

    def test_arrival_on_original_deadline_completes_and_nonarrival_times_out(self):
        for arrived in (True, False):
            with self.subTest(arrived=arrived):
                memory, _, instance = self.start()
                state = self.state((16., 16., 16., 16.) if arrived else (7., 8., 16., 16.), 144)
                events = self.observe(memory, self.context(state))
                reason = 'support_connected' if arrived else 'task_timeout'
                self.assertIn(f'{reason}:{instance.instance_id}', events)
                self.assertNotIn(instance.instance_id, memory.instances)

    def test_blocked_safe_member_can_continue_but_stall_budget_still_expires(self):
        memory, skill, instance = self.start()
        state = self.state(loop=88)
        payload = with_questions(state)
        for action in ('2', '3', '4', '5'):
            payload['questions']['u0']['criteria'].pop(action)
        context = Context(build_snapshot(payload, 88))
        self.observe(memory, context)
        plan = skill.propose(context, memory, Telemetry())[0]
        self.assertEqual(plan.current_actions['u0'], 1)
        memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
        self.assertEqual(instance.start_loop, 80)
        events = self.observe(memory, self.context(self.state(loop=104)))
        self.assertIn(f'no_progress:{instance.instance_id}', events)

    def test_unreachable_joiners_do_not_create_an_unbounded_promise(self):
        context = self.context(self.state((0., 1., 16., 16.)))
        self.assertEqual(CombatRejoinSkill({}).propose(context, TacticalMemory(), Telemetry()), [])
        legacy = {'common_combat': {'local_cover_enabled': False, 'shared_front_enabled': False}}
        self.assertTrue(CombatRejoinSkill(legacy).propose(context, TacticalMemory(), Telemetry()))

    def test_unreachable_joiner_does_not_cancel_the_feasible_subgroup(self):
        context = self.context(self.state((0., 8., 16., 16.)))
        plans = CombatRejoinSkill({}).propose(context, TacticalMemory(), Telemetry())
        self.assertTrue(plans)
        self.assertTrue(all('u0' not in p.controlled_actors for p in plans))
        self.assertTrue(any(set(p.controlled_actors) == {'u1', 'u2', 'u3'} for p in plans))

    def test_full_health_exposed_receiver_cannot_borrow_safe_receivers_reserve(self):
        state = self.state((10., 11., 16., 16.))
        state['visible_enemies']['e0'] = enemy((18., 0.), 'colossus')
        state['allies']['u3'].update(health=500., max_health=500.)
        context = self.context(state)
        legacy = {'common_combat': {'joint_assessment_enabled': False}}
        plans = CombatRejoinSkill(legacy).propose(context, TacticalMemory(), Telemetry())
        self.assertTrue(plans)
        self.assertTrue(all('u2' not in p.controlled_actors for p in plans))
        self.assertTrue(any('u3' in p.goal_data['supporters'] for p in plans))
        # The joint mode retains the declaration; the final assessment still
        # keeps each receiver's own reserve (covered in joint contract tests).
        joint = CombatRejoinSkill({}).propose(context, TacticalMemory(), Telemetry())
        self.assertTrue(any('u2' in p.controlled_actors for p in joint))

    def test_external_override_cannot_commit_pruned_identity(self):
        memory, skill, instance = self.start()
        state = self.state((8., 9., 16., 16.), 88)
        state['allies']['u2']['health'] = 8.
        context = self.context(state)
        self.observe(memory, context)
        plan = skill.propose(context, memory, Telemetry())[0]
        memory.commit(context.snapshot, dict(plan.current_actions, u0=1), [plan], context=context)
        self.assertNotIn(instance.instance_id, memory.instances)


if __name__ == '__main__':
    unittest.main()
