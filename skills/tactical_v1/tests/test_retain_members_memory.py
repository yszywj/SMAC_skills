"""A shrinking rejoin task retains its original identity and fixed budget."""
import copy
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.contracts import Commitment, ResourceClaims
from skills.tactical_v1.evaluator import Evaluator
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.combat_rejoin import CombatRejoinSkill
from skills.tactical_v1.skills.common import make_plan
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import scene, enemy, with_questions


class RetainMembersMemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = TacticalMemory()
        s = scene([((0., 0.), 'marine'), ((5., 0.), 'marine'),
                   ((6., 0.), 'marine'), ((-1., 0.), 'marine')],
                  {'e0': enemy((12., 0.))})
        for actor, unit in s['allies'].items():
            unit['tag'] = actor + '-original'
        self.state = s
        self.snapshot = self.context(s).snapshot
        self.extra = {'local_cover_enabled': True, 'joiners': ['u0'],
            'supporters': ['u1', 'u2'], 'anchor': [5., 0.],
            'roles': {'u0': 'join', 'u1': 'cover', 'u2': 'cover'},
            'targets_by_actor': {'u0': ['e0'], 'u1': ['e0'], 'u2': ['e0']},
            'joiner_progress': {'u0': {'best_error': 5., 'last_progress_loop': 72,
                                      'observed_progress': True}},
            'best_support_gap': 4., 'last_support_progress_loop': 72,
            'no_progress_loops': 24, 'min_loops': 8}
        self.task = self.memory.create_instance('combat_rejoin', ['u0', 'u1', 'u2'],
            'old-duty', 'rejoining', 64, 48,
            tuple(self.snapshot.allies[a].tag for a in ('u0', 'u1', 'u2')),
            goal_kind='area', extra=copy.deepcopy(self.extra))

    def context(self, s):
        return Context(build_snapshot(with_questions(s), s['game_loop']))

    def delta(self, actors=('u0', 'u1')):
        data = copy.deepcopy(self.extra)
        data.update(joiners=['u0'], supporters=list(actors[1:]),
                    best_support_gap=0., last_support_progress_loop=80,
                    no_progress_loops=1000, retry_key=('new', 'fake', 'key'))
        return {'kind': 'retain_members', 'instance_id': self.task.instance_id,
                'actors': list(actors), 'extra': data}

    def test_commit_prunes_members_but_never_restarts_deadline_or_progress(self):
        old = copy.deepcopy(self.task)
        self.memory.apply_delta(self.snapshot, self.delta(), None)
        self.assertEqual(self.task.actors, ('u0', 'u1'))
        self.assertEqual(self.task.ally_tags, old.ally_tags[:2])
        for key in ('start_loop', 'phase_start_loop', 'last_progress_loop',
                    'max_loops', 'goal', 'goal_kind', 'instance_id'):
            self.assertEqual(getattr(self.task, key), getattr(old, key))
        for key in ('retry_key', 'best_support_gap', 'last_support_progress_loop',
                    'no_progress_loops', 'joiner_progress', 'anchor'):
            self.assertEqual(self.task.extra[key], old.extra[key])
        self.assertNotIn('u2', self.task.extra['roles'])
        self.assertNotIn('u2', self.task.extra['targets_by_actor'])
        self.assertIsNone(self.memory.instance_for('combat_rejoin', 'u2'))

    def test_cannot_add_member_or_inherit_replacement_tag(self):
        before = copy.deepcopy(self.task)
        self.memory.apply_delta(self.snapshot, self.delta(('u0', 'u3')), None)
        self.assertEqual(self.task, before)
        changed = copy.deepcopy(self.state)
        changed['allies']['u1']['tag'] = 'replacement'
        self.memory.apply_delta(self.context(changed).snapshot, self.delta(), None)
        self.assertEqual(self.task, before)

    def test_invalid_member_waits_for_committed_subset_only_with_both_roles(self):
        s = copy.deepcopy(self.state)
        s['allies']['u2']['alive'] = False
        self.assertEqual(self.memory.retire_invalid(self.context(s).snapshot), [])
        self.assertEqual(self.task.actors, ('u0', 'u1', 'u2'))  # observation is not commit
        s['allies']['u1']['alive'] = False
        self.assertEqual(self.memory.retire_invalid(self.context(s).snapshot),
                         [self.task.instance_id])
        self.assertFalse(self.memory.instances)

    def test_disabled_flag_keeps_original_all_member_lifecycle(self):
        self.task.extra['local_cover_enabled'] = False
        before = copy.deepcopy(self.task)
        self.memory.apply_delta(self.snapshot, self.delta(), None)
        self.assertEqual(self.task, before)
        s = copy.deepcopy(self.state)
        s['allies']['u2']['alive'] = False
        self.memory.retire_invalid(self.context(s).snapshot)
        self.assertFalse(self.memory.instances)


class RejoinBoundaryTests(unittest.TestCase):
    def context(self, state):
        return Context(build_snapshot(with_questions(state), state['game_loop']))

    def observe(self, memory, state):
        context = self.context(state)
        memory.begin_step(context.snapshot)
        memory.observe(context.snapshot)
        memory.update_task_facts(context.snapshot, context)
        return context

    def test_escape_support_credit_is_not_cancelled_by_increasing_weapon_gap(self):
        state = scene([((8., 0.), 'stalker'), ((16., 0.), 'stalker')],
                      {'e0': enemy((0., 0.))}, distance=2., enemy_can_reach=False)
        context = self.context(state)
        data = {'joiners': ['u0'], 'supporters': ['u1'], 'support_radius': 6.,
                'roles': {'u0': 'escape', 'u1': 'cover'}, 'local_cover_enabled': True,
                'targets_by_actor': {'u0': ['e0'], 'u1': ['e0']},
                'objective': {'kind': 'combat_rejoin'}}
        actions = {'u0': 4, 'u1': 1}
        plan = make_plan('combat_rejoin', tuple(actions), 'escape-to-support', actions, [],
            ResourceClaims(controls=tuple(actions)), Commitment('rejoining'), {},
            context.snapshot.snapshot_id, 'test', goal_kind='area', goal_data=data)
        # No enemy can attack this pair and the receiver cannot shoot. The
        # new contract grants no combat-cover credit for this empty retreat.
        self.assertEqual(Evaluator().score_joint(context, [plan], actions)[1]['link'], 0.)
        evaluator = Evaluator(common_combat={'joint_assessment_enabled': False})
        enabled = evaluator.score_joint(context, [plan], actions)[1]['link']
        self.assertGreater(enabled, 0.)
        self.assertLessEqual(enabled, evaluator.weights['link'])  # one credit, not two
        plan.goal_data['local_cover_enabled'] = False
        legacy = evaluator.score_joint(context, [plan], actions)[1]['link']
        self.assertAlmostEqual(legacy, 0.)  # positive geometry and negative weapon ETA cancel

    def test_committed_retained_subgroups_share_original_timeout_cooldown(self):
        def state(loop, missing=False):
            s = scene([((x, 0.), 'stalker') for x in (7., 8., 16., 16.)],
                      {'e0': enemy((18., 0.))}, loop=loop)
            if loop > 80:
                s['allies']['u2']['health'] = 8.
            if missing:
                s['allies']['u0']['alive'] = False
            return s
        memory, skill = TacticalMemory(), CombatRejoinSkill({})
        context = self.observe(memory, state(80))
        plan = skill.propose(context, memory, Telemetry())[0]
        memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
        instance = memory.instances[plan.instance_id]
        original = instance.extra['retry_key']
        deadline = instance.start_loop + instance.max_loops
        retained_keys = []
        for loop, missing in ((88, False), (96, True)):
            context = self.observe(memory, state(loop, missing))
            plan = skill.propose(context, memory, Telemetry())[0]
            self.assertEqual(plan.proposed_memory_delta['kind'], 'retain_members')
            memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
            goal = '+'.join(plan.goal_data['joiners']) + '->' + '+'.join(plan.goal_data['supporters'])
            retained_keys.append(memory.task_retry_key(skill.name, plan.controlled_actors,
                                                       'internal', goal))
            self.assertEqual(instance.start_loop + instance.max_loops, deadline)
        context = self.context(state(deadline, True))
        memory.begin_step(context.snapshot)
        memory.observe(context.snapshot)
        memory.finish_task(instance.instance_id, 'timeout', context.snapshot)
        until = deadline + 2 * context.H
        for key in (original, *retained_keys):
            self.assertEqual(memory.retry_records[key], until)
            self.assertTrue(memory.retry_blocked(*key, deadline))
            self.assertFalse(memory.retry_blocked(*key, until))
        self.assertFalse(skill.propose(context, memory, Telemetry()))
        # Expiration releases the same feasible duty, with no permanent lock.
        later = self.observe(memory, state(until, True))
        self.assertTrue(skill.propose(later, memory, Telemetry()))

if __name__ == '__main__':
    unittest.main()
