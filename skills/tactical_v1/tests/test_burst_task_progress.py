"""S3 task progress is observed relative closure, never inferred detonation."""
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.contracts import Commitment, ResourceClaims
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.common import make_plan
from .fixtures import scene, enemy, with_questions


class BurstTaskProgressTests(unittest.TestCase):
    def context(self, memory, x=0., target_x=10., loop=80, tag=123):
        state = scene([((x, 0.), 'baneling')],
            enemies={'e0': enemy((target_x, 0.), tag=tag)}, loop=loop, attack_range=.1)
        c = Context(build_snapshot(with_questions(state), loop))
        memory.begin_step(c.snapshot)
        memory.observe(c.snapshot)
        memory.update_task_facts(c.snapshot, c)
        return c

    def commit(self, memory, c, instance=None, phase='approaching'):
        delta = ({'kind': 'create', 'skill': 'burst_engage', 'actors': ['u0'],
            'goal': 'e0', 'goal_kind': 'enemy', 'phase': phase, 'max_loops': 48,
            'extra': {'no_progress_loops': 24}} if instance is None else
            {'kind': 'continue', 'instance_id': instance.instance_id})
        plan = make_plan('burst_engage', ('u0',), 'e0', {'u0': 4}, [],
            ResourceClaims(controls=('u0',)), Commitment(phase), {},
            c.snapshot.snapshot_id, 'test', goal_kind='enemy', memory_delta=delta,
            instance_id=instance.instance_id if instance else None)
        memory.commit(c.snapshot, plan.current_actions, [plan], context=c)
        return memory.instance_for('burst_engage', 'u0')

    def test_same_speed_chase_times_out_without_false_displacement_progress(self):
        memory = TacticalMemory()
        c = self.context(memory)
        task = self.commit(memory, c)
        for loop, x in ((88, 1.), (96, 2.)):
            c = self.context(memory, x, 10.+x, loop)
            self.assertEqual(task.last_progress_loop, 80)
            self.commit(memory, c, task)
        self.context(memory, 3., 13., 104)
        self.assertEqual(task.termination_reason, 'no_progress')
        self.assertNotIn(task.instance_id, memory.instances)
        self.assertTrue(memory.retry_blocked(*memory.task_retry_key(
            'burst_engage', ['u0'], 'enemy', 'e0'), 104))

    def test_actual_relative_closure_counts_even_if_actor_stands_still(self):
        memory = TacticalMemory()
        task = self.commit(memory, self.context(memory))
        self.context(memory, 0., 9., 88)
        self.assertEqual(task.last_progress_loop, 88)
        self.assertEqual((task.start_loop, task.max_loops), (80, 48))

    def test_target_replacement_is_lost_not_success_and_transition_does_not_renew(self):
        memory = TacticalMemory()
        c = self.context(memory)
        task = self.commit(memory, c)
        c = self.context(memory, 1., 9., 88)
        memory.apply_delta(c.snapshot, {'kind': 'transition', 'instance_id': task.instance_id,
                           'phase': 'bursting'}, None, c)
        self.assertEqual((task.start_loop, task.max_loops, task.phase_start_loop), (80, 48, 88))
        self.context(memory, 1., 9., 96, tag=999)
        self.assertEqual(task.termination_reason, 'target_lost')
        self.assertNotIn(task.instance_id, memory.instances)


if __name__ == '__main__':
    unittest.main()
