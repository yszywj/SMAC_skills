"""A moving enemy must not turn actor translation into closing progress."""
import copy
import unittest
from .fixtures import scene, enemy, with_questions
from ..context import Context, build_snapshot
from ..memory import TacticalMemory
from ..skill_manager import SkillManager
from ..telemetry import Telemetry


def observation(loop, actor_x, target_x, tag='enemy-a', alternate=False):
    enemies = {'e0': enemy((target_x, 0.), tag=tag)}
    if alternate:
        enemies['e1'] = enemy((target_x+.2, 1.), tag='enemy-b')
    state = scene([((actor_x, 0.), 'marine')], enemies, loop=loop)
    return Context(build_snapshot(with_questions(state), loop))


class RelativeApproachTests(unittest.TestCase):
    def setUp(self):
        self.skill = SkillManager().skills['coordinated_engage']
        self.memory = TacticalMemory()
        self.telemetry = Telemetry()
        c = observation(80, 0., 6.5)
        plan = next(p for p in self.skill.propose(c, self.memory, self.telemetry)
                    if p.commitment.phase == 'approaching')
        self.memory.commit(c.snapshot, plan.current_actions, [plan], context=c)
        self.instance = self.memory.instances[plan.instance_id]

    def test_committed_history_records_both_sides_at_the_same_loop(self):
        self.assertEqual(self.instance.extra['approach_history'][0][0], 80)
        self.assertEqual(self.instance.extra['approach_target_history'][0],
                         (80, 'e0', 'enemy-a', (6.5, 0.)))

    def test_equal_speed_chase_is_no_progress(self):
        self.assertTrue(self.skill._blocked(observation(104, 3., 9.5), self.memory, 'u0', 'e0'))

    def test_actual_closing_is_progress(self):
        self.assertFalse(self.skill._blocked(observation(104, 1., 6.5), self.memory, 'u0', 'e0'))

    def test_enemy_approach_is_relative_closing_even_if_actor_did_not_move(self):
        self.assertFalse(self.skill._blocked(observation(104, 0., 5.5), self.memory, 'u0', 'e0'))

    def test_new_identity_or_missing_synchronized_history_is_not_stall_evidence(self):
        self.assertFalse(self.skill._blocked(observation(104, 3., 9.5, 'replacement'), self.memory, 'u0', 'e0'))
        self.instance.extra['approach_target_history'].clear()
        self.assertFalse(self.skill._blocked(observation(104, 3., 9.5), self.memory, 'u0', 'e0'))

    def test_stalled_target_does_not_override_a_local_alternative_or_mutate_query(self):
        c = observation(104, 3., 9.5, alternate=True)
        before = copy.deepcopy(self.memory.__dict__)
        plans = self.skill.propose(c, self.memory, self.telemetry)
        self.assertTrue(plans)
        self.assertEqual({p.goal for p in plans}, {'e1'})
        self.assertEqual(self.memory.__dict__, before)
        self.assertEqual(self.instance.start_loop, 80)


if __name__ == '__main__':
    unittest.main()
