"""Counterexamples from the joint-combat review, all in memory/no SC2."""
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.evaluator import Evaluator
from skills.tactical_v1.joint_combat import assess_joint
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.recovery import RecoveryMemory
from skills.tactical_v1.skills.burst_engage import BurstEngageSkill
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import enemy, scene, with_questions
from .test_burst_mechanism import native_scene


def context(raw):
    return Context(build_snapshot(with_questions(raw), raw['game_loop']))


class JointReviewRegressions(unittest.TestCase):
    def support_scene(self):
        return scene([((0., 0.), 'marine'), ((4., 0.), 'marine')], {
            'e0': enemy((1., 0.), 'hydralisk', health=90.),
            'e1': enemy((9., 0.), 'hydralisk', health=90.)})

    def test_stop_cannot_repair_support_by_inventing_a_new_target(self):
        raw = self.support_scene()
        # The peer is firing into the other fight. Stopping must not make it
        # cover e0 just because e0 is physically attackable from this position.
        raw['allies']['u1']['orders'] = [{'kind': 'attack', 'target_unit': 'e1'}]
        c = context(raw)
        attack = assess_joint(c, {'u0': 6, 'u1': 7})['actors']['u0']
        stopped = assess_joint(c, {'u0': 6, 'u1': 1})
        self.assertEqual(set(stopped['actors']['u1']['target_etas']), {'e1'})
        self.assertEqual(stopped['actors']['u0']['support_after'], 0.)
        self.assertGreaterEqual(stopped['actors']['u0']['support_loss'], attack['support_loss'])

    def test_stop_without_current_attack_or_with_queued_attack_promises_no_cover(self):
        for orders in ([], [{'kind': 'move'}, {'kind': 'attack', 'target_unit': 'e0'}]):
            with self.subTest(orders=orders):
                raw = self.support_scene()
                raw['allies']['u1']['orders'] = orders
                result = assess_joint(context(raw), {'u0': 6, 'u1': 1})
                self.assertEqual(result['actors']['u1']['target_etas'], {})
                self.assertEqual(result['actors']['u0']['support_after'], 0.)

    def test_stopping_an_existing_attack_cannot_offer_more_cover_than_firing(self):
        raw = self.support_scene()
        raw['allies']['u1']['orders'] = [{'kind': 'attack', 'target_unit': 'e0'}]
        c = context(raw)
        attack = assess_joint(c, {'u0': 6, 'u1': 6})['actors']['u0']
        stopped = assess_joint(c, {'u0': 6, 'u1': 1})['actors']['u0']
        self.assertGreater(attack['support_after'], 0.)
        self.assertLessEqual(stopped['support_after'], .5 * attack['support_after'])
        self.assertGreaterEqual(stopped['support_loss'], attack['support_loss'])

    def test_last_heal_interval_when_medivac_dies_cannot_train_passive_regeneration(self):
        raw = scene([((0., 0.), 'marine'), ((0., 1.), 'marine'), ((1., 0.), 'medivac')])
        for actor in ('u0', 'u1'):
            raw['allies'][actor].update(health=20., health_lost_since_previous_decision=1.)
        memory = RecoveryMemory()
        memory.observe(context(raw))
        raw['game_loop'] += 8
        raw['allies']['u2']['alive'] = False
        for actor in ('u0', 'u1'):
            raw['allies'][actor].update(health=22., health_lost_since_previous_decision=0.)
        c = context(raw)
        memory.observe(c)
        self.assertNotIn((48, 'health'), memory.models)
        self.assertEqual(memory.opportunity(c, 'u0')['gain'], 0.)
        self.assertFalse(memory.opportunity(c, 'u0')['known'])

    def test_verified_passive_recovery_has_no_stop_only_reward(self):
        raw = scene([((0., 0.), 'marine')],
                    {'e0': enemy((4., 0.), 'hydralisk')}, enemy_can_reach=False)
        raw['allies']['u0']['health'] = 20.
        raw['unit_type_profiles']['48']['passive_recovery'] = {
            'health': {'verified': True, 'rate_per_loop': .125, 'delay_loops': 0}}
        c, evaluator = context(raw), Evaluator()
        values = [evaluator.score_joint(c, [], {'u0': action})[1]['passive_recovery']
                  for action in (1, 2, 6)]
        self.assertGreater(values[0], 0.)
        self.assertEqual(values, [values[0]] * 3)

    def test_unknown_passive_recovery_is_zero_for_stop_move_and_attack(self):
        raw = scene([((0., 0.), 'marine')],
                    {'e0': enemy((4., 0.), 'hydralisk')}, enemy_can_reach=False)
        raw['allies']['u0']['health'] = 20.
        c, evaluator = context(raw), Evaluator()
        for action in (1, 2, 6):
            self.assertEqual(evaluator.score_joint(c, [], {'u0': action})[1]['passive_recovery'], 0.)

    def test_baseline_burst_commits_keep_identity_and_deadline_without_predicted_progress(self):
        memory = TacticalMemory()

        def observe(loop, x):
            raw = native_scene([((x, 0.), 'baneling')],
                {'e0': enemy((10., 0.), 'hydralisk', health=90., radius=.625, tag=101)},
                loop=loop, enemy_can_reach=False)
            c = context(raw)
            memory.begin_step(c.snapshot)
            memory.observe(c.snapshot, {'u0': 0})
            memory.update_task_facts(c.snapshot, c)
            return c

        first = observe(80, 0.)
        plan = next(p for p in BurstEngageSkill({}).propose(first, memory, Telemetry())
                    if p.current_actions['u0'] == 4)
        memory.commit(first.snapshot, {'u0': 4}, [plan], context=first)
        task = memory.instance_for('burst_engage', 'u0')
        identity = task.instance_id
        deadline = task.start_loop + task.max_loops
        self.assertEqual(deadline, 128)

        for loop, x in ((88, 1.), (96, 1.), (104, 2.), (112, 3.), (120, 4.)):
            c = observe(loop, x)
            last_observed_progress = task.last_progress_loop
            executed = task.executed_steps
            # An empty accepted-plan list is the real baseline completion
            # path: semantic continuation must not require an S3 label.
            memory.commit(c.snapshot, {'u0': 4}, [], context=c)
            self.assertIs(memory.instance_for('burst_engage', 'u0'), task)
            self.assertEqual(task.instance_id, identity)
            self.assertEqual(task.start_loop + task.max_loops, deadline)
            self.assertEqual(task.last_progress_loop, last_observed_progress)
            self.assertEqual(task.executed_steps, executed + 1)
            self.assertEqual(task.extra['approach_target_history'][-1],
                             (loop, 'e0', 101, (10., 0.)))
            memory.commit(c.snapshot, {'u0': 4}, [], context=c)
            self.assertEqual(task.executed_steps, executed + 1, 'same-clock commit remains idempotent')
            if loop == 96:
                self.assertEqual(task.last_progress_loop, 88, 'a blocked step did not create progress')
        observe(128, 5.)
        self.assertNotIn(identity, memory.instances)
        self.assertEqual(task.termination_reason, 'timeout')


if __name__ == '__main__':
    unittest.main()
