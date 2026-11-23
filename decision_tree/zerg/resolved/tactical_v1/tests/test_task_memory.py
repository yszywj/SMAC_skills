"""Regression checks across task creation, failure, retry and final commit."""
import copy
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.contracts import Commitment, ResourceClaims
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.common import make_plan
from .fixtures import enemy, scene, with_questions


class TaskMemoryTests(unittest.TestCase):
    def context(self, loop=80):
        state = scene([((0., 0.), "marine"), ((1., 0.), "marine")],
                      enemies={"e0": enemy((10., 0.))}, loop=loop)
        return Context(build_snapshot(with_questions(state), loop))

    def plan(self, goal="e0", phase="approaching", actors=("u0",)):
        return make_plan(
            "coordinated_engage", actors, goal, {a: 4 for a in actors}, [],
            ResourceClaims(controls=actors), Commitment(phase), {}, 80, "test",
            goal_kind="enemy", memory_delta={
                "kind": "create", "skill": "coordinated_engage",
                "actors": list(actors), "goal": goal, "goal_kind": "enemy",
                "phase": phase, "max_loops": 96})

    def test_failed_task_key_round_trips_without_splitting_actor_string(self):
        for actors in (("u0",), ("u1", "u0")):
            with self.subTest(actors=actors):
                memory = TacticalMemory()
                snapshot = self.context().snapshot
                instance = memory.create_instance(
                    "coordinated_engage", actors, "e0", "waiting", 64, 16,
                    goal_kind="enemy")
                memory.finish_task(instance.instance_id, "timeout", snapshot)
                key = memory.task_retry_key(
                    "coordinated_engage", actors, "enemy", "e0")
                before = copy.deepcopy(memory.retry_records)
                self.assertTrue(memory.retry_blocked(*key, 80))
                self.assertTrue(memory.retry_blocked(key[0], actors, key[2], 80))
                self.assertFalse(memory.retry_blocked(*key, before[key]))
                self.assertEqual(memory.retry_records, before)
                # Recording an already-normalized key must work as well.
                memory.record_retry(*key, 80, 16)
                self.assertEqual(set(memory.retry_records), {key})
                self.assertEqual(memory.retry_records[key], 96)

    def test_create_commit_checks_the_same_typed_key_written_on_failure(self):
        cases = [
            ("enemy", "e0", {}, "e1", {}),
            ("position", "waypoint", {"waypoint": (20., 0.)},
             "waypoint", {"waypoint": (40., 0.)}),
            ("group_link", "merge", {"group_a": ["u0"], "group_b": ["u1"]},
             "merge", {"group_a": ["u0"], "group_b": ["u2"]}),
        ]
        for goal_kind, goal, extra, other_goal, other_extra in cases:
            with self.subTest(goal_kind=goal_kind):
                memory = TacticalMemory()
                snapshot = self.context().snapshot
                delta = {"kind": "create", "skill": "test", "actors": ["u0"],
                         "goal_kind": goal_kind, "goal": goal, "extra": extra}
                memory.apply_delta(snapshot, delta, None)
                instance = next(iter(memory.instances.values()))
                memory.finish_task(instance.instance_id, "timeout", snapshot)
                memory.apply_delta(snapshot, delta, None)
                self.assertFalse(memory.instances, "failed task restarted during cooldown")
                memory.apply_delta(snapshot, dict(delta, goal=other_goal,
                                                  extra=other_extra), None)
                self.assertEqual(len(memory.instances), 1)

    def test_explicit_stable_duty_key_is_used_at_creation_and_failure(self):
        memory = TacticalMemory()
        snapshot = self.context().snapshot
        key = memory.task_retry_key("heal_support", ["u0"], "internal", "squad:u1")
        delta = {"kind": "create", "skill": "heal_support", "actors": ["u0"],
                 "goal": "escort", "goal_kind": "position", "phase": "escort",
                 "extra": {"waypoint": (20., 0.), "retry_key": key}}
        memory.apply_delta(snapshot, delta, None)
        instance = next(iter(memory.instances.values()))
        memory.finish_task(instance.instance_id, "timeout", snapshot)
        delta["extra"]["waypoint"] = (40., 0.)
        memory.apply_delta(snapshot, delta, None)
        self.assertFalse(memory.instances)
        self.assertTrue(memory.retry_blocked(*key, snapshot.game_loop))

    def test_first_commit_is_counted_and_repeated_commit_is_idempotent(self):
        memory = TacticalMemory()
        context = self.context()
        memory.begin_step(context.snapshot)
        plan = self.plan()
        memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
        self.assertIsNotNone(plan.instance_id)
        instance = memory.instances[plan.instance_id]
        self.assertEqual(instance.executed_steps, 1)
        memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
        self.assertEqual(instance.executed_steps, 1)
        self.assertEqual(len(memory.instances), 1)

    def test_changed_final_atomic_action_does_not_create_or_keep_task(self):
        memory = TacticalMemory()
        context = self.context()
        memory.begin_step(context.snapshot)
        plan = self.plan(actors=("u0", "u1"))
        memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
        context = self.context(88)
        memory.begin_step(context.snapshot)
        plan.proposed_memory_delta = {"kind": "continue", "instance_id": plan.instance_id}
        memory.commit(context.snapshot, {"u0": 4, "u1": 1}, [plan], context=context)
        self.assertFalse(memory.instances, "an externally changed member cancels the whole task")
        self.assertFalse(memory.retry_records)

    def test_retarget_failure_is_recorded_for_new_target(self):
        memory = TacticalMemory()
        context = self.context()
        instance = memory.create_instance("burst_engage", ["u0"], "e0", "bursting",
                                          64, 16, goal_kind="enemy")
        memory.apply_delta(context.snapshot, {
            "kind": "retarget", "instance_id": instance.instance_id,
            "goal": "e1", "goal_kind": "enemy"}, None)
        memory.finish_task(instance.instance_id, "timeout", context.snapshot)
        self.assertTrue(memory.retry_blocked(*memory.task_retry_key(
            "burst_engage", ["u0"], "enemy", "e1"), 80))
        self.assertFalse(memory.retry_blocked(*memory.task_retry_key(
            "burst_engage", ["u0"], "enemy", "e0"), 80))

    def test_typed_retry_delta_blocks_creation(self):
        memory = TacticalMemory()
        snapshot = self.context().snapshot
        plan = self.plan()
        memory.apply_delta(snapshot, dict(plan.proposed_memory_delta,
                                         kind="retry", cooldown_loops=16), None)
        memory.apply_delta(snapshot, plan.proposed_memory_delta, plan)
        self.assertFalse(memory.instances)
        self.assertIsNone(plan.instance_id)

    def test_escort_deadline_uses_current_squad_position(self):
        for current_position, should_fail in (((3., 0.), False), ((20., 0.), True)):
            with self.subTest(position=current_position):
                memory = TacticalMemory()
                instance = memory.create_instance(
                    "heal_support", ["u0"], "escort", "escort", 32, 48,
                    goal_kind="position", extra={
                        "waypoint": (9., 0.), "escort_members": ["u1"],
                        "follow_distance": 5.})
                state = scene([((5., 0.), "medivac"), (current_position, "marine")],
                              loop=80)
                context = Context(build_snapshot(with_questions(state), 1))
                memory.update_task_facts(context.snapshot, context)
                self.assertNotIn(instance.instance_id, memory.instances)
                self.assertEqual(bool(memory.retry_records), should_fail)


if __name__ == "__main__":
    unittest.main()
