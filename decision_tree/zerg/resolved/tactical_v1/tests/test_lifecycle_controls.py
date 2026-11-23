"""Configured lifecycle limits govern actual observation/commit behaviour."""
import json
import tempfile
import unittest
from pathlib import Path

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skill_manager import SkillManager
from .fixtures import enemy, scene, with_questions


class LifecycleControlTests(unittest.TestCase):
    def context(self, loop, x=0., enemy_x=20.):
        return Context(build_snapshot(with_questions(scene(
            [((x, 0.), "marine")], enemies={"e0": enemy((enemy_x, 0.))},
            loop=loop)), loop))

    def test_retry_configuration_survives_manager_reset_and_controls_failure(self):
        for multiplier in (0., 2., 10.):
            with self.subTest(multiplier=multiplier), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "config.json"
                path.write_text(json.dumps({"lifecycle": {
                    "retry_cooldown_h": multiplier}}), encoding="utf8")
                manager = SkillManager(tactical_config=str(path))
                manager.reset()
                context = self.context(80)
                plan = manager.skills["coordinated_engage"].propose(
                    context, manager.memory, manager.telemetry)[0]
                manager.memory.commit(context.snapshot, plan.current_actions,
                                      [plan], context=context)
                instance = manager.memory.instance_by_id(plan.instance_id)
                self.assertEqual(instance.extra["retry_cooldown_loops"], int(8 * multiplier))
                manager.memory.finish_task(instance.instance_id, "timeout", context.snapshot)
                key = manager.memory.task_retry_key(
                    "coordinated_engage", ["u0"], "enemy", "e0")
                self.assertEqual(manager.memory.retry_records[key], 80 + int(8 * multiplier))
                self.assertEqual(manager.memory.retry_blocked(*key, 80), multiplier > 0)

    def test_recent_stall_uses_window_boundary_not_oldest_position(self):
        memory = TacticalMemory()
        memory.move_history["u0"] = [(8, (0., 0.)), (16, (5., 0.)),
                                     (24, (5., 0.)), (32, (5., 0.))]
        self.assertEqual(memory.displacement("u0", 16, self.context(40, 5.).snapshot), 0.)

    def test_stuck_approach_exits_before_total_deadline(self):
        manager = SkillManager()
        context = self.context(80)
        manager.memory.begin_step(context.snapshot)
        manager.memory.observe(context.snapshot)
        plan = manager.skills["coordinated_engage"].propose(
            context, manager.memory, manager.telemetry)[0]
        manager.memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
        instance = manager.memory.instance_by_id(plan.instance_id)
        self.assertEqual(instance.max_loops, 96)
        context = self.context(104)
        events = manager.memory.update_task_facts(context.snapshot, context)
        self.assertIsNone(manager.memory.instance_by_id(instance.instance_id))
        self.assertIn(f"task_no_progress:{instance.instance_id}", events)

    def test_observed_progress_on_stall_boundary_prevents_false_failure(self):
        manager = SkillManager()
        context = self.context(80)
        manager.memory.begin_step(context.snapshot)
        plan = manager.skills["coordinated_engage"].propose(
            context, manager.memory, manager.telemetry)[0]
        manager.memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
        context = self.context(104, x=1.)
        manager.memory.update_task_facts(context.snapshot, context)
        instance = manager.memory.instance_by_id(plan.instance_id)
        self.assertIsNotNone(instance)
        self.assertEqual(instance.last_progress_loop, 104)

    def test_lifecycle_counts_only_real_commits_and_exits(self):
        manager = SkillManager()
        for loop in (80, 88):
            context = self.context(loop)
            manager.memory.begin_step(context.snapshot)
            plan = manager.skills["coordinated_engage"].propose(
                context, manager.memory, manager.telemetry)[0]
            manager.memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
            manager.memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
        counts = manager.memory.lifecycle_counts["coordinated_engage"]
        self.assertEqual(counts, {"created": 1, "continued": 1, "exited": 0})
        self.assertEqual(manager.memory.last_committed_loop, 88)
        manager.memory.finish_task(plan.instance_id, "timeout", context.snapshot)
        manager.memory.finish_task(plan.instance_id, "timeout", context.snapshot)
        self.assertEqual(counts["exited"], 1)

    def test_explicit_drop_closes_rally_encounter_through_normal_termination(self):
        memory = TacticalMemory()
        snapshot = self.context(80).snapshot
        instance = memory.create_instance(
            "precombat_rally", ["u0"], "encounter:1", "rallying", 80, 64,
            goal_kind="area", extra={"encounter_id": 1})
        memory.apply_delta(snapshot, {"kind": "drop",
                                     "instance_id": instance.instance_id,
                                     "reason": "cancelled"}, None)
        self.assertEqual(memory.instances, {})
        self.assertEqual(memory.encounters.closed_ids, {1: "cancelled"})
        self.assertFalse(memory.encounters.can_rally())


if __name__ == "__main__":
    unittest.main()
