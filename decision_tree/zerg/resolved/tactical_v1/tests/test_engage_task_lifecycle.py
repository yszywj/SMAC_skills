"""G2 task continuity through real proposal, observation and commit phases."""
import copy
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skill_manager import SkillManager
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import enemy, scene, with_questions


class EngageTaskLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.memory = TacticalMemory()
        self.skill = SkillManager().skills["coordinated_engage"]
        self.telemetry = Telemetry()

    def observe(self, loop, enemies=None):
        # Both actors are outside contact and one move from a common frontage.
        # u0 is still cooling; a short precontact synchronization is admissible.
        state = scene([((2.7, 0.), "marine"), ((2.1, 0.), "marine")],
                      enemies=enemies or {"e0": enemy((8., 0.))}, loop=loop)
        state["allies"]["u0"].update(weapon_cooldown_loops=12., weapon_ready=False)
        context = Context(build_snapshot(with_questions(state), loop))
        self.memory.begin_step(context.snapshot)
        self.memory.observe(context.snapshot)
        self.memory.update_task_facts(context.snapshot, context)
        return context

    def proposals(self, context):
        return [plan for plan in self.skill.propose(
            context, self.memory, self.telemetry)
                if plan.controlled_actors == ("u0",)]

    def select(self, context, phase):
        return next(plan for plan in self.proposals(context)
                    if plan.commitment.phase == phase)

    def commit(self, context, phase):
        plan = self.select(context, phase)
        self.memory.commit(context.snapshot, dict(plan.current_actions),
                           [plan], context=context)
        return plan

    def test_wait_continues_same_instance_and_proposals_are_read_only(self):
        first = self.commit(self.observe(80), "waiting")
        original = copy.deepcopy(self.memory.instances[first.instance_id])
        context = self.observe(88)
        before = copy.deepcopy(self.memory.__dict__)
        plan = self.select(context, "waiting")
        self.assertEqual(self.memory.__dict__, before)
        self.assertEqual(plan.instance_id, first.instance_id)
        self.assertEqual(plan.proposed_memory_delta["kind"], "continue")
        self.memory.commit(context.snapshot, dict(plan.current_actions),
                           [plan], context=context)
        self.assertEqual(list(self.memory.instances), [first.instance_id])
        current = self.memory.instances[first.instance_id]
        self.assertEqual(current.start_loop, original.start_loop)
        self.assertEqual(current.phase_start_loop, original.phase_start_loop)
        self.assertEqual(current.executed_steps, 2)

    def test_wait_expires_and_cannot_restart_until_cooldown_ends(self):
        first = self.commit(self.observe(80), "waiting")
        self.commit(self.observe(88), "waiting")
        context = self.observe(96)
        self.assertNotIn(first.instance_id, self.memory.instances)
        self.assertEqual(self.proposals(context), [])
        key = self.memory.task_retry_key(
            self.skill.name, ["u0"], "enemy", "e0")
        until = self.memory.retry_records[key]
        self.assertEqual(self.proposals(self.observe(until - 1)), [])
        restarted = self.commit(self.observe(until), "waiting")
        self.assertNotEqual(restarted.instance_id, first.instance_id)
        self.assertEqual(self.memory.instances[restarted.instance_id].start_loop,
                         until)

    def test_contact_on_wait_deadline_completes_without_cooldown(self):
        first = self.commit(self.observe(80), "waiting")
        self.commit(self.observe(88), "waiting")
        state = scene([((4., 0.), "marine"), ((6., 0.), "marine")],
                      enemies={"e0": enemy((8., 0.))}, loop=96)
        context = Context(build_snapshot(with_questions(state), 96))
        self.memory.begin_step(context.snapshot)
        self.memory.observe(context.snapshot)
        events = self.memory.update_task_facts(context.snapshot, context)
        self.assertNotIn(first.instance_id, self.memory.instances)
        self.assertFalse(self.memory.retry_records)
        self.assertFalse(any("timeout" in event for event in events))

    def test_wait_phase_gets_own_deadline_after_long_approach(self):
        # Isolate phase timing; the stationary fixture otherwise correctly
        # reaches the shorter no-progress deadline during this long approach.
        self.memory.no_progress_window_h = 20.0
        first = self.commit(self.observe(80), "approaching")
        self.commit(self.observe(104), "approaching")
        waiting = self.commit(self.observe(112), "waiting")
        self.assertEqual(waiting.instance_id, first.instance_id)
        instance = self.memory.instances[first.instance_id]
        self.assertEqual(instance.start_loop, 80)
        self.assertEqual(instance.phase_start_loop, 112)
        self.assertEqual(instance.max_loops, 96)
        self.commit(self.observe(120), "waiting")
        self.assertEqual(instance.phase_start_loop, 112)
        self.observe(128)
        self.assertNotIn(first.instance_id, self.memory.instances)

    def test_leaving_wait_clears_phase_deadline_but_keeps_total_budget(self):
        first = self.commit(self.observe(80), "waiting")
        approach = self.commit(self.observe(88), "approaching")
        self.assertEqual(approach.instance_id, first.instance_id)
        instance = self.memory.instances[first.instance_id]
        self.assertEqual(instance.phase, "approaching")
        self.assertEqual(instance.phase_start_loop, 88)
        self.assertIsNone(instance.extra["phase_max_loops"])
        self.commit(self.observe(104), "approaching")
        self.assertEqual(instance.start_loop, 80)
        self.assertEqual(instance.phase_start_loop, 88)
        self.observe(176)
        self.assertNotIn(first.instance_id, self.memory.instances)

    def test_failed_target_does_not_block_an_independent_enemy(self):
        enemies = {"e0": enemy((8., 0.)), "e1": enemy((10., 0.))}
        context = self.observe(80, enemies)
        first = self.commit(context, "approaching")
        self.memory.finish_task(first.instance_id, "timeout", context.snapshot)
        context = self.observe(88, enemies)
        plans = self.proposals(context)
        self.assertTrue(plans)
        self.assertEqual({plan.goal for plan in plans}, {"e1"})
        second = self.commit(context, "approaching")
        self.memory.finish_task(second.instance_id, "timeout", context.snapshot)
        self.assertEqual(self.proposals(self.observe(95, enemies)), [])
        # e0's configured 2H retry interval ends at loop 96; e1 is
        # independently blocked until loop 104.
        self.assertEqual({plan.goal for plan in self.proposals(
            self.observe(96, enemies))}, {"e0"})

    def test_changed_target_creates_new_identity_only_on_commit(self):
        first = self.commit(self.observe(80, {
            "e0": enemy((8., 0.)), "e1": enemy((12., 0.))}), "waiting")
        context = self.observe(88, {
            "e0": enemy((16., 0.)), "e1": enemy((8., 0.))})
        plan = self.select(context, "waiting")
        self.assertEqual(plan.goal, "e1")
        self.assertIsNone(plan.instance_id)
        self.assertIn(first.instance_id, self.memory.instances)
        self.memory.commit(context.snapshot, dict(plan.current_actions),
                           [plan], context=context)
        self.assertNotEqual(plan.instance_id, first.instance_id)
        self.assertEqual(list(self.memory.instances), [plan.instance_id])
        self.assertEqual(self.memory.instances[plan.instance_id].goal, "e1")
        self.assertFalse(self.memory.retry_records)


if __name__ == "__main__":
    unittest.main()
