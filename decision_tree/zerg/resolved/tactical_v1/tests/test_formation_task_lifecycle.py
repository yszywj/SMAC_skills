"""Singleton G6 adjustments keep their identity through final-action commit."""
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skill_manager import SkillManager
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import scene, with_questions


class FormationTaskLifecycleTests(unittest.TestCase):
    def test_repeated_adjustment_preserves_original_deadline(self):
        skill = SkillManager().skills["local_formation"]
        memory = TacticalMemory()
        telemetry = Telemetry()
        first_id = None
        # Blocked movement keeps the same adjustment relevant for three steps.
        for step, loop in enumerate((80, 88, 96), 1):
            state = scene([((0., 0.), "marine"), ((0.5, 0.), "marine")], loop=loop, distance=2.)
            context = Context(build_snapshot(with_questions(state), loop))
            memory.begin_step(context.snapshot)
            memory.observe(context.snapshot)
            memory.update_task_facts(context.snapshot, context)
            plan = next(p for p in skill.propose(context, memory, telemetry)
                        if p.controlled_actors == ("u0",))
            if first_id is not None:
                self.assertEqual(plan.instance_id, first_id)
                self.assertFalse(plan.proposed_memory_delta)
            memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
            if first_id is None:
                first_id = plan.instance_id
            self.assertEqual(list(memory.instances), [first_id])
            instance = memory.instances[first_id]
            self.assertEqual(instance.start_loop, 80)
            self.assertEqual(instance.phase_start_loop, 80)
            self.assertEqual(instance.executed_steps, step)
            self.assertEqual(instance.max_loops, 24)

        state["game_loop"] = 104
        context = Context(build_snapshot(with_questions(state), 104))
        memory.begin_step(context.snapshot)
        memory.observe(context.snapshot)
        memory.update_task_facts(context.snapshot, context)
        self.assertNotIn(first_id, memory.instances)
        key = memory.task_retry_key("local_formation", ["u0"], "internal", "formation")
        self.assertTrue(memory.retry_blocked(*key, 104))


if __name__ == "__main__":
    unittest.main()
