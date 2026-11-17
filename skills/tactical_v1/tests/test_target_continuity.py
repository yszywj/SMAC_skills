"""A real attack target survives a cooldown task without becoming a hard lock."""
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.combat_reposition import CombatRepositionSkill
from skills.tactical_v1.skills.focus_fire import FocusFireSkill
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import enemy, scene, with_questions


class TargetContinuityTests(unittest.TestCase):
    def after_reposition(self):
        memory = TacticalMemory()
        state = scene([((0., 0.), "stalker")], loop=80,
                      enemies={"e0": enemy((2., 0.), "stalker"),
                               "e1": enemy((4., 0.), "stalker")}, attack_range=6.)

        def observe():
            context = Context(build_snapshot(with_questions(state), state["game_loop"]))
            memory.begin_step(context.snapshot)
            memory.observe(context.snapshot)
            memory.update_task_facts(context.snapshot, context)
            return context

        context = observe()
        shot = next(p for p in FocusFireSkill({}).propose(context, memory, Telemetry())
                    if p.goal == "e1" and p.controlled_actors == ("u0",))
        memory.commit(context.snapshot, shot.current_actions, [shot], context=context)
        state["game_loop"] = 88
        state["allies"]["u0"]["weapon_cooldown_loops"] = 20.
        context = observe()
        move = CombatRepositionSkill({}).propose(context, memory, Telemetry())[0]
        self.assertEqual(move.goal, "e1")
        memory.commit(context.snapshot, move.current_actions, [move], context=context)
        self.assertIsNone(memory.instance_for("focus_fire", "u0"))
        state["game_loop"] = 96
        state["allies"]["u0"]["weapon_cooldown_loops"] = 0.
        return state, memory, observe

    def test_original_target_keeps_a_candidate_and_soft_switch_cost_after_g3(self):
        _, memory, observe = self.after_reposition()
        context = observe()
        limited = FocusFireSkill({"search": {"max_candidates_per_unit": 1}})
        plans = limited.propose(context, memory, Telemetry())
        self.assertEqual([p.goal for p in plans], ["e1"])
        choices = FocusFireSkill({}).propose(context, memory, Telemetry())
        cost = {p.goal: p.features["switch_cost"] for p in choices}
        self.assertEqual(cost["e1"], 0.)
        self.assertGreater(cost["e0"], 0.)

    def test_new_immediate_kill_can_displace_remembered_target(self):
        state, memory, observe = self.after_reposition()
        state["visible_enemies"]["e0"].update(health=1., shield=0.)
        context = observe()
        limited = FocusFireSkill({"search": {"max_candidates_per_unit": 1}})
        plans = limited.propose(context, memory, Telemetry())
        self.assertEqual([p.goal for p in plans], ["e0"])


if __name__ == "__main__":
    unittest.main()
