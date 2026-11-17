"""Combat-model and lifecycle regressions grounded in the 10v10 audit."""
import math
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.coordinated_engage import CoordinatedEngageSkill
from skills.tactical_v1.skills.combat_reposition import CombatRepositionSkill
from skills.tactical_v1.skills.disengage_reenter import DisengageReenterSkill, ESCAPE
from skills.tactical_v1.skills.regroup import RegroupSkill
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import enemy, scene, with_questions


class CombatSemanticsTests(unittest.TestCase):
    def context(self, state, memory=None):
        context = Context(build_snapshot(with_questions(state), state["game_loop"]))
        if memory is not None:
            memory.begin_step(context.snapshot)
            memory.observe(context.snapshot)
            memory.update_task_facts(context.snapshot, context)
        return context

    def proposals(self, skill, context, memory):
        return skill.propose(context, memory, Telemetry())

    def test_catalog_speed_matches_recorded_eight_loop_travel(self):
        # Native speeds and observed movement from the real 6z4s trace:
        # zealot step2/u2 -> 1.125; stalker step1/u9 -> 1.477 (rounded log).
        for name, unit_id, speed, expected in (
                ("zealot", 73, 2.25, 1.125),
                ("stalker", 74, 2.953125, 1.4765625)):
            with self.subTest(unit=name):
                state = scene([((0., 0.), name)], distance=2.)
                state["unit_type_profiles"][str(unit_id)]["movement_speed_native"] = speed
                context = self.context(state)
                self.assertAlmostEqual(context.predict_position("u0", 4).position[0], expected)
                # Longer windows stop at the command destination.
                self.assertEqual(context.predict_position("u0", 4, horizon=64).position,
                                 (2., 0.))

    def test_melee_slots_are_inside_real_contact_and_angularly_separated(self):
        state = scene([((0., 0.), "zealot")],
                      enemies={"e0": enemy((8., 0.), "stalker")})
        state["relations"]["u0"]["e0"]["native_weapon_range"] = .10009765625
        state["visible_enemies"]["e0"]["radius"] = .625
        context = self.context(state)
        skill = CoordinatedEngageSkill({})
        positions = {}
        for index in range(12):
            position, slot = skill._slot(context, "u0", "e0", index)
            self.assertIsNotNone(position)
            gap = math.dist(position, (8., 0.))
            self.assertGreaterEqual(gap, 1.125)
            self.assertLess(gap, 1.22509765625)
            positions[slot] = position
        self.assertGreaterEqual(len(positions), 3)
        for i, a in enumerate(positions.values()):
            for b in list(positions.values())[i + 1:]:
                # Native body diameter, rather than the former fixed 1.4
                # spacing, determines usable melee frontage.
                self.assertGreaterEqual(math.dist(a, b), 2 * context.snapshot.allies["u0"].radius - 1e-8)

    def test_no_progress_uses_latest_sample_at_window_boundary(self):
        state = scene([((5., 0.), "zealot")],
                      enemies={"e0": enemy((10., 0.))}, loop=112, attack_range=2.)
        context = self.context(state)
        memory = TacticalMemory()
        memory.move_history["u0"] = [(80, (0., 0.)), (88, (5., 0.)),
                                     (96, (5., 0.)), (104, (5., 0.)), (112, (5., 0.))]
        instance = memory.create_instance("coordinated_engage", ["u0"], "e0",
                                          "approaching", 80, 96, goal_kind="enemy")
        instance.extra["approach_history"] = [
            (loop, {"u0": position}) for loop, position in memory.move_history["u0"][:-1]]
        instance.extra['approach_target_history'] = [
            (loop, 'e0', None, (10., 0.)) for loop, _ in memory.move_history['u0'][:-1]]
        self.assertTrue(CoordinatedEngageSkill({})._blocked(context, memory, "u0", "e0"))

    def test_g3_retarget_updates_task_and_failure_key_without_renewing_budget(self):
        memory = TacticalMemory()
        state = scene([((0., 0.), "marine")],
                      enemies={"e0": enemy((4., 0.)), "e1": enemy((2., 0.))}, loop=80)
        state["allies"]["u0"]["weapon_cooldown_loops"] = 20.
        context = self.context(state, memory)
        initial = memory.create_instance("combat_reposition", ["u0"], "e0",
                                         "repositioning", 80, 64, goal_kind="enemy")
        state["game_loop"] = 88
        context = self.context(state, memory)
        plan = self.proposals(CombatRepositionSkill({}), context, memory)[0]
        self.assertEqual(plan.goal, "e1")
        self.assertEqual(plan.proposed_memory_delta["kind"], "retarget")
        memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
        current = memory.instances[initial.instance_id]
        self.assertEqual((current.start_loop, current.max_loops), (80, 64))
        self.assertEqual((current.goal, current.extra["target_id"]), ("e1", "e1"))
        memory.finish_task(current.instance_id, "timeout", context.snapshot)
        self.assertTrue(memory.retry_blocked(*memory.task_retry_key(
            "combat_reposition", ["u0"], "enemy", "e1"), 88))
        plans = self.proposals(CombatRepositionSkill({}), context, memory)
        self.assertTrue(plans)
        self.assertEqual({p.goal for p in plans}, {"e0"})

    def danger_state(self, loop=80, ready=False, severe=False):
        state = scene([((0., 0.), "marine")], distance=2., loop=loop,
                      enemies={"e0": enemy((5., 0.), "marine"),
                               "e1": enemy((5., .5), "marine")}, attack_range=6.)
        state["allies"]["u0"]["weapon_cooldown_loops"] = 0. if ready else 24.
        if severe:
            state["allies"]["u0"].update(health=5., health_lost_since_previous_decision=10.)
        return state

    def test_g4_does_not_cancel_healthy_engaged_volley_for_two_threats(self):
        memory = TacticalMemory()
        context = self.context(self.danger_state(ready=True), memory)
        skill = DisengageReenterSkill({})
        self.assertEqual(len(context.threats_to("u0")), 2)
        self.assertEqual(self.proposals(skill, context, memory), [])
        memory.create_instance(skill.name, ["u0"], "safe", ESCAPE, 72, 96)
        self.assertEqual(self.proposals(skill, context, memory), [])

    def test_g4_severe_danger_can_still_override_ready_weapon(self):
        memory = TacticalMemory()
        context = self.context(self.danger_state(ready=True, severe=True), memory)
        plans = self.proposals(DisengageReenterSkill({}), context, memory)
        self.assertTrue(plans)
        self.assertTrue(all(p.commitment.phase == ESCAPE for p in plans))

    def test_g4_escape_keeps_phase_deadline_then_fails_into_cooldown(self):
        memory = TacticalMemory()
        skill = DisengageReenterSkill({})
        for loop in (80, 88, 96):
            state = self.danger_state(loop)
            # A bounded low-reserve escape, not the former "two enemies"
            # trigger on an untouched healthy unit.
            state['allies']['u0'].update(health=20., health_lost_since_previous_decision=3.)
            context = self.context(state, memory)
            plan = self.proposals(skill, context, memory)[0]
            memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
            instance = memory.instances[plan.instance_id]
            self.assertEqual(instance.phase_start_loop, 80)
            self.assertEqual(instance.extra["phase_max_loops"], 24)
            self.assertEqual(instance.start_loop, 80)
        state = self.danger_state(104)
        state['allies']['u0'].update(health=20., health_lost_since_previous_decision=3.)
        context = self.context(state, memory)
        self.assertFalse(memory.instances)
        self.assertTrue(memory.retry_records)
        self.assertEqual(self.proposals(skill, context, memory), [])
        self.assertIsNone(skill._phase(context, "u0", instance))

    def test_g4_does_not_move_when_no_escape_reduces_danger(self):
        state = scene([((0., 0.), "marine")],
                      enemies={"e0": enemy((1., 0.), "marine"),
                               "e1": enemy((-1., 0.), "marine")})
        state["allies"]["u0"]["weapon_cooldown_loops"] = 24.
        context = self.context(state)
        self.assertEqual(self.proposals(DisengageReenterSkill({}), context,
                                        TacticalMemory()), [])

    def test_support_groups_do_not_hide_long_transitive_chain(self):
        state = scene([((float(x), 0.), "marine") for x in (0, 7, 14, 21)])
        context = self.context(state)
        self.assertEqual(len(context.engagement_groups(8.)), 1)
        self.assertEqual(len(context.support_groups(8.)), 2)
        for group in context.support_groups(8.):
            for a in group:
                for b in group:
                    self.assertLessEqual(math.dist(context.snapshot.allies[a].position,
                                                   context.snapshot.allies[b].position), 8.)

    def test_g5_isolated_fighter_gets_support_repair_but_supported_front_does_not(self):
        state = scene([((0., 0.), "marine"), ((12., 0.), "marine"),
                       ((13., 0.), "marine")],
                      enemies={"e0": enemy((3., 0.), "marine"),
                               "e1": enemy((15., 0.), "marine")})
        context = self.context(state)
        plans = self.proposals(RegroupSkill({}), context, TacticalMemory())
        self.assertTrue(plans)
        self.assertEqual({p.controlled_actors for p in plans}, {("u0",)})
        self.assertTrue(all(p.current_actions["u0"] == 4 for p in plans))
        self.assertTrue(all(p.goal_data["group_b"] == ["u1", "u2"] for p in plans))


if __name__ == "__main__":
    unittest.main()
