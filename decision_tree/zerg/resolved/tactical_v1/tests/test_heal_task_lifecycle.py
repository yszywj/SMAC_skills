"""S2 task lifetimes through real proposal, observation and commit paths."""
import copy
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.heal_support import HealSupportSkill
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import enemy, heal_candidates, scene, with_questions


class HealTaskLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.memory = TacticalMemory()
        self.skill = HealSupportSkill({})
        self.sequence = 0

    def state(self, phase, loop=80):
        state = scene([((0., 0.), "medivac"), ((9., 0.), "marine")],
                      loop=loop, enemies={"e0": enemy((2., 0.), "marine")}
                      if phase == "escape" else None)
        state["allies"]["u0"]["energy"] = 50.
        if phase in ("meet_patient", "sustained_heal"):
            state["allies"]["u1"]["health"] = 20.
        if phase == "sustained_heal":
            state["allies"]["u1"]["position"] = [2., 0.]
            heal_candidates(state, "u0")
        return state

    def frame(self, state):
        self.sequence += 1
        payload = with_questions(copy.deepcopy(state))
        for actor, keep in state.get("keep_heal_order_candidates", {}).items():
            payload["questions"][actor]["criteria"][str(keep["action_id"])] = "keep"
        snapshot = build_snapshot(payload, self.sequence)
        context = Context(snapshot)
        self.memory.begin_step(snapshot)
        risk = {a: len(context.risk_at(a, unit.position).threats)
                for a, unit in snapshot.allies.items()}
        self.memory.observe(snapshot, risk)
        self.memory.update_task_facts(snapshot, context)
        return context, self.skill.propose(context, self.memory, Telemetry())

    def choose(self, plans, phase, action=None):
        return next(p for p in plans if p.commitment.phase == phase
                    and (phase != "meet_patient" or p.atomic)
                    and (action is None or p.current_actions["u0"] == action))

    def commit(self, context, plan):
        actions = {a: 1 for a in context.snapshot.allies}
        actions.update(plan.current_actions)
        self.memory.commit(context.snapshot, actions, [plan], context=context)
        return self.memory.instance_by_id(plan.instance_id)

    def test_movement_duties_create_and_continue_one_bounded_task(self):
        for phase, actors, max_loops in (("meet_patient", ("u0", "u1"), 48),
                                         ("escort", ("u0",), 48),
                                         ("escape", ("u0",), 24)):
            with self.subTest(phase=phase):
                self.setUp()
                for step in range(3):
                    context, plans = self.frame(self.state(phase, 80 + 8 * step))
                    plan = self.choose(plans, phase)
                    self.assertEqual(plan.proposed_memory_delta["kind"],
                                     "create" if step == 0 else "continue")
                    instance = self.commit(context, plan)
                    if step == 0:
                        identity = instance.instance_id
                    self.assertEqual(instance.instance_id, identity)
                    self.assertEqual(instance.actors, actors)
                    self.assertEqual(instance.start_loop, 80)
                    self.assertEqual(instance.phase_start_loop, 80)
                    self.assertEqual(instance.max_loops, max_loops)
                    self.assertEqual(instance.executed_steps, step + 1)
                    self.assertEqual(instance.budget_loops, 8 * step)
                    self.assertEqual(len(self.memory.instances), 1)

    def test_escort_waypoint_changes_and_hold_keep_identity_and_budget(self):
        context, plans = self.frame(self.state("escort"))
        instance = self.commit(context, self.choose(plans, "escort"))
        identity, retry_key = instance.instance_id, instance.extra["retry_key"]
        state = scene([((0., 0.), "medivac"), ((17., 0.), "marine")], loop=88)
        context, plans = self.frame(state)
        instance = self.commit(context, self.choose(plans, "escort"))
        self.assertEqual(instance.instance_id, identity)
        self.assertEqual(instance.extra["waypoint"], [17., 0.])
        self.assertEqual(instance.extra["retry_key"], retry_key)
        state = scene([((13., 0.), "medivac"), ((17., 0.), "marine")], loop=96)
        context, plans = self.frame(state)
        instance = self.commit(context, self.choose(plans, "escort", action=1))
        self.assertEqual(instance.instance_id, identity)
        self.assertEqual(instance.start_loop, 80)
        self.assertEqual(instance.executed_steps, 3)

    def test_escort_hold_creates_task_when_no_task_exists(self):
        state = scene([((0., 0.), "medivac"), ((3., 0.), "marine")])
        context, plans = self.frame(state)
        instance = self.commit(context, self.choose(plans, "escort", action=1))
        self.assertEqual(instance.phase, "escort")
        self.assertEqual(instance.goal_kind, "position")
        self.assertEqual(instance.extra["waypoint"], [3., 0.])

    def test_healing_escort_and_escape_replace_previous_duty(self):
        previous = None
        for step, phase in enumerate(("sustained_heal", "escort", "escape",
                                      "sustained_heal")):
            context, plans = self.frame(self.state(phase, 80 + 8 * step))
            plan = self.choose(plans, phase)
            instance = self.commit(context, plan)
            self.assertEqual(instance.phase, phase)
            self.assertEqual(instance.start_loop, 80 + 8 * step)
            self.assertEqual(len(self.memory.instances), 1)
            if previous:
                self.assertNotEqual(instance.instance_id, previous.instance_id)
                self.assertNotIn(previous.instance_id, self.memory.instances)
            previous = instance

    def test_keep_heal_creates_then_continues_the_same_healing_task(self):
        state = self.state("sustained_heal")
        state["action_contract"]["keep_heal_order"] = {
            "enabled": True, "action_id": 16}
        state["keep_heal_order_candidates"] = {
            "u0": {"action_id": 16, "target": "u1"}}
        context, plans = self.frame(state)
        instance = self.commit(context, self.choose(plans, "sustained_heal", 16))
        identity = instance.instance_id
        state["game_loop"] = 88
        context, plans = self.frame(state)
        # Both the observed-order and ordinary healing action share its duty.
        healing = [p for p in plans if p.commitment.phase == "sustained_heal"]
        self.assertEqual({p.instance_id for p in healing}, {identity})
        self.assertEqual({p.proposed_memory_delta["kind"] for p in healing},
                         {"continue"})
        instance = self.commit(context, self.choose(plans, "sustained_heal", 7))
        self.assertEqual(instance.instance_id, identity)
        self.assertEqual(instance.start_loop, 80)

    def test_meeting_completion_on_deadline_releases_patient_for_healing(self):
        context, plans = self.frame(self.state("meet_patient"))
        meeting = self.commit(context, self.choose(plans, "meet_patient"))
        context, plans = self.frame(self.state("sustained_heal", 128))
        self.assertNotIn(meeting.instance_id, self.memory.instances)
        self.assertFalse(self.memory.retry_records)
        healing = self.commit(context, self.choose(plans, "sustained_heal"))
        self.assertEqual(healing.actors, ("u0",))
        self.assertNotEqual(healing.instance_id, meeting.instance_id)

    def test_timeout_blocks_same_duty_but_allows_other_duties(self):
        for phase, deadline in (("meet_patient", 128), ("escort", 128),
                                ("escape", 104)):
            with self.subTest(phase=phase):
                self.setUp()
                context, plans = self.frame(self.state(phase))
                instance = self.commit(context, self.choose(plans, phase))
                state = self.state(phase, deadline)
                if phase == "escort":
                    # Moving into another waypoint cell cannot bypass cooldown.
                    state["allies"]["u1"]["position"] = [17., 0.]
                _, plans = self.frame(state)
                self.assertNotIn(instance.instance_id, self.memory.instances)
                self.assertTrue(self.memory.retry_records)
                self.assertFalse(any(p.commitment.phase == phase for p in plans))
                _, other_plans = self.frame(self.state("sustained_heal", deadline))
                self.assertTrue(any(p.commitment.phase == "sustained_heal"
                                    for p in other_plans),
                                "only the failed duty should be blocked")

    def test_escape_safe_on_deadline_finishes_without_retry_penalty(self):
        context, plans = self.frame(self.state("escape"))
        instance = self.commit(context, self.choose(plans, "escape"))
        self.frame(self.state("escort", 104))
        self.assertNotIn(instance.instance_id, self.memory.instances)
        self.assertFalse(self.memory.retry_records)

    def test_escort_in_band_on_deadline_finishes_without_failure_cooldown(self):
        context, plans = self.frame(self.state("escort"))
        instance = self.commit(context, self.choose(plans, "escort"))
        state = scene([((5., 0.), "medivac"), ((9., 0.), "marine")], loop=128)
        _, plans = self.frame(state)
        self.assertNotIn(instance.instance_id, self.memory.instances)
        self.assertFalse(self.memory.retry_records)
        self.assertEqual(self.choose(plans, "escort").proposed_memory_delta["kind"],
                         "create")

    def test_proposals_do_not_mutate_existing_tasks(self):
        context, plans = self.frame(self.state("meet_patient"))
        self.commit(context, self.choose(plans, "meet_patient"))
        context, _ = self.frame(self.state("meet_patient", 88))
        before = copy.deepcopy(self.memory.__dict__)
        first = self.skill.propose(context, self.memory, Telemetry())
        second = self.skill.propose(context, self.memory, Telemetry())
        self.assertEqual(first, second)
        self.assertEqual(self.memory.__dict__, before)

    def test_changed_final_atomic_action_does_not_create_meeting_task(self):
        context, plans = self.frame(self.state("meet_patient"))
        plan = self.choose(plans, "meet_patient")
        actions = dict(plan.current_actions)
        actions["u1"] = 1
        self.memory.commit(context.snapshot, actions, [plan], context=context)
        self.assertFalse(self.memory.instances)


if __name__ == "__main__":
    unittest.main()
