"""Observed encounter boundaries and the real Adaptive rally/commit path."""
import copy
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.encounters import EncounterTracker
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skill_manager import SkillManager
from skills.tactical_v1.skills.precombat_rally import PreCombatRallySkill
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import enemy, scene, with_questions


class PreCombatRallyTests(unittest.TestCase):
    def state(self, xs=(0., 8., 16.), enemy_x=30., loop=80, step_mul=8):
        return scene([((x, 0.), "stalker") for x in xs],
                     enemies={} if enemy_x is None else {"e0": enemy((enemy_x, 0.))},
                     loop=loop, step_mul=step_mul, map_name="10gen_protoss")

    def context(self, **kwargs):
        state = self.state(**kwargs)
        return Context(build_snapshot(with_questions(state), state["game_loop"]))

    def observe(self, memory, context):
        memory.begin_step(context.snapshot)
        memory.observe(context.snapshot)
        return memory.update_task_facts(context.snapshot, context)

    def start(self, config=None):
        config = config or {}
        memory = TacticalMemory(encounter_config=config)
        skill = PreCombatRallySkill(config)
        context = self.context()
        self.observe(memory, context)
        plan = skill.propose(context, memory, Telemetry())[0]
        memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
        return memory, skill, plan.instance_id

    def test_scattered_group_has_atomic_stable_anchor_and_shared_goal(self):
        memory = TacticalMemory()
        context = self.context()
        self.observe(memory, context)
        plan = PreCombatRallySkill({}).propose(context, memory, Telemetry())[0]
        self.assertTrue(plan.atomic)
        self.assertEqual(plan.controlled_actors, ("u0", "u1", "u2"))
        self.assertEqual(plan.current_actions, {"u0": 4, "u1": 1, "u2": 5})
        self.assertEqual(plan.goal_data["anchor"], [8., 0.])
        self.assertEqual(plan.goal_data["objective"], {"kind": "rally"})
        self.assertEqual(plan.proposed_memory_delta["kind"], "create")

    def test_compact_unseen_distant_and_already_fighting_groups_skip(self):
        cases = [dict(xs=(0., 1., 2.), enemy_x=15.),
                 dict(enemy_x=None), dict(enemy_x=60.), dict(enemy_x=18.)]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                memory = TacticalMemory()
                context = self.context(**kwargs)
                self.observe(memory, context)
                self.assertEqual(PreCombatRallySkill({}).propose(
                    context, memory, Telemetry()), [])

    def test_recent_damage_aborts_preparation(self):
        state = self.state()
        state["allies"]["u0"]["health_lost_since_previous_decision"] = 1.
        memory = TacticalMemory()
        context = Context(build_snapshot(with_questions(state), 1))
        self.observe(memory, context)
        self.assertEqual(PreCombatRallySkill({}).propose(context, memory, Telemetry()), [])

    def test_visible_suicide_effect_aborts_preparation_without_ready_attack(self):
        state = scene([((x, 0.), "stalker") for x in (0., 8., 16.)],
                      enemies={"e0": enemy((18.5, 0.), name="baneling")},
                      attack_range=1.)
        # Native Banelings have an effect attack, not the fixture's synthetic
        # two-range gun. Use the catalog contract for the calibrated blast.
        state["unit_type_profiles"]["9"] = {
            "known": True, "movement_speed_native": 2.5, "armor": 0.,
            "attributes": [3], "weapons": []}
        context = Context(build_snapshot(with_questions(state), 1))
        memory = TacticalMemory()
        self.observe(memory, context)
        self.assertFalse(context.can_issue_attack("u2", "e0"))
        self.assertEqual(context.risk_at("u2", (16., 0.)).effect_weapon_near, ("e0",))
        self.assertEqual(PreCombatRallySkill({}).propose(context, memory, Telemetry()), [])

    def test_remote_straggler_is_not_a_member_of_frozen_majority(self):
        memory = TacticalMemory()
        context = self.context(xs=(0., 8., 16., 60.))
        self.observe(memory, context)
        plan = PreCombatRallySkill({}).propose(context, memory, Telemetry())[0]
        self.assertEqual(plan.controlled_actors, ("u0", "u1", "u2"))
        self.assertEqual(plan.goal_data["anchor"], [8., 0.])

    def test_proposals_are_read_only_for_new_and_live_tasks(self):
        memory = TacticalMemory()
        context = self.context()
        skill = PreCombatRallySkill({})
        self.observe(memory, context)
        for active in (False, True):
            before = copy.deepcopy(memory.__dict__)
            plans = skill.propose(context, memory, Telemetry())
            skill.propose(context, memory, Telemetry())
            self.assertEqual(memory.__dict__, before)
            if not active:
                memory.commit(context.snapshot, plans[0].current_actions, plans,
                              context=context)

    def test_continuation_does_not_restart_budget_and_stall_closes_encounter(self):
        memory, skill, instance_id = self.start()
        for count, loop in enumerate((88, 96), 2):
            context = self.context(loop=loop)
            self.observe(memory, context)
            plan = skill.propose(context, memory, Telemetry())[0]
            self.assertEqual(plan.instance_id, instance_id)
            self.assertEqual(plan.proposed_memory_delta,
                             {"kind": "continue", "instance_id": instance_id})
            memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
            instance = memory.instances[instance_id]
            self.assertEqual(instance.start_loop, 80)
            self.assertEqual(instance.max_loops, 64)
            self.assertEqual(instance.executed_steps, count)
        self.observe(memory, self.context(loop=104))
        self.assertNotIn(instance_id, memory.instances)
        self.assertEqual(memory.encounters.closed_ids, {1: "no_progress"})
        # A failed approach never refreshes its budget merely by waiting.
        context = self.context(loop=800)
        self.observe(memory, context)
        self.assertEqual(memory.encounters.current_id, 1)
        self.assertEqual(skill.propose(context, memory, Telemetry()), [])

    def test_in_place_shuffling_is_not_assembly_progress(self):
        memory, _, instance_id = self.start()
        # Every unit moves, but the frozen group's total readiness gap does not improve.
        self.observe(memory, self.context(xs=(-1., 9., 15.), loop=104))
        self.assertNotIn(instance_id, memory.instances)
        self.assertEqual(memory.encounters.closed_ids[1], "no_progress")

    def test_anchor_and_members_survive_a_moving_anchor_unit(self):
        memory, skill, instance_id = self.start()
        context = self.context(xs=(1., 9., 15., 9.), loop=88)
        self.observe(memory, context)
        plan = skill.propose(context, memory, Telemetry())[0]
        self.assertEqual(plan.instance_id, instance_id)
        self.assertEqual(plan.goal_data["anchor"], [8., 0.])
        self.assertEqual(plan.controlled_actors, ("u0", "u1", "u2"))

    def test_readiness_on_deadline_completes_without_failure_retry(self):
        memory, skill, instance_id = self.start({"precombat_rally": {"max_h": 1.}})
        context = self.context(xs=(4., 8., 12.), loop=88)
        self.observe(memory, context)
        self.assertNotIn(instance_id, memory.instances)
        self.assertEqual(memory.encounters.closed_ids, {1: "rally_ready"})
        self.assertEqual(memory.retry_records, {})
        self.assertEqual(skill.propose(context, memory, Telemetry()), [])

    def test_total_budget_still_expires_when_group_moves_too_slowly(self):
        memory, skill, instance_id = self.start({"precombat_rally": {"max_h": 2.}})
        context = self.context(xs=(.5, 8., 15.5), loop=88)
        self.observe(memory, context)
        plan = skill.propose(context, memory, Telemetry())[0]
        memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
        self.observe(memory, self.context(xs=(1., 8., 15.), loop=96))
        self.assertNotIn(instance_id, memory.instances)
        self.assertEqual(memory.encounters.closed_ids[1], "timeout")

    def test_changed_final_action_cannot_partially_create_group_task(self):
        memory = TacticalMemory()
        context = self.context()
        self.observe(memory, context)
        plan = PreCombatRallySkill({}).propose(context, memory, Telemetry())[0]
        changed = dict(plan.current_actions, u0=1)
        memory.commit(context.snapshot, changed, [plan], context=context)
        self.assertEqual(memory.instances, {})
        self.assertEqual(memory.encounters.closed_ids, {})

    def test_contact_interrupts_live_rally_before_minimum_commitment(self):
        memory, skill, instance_id = self.start()
        context = self.context(enemy_x=18., loop=88)
        self.observe(memory, context)
        self.assertNotIn(instance_id, memory.instances)
        self.assertEqual(memory.encounters.closed_ids, {1: "engagement_started"})
        self.assertEqual(skill.propose(context, memory, Telemetry()), [])

    def test_member_loss_closes_the_encounter(self):
        memory, _, instance_id = self.start()
        state = self.state(loop=88)
        state["allies"]["u0"]["alive"] = False
        context = Context(build_snapshot(with_questions(state), 88))
        self.observe(memory, context)
        self.assertNotIn(instance_id, memory.instances)
        self.assertEqual(memory.encounters.closed_ids, {1: "member_lost"})

    def test_contact_without_previous_rally_then_visibility_flicker_does_not_arm(self):
        tracker = EncounterTracker()
        tracker.observe(self.context(enemy_x=18.))
        tracker.observe(self.context(enemy_x=None, loop=88))
        self.assertFalse(tracker.can_rally())
        tracker.observe(self.context(enemy_x=18., loop=96))
        tracker.observe(self.context(enemy_x=None, loop=104))
        tracker.observe(self.context(enemy_x=30., loop=128))
        self.assertEqual(tracker.current_id, 1)
        self.assertFalse(tracker.can_rally())
        tracker.observe(self.context(enemy_x=30., loop=136))
        self.assertEqual(tracker.current_id, 2)
        self.assertTrue(tracker.can_rally())

    def test_second_encounter_rearms_after_real_disengagement_only(self):
        memory, skill, first_id = self.start()
        self.observe(memory, self.context(enemy_x=18., loop=88))
        self.assertNotIn(first_id, memory.instances)
        for loop in (96, 104, 112, 120):
            context = self.context(loop=loop)
            self.observe(memory, context)
            self.assertEqual(skill.propose(context, memory, Telemetry()), [])
        context = self.context(loop=128)
        self.observe(memory, context)
        plan = skill.propose(context, memory, Telemetry())[0]
        self.assertEqual(plan.goal_data["encounter_id"], 2)
        memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
        self.assertNotEqual(plan.instance_id, first_id)
        self.assertEqual(memory.instances[plan.instance_id].start_loop, 128)

    def test_rearm_budget_is_frozen_across_horizon_change(self):
        tracker = EncounterTracker()
        tracker.observe(self.context(enemy_x=18.))
        tracker.observe(self.context(loop=88, step_mul=8))
        tracker.observe(self.context(loop=104, step_mul=1))
        self.assertEqual(tracker.current_id, 1)
        tracker.observe(self.context(loop=120, step_mul=1))
        self.assertEqual(tracker.current_id, 2)
        before = copy.deepcopy(tracker.__dict__)
        tracker.observe(self.context(loop=120, step_mul=1))
        self.assertEqual(tracker.__dict__, before)

    def test_real_manager_selects_atomic_rally_then_combat_takes_over(self):
        manager = SkillManager()
        actions, summary = manager.step(with_questions(self.state()), 3)
        self.assertEqual(actions, [4, 1, 5])
        self.assertEqual({row["skill"] for row in summary["units"].values()},
                         {"precombat_rally"})
        manager.commit_final_actions(actions)
        before = copy.deepcopy(manager.memory.__dict__)
        actions, _ = manager.step(with_questions(self.state()), 3)
        manager.commit_final_actions(actions)
        self.assertEqual(manager.memory.__dict__, before)
        actions, summary = manager.step(with_questions(self.state(enemy_x=18., loop=88)), 3)
        self.assertNotIn("precombat_rally", {row["skill"]
                                            for row in summary["units"].values()})
        self.assertEqual(actions[2], 6)  # ready frontline fires instead of holding
        manager.commit_final_actions(actions)
        self.assertFalse(any(i.skill == "precombat_rally"
                             for i in manager.memory.instances.values()))


if __name__ == "__main__":
    unittest.main()
