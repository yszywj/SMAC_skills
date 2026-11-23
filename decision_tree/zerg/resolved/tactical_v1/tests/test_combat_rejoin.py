"""In-combat group duties through real proposal, observation and commit."""
import copy
import json
import unittest
from pathlib import Path

from skills.tactical_v1.combat_groups import group_status, support_readiness
from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skill_manager import SkillManager
from skills.tactical_v1.skills.combat_rejoin import CombatRejoinSkill
from skills.tactical_v1.telemetry import Telemetry
from skills.tactical_v1.validator import validate_actions, validate_plans
from .fixtures import enemy, scene, with_questions


def legacy_config(config=None):
    """Keep the r2 whole-group lifecycle as an explicit ablation regression."""
    result = dict(config or {})
    result['common_combat'] = dict(result.get('common_combat', {}),
        shared_front_enabled=False, local_cover_enabled=False)
    return result


class CombatRejoinTests(unittest.TestCase):
    def state(self, xs=(0., 1., 14.), enemy_x=18., loop=80, step_mul=8):
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
        memory = TacticalMemory()
        skill = CombatRejoinSkill(legacy_config(config))
        context = self.context()
        self.observe(memory, context)
        plan = skill.propose(context, memory, Telemetry())[0]
        memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
        return memory, skill, plan.instance_id

    def test_front_keeps_firing_and_rear_joins_in_one_atomic_plan(self):
        context = self.context()
        plan = CombatRejoinSkill(legacy_config()).propose(context, TacticalMemory(), Telemetry())[0]
        self.assertTrue(plan.atomic)
        self.assertEqual(plan.current_actions, {"u0": 4, "u1": 4, "u2": 6})
        self.assertEqual(plan.goal_data["roles"], {"u0": "join", "u1": "join", "u2": "cover"})
        self.assertEqual(plan.goal_data["objective"]["kind"], "combat_rejoin")
        manager = SkillManager()
        self.assertEqual(validate_plans([plan], context, plan.current_actions,
                                        manager.evaluator.effects(context, plan.current_actions)), [])

    def test_precontact_compact_and_no_visible_enemy_skip(self):
        for kwargs in (dict(enemy_x=30.), dict(xs=(13., 14., 15.)), dict(enemy_x=None)):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(CombatRejoinSkill(legacy_config()).propose(
                    self.context(**kwargs), TacticalMemory(), Telemetry()), [])

    def test_global_rally_veto_does_not_block_combat_joining(self):
        context = self.context()
        memory = TacticalMemory()
        self.observe(memory, context)
        self.assertFalse(memory.encounters.can_rally())
        self.assertTrue(CombatRejoinSkill(legacy_config()).propose(context, memory, Telemetry()))

    def test_wounded_cooling_pair_can_escape_to_stronger_group(self):
        state = self.state(xs=(0., 1., 12., 13., 14.), enemy_x=-2.)
        for actor in ("u0", "u1"):
            state["allies"][actor]["weapon_cooldown_loops"] = 40.
            state["allies"][actor]["weapon_ready"] = False
        context = Context(build_snapshot(with_questions(state), 1))
        plans = CombatRejoinSkill(legacy_config()).propose(context, TacticalMemory(), Telemetry())
        escape = next(p for p in plans if p.goal_data["roles"].get("u0") == "escape")
        self.assertEqual(escape.goal_data["joiners"], ["u0", "u1"])
        self.assertEqual(escape.current_actions["u0"], 4)
        self.assertEqual(escape.current_actions["u1"], 4)
        self.assertEqual(group_status(context, ("u0", "u1"))["ready_shooters"], 0)

    def test_healers_and_suicide_units_are_excluded(self):
        state = scene([((0., 0.), "stalker"), ((1., 0.), "stalker"),
                       ((14., 0.), "stalker"), ((13., 1.), "medivac"),
                       ((13., 2.), "baneling")], enemies={"e0": enemy((18., 0.))})
        context = Context(build_snapshot(with_questions(state), 1))
        plans = CombatRejoinSkill(legacy_config()).propose(context, TacticalMemory(), Telemetry())
        self.assertTrue(plans)
        self.assertTrue(all("u3" not in p.controlled_actors and "u4" not in p.controlled_actors
                            for p in plans))

    def test_member_cap_preserves_whole_duties(self):
        skill = CombatRejoinSkill(legacy_config({"combat_rejoin": {"max_members": 2}}))
        self.assertEqual(skill.propose(self.context(), TacticalMemory(), Telemetry()), [])

    def test_new_and_live_proposals_are_read_only(self):
        memory, skill, _ = self.start()
        for current in (TacticalMemory(), memory):
            before = copy.deepcopy(current.__dict__)
            self.assertTrue(skill.propose(self.context(), current, Telemetry()))
            skill.propose(self.context(), current, Telemetry())
            self.assertEqual(current.__dict__, before)

    def test_continuation_freezes_members_anchor_roles_and_budgets(self):
        memory, skill, instance_id = self.start()
        instance = memory.instances[instance_id]
        frozen = {k: copy.deepcopy(instance.extra[k]) for k in (
            "joiners", "supporters", "anchor", "roles", "no_progress_loops", "min_loops")}
        context = self.context(xs=(1., 2., 15., 3.), loop=88, step_mul=16)
        self.observe(memory, context)
        plan = skill.propose(context, memory, Telemetry())[0]
        self.assertEqual(plan.instance_id, instance_id)
        self.assertEqual(plan.proposed_memory_delta, {"kind": "continue", "instance_id": instance_id})
        memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
        self.assertEqual(instance.start_loop, 80)
        self.assertEqual(instance.phase_start_loop, 80)
        self.assertEqual(instance.max_loops, 64)
        self.assertEqual(instance.executed_steps, 2)
        self.assertEqual({k: instance.extra[k] for k in frozen}, frozen)
        self.assertNotIn("u3", instance.actors)

    def test_blocked_group_exits_and_same_duty_cannot_reset_stall_budget(self):
        memory, skill, instance_id = self.start()
        for loop in (88, 96):
            context = self.context(loop=loop)
            self.observe(memory, context)
            plan = skill.propose(context, memory, Telemetry())[0]
            memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
            self.assertEqual(plan.instance_id, instance_id)
        events = self.observe(memory, self.context(loop=104))
        self.assertNotIn(instance_id, memory.instances)
        self.assertTrue(any("no_progress" in event for event in events))
        self.assertEqual(skill.propose(self.context(loop=104), memory, Telemetry()), [])

    def test_connected_on_total_deadline_completes_without_retry(self):
        memory, _, instance_id = self.start({"combat_rejoin": {"max_h": 1.}})
        # Actual weapon/mask readiness is required as well as the support ring.
        context = self.context(xs=(14., 14., 14.), loop=88)
        events = self.observe(memory, context)
        self.assertNotIn(instance_id, memory.instances)
        self.assertIn(f"support_connected:{instance_id}", events)
        self.assertEqual(memory.retry_records, {})

    def test_potential_contact_on_deadline_is_not_completed_support(self):
        memory, _, instance_id = self.start({"combat_rejoin": {"max_h": 1.}})
        events = self.observe(memory, self.context(xs=(12., 13., 14.), loop=88))
        self.assertNotIn(f"support_connected:{instance_id}", events)
        self.assertIn(f"task_timeout:{instance_id}", events)

    def test_observed_movement_does_not_remove_total_deadline(self):
        memory, skill, instance_id = self.start({"combat_rejoin": {"max_h": 2.}})
        context = self.context(xs=(1., 2., 14.), loop=88)
        self.observe(memory, context)
        plan = skill.propose(context, memory, Telemetry())[0]
        memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
        events = self.observe(memory, self.context(xs=(2., 3., 14.), loop=96))
        self.assertNotIn(instance_id, memory.instances)
        self.assertTrue(any("timeout" in event for event in events))

    def test_command_endpoint_cannot_complete_without_observed_connection(self):
        memory, _, instance_id = self.start()
        self.assertEqual(support_readiness(self.context(), ["u0", "u1"], ["u2"], 6.)["fraction"], 0.)
        self.observe(memory, self.context(loop=88))
        self.assertIn(instance_id, memory.instances)

    def test_member_loss_ends_the_whole_task(self):
        memory, _, instance_id = self.start()
        state = self.state(loop=88)
        state["allies"]["u1"]["alive"] = False
        self.observe(memory, Context(build_snapshot(with_questions(state), 88)))
        self.assertNotIn(instance_id, memory.instances)

    def test_cover_emergency_interrupts_before_minimum_commitment(self):
        memory, _, instance_id = self.start()
        state = self.state(enemy_x=16., loop=88)
        state["allies"]["u2"]["health"] = 8.
        state["allies"]["u2"]["health_lost_since_previous_decision"] = 10.
        events = self.observe(memory, Context(build_snapshot(with_questions(state), 88)))
        self.assertNotIn(instance_id, memory.instances)
        self.assertIn(f"cover_emergency:{instance_id}", events)
        self.assertTrue(memory.retry_records)

    def test_external_override_cannot_create_or_partially_continue_group(self):
        for active in (False, True):
            with self.subTest(active=active):
                if active:
                    memory, skill, instance_id = self.start()
                    context = self.context(loop=88)
                else:
                    memory, skill, instance_id = TacticalMemory(), CombatRejoinSkill(legacy_config()), None
                    context = self.context()
                self.observe(memory, context)
                plan = skill.propose(context, memory, Telemetry())[0]
                memory.commit(context.snapshot, dict(plan.current_actions, u0=1), [plan], context=context)
                self.assertEqual(memory.instances, {})

    def test_one_blocked_member_cannot_commit_other_members_alone(self):
        payload = with_questions(self.state())
        for action in ("2", "3", "4", "5"):
            payload["questions"]["u0"]["criteria"].pop(action)
        context = Context(build_snapshot(payload, 1))
        self.assertEqual(CombatRejoinSkill(legacy_config()).propose(context, TacticalMemory(), Telemetry()), [])

    def test_temporary_rear_block_keeps_live_group_identity_and_budget(self):
        memory, skill, instance_id = self.start()
        payload = with_questions(self.state(loop=88))
        for action in ("2", "3", "4", "5"):
            payload["questions"]["u0"]["criteria"].pop(action)
        context = Context(build_snapshot(payload, 88))
        self.observe(memory, context)
        plan = skill.propose(context, memory, Telemetry())[0]
        self.assertEqual(plan.current_actions, {"u0": 1, "u1": 4, "u2": 6})
        self.assertEqual(plan.instance_id, instance_id)
        memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
        self.assertEqual(memory.instances[instance_id].start_loop, 80)
        self.assertEqual(memory.instances[instance_id].max_loops, 64)

    def test_real_manager_selects_and_commits_one_joint_duty(self):
        manager = SkillManager()
        # This checks the original single-task lifecycle, not which labels
        # win under the new action-based joint objective.
        manager.config['common_combat']['joint_assessment_enabled'] = False
        manager.evaluator.common_combat['joint_assessment_enabled'] = False
        manager.memory.combat_progress.joint_enabled = False
        manager.skills['combat_rejoin'] = CombatRejoinSkill(legacy_config(manager.config))
        actions, summary = manager.step(with_questions(self.state()), 3)
        self.assertEqual(actions, [4, 4, 6])
        self.assertEqual({u["skill"] for u in summary["units"].values()}, {"combat_rejoin"})
        self.assertEqual(len({u["plan"] for u in summary["units"].values()}), 1)
        manager.commit_final_actions(actions)
        self.assertEqual(len(manager.memory.instances), 1)
        instance = next(iter(manager.memory.instances.values()))
        self.assertEqual(instance.actors, ("u0", "u1", "u2"))

    def test_recorded_scattered_contact_seeds_176_and_183_offer_joint_cover(self):
        path = Path(__file__).resolve().parents[3] / (
            "runs/tactical_protoss_10v10_3x20_postfix_20261113/2z6s2c/events.jsonl")
        if not path.exists():
            self.skipTest("local recorded-run snapshots are not distributed with source")
        snapshots = {}
        with path.open(encoding="utf8") as source:
            for line in source:
                event = json.loads(line)
                if event.get("episode") in (7, 14) and event.get("step") == 0 \
                        and event.get("event") == "skill_observation":
                    snapshots[event["episode"]] = event["payload"]
                    if len(snapshots) == 2:
                        break
        self.assertEqual(set(snapshots), {7, 14})
        for episode, payload in snapshots.items():
            with self.subTest(seed=169 + episode):
                manager = SkillManager()
                context = Context(build_snapshot(payload, 1))
                plans = manager.skills["combat_rejoin"].propose(context, manager.memory, Telemetry())
                self.assertTrue(plans)
                self.assertTrue(all(p.atomic and p.goal_data["joiners"] and p.goal_data["supporters"] for p in plans))
                actions, summary = manager.step(payload, 10)
                self.assertEqual(validate_actions(actions, context, 10), [])
                # The old oversized link reward forced a whole-team duty.
                # A valid competing attack/approach may now beat it; existing
                # front fire and atomicity must survive either selection.
                front = 'u1' if episode == 7 else 'u9'
                self.assertEqual(context.decode(front, actions[int(front[1:])]).kind, 'attack')
                self.assertTrue(any(a in (2, 3, 4, 5) for a in actions))
                manager.commit_final_actions(actions)
                for task in manager.memory.instances.values():
                    if task.skill == 'combat_rejoin':
                        self.assertGreaterEqual(len(task.actors), 2)
                        self.assertTrue(all(summary['units'][a]['skill'] == 'combat_rejoin'
                                            for a in task.actors))


if __name__ == "__main__":
    unittest.main()
