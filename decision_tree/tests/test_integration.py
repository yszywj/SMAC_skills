"""Integration against actual legacy scheduling and actual skill executors."""
import copy
import unittest

from decision_tree.integration import SkillTreeController
from decision_tree.schema import validate_policy
from sc2_skill_adapter.bridge.policy import nearest_actions
from sc2_skill_adapter.bridge.skill_runtime import load_manager
from skills.current.tests.test_group_skills import scene, enemy_scene, advance
from decision_tree.tests.test_invocation import combat_scene


def policy(mode="base", **extra):
    return {"schema_version": 1, "kind": "skill_return_policy",
            "supported_races": ["protoss"],
            "gate": {"phases": ["engage"], "min_game_loop": 0},
            "candidates": [{"id": "priority", "skill": "adaptive_focus_fire",
                            "priority_target": "e1", "max_steps": 2}],
            "selection": {"mode": mode, **({"candidate_id": "priority"} if mode == "candidate" else {})},
            **extra}


def observed(state, race="protoss"):
    """Complete real allocator fixture with the adapter's legal-choice format."""
    result = {"state": copy.deepcopy(state), "questions": {}}
    for actor, unit in result["state"]["allies"].items():
        unit["type"] = "Stalker" if race == "protoss" else "Marine"
        criteria = {"1": "Stop"}
        for action, move in result["state"]["movement_candidates"][actor].items():
            criteria[action] = "Move"
            move.setdefault("distances_to_visible_enemies", {
                target: abs(move["destination"][0] - enemy["position"][0])
                + abs(move["destination"][1] - enemy["position"][1])
                for target, enemy in result["state"]["visible_enemies"].items()})
            move.setdefault("distance_to_search_location", 0.0)
        for target, relation in result["state"]["relations"].get(actor, {}).items():
            if relation.get("smac_attack_available"):
                criteria[str(6 + int(target[1:]))] = "Attack"
        result["questions"][actor] = {"criteria": criteria}
    return result


def legacy_step(manager, request, count, posture="balanced"):
    """The runner's existing no-tree path, including spatial same-frame handoff."""
    spatial = manager.spatial_dispatch(request) if manager.spatial_params["auto_spatial_dispatch"] else None
    if not manager.active:
        manager.select(spatial or "adaptive_focus_fire", posture=posture)
    actions, log = manager.step(request, count)
    if isinstance(actions, dict):
        if log.get("status") == "interrupted" and not actions:
            manager.select("adaptive_focus_fire", posture=posture)
            return manager.step(request, count)
        merged = nearest_actions(request, count)
        for actor, action in actions.items():
            merged[actor] = action
        log = {**log, "base_policy": "nearest", "merged_units": sorted(actions)}
        actions = merged
    return actions, log


class SkillTreeIntegrationTests(unittest.TestCase):
    def manager(self, **kwargs):
        return load_manager()(**kwargs)

    def assert_shared_memory_equal(self, actual, expected):
        for name in ("unit_states", "target_assignments", "all_in", "stall_steps", "contacted",
                     "_last_move", "_last_enemy_health", "_last_seen_enemies", "_race_memory",
                     "_regroup_memory", "_search_memory", "_dispatch_state", "elapsed_steps",
                     "current_skill", "termination_reason", "mode_remaining", "stats"):
            self.assertEqual(getattr(actual, name), getattr(expected, name), name)

    def test_base_tree_matches_legacy_across_windows_and_memory(self):
        for enabled in (False, True):
            with self.subTest(engagement=enabled):
                options = {"combat_params": {"engagement_commitment": enabled}}
                old, new = self.manager(**options), self.manager(**options)
                tree = SkillTreeController(new, policy(), race="protoss", posture="defensive")
                state = combat_scene()
                opportunities = []
                for _ in range(14):
                    request = observed(state)
                    want, original_log = legacy_step(old, copy.deepcopy(request), 1, "defensive")
                    actual, log, events = tree.step(copy.deepcopy(request), 1, nearest_actions)
                    self.assertEqual(actual, want)
                    self.assertEqual({k: v for k, v in log.items() if not k.startswith("skill_tree_")}, original_log)
                    self.assert_shared_memory_equal(new, old)
                    opportunities.extend(e for e in events if e["opportunity"])
                    state["game_loop"] += 8
                self.assertEqual(len(opportunities), 1)  # BASE also consumes the shared gate.

    def test_spatial_base_parity_and_same_frame_contact_handoff(self):
        kwargs = {"spatial_params": {"enable_regroup": True, "enable_cohesive_search": True,
                                      "auto_spatial_dispatch": True}}
        old, new = self.manager(**kwargs), self.manager(**kwargs)
        tree = SkillTreeController(new, policy(), race="protoss")
        state = scene([(0., 0.), (20., 0.)])
        modes = []
        for step in range(9):
            if step == 3:
                positions = [u["position"] for u in state["allies"].values()]
                state = scene(positions, {"e0": enemy_scene((30., 0.))}, loop=state["game_loop"])
            request = observed(state)
            want, original_log = legacy_step(old, copy.deepcopy(request), 2)
            actual, log, events = tree.step(copy.deepcopy(request), 2, nearest_actions)
            self.assertEqual(actual, want)
            self.assertEqual({k: v for k, v in log.items() if not k.startswith("skill_tree_")}, original_log)
            old.commit_final_actions(want)
            new.commit_final_actions(actual)
            self.assert_shared_memory_equal(new, old)
            modes.append(new.current_skill)
            if step == 3:
                self.assertEqual(len(events), 1)
                self.assertFalse(events[0]["opportunity"])
            advance(state, {f"u{i}": action for i, action in enumerate(actual)})
        self.assertEqual(modes[0], "regroup")
        self.assertEqual(modes[3], "adaptive_focus_fire")

    def test_forced_candidate_changes_one_complete_call_only(self):
        manager = self.manager()
        tree = SkillTreeController(manager, policy("candidate"), race="protoss")
        state = combat_scene()
        targets, decisions = [], []
        for _ in range(9):
            actions, log, events = tree.step(observed(state), 1, nearest_actions)
            targets.append(log["attack_assignments"]["u0"])
            decisions.extend(events)
            state["game_loop"] += 8
        self.assertEqual(targets[:2], ["e1", "e1"])
        self.assertTrue(all(target == "e0" for target in targets[2:]))
        self.assertEqual(sum(e["selected_id"] != "BASE" for e in decisions), 1)
        self.assertEqual(tree.opportunities, 1)
        self.assertEqual(manager.max_steps, 4)

    def test_unsupported_race_does_not_apply_initial_or_candidate_tree(self):
        tree_policy = policy("candidate", initial_tree={"call": {
            "skill": "adaptive_focus_fire", "priority_target": "e1"}})
        old, new = self.manager(), self.manager()
        tree = SkillTreeController(new, tree_policy, race="terran")
        state = combat_scene()
        for _ in range(6):
            request = observed(state, "terran")
            expected, _ = legacy_step(old, copy.deepcopy(request), 1)
            actual, _, events = tree.step(request, 1, nearest_actions)
            self.assertEqual(actual, expected)
            self.assert_shared_memory_equal(new, old)
            self.assertTrue(all(e["reason"] == "unsupported_race" for e in events))
            state["game_loop"] += 8
        self.assertEqual(tree.opportunities, 0)

    def test_preview_changes_neither_live_manager_nor_observation(self):
        manager = self.manager(combat_params={"engagement_commitment": True})
        tree = SkillTreeController(manager, policy(), race="protoss")
        request = observed(combat_scene())
        before_manager, before_request = copy.deepcopy(manager.__dict__), copy.deepcopy(request)
        actions = tree._preview({"skill": "adaptive_focus_fire", "priority_target": "e1"},
                                request, 1, nearest_actions)
        self.assertEqual(actions["actions"], [7])
        self.assertEqual(manager.__dict__, before_manager)
        self.assertEqual(request, before_request)

    def test_identical_action_candidate_does_not_consume_first_action_gate(self):
        p = policy("candidate")
        p["candidates"][0]["priority_target"] = "e0"
        tree = SkillTreeController(self.manager(), p, race="protoss")
        _, _, events = tree.step(observed(combat_scene()), 1, nearest_actions)
        self.assertFalse(events[0]["opportunity"])
        self.assertEqual(events[0]["selected_id"], "BASE")
        self.assertEqual(tree.opportunities, 0)

    def test_call_config_gate_can_schedule_duration_only_experiment(self):
        p = policy("candidate")
        p["gate"]["effect_check"] = "call_config"
        p["candidates"][0]["priority_target"] = "e0"
        tree = SkillTreeController(self.manager(), p, race="protoss")
        _, _, events = tree.step(observed(combat_scene()), 1, nearest_actions)
        event = events[0]
        self.assertTrue(event["opportunity"])
        self.assertEqual(event["selected_id"], "priority")
        self.assertEqual(event["base_actions"], event["candidate_actions"]["priority"])
        self.assertEqual(tree.manager.max_steps, 2)
        # This is an eligible configuration experiment, not evidence of a
        # changed trajectory or improved return. Dataset auditing checks that.

    def test_model_positive_and_nonpositive_leaves_respect_same_gate(self):
        for gain, target in ((1.0, "e1"), (0.0, "e0"), (-1.0, "e0")):
            with self.subTest(gain=gain):
                p = policy()
                p["selection"] = {"mode": "model", "model": {
                    "schema_version": 1, "kind": "skill_gain_tree",
                    "candidate_ids": ["priority"], "root": {"gains": {"priority": gain}}}}
                tree = SkillTreeController(self.manager(), p, race="protoss")
                _, log, events = tree.step(observed(combat_scene()), 1, nearest_actions)
                self.assertEqual(log["attack_assignments"], {"u0": target})
                self.assertTrue(events[0]["opportunity"])
                self.assertEqual(tree.opportunities, 1)

    def test_initial_target_rule_is_bound_before_execution(self):
        # The rule must override the explicit old priority just as candidate binding does.
        p = policy(initial_tree={"call": {"skill": "adaptive_focus_fire",
                                          "priority_target": "e1", "target_rule": "lowest_effective_health"}})
        p["gate"]["phases"] = ["search"]
        tree = SkillTreeController(self.manager(), p, race="protoss")
        _, log, _ = tree.step(observed(combat_scene()), 1, nearest_actions)
        self.assertEqual(log["attack_assignments"], {"u0": "e0"})

    def test_initial_keep_rejected_before_game(self):
        with self.assertRaises(ValueError):
            validate_policy(policy(initial_tree={"call": {"operation": "keep"}}))

    def test_unavailable_positioning_candidate_falls_back_without_mutation(self):
        p = policy("candidate")
        p["candidates"][0]["movement_profile"] = {"spacing": "spread"}
        tree = SkillTreeController(self.manager(), p, race="protoss")
        _, log, events = tree.step(observed(combat_scene()), 1, nearest_actions)
        self.assertEqual(log["attack_assignments"], {"u0": "e0"})
        self.assertFalse(events[0]["opportunity"])
        self.assertEqual(tree.opportunities, 0)


if __name__ == "__main__":
    unittest.main()
