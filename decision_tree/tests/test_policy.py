"""Checks decision information boundaries, legal target binding and fail-closed selection."""

from copy import deepcopy
import math
from pathlib import Path
import unittest

from decision_tree.features import FEATURE_NAMES, extract_features, infer_races, phase_name
from decision_tree.policy import bind_call, candidate_calls, choose, initial_call, predict_gains
from decision_tree.schema import load_policy, policy_hash, validate_model

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "policy_v1.json"


def snapshot():
    return {"game_loop": 40, "allies": {
        "u0": {"type": "Stalker", "unit_type_id": 74, "alive": True, "health": 70, "max_health": 80,
               "shield": 20, "max_shield": 80, "position": [0, 0], "weapon_ready": True,
               "health_lost_since_previous_decision": 10},
        "u1": {"type": "Zealot", "unit_type_id": 73, "alive": True, "health": 100, "max_health": 100,
               "shield": 50, "max_shield": 50, "position": [1, 0], "weapon_ready": False}},
        "visible_enemies": {"e0": {"health": 5, "shield": 0, "alive": True},
                            "e1": {"health": 10, "shield": 20, "alive": True}},
        "relations": {"u0": {"e0": {"smac_attack_available": False}, "e1": {"smac_attack_available": True}},
                      "u1": {"e0": {"smac_attack_available": False}}},
        "unit_type_profiles": {"74": {"weapons": [{"target_type": "any", "range": 6}]},
                               "73": {"weapons": [{"target_type": "ground", "range": .1}]}}}


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy(CONFIG)
        self.state = snapshot()
        self.base = {"operation": "start", "skill": "adaptive_focus_fire", "posture": "balanced",
                     "max_steps": 4, "adaptive_params": {"max_retreat_steps": 2}}

    def test_schema_is_finite_public_and_pure(self):
        memory = {"contacted": False, "last_visible_loop": 10}
        before = deepcopy((self.state, memory))
        features = extract_features(self.state, memory)
        self.assertEqual(list(features), list(FEATURE_NAMES))
        self.assertTrue(all(math.isfinite(v) for v in features.values()))
        self.assertEqual((self.state, memory), before)
        self.assertEqual(features["melee_fraction"], .5)
        self.assertEqual(features["can_attack_air_fraction"], .5)
        self.assertEqual(features["can_attack_ground_fraction"], 1)
        self.state["final_return"] = 999
        self.state["hidden_enemies"] = {"e2": {"health": 1000}}
        self.assertEqual(extract_features(self.state, memory), features)

    def test_phase_separates_visibility_attack_range_and_history(self):
        self.assertEqual(phase_name(extract_features(self.state)), "engage")
        self.state["relations"] = {}
        self.assertEqual(phase_name(extract_features(self.state)), "approach")
        self.state["visible_enemies"] = {}
        self.assertEqual(phase_name(extract_features(self.state)), "search")
        features = extract_features(self.state, {"contacted": True, "last_visible_loop": 30})
        self.assertEqual(phase_name(features), "lost_contact")
        self.assertEqual(features["loops_since_visible"], 10)

    def test_target_binding_excludes_visible_but_illegal_low_health_enemy(self):
        before = deepcopy((self.policy, self.state, self.base))
        proposals = candidate_calls(self.policy, self.state, self.base)
        self.assertEqual(proposals[0]["call"]["priority_target"], "e1")
        self.assertNotIn("target_rule", proposals[0]["call"])
        self.assertEqual(proposals[1]["call"]["adaptive_params"]["max_retreat_steps"], 3)
        self.assertEqual((self.policy, self.state, self.base), before)

    def test_unknown_race_no_gate_and_keep_are_not_opportunities(self):
        self.state["allies"]["u0"]["type"] = "unknown_custom_unit"
        self.assertEqual(infer_races(self.state), ("protoss", "unknown"))
        self.assertEqual(candidate_calls(self.policy, self.state, self.base), [])
        self.state = snapshot()
        self.assertEqual(candidate_calls(self.policy, self.state, {"operation": "keep"}), [])
        self.state["relations"] = {}
        self.assertEqual(candidate_calls(self.policy, self.state, self.base), [])

    def model(self, weak=.2, protect=-.1):
        return {"schema_version": 1, "kind": "skill_gain_tree", "candidate_ids": ["focus_weakest", "protect"],
                "feature_names": list(FEATURE_NAMES), "root": {"feature": "ally_recent_loss_ratio", "threshold": .5,
                "left": {"gains": {"focus_weakest": weak, "protect": protect}},
                "right": {"gains": {"focus_weakest": -1, "protect": 1}}}}

    def test_model_selects_only_available_positive_gain(self):
        self.policy["selection"].update(mode="model", model=self.model())
        f = extract_features(self.state)
        available = candidate_calls(self.policy, self.state, self.base, f)
        self.assertEqual(choose(self.policy, f, available, self.base)["candidate_id"], "focus_weakest")
        self.assertEqual(choose(self.policy, f, available[1:], self.base)["candidate_id"], "BASE")
        self.policy["selection"]["min_gain"] = .2
        self.assertEqual(choose(self.policy, f, available, self.base)["candidate_id"], "BASE")
        del f["ally_recent_loss_ratio"]
        self.assertEqual(choose(self.policy, f, available, self.base)["reason"], "invalid_model_features")

    def test_negative_zero_labels_and_base_mode_remain_base(self):
        f = extract_features(self.state)
        available = candidate_calls(self.policy, self.state, self.base, f)
        self.assertEqual(choose(self.policy, f, available, self.base)["candidate_id"], "BASE")
        self.policy["selection"].update(mode="model", model=self.model(0, -.2))
        self.assertEqual(predict_gains(self.policy["selection"]["model"], f)["protect"], -.2)
        self.assertEqual(choose(self.policy, f, available, self.base)["candidate_id"], "BASE")

    def test_validation_rejects_leakage_nonfinite_missing_candidate_and_bad_branches(self):
        for mutate in (
            lambda m: m["root"].update(feature="final_return"),
            lambda m: m["root"].update(threshold=float("nan")),
            lambda m: m["root"]["left"]["gains"].pop("protect"),
            lambda m: m["root"].pop("right"),
        ):
            model = self.model()
            mutate(model)
            with self.assertRaises(ValueError):
                validate_model(model)
        model = self.model()
        model["root"]["left"] = model["root"]
        with self.assertRaises(ValueError):
            validate_model(model)

    def test_policy_validation_no_unknown_scope_params_or_negative_margin(self):
        for mutate in (
            lambda p: p["candidates"][0].update(actor_scope=["u0"]),
            lambda p: p["candidates"][0].update(adaptive_params={"invented_knob": 1}),
            lambda p: p["selection"].update(min_gain=-.1),
            lambda p: p["selection"].update(max_interventions_per_episode=2),
            lambda p: p["candidates"][0].update(id="../outside"),
            lambda p: p["candidates"][0].update(id="base"),
            lambda p: p.update(min_gain=5),
            lambda p: p.update(initial_tree={"call": {"operation": "keep"}}),
            lambda p: p["candidates"][0].update(movement_profile=None),
            lambda p: p["candidates"][0].update(skill=[]),
            lambda p: p["gate"].update(phases=[{}]),
            lambda p: p.update(schema_version=True),
        ):
            policy = deepcopy(self.policy)
            mutate(policy)
            with self.assertRaises(ValueError):
                load_policy(policy)

    def test_initial_semantic_target_call_is_bound_before_execution(self):
        template = {"skill": "adaptive_focus_fire", "target_rule": "lowest_effective_health"}
        bound = bind_call(template, self.state, self.base)
        self.assertEqual(bound["priority_target"], "e1")
        self.assertEqual(bound["max_steps"], 4)
        self.assertNotIn("target_rule", bound)
        self.assertIsNone(bind_call({"skill": "adaptive_focus_fire", "priority_target": "e99"}, self.state))
        self.assertIsNone(bind_call({"skill": "adaptive_focus_fire", "priority_target": "e0"}, self.state))
        self.state["relations"] = {}
        self.assertIsNone(bind_call(template, self.state, self.base))

    def test_default_delegation_and_example_are_distinct(self):
        self.assertEqual(initial_call(self.policy, extract_features(self.state)), {"delegate": "legacy"})
        example = load_policy(CONFIG.with_name("phase_tree_example.json"))
        self.state["visible_enemies"] = {}
        self.state["relations"] = {}
        self.assertEqual(initial_call(example, extract_features(self.state))["skill"], "cohesive_search")
        self.assertNotEqual(policy_hash(example), policy_hash(self.policy))
        copied = load_policy(self.policy)
        copied["candidates"][0]["id"] = "changed"
        self.assertEqual(self.policy["candidates"][0]["id"], "focus_weakest")


if __name__ == "__main__":
    unittest.main()
