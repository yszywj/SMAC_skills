"""Behavioral tests for the declared repeated-STOP development hypothesis."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from medivac_tree import MedivacTree
from medivac_tree.candidates import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, validate_candidate
from medivac_tree.idle_recovery import DEFAULT_EVENT, evaluate, observed_context, stop_windows
from medivac_tree.selector import comparison_contract_hash, validate_policy
from medivac_tree.tests.fixtures import observe_heal, scene, unit


def idle_scene():
    # All progress toward the wounded ally enters one visible threat envelope.
    # Original APPROACH permits no exposure increase; original SUPPORT stops.
    return scene([unit("Medivac"), unit("Marine", (0, 8), health=20)],
                 {"e0": unit("Marine", (0, 6.5))})


class IdleRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.plain = MedivacTree()
        self.counter = 0

    def document(self, mode="base", *, event=None, gain=0.):
        candidate = {"intent": "APPROACH", "base_intents": ["STOP"], "min_game_loop": 0,
                     "generator": "idle_recovery", "event": {**DEFAULT_EVENT, **(event or {})}}
        document = {"schema_version": 1, "version": "idle-test", "policy_kind": "paired_return_selector",
                    "base_tree": deepcopy(self.plain.document), "candidate": candidate, "selection": {"mode": mode}}
        if mode == "model":
            document["selection"]["model"] = {
                "schema_version": 1, "model_kind": "gain_regression_tree", "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_names": list(FEATURE_NAMES),
                "comparison_contract_hash": comparison_contract_hash(document["base_tree"], candidate),
                "root": {"id": "gain", "gain": gain, "n_samples": 4},
            }
        return document

    def load(self, document):
        path = Path(self.directory.name) / f"policy-{self.counter}.json"
        self.counter += 1
        path.write_text(json.dumps(document), encoding="utf-8")
        return MedivacTree(path)

    def advance(self, tree, state, count=3, memory=None):
        results = []
        for _ in range(count):
            result = tree.decide(state, len(state["allies"]), memory)
            results.append(result)
            memory = result["memory"]
            state["game_loop"] += state.get("step_mul", 8)
        return results

    def test_third_stop_proposes_new_legal_binding_without_rewriting_base(self):
        tree = self.load(self.document("candidate"))
        results = self.advance(tree, idle_scene())
        for index, result in enumerate(results[:2], 1):
            decision = result["decisions"]["u0"]
            self.assertEqual(result["actions"]["u0"], 1)
            self.assertEqual(decision["candidate_audit"]["stop_windows"], index)
            self.assertNotIn("return_selection", decision)
            self.assertFalse(result["memory"]["return_opportunity_consumed"])
        decision = results[2]["decisions"]["u0"]
        self.assertEqual(decision["intent"], "APPROACH")
        self.assertEqual(decision["action"], 2)
        self.assertNotIn("APPROACH", decision["bindings"])
        self.assertFalse(decision["facts"]["has_approach"])
        self.assertEqual(decision["return_selection"]["base"], {"intent": "STOP", "action": 1, "target": None})
        self.assertEqual(decision["candidate_audit"]["stop_windows"], 3)
        self.assertEqual(decision["memory_after"]["idle_stop_windows"], 0)
        self.assertEqual(decision["memory_after"]["approach"]["gap"], 8.)
        self.assertEqual(decision["memory_after"]["approach"]["windows"], 1)
        self.assertEqual(results[2]["reservations"]["u1"], "u0")
        features = decision["return_selection"]["features"]
        self.assertEqual(set(features), set(FEATURE_NAMES))
        self.assertEqual(features["stop_windows"], 3.)
        self.assertEqual(features["candidate_next_gap"], 7.)
        self.assertEqual(features["candidate_progress"], 1.)
        self.assertEqual(features["candidate_exposure_delta"], 1.)
        self.assertEqual(features["candidate_movement_known"], 1.)

    def test_base_arm_retains_actions_facts_and_real_history(self):
        tree = self.load(self.document("base"))
        state = idle_scene()
        plain_memory, paired_memory = None, None
        for _ in range(5):
            ordinary = self.plain.decide(state, 2, plain_memory)
            paired = tree.decide(state, 2, paired_memory)
            self.assertEqual(ordinary["actions"], paired["actions"])
            self.assertEqual(ordinary["reservations"], paired["reservations"])
            a, b = ordinary["decisions"]["u0"], paired["decisions"]["u0"]
            self.assertEqual(a["facts"], b["facts"])
            self.assertEqual(a["bindings"], b["bindings"])
            self.assertEqual(a["memory_after"], {k: v for k, v in b["memory_after"].items() if k != "idle_stop_windows"})
            self.assertEqual(b["memory_after"]["approach"], {})
            self.assertNotIn("u1", paired["reservations"])
            plain_memory, paired_memory = ordinary["memory"], paired["memory"]
            state["game_loop"] += 8

    def test_unselected_model_candidate_does_not_commit_approach(self):
        for gain in (-1., 0.):
            with self.subTest(gain=gain):
                result = self.advance(self.load(self.document("model", gain=gain)), idle_scene())[-1]
                decision = result["decisions"]["u0"]
                self.assertEqual(decision["return_selection"]["selected"], "BASE")
                self.assertEqual(decision["memory_after"]["approach"], {})
                self.assertEqual(decision["memory_after"]["idle_stop_windows"], 3)
                self.assertEqual(result["reservations"], {})

    def test_one_intervention_only_and_observed_progress_is_not_invented(self):
        tree = self.load(self.document("candidate"))
        state = idle_scene()
        results = self.advance(tree, state, 5)
        self.assertEqual(sum("return_selection" in r["decisions"]["u0"] for r in results), 1)
        later = results[3]["decisions"]["u0"]
        self.assertEqual(later["candidate_audit"]["reason"], "episode_opportunity_consumed")
        self.assertIn("approach_expired_or_no_observed_progress", later["binding_details"]["approach_rejections"])
        self.assertEqual(state["allies"]["u0"]["position"], [0, 0])

    def test_other_medical_reservation_is_not_stolen(self):
        state = scene([unit("Medivac"), unit("Medivac", (0, 7)), unit("Marine", (0, 8), health=20)],
                      {"e0": unit("Marine", (0, 6.5))})
        observe_heal(state, "u1", "u2", keep=False)
        result = self.advance(self.load(self.document("candidate")), state)[-1]
        decision = result["decisions"]["u0"]
        self.assertEqual(decision["intent"], "STOP")
        self.assertEqual(decision["candidate_audit"]["reason"], "no_unreserved_wounded")
        self.assertEqual(decision["candidate_audit"]["reserved_patients"][0]["reserved_by"], "u1")
        self.assertFalse(result["memory"]["return_opportunity_consumed"])

    def test_idle_medic_leaves_direct_heal_target_for_nearby_medic(self):
        state = scene([unit("Medivac"), unit("Medivac", (0, 7)),
                       unit("Marine", (0, 8), health=20), unit("Marine", (0, 12), health=25)],
                      {"e0": unit("Marine", (0, 6.5))})
        base = self.advance(self.load(self.document("base")), deepcopy(state))[-1]
        candidate = self.advance(self.load(self.document("candidate")), deepcopy(state))[-1]
        decision = candidate["decisions"]["u0"]
        self.assertEqual(decision["target"], "u3")
        self.assertEqual(decision["intent"], "APPROACH")
        self.assertEqual(candidate["decisions"]["u1"]["target"], "u2")
        self.assertEqual(candidate["actions"]["u1"], base["actions"]["u1"])
        self.assertEqual([a for a in candidate["actions"] if candidate["actions"][a] != base["actions"][a]], ["u0"])
        covered = decision["candidate_audit"]["covered_patients"]
        self.assertEqual(covered[0]["target"], "u2")
        self.assertEqual(covered[0]["healers"], [{"actor": "u1", "action": 8}])
        # The observation feature continues to count all unreserved patients;
        # it is not silently redefined as the filtered proposal count.
        self.assertEqual(decision["return_selection"]["features"]["wounded_unreserved_count"], 2.)

    def test_all_patients_directly_healable_elsewhere_spends_no_opportunity(self):
        state = scene([unit("Medivac"), unit("Medivac", (0, 7)), unit("Marine", (0, 8), health=20)],
                      {"e0": unit("Marine", (0, 6.5))})
        result = self.advance(self.load(self.document("candidate")), state)[-1]
        decision = result["decisions"]["u0"]
        self.assertEqual(decision["candidate_audit"]["reason"], "no_uncovered_wounded")
        self.assertEqual(decision["candidate_audit"]["wounded_unreserved_count"], 1)
        self.assertEqual(result["actions"]["u0"], 1)
        self.assertFalse(result["memory"]["return_opportunity_consumed"])

    def test_other_medic_without_energy_does_not_hide_approach_candidate(self):
        state = scene([unit("Medivac"), unit("Medivac", (0, 7), energy=0),
                       unit("Marine", (0, 8), health=20)], {"e0": unit("Marine", (0, 6.5))})
        result = self.advance(self.load(self.document("candidate")), state)[-1]
        decision = result["decisions"]["u0"]
        self.assertEqual(decision["intent"], "APPROACH")
        self.assertEqual(decision["target"], "u2")
        self.assertEqual(decision["candidate_audit"]["covered_patients"], [])

    def test_missing_other_medical_legality_never_means_uncovered(self):
        for missing in ("legal_actions", "heal_candidates", "energy"):
            with self.subTest(missing=missing):
                state = scene([unit("Medivac"), unit("Medivac", (0, 7)),
                               unit("Marine", (0, 8), health=20)], {"e0": unit("Marine", (0, 6.5))})
                decision = self.plain.decide(state, 3)["decisions"]["u0"]
                if missing == "energy":
                    state["allies"]["u1"].pop("energy")
                else:
                    state[missing].pop("u1")
                proposal, audit = evaluate(validate_candidate(self.document()["candidate"]), state, "u0",
                                           set(state["legal_actions"]["u0"]), {}, {}, decision["visible_threats"],
                                           "STOP", decision["binding"], 3)
                self.assertIsNone(proposal)
                self.assertEqual(audit["reason"], "other_medical_legality_unknown")
                self.assertEqual(audit["coverage_uncertain_medics"][0]["actor"], "u1")

    def test_masked_other_heal_candidate_is_not_direct_coverage(self):
        state = scene([unit("Medivac"), unit("Medivac", (0, 7)),
                       unit("Marine", (0, 8), health=20)], {"e0": unit("Marine", (0, 6.5))})
        decision = self.plain.decide(state, 3)["decisions"]["u0"]
        state["legal_actions"]["u1"].remove(8)
        proposal, audit = evaluate(validate_candidate(self.document()["candidate"]), state, "u0",
                                   set(state["legal_actions"]["u0"]), {}, {}, decision["visible_threats"],
                                   "STOP", decision["binding"], 3)
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal["binding"]["target"], "u2")
        self.assertEqual(audit["covered_patients"], [])

    def test_low_health_energy_and_active_order_prevent_candidate(self):
        for change in ("health", "energy", "order"):
            with self.subTest(change=change):
                state = idle_scene()
                if change == "health":
                    state["allies"]["u0"]["health"] = 75
                elif change == "energy":
                    state["allies"]["u0"]["energy"] = 4
                else:
                    observe_heal(state, "u0", "u1", keep=False)
                result = self.advance(self.load(self.document("candidate")), state)[-1]
                self.assertNotIn("return_selection", result["decisions"]["u0"])
                self.assertFalse(result["memory"]["return_opportunity_consumed"])
                if change in ("energy", "order"):
                    self.assertEqual(result["memory"]["actors"]["u0"]["idle_stop_windows"], 0)

    def test_energy_order_and_non_stop_break_consecutive_window(self):
        state = idle_scene()
        old = {"tag": state["allies"]["u0"]["tag"], "loop": 0,
               "last_action": 1, "last_intent": "STOP", "idle_stop_windows": 2}
        binding = {"action": 1}
        self.assertEqual(stop_windows(state, "u0", old, "STOP", binding, DEFAULT_EVENT), 3)
        for key, value in (("loop", -8), ("last_action", 2), ("last_intent", "SUPPORT"), ("tag", "new-tag")):
            self.assertEqual(stop_windows(state, "u0", {**old, key: value}, "STOP", binding, DEFAULT_EVENT), 1)
        self.assertEqual(stop_windows(state, "u0", old, "SUPPORT", binding, DEFAULT_EVENT), 0)
        state["allies"]["u0"]["energy"] = 0
        self.assertEqual(stop_windows(state, "u0", old, "STOP", binding, DEFAULT_EVENT), 0)
        state["allies"]["u0"]["energy"] = 50
        state["allies"]["u0"]["orders"] = [{"ability_id": 386, "target_unit": "u1"}]
        self.assertEqual(stop_windows(state, "u0", old, "STOP", binding, DEFAULT_EVENT), 0)

    def test_illegal_progress_exposure_and_distance_constraints_do_not_consume(self):
        for kind in ("illegal", "progress", "exposure", "distance"):
            with self.subTest(kind=kind):
                state = idle_scene()
                event = {}
                if kind == "illegal":
                    state["legal_actions"]["u0"].remove(2)
                elif kind == "progress":
                    event["min_progress"] = 2
                elif kind == "exposure":
                    event["max_extra_exposure"] = 0
                else:
                    event["max_target_distance"] = 6
                result = self.advance(self.load(self.document("candidate", event=event)), state)[-1]
                self.assertNotIn("return_selection", result["decisions"]["u0"])
                self.assertFalse(result["memory"]["return_opportunity_consumed"])

    def test_no_candidate_does_not_spend_future_opportunity(self):
        state = idle_scene()
        state["legal_actions"]["u0"].remove(2)
        tree = self.load(self.document("candidate"))
        result = self.advance(tree, state)[-1]
        state["legal_actions"]["u0"].append(2)
        later = tree.decide(state, 2, result["memory"])
        self.assertEqual(later["decisions"]["u0"]["return_selection"]["selected"], "CANDIDATE")

    def test_gap_and_episode_reset_require_new_three_stop_sequence(self):
        tree = self.load(self.document("candidate"))
        state = idle_scene()
        first = self.advance(tree, state, 2)[-1]
        state["game_loop"] += 8
        reset = tree.decide(state, 2, first["memory"])
        self.assertEqual(reset["decisions"]["u0"]["candidate_audit"]["stop_windows"], 1)
        self.assertNotIn("return_selection", reset["decisions"]["u0"])
        state["episode"] = 2
        state["game_loop"] = 8
        fresh = self.advance(tree, state, 3, reset["memory"])[-1]
        self.assertIn("return_selection", fresh["decisions"]["u0"])

    def test_cache_and_cross_arm_pre_intervention_fingerprints(self):
        state = idle_scene()
        trees = [self.load(self.document(mode)) for mode in ("base", "candidate")]
        results = [self.advance(tree, deepcopy(state))[-1] for tree in trees]
        a, b = [r["decisions"]["u0"]["return_selection"] for r in results]
        for name in ("features", "state_fingerprint", "memory_fingerprint", "reservations_fingerprint", "context_fingerprint"):
            self.assertEqual(a[name], b[name])
        state["game_loop"] = 24
        before = deepcopy(results[1]["memory"])
        self.assertEqual(trees[1].decide(state, 2, before), results[1])
        self.assertEqual(before, results[1]["memory"])

    def test_pure_generator_does_not_mutate_observations_or_reservations(self):
        state = idle_scene()
        decision = self.plain.decide(state, 2)["decisions"]["u0"]
        reservations = {}
        inputs = deepcopy((state, decision, reservations))
        config = validate_candidate(self.document()["candidate"])
        context = observed_context(state, "u0", reservations, decision["visible_threats"])
        proposal, audit = evaluate(config, state, "u0", set(state["legal_actions"]["u0"]), {}, reservations,
                                   decision["visible_threats"], "STOP", decision["binding"], 3, context)
        self.assertEqual((state, decision, reservations), inputs)
        self.assertEqual(proposal["binding"]["target"], "u1")
        self.assertEqual(audit["reason"], "proposed")

    def test_new_schema_rejects_old_feature_model_and_invalid_events(self):
        self.assertEqual(FEATURE_SCHEMA_VERSION, 2)
        self.assertEqual(len(FEATURE_NAMES), 46)
        document = self.document("model", gain=1.)
        document["selection"]["model"]["feature_schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "feature schema"):
            validate_policy(document)
        for key, value in (("min_stop_windows", 0), ("min_energy", .5), ("min_health_ratio", 1.1),
                           ("max_target_distance", 0), ("min_progress", 0), ("max_extra_exposure", -1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_policy(self.document(event={key: value}))

    def test_old_binding_candidate_configuration_stays_supported(self):
        old = {"intent": "HEAL_NEW", "base_intents": ["HEAL_CURRENT"], "min_game_loop": 0}
        self.assertEqual(validate_candidate(old), old)
        document = self.document()
        document["candidate"] = old
        tree = self.load(document)
        state = scene([unit("Medivac"), unit("Marine", (1, 0), health=20), unit("Marine", (0, 1), health=25)])
        observe_heal(state, "u0", "u1", keep=False)
        result = tree.decide(state, 3)
        self.assertIn("return_selection", result["decisions"]["u0"])
        self.assertNotIn("idle_stop_windows", result["memory"]["actors"]["u0"])
        self.assertNotIn("candidate_audit", result["decisions"]["u0"])


if __name__ == "__main__":
    unittest.main()
