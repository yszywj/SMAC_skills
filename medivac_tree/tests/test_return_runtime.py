"""One real choice per episode, with no speculative history or coordination."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from medivac_tree import MedivacTree
from medivac_tree.candidates import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from medivac_tree.selector import comparison_contract_hash, validate_policy
from medivac_tree.tests.fixtures import observe_heal, scene, unit


def paired_scene(keep=True):
    state = scene([unit("Medivac"), unit("Marine", (1, 0), health=20),
                   unit("Marine", (0, 2), health=25)])
    observe_heal(state, "u0", "u1", keep=keep)
    return state


class ReturnRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.plain = MedivacTree()
        self.base = deepcopy(self.plain.document)
        self.counter = 0

    def document(self, mode="base", gain=None, candidate=None):
        config = candidate or {"intent": "HEAL_NEW", "base_intents": ["KEEP_HEAL", "HEAL_CURRENT"], "min_game_loop": 0}
        selection = {"mode": mode}
        if mode == "model":
            selection["model"] = {
                "schema_version": 1, "model_kind": "gain_regression_tree", "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_names": list(FEATURE_NAMES),
                "comparison_contract_hash": comparison_contract_hash(self.base, config),
                "root": {"id": "gain", "gain": gain, "n_samples": 4},
            }
        return {"schema_version": 1, "version": "paired-test", "policy_kind": "paired_return_selector",
                "base_tree": deepcopy(self.base), "candidate": config, "selection": selection}

    def load(self, document):
        path = Path(self.directory.name) / f"policy-{self.counter}.json"
        self.counter += 1
        path.write_text(json.dumps(document, allow_nan=False), encoding="utf-8")
        return MedivacTree(path)

    @staticmethod
    def opportunities(result):
        return [decision["return_selection"] for decision in result["decisions"].values()
                if "return_selection" in decision]

    def test_base_arm_is_behaviorally_equal_to_unwrapped_tree(self):
        paired = self.load(self.document())
        state = paired_scene()
        before = deepcopy(state)
        memories = [None, None]
        for loop in (8, 16, 24, 32):
            state["game_loop"] = loop
            results = [tree.decide(state, 3, memories[index]) for index, tree in enumerate((self.plain, paired))]
            for key in ("actions", "assignments", "reservations", "processing_order"):
                self.assertEqual(results[0][key], results[1][key])
            self.assertEqual(results[0]["memory"]["actors"], results[1]["memory"]["actors"])
            memories = [result["memory"] for result in results]
        before["game_loop"] = 32
        self.assertEqual(state, before)

    def test_candidate_intervenes_once_and_next_step_uses_base(self):
        tree = self.load(self.document("candidate"))
        state = paired_scene()
        first = tree.decide(state, 3)
        self.assertEqual(first["actions"]["u0"], 8)
        self.assertEqual(first["decisions"]["u0"]["intent"], "HEAL_NEW")
        self.assertTrue(first["memory"]["return_opportunity_consumed"])
        self.assertEqual(first["memory"]["actors"]["u0"]["last_target"], "u2")
        state["game_loop"] += 8
        second = tree.decide(state, 3, first["memory"])
        expected = self.plain.decide(state, 3, {**first["memory"], "tree_hash": self.plain.tree_hash})
        self.assertEqual(second["actions"], expected["actions"])
        self.assertEqual(second["memory"]["actors"], expected["memory"]["actors"])
        self.assertEqual(self.opportunities(second), [])

    def test_unselected_candidate_does_not_reserve_or_write_history(self):
        tree = self.load(self.document("base"))
        state = paired_scene()
        result = tree.decide(state, 3)
        plain = self.plain.decide(state, 3)
        self.assertNotIn("u2", result["reservations"])
        self.assertEqual(result["memory"]["actors"], plain["memory"]["actors"])
        opportunity = self.opportunities(result)[0]
        self.assertEqual(opportunity["selected"], "BASE")
        self.assertEqual(opportunity["candidate"]["target"], "u2")
        self.assertEqual(set(opportunity["features"]), set(FEATURE_NAMES))
        self.assertEqual(opportunity["features"]["candidate_target_distance"], 2.)

    def test_only_first_eligible_actor_can_intervene(self):
        state = scene([unit("Medivac"), unit("Medivac", (0, 1)),
                       unit("Marine", (1, 0), health=20), unit("Marine", (1, 1), health=22),
                       unit("Marine", (2, 0), health=25), unit("Marine", (2, 1), health=26)])
        observe_heal(state, "u0", "u2")
        observe_heal(state, "u1", "u3")
        tree = self.load(self.document("candidate"))
        result = tree.decide(state, 6)
        self.assertEqual(len(self.opportunities(result)), 1)
        self.assertEqual(result["assignments"], {"u0": "u4", "u1": "u3"})
        expected = self.plain.decide(state, 6)
        self.assertEqual(result["decisions"]["u1"]["intent"], expected["decisions"]["u1"]["intent"])
        self.assertNotIn("u5", result["reservations"])

    def test_cache_is_repeatable_and_inputs_are_not_mutated(self):
        tree = self.load(self.document("candidate"))
        state = paired_scene()
        first = tree.decide(state, 3)
        saved = deepcopy(first["memory"])
        second = tree.decide(state, 3, first["memory"])
        self.assertEqual(first, second)
        self.assertEqual(first["memory"], saved)
        changed = deepcopy(state)
        changed["allies"]["u2"]["health"] = 24
        with self.assertRaisesRegex(ValueError, "changed snapshot"):
            tree.decide(changed, 3, first["memory"])

    def test_no_legal_distinct_candidate_leaves_opportunity_unconsumed(self):
        tree = self.load(self.document("candidate"))
        state = paired_scene()
        state["legal_actions"]["u0"].remove(8)
        result = tree.decide(state, 3)
        self.assertFalse(result["memory"]["return_opportunity_consumed"])
        self.assertEqual(self.opportunities(result), [])
        state["game_loop"] += 8
        state["legal_actions"]["u0"].append(8)
        self.assertEqual(len(self.opportunities(tree.decide(state, 3, result["memory"]))), 1)

    def test_gate_requires_min_loop_base_intent_and_different_action(self):
        cases = [
            {"intent": "HEAL_NEW", "base_intents": ["KEEP_HEAL"], "min_game_loop": 16},
            {"intent": "HEAL_NEW", "base_intents": ["STOP"], "min_game_loop": 0},
            {"intent": "KEEP_HEAL", "base_intents": ["KEEP_HEAL"], "min_game_loop": 0},
        ]
        for config in cases:
            with self.subTest(config=config):
                result = self.load(self.document("candidate", candidate=config)).decide(paired_scene(), 3)
                self.assertFalse(result["memory"]["return_opportunity_consumed"])
                self.assertEqual(self.opportunities(result), [])

    def test_positive_negative_and_tie_model_gains_preserve_base_semantics(self):
        for gain in (-2., 0., 2.):
            with self.subTest(gain=gain):
                tree = self.load(self.document("model", gain=gain))
                result = tree.decide(paired_scene(keep=False), 3)
                decision = result["decisions"]["u0"]
                expected = "HEAL_NEW" if gain > 0 else "HEAL_CURRENT"
                self.assertEqual(decision["intent"], expected)
                self.assertEqual(decision["action"], 8 if gain > 0 else 7)
                opportunity = self.opportunities(result)[0]
                self.assertEqual(opportunity["predicted_gain"], gain)
                self.assertTrue(opportunity["path"])
                self.assertTrue(result["memory"]["return_opportunity_consumed"])

    def test_model_split_uses_observed_candidate_features(self):
        document = self.document("model", gain=0.)
        document["selection"]["model"]["root"] = {
            "id": "split", "feature": "candidate_target_hp_ratio", "op": "<=", "threshold": .6,
            "yes": {"id": "yes", "gain": 1., "n_samples": 4},
            "no": {"id": "no", "gain": -1., "n_samples": 4},
        }
        result = self.load(document).decide(paired_scene(), 3)
        self.assertEqual(self.opportunities(result)[0]["selected"], "CANDIDATE")
        self.assertTrue(self.opportunities(result)[0]["path"][0]["result"])

    def test_cross_arm_fingerprints_match_before_delayed_intervention(self):
        config = {"intent": "HEAL_NEW", "base_intents": ["KEEP_HEAL", "HEAL_CURRENT"], "min_game_loop": 16}
        trees = [self.load(self.document(mode, candidate=config)) for mode in ("base", "candidate")]
        self.assertNotEqual(trees[0].tree_hash, trees[1].tree_hash)
        self.assertEqual(trees[0].metadata["comparison_contract_hash"], trees[1].metadata["comparison_contract_hash"])
        state = paired_scene()
        memories = [tree.decide(state, 3)["memory"] for tree in trees]
        state["game_loop"] = 16
        logs = [self.opportunities(tree.decide(state, 3, memory))[0] for tree, memory in zip(trees, memories)]
        for key in ("state_fingerprint", "memory_fingerprint", "reservations_fingerprint", "context_fingerprint", "features", "base", "candidate"):
            self.assertEqual(logs[0][key], logs[1][key], key)
        self.assertEqual(logs[0]["selected"], "BASE")
        self.assertEqual(logs[1]["selected"], "CANDIDATE")

    def test_episode_reset_rearms_but_same_episode_history_gap_does_not(self):
        tree = self.load(self.document("candidate"))
        state = paired_scene()
        first = tree.decide(state, 3)
        state["game_loop"] += 32
        gap = tree.decide(state, 3, first["memory"])
        self.assertEqual(gap["memory_reset_reason"], "noncontiguous_clock")
        self.assertEqual(self.opportunities(gap), [])
        state["episode"] = 2
        state["game_loop"] = 8
        fresh = tree.decide(state, 3, gap["memory"])
        self.assertEqual(len(self.opportunities(fresh)), 1)

    def test_mode_change_does_not_reuse_other_policy_cache(self):
        state = paired_scene()
        base = self.load(self.document("base")).decide(state, 3)
        tree = self.load(self.document("candidate"))
        result = tree.decide(state, 3, base["memory"])
        self.assertEqual(result["memory_reset_reason"], "schema_or_tree_changed")
        self.assertEqual(self.opportunities(result)[0]["selected"], "CANDIDATE")

    def test_contract_rejects_changed_candidate_model_and_deployed_sources(self):
        document = self.document("model", gain=1.)
        changed = deepcopy(document)
        changed["candidate"]["min_game_loop"] = 16
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            self.load(changed)
        with patch("medivac_tree.selector.runtime_source_hashes", return_value={"runtime.py": "0" * 64}):
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                validate_policy(document)
        document["selection"]["model"]["root"]["gain"] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_policy(document)

    def test_contract_excludes_envelope_version_and_mode(self):
        base = self.document("base")
        candidate = self.document("candidate")
        candidate["version"] = "different-artifact-name"
        left, right = (validate_policy(document) for document in (base, candidate))
        self.assertEqual(left["comparison_contract_hash"], right["comparison_contract_hash"])
        self.assertIn("runtime.py", left["runtime_source_hashes"])
        self.assertEqual(len(left["runtime_source_hashes"]["runtime.py"]), 64)
        self.assertEqual(len(FEATURE_NAMES), 46)


if __name__ == "__main__":
    unittest.main()
