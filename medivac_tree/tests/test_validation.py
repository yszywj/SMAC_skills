"""Reject malformed executable trees and prove the JSON drives decisions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from medivac_tree import MedivacTree
from medivac_tree.tests.fixtures import scene, unit
from medivac_tree.validate import validate_tree


def document():
    return {
        "schema_version": 1, "version": "test-fixture-v1", "parameters": {},
        "root": {"id": "alive", "feature": "alive", "op": "==", "threshold": True,
                 "yes": {"id": "stop", "intent": "STOP"},
                 "no": {"id": "noop", "intent": "NOOP"}},
    }


class ValidationTests(unittest.TestCase):
    def assert_rejected(self, tree):
        with self.assertRaises(ValueError):
            validate_tree(tree)

    def test_closed_tree_schema_accepts_only_known_features_and_operators(self):
        validate_tree(document())
        for key, value in [("feature", "hidden_state"), ("op", "eval"),
                           ("threshold", "__import__('os')"), ("threshold", float("nan"))]:
            with self.subTest(key=key, value=value):
                tree = document()
                tree["root"][key] = value
                self.assert_rejected(tree)

    def test_unsupported_leaf_and_extra_leaf_effects_are_rejected(self):
        tree = document()
        tree["root"]["yes"]["intent"] = "ATTACK"
        self.assert_rejected(tree)
        tree = document()
        tree["root"]["yes"]["python"] = "arbitrary code"
        self.assert_rejected(tree)

    def test_missing_child_is_rejected(self):
        tree = document()
        del tree["root"]["no"]
        self.assert_rejected(tree)

    def test_duplicate_node_ids_and_in_memory_cycles_are_rejected(self):
        tree = document()
        tree["root"]["yes"]["id"] = tree["root"]["no"]["id"]
        self.assert_rejected(tree)
        cyclic = document()
        cyclic["root"]["yes"] = cyclic["root"]
        self.assert_rejected(cyclic)

    def test_orphan_node_collection_outside_root_is_rejected(self):
        tree = document()
        tree["nodes"] = [{"id": "unreachable", "intent": "STOP"}]
        self.assert_rejected(tree)

    def test_parameters_have_semantic_domains(self):
        for parameters in ({"unknown_setting": 1}, {"keep_stall_windows": 0},
                           {"rescue_cooldown_windows": 1.5}, {"danger_hp_ratio": 1.1},
                           {"energy_drop": float("inf")}, {"energy_drop": True}):
            with self.subTest(parameters=parameters):
                tree = document()
                tree["parameters"] = parameters
                self.assert_rejected(tree)

    def test_validation_does_not_mutate_input_document(self):
        tree = document()
        before = copy.deepcopy(tree)
        validate_tree(tree)
        self.assertEqual(tree, before)

    def test_supplied_json_changes_executable_leaf_and_trace(self):
        # A wounded legal patient would normally produce a heal. This valid
        # STOP-only policy must instead run the supplied leaf, not hidden rules.
        tree = document()
        tree["root"] = {"id": "explicit_stop", "intent": "STOP"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tree.json"
            path.write_text(json.dumps(tree), encoding="utf-8")
            runtime = MedivacTree(tree_path=path)
            result = runtime.decide(scene([unit("Medivac"), unit("Marine", (1, 0), health=20)]), 2)
        self.assertEqual(result["actions"], {"u0": 1})
        self.assertEqual(result["assignments"], {})
        self.assertEqual(result["decisions"]["u0"]["leaf"], "explicit_stop")
        self.assertEqual(result["decisions"]["u0"]["intent"], "STOP")
        self.assertEqual(result["decisions"]["u0"]["action"], 1)

    def test_leaf_without_required_binding_cannot_emit_fake_action(self):
        tree = document()
        tree["root"] = {"id": "unbound_heal", "intent": "HEAL_NEW"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tree.json"
            path.write_text(json.dumps(tree), encoding="utf-8")
            runtime = MedivacTree(tree_path=path)
            with self.assertRaises(ValueError):
                runtime.decide(scene([unit("Medivac")]), 1)


if __name__ == "__main__":
    unittest.main()
