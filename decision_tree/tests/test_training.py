"""Training checks: independent scenario support, complete labels and deployment parity."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from decision_tree.learning.train import (
    EVIDENCE_KIND, evaluate_paired_rows, fit_gain_tree, group_split, json_hash,
    load_datasets, main, train_policy, tree_counts, validate_rows,
)


CANDIDATES = ["focus_weakest", "protect"]


def row(group, seed=0, x=0., gains=(1., -1.), features=None):
    return {"group_id": group, "sc2_seed": seed, "features": features or {"x": x},
            "base_return": 10., "candidate_returns": {c: 10. + g for c, g in zip(CANDIDATES, gains)},
            "gains": dict(zip(CANDIDATES, gains)), "evidence_kind": EVIDENCE_KIND,
            "gate": {"game_loop": 32, "invocation": 2}}


def real_features(x):
    from decision_tree.features import FEATURE_NAMES
    values = dict.fromkeys(FEATURE_NAMES, 0.)
    values[FEATURE_NAMES[0]] = x
    return values


def policy_contract():
    return {"schema_version": 1, "kind": "skill_return_policy", "supported_races": ["protoss"],
            "gate": {"phases": ["engage"], "min_game_loop": 0},
            "candidates": [{"id": "focus_weakest", "skill": "adaptive_focus_fire",
                            "target_rule": "lowest_effective_health", "adaptive_params": {}},
                           {"id": "protect", "skill": "adaptive_focus_fire", "target_rule": "default",
                            "posture": "defensive", "adaptive_params": {"max_retreat_steps": 3}}]}


def dataset(rows):
    from decision_tree.features import FEATURE_NAMES
    context = {"runtime_sha256": "fixture-only", "environment": {"name": "fixture"}}
    return {"schema_version": 1, "kind": "skill_call_pairs", "context": context,
            "game_version": "fixture-sc2-version",
            "context_hash": json_hash(context), "policy_contract": policy_contract(),
            "candidate_ids": CANDIDATES[:], "feature_names": list(FEATURE_NAMES),
            "rows": rows, "rejected": []}


def write_dataset(directory, name, data):
    path = Path(directory) / name
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path


class TrainingTests(unittest.TestCase):
    def test_multioutput_split_predicts_opposing_preferences(self):
        rows = [row(f"g{i}", x=float(i >= 4), gains=(2., -1.) if i < 4 else (-2., 3.)) for i in range(8)]
        model = fit_gain_tree(rows, CANDIDATES, ["x"], context_hash="fixture")
        self.assertEqual(tree_counts(model["root"]), {"nodes": 3, "predicates": 1, "leaves": 2, "depth": 1})
        self.assertEqual(model["root"]["left"]["gains"], {"focus_weakest": 2., "protect": -1.})
        self.assertEqual(model["root"]["right"]["gains"], {"focus_weakest": -2., "protect": 3.})

    def test_extra_seeds_do_not_make_one_scenario_dominate(self):
        rows = [row("many", seed=i, gains=(8., 0.)) for i in range(9)]
        rows += [row(f"single{i}", gains=(0., 0.)) for i in range(3)]
        model = fit_gain_tree(rows, CANDIDATES, ["x"], context_hash="fixture", max_depth=0)
        self.assertAlmostEqual(model["root"]["gains"]["focus_weakest"], 2.)
        self.assertEqual(model["root"]["n_groups"], 4)

    def test_many_rows_from_few_scenarios_cannot_satisfy_leaf_coverage(self):
        rows = [row(f"g{i % 2}", seed=i, x=i % 2) for i in range(20)]
        with self.assertRaisesRegex(ValueError, "coverage"):
            fit_gain_tree(rows, CANDIDATES, ["x"], context_hash="fixture")

    def test_splits_cannot_create_leaves_with_insufficient_groups(self):
        rows = [row(f"g{i}", x=float(i == 0), gains=(10., 0.) if i == 0 else (0., 0.)) for i in range(6)]
        model = fit_gain_tree(rows, CANDIDATES, ["x"], context_hash="fixture")
        self.assertEqual(tree_counts(model["root"])["nodes"], 1)

    def test_group_split_keeps_all_seeds_together(self):
        rows = [row(f"g{i}", seed=j) for i in range(10) for j in range(3)]
        train, validation = group_split(rows)
        self.assertFalse({r["group_id"] for r in train} & {r["group_id"] for r in validation})
        self.assertEqual(len(train) + len(validation), len(rows))
        self.assertEqual(group_split(rows), (train, validation))

    def test_incomplete_nonfinite_and_wrong_gain_labels_rejected(self):
        variants = []
        missing = row("g"); missing["candidate_returns"].pop("protect"); variants.append(missing)
        mismatched = row("g"); mismatched["gains"]["protect"] = 0.; variants.append(mismatched)
        nan = row("g"); nan["features"]["x"] = float("nan"); variants.append(nan)
        boolean = row("g"); boolean["features"]["x"] = True; variants.append(boolean)
        unaudited = row("g"); unaudited["evidence_kind"] = "same_seed_only"; variants.append(unaudited)
        no_gate = row("g"); no_gate["gate"] = {}; variants.append(no_gate)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_rows([value], CANDIDATES, ["x"])

    def test_duplicate_scenario_seed_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_rows([row("g"), row("g")], CANDIDATES, ["x"])

    def test_all_zero_and_negative_predictions_keep_base(self):
        from decision_tree.features import FEATURE_NAMES
        for gains in ((0., 0.), (-1., -2.)):
            rows = [row(f"g{i}", gains=gains, features=real_features(0.)) for i in range(6)]
            model = fit_gain_tree(rows, CANDIDATES, list(FEATURE_NAMES), context_hash="0" * 64)
            stats = evaluate_paired_rows(model, rows)
            self.assertEqual(stats["selection_counts"]["BASE"], 6)
            self.assertEqual(stats["selector_mean_gain"], 0.)
            self.assertEqual(tree_counts(model["root"])["nodes"], 1)

    def test_context_tamper_and_duplicate_dataset_rejected(self):
        rows = [row(f"g{i}", features=real_features(i % 2)) for i in range(8)]
        with tempfile.TemporaryDirectory() as directory:
            data = dataset(rows)
            path = write_dataset(directory, "valid.json", data)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_datasets([path, path])
            data["context"]["runtime_sha256"] = "changed"
            invalid = write_dataset(directory, "bad.json", data)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_datasets([invalid])

    def test_engine_version_is_required_and_must_match_across_batches(self):
        data = dataset([row("g", features=real_features(0.))])
        with tempfile.TemporaryDirectory() as directory:
            valid = write_dataset(directory, "valid.json", data)
            for i, invalid_version in enumerate((None, "", "  ", 123)):
                invalid = deepcopy(data)
                invalid["game_version"] = invalid_version
                path = write_dataset(directory, f"missing-{i}.json", invalid)
                with self.assertRaisesRegex(ValueError, "game_version"):
                    load_datasets([path])
            other = deepcopy(data)
            other["game_version"] = "other-sc2-build"
            other["rows"][0]["group_id"] = "other-group"
            path = write_dataset(directory, "other.json", other)
            with self.assertRaisesRegex(ValueError, "SC2 game versions"):
                load_datasets([valid, path])

    def test_unregistered_features_rejected_before_fitting(self):
        data = dataset([row("g", features=real_features(1.))])
        data["feature_names"].append("future_reward")
        data["rows"][0]["features"]["future_reward"] = 10.
        with tempfile.TemporaryDirectory() as directory:
            path = write_dataset(directory, "bad.json", data)
            with self.assertRaisesRegex(ValueError, "allowlist"):
                load_datasets([path])

    def test_sealed_test_is_never_fitting_or_internal_validation_data(self):
        data = dataset([row("g", features=real_features(1.))])
        with tempfile.TemporaryDirectory() as directory:
            for split in ("validation", "test"):
                data["split"] = split
                path = write_dataset(directory, split + ".json", data)
                with self.assertRaisesRegex(ValueError, "sealed"):
                    load_datasets([path])
                if split == "test":
                    with self.assertRaisesRegex(ValueError, "sealed"):
                        load_datasets([path], purpose="validation")
                else:
                    self.assertEqual(len(load_datasets([path], purpose="validation")[0]), 1)

    def test_training_outputs_loadable_model_and_preserves_validation_boundary(self):
        from decision_tree.policy import predict_gains
        from decision_tree.schema import load_policy
        rows = [row(f"g{i}", x=i % 2, gains=(2., -1.) if i % 2 else (-2., 3.),
                    features=real_features(i % 2)) for i in range(24)]
        with tempfile.TemporaryDirectory() as directory:
            path = write_dataset(directory, "data.json", dataset(rows))
            policy, model, report = train_policy([path])
            self.assertEqual(policy["selection"]["mode"], "model")
            self.assertEqual(report["status"], "trained_not_game_validated")
            self.assertFalse(set(model["training"]["train_groups"]) & set(model["training"]["validation_groups"]))
            self.assertEqual(model["context_hash"], json_hash(model["context"]))
            self.assertEqual(model["game_version"], "fixture-sc2-version")
            self.assertEqual(report["game_version"], model["game_version"])
            self.assertEqual(policy["selection"]["model"]["game_version"], model["game_version"])
            self.assertEqual(load_policy(policy), policy)
            self.assertEqual(set(predict_gains(model, rows[0]["features"])), set(CANDIDATES))
            self.assertEqual(report["model_sha256"], json_hash(model))
            output = Path(directory) / "fit"
            self.assertEqual(main(["--dataset", str(path), "--output-dir", str(output)]), 0)
            self.assertTrue(all((output / name).is_file() for name in ("policy.json", "model.json", "report.json")))
            with self.assertRaises(SystemExit):
                main(["--dataset", str(path), "--output-dir", str(output)])

    def test_explicit_validation_rejects_same_scenarios_with_new_seeds(self):
        rows = [row(f"g{i}", features=real_features(i % 2)) for i in range(8)]
        validation = deepcopy(rows)
        for value in validation:
            value["sc2_seed"] = 100
        with tempfile.TemporaryDirectory() as directory:
            first = write_dataset(directory, "train.json", dataset(rows))
            second = write_dataset(directory, "val.json", dataset(validation))
            with self.assertRaisesRegex(ValueError, "overlap"):
                train_policy([first], validation_paths=[second])


if __name__ == "__main__":
    unittest.main()
