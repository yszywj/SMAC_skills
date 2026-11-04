"""Return-learning semantics using synthetic fixtures, never game evidence."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from medivac_tree.candidates import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from medivac_tree.experiment import context_hash, experiment_context, validate_deployment
from medivac_tree.learning.train import (EVIDENCE_KIND, evaluate_paired_rows, fit_gain_tree,
                                        group_split, load_datasets, main, train_policy, validate_rows)
from medivac_tree.selector import comparison_contract_hash, predict_gain, select, validate_model, validate_policy


CONTRACT = "a" * 64


def row(group, seed, x, gain, contract=CONTRACT):
    features = dict.fromkeys(FEATURE_NAMES, 0.)
    features["health_ratio"] = float(x)
    return {"features": features, "target_gain": float(gain), "base_return": 10.,
            "candidate_return": 10. + gain, "group_id": group, "sc2_seed": seed,
            "comparison_contract_hash": contract, "evidence_kind": EVIDENCE_KIND}


def execution_metadata(policy="base.json"):
    return {"source_hashes": {"adapter.py": "a" * 64, "skills/ground.py": "b" * 64, "C:/synthetic/" + policy: "c" * 64},
            "medivac_tree_metadata": {"tree_path": "C:/synthetic/" + policy},
            "skill_source_hashes": {"skills/ground.py": "b" * 64}, "env_args": {"difficulty": 7},
            "skill_parameters": {"posture": "balanced"}, "versions": {"simulator": "v1"},
            "observation_protocol": "observed-v1", "action_extensions": {"keep_heal": True},
            "method": "adaptive-skill", "skill_version": "current", "native_teacher_control": "off",
            "medivac_policy": "tree", "medical_patient_coordination": False,
            "unit_value_mode": "contextual", "manifest_sha256": "d" * 64, "scenario_count": 4}


def data(rows, contract=CONTRACT, context=None):
    context = context or experiment_context(execution_metadata())
    return {"schema_version": 1, "evidence_kind": EVIDENCE_KIND,
            "comparison_contract_hash": contract,
            "experiment_context": context, "experiment_context_hash": context_hash(context),
            "feature_schema": {"version": FEATURE_SCHEMA_VERSION, "names": list(FEATURE_NAMES)},
            "rows": rows, "rejected": []}


def envelope():
    path = Path(__file__).parents[1] / "tree_v0.json"
    return {"schema_version": 1, "version": "synthetic-training-fixture",
            "policy_kind": "paired_return_selector", "base_tree": json.loads(path.read_text(encoding="utf-8")),
            "candidate": {"intent": "HEAL_NEW", "base_intents": ["KEEP_HEAL", "HEAL_CURRENT"],
                          "min_game_loop": 0}, "selection": {"mode": "base"}}


class ReturnTrainingTests(unittest.TestCase):
    def test_tree_learns_negative_and_positive_return_regions_and_runtime_agrees(self):
        rows = [row(f"g{i}", i, .2 if i < 4 else .8, -2 if i < 4 else 3) for i in range(8)]
        before = deepcopy(rows)
        model = fit_gain_tree(rows, CONTRACT)
        validate_model(model, CONTRACT)
        self.assertEqual(model["root"]["feature"], "health_ratio")
        self.assertEqual(predict_gain(model, rows[0]["features"])[0], -2.)
        self.assertEqual(predict_gain(model, rows[-1]["features"])[0], 3.)
        self.assertEqual(select({"mode": "model", "model": model}, rows[0]["features"])[0], "BASE")
        self.assertEqual(select({"mode": "model", "model": model}, rows[-1]["features"])[0], "CANDIDATE")
        self.assertEqual({v["n_groups"] for v in model["training"]["leaves"].values()}, {4})
        self.assertEqual(rows, before)

    def test_repeated_seeds_do_not_satisfy_minimum_independent_leaf_coverage(self):
        # Plenty of rows but only two scenarios on either side: no split.
        rows = [row(f"g{i % 4}", i, float(i % 4 >= 2), -2 if i % 4 < 2 else 2) for i in range(40)]
        model = fit_gain_tree(rows, CONTRACT)
        self.assertIn("gain", model["root"])
        self.assertEqual(model["training"]["leaves"]["r"]["n_groups"], 4)
        with self.assertRaisesRegex(ValueError, "coverage"):
            fit_gain_tree(rows[:3], CONTRACT)

    def test_each_scenario_has_equal_total_training_weight(self):
        rows = [row("frequent", i, .5, 9) for i in range(10)]
        rows += [row("rare", 20, .5, -1)]
        model = fit_gain_tree(rows, CONTRACT, min_samples_leaf=1, min_leaf_groups=1)
        self.assertAlmostEqual(model["root"]["gain"], 4.)

    def test_zero_prediction_keeps_base_and_report_retains_all_signs(self):
        rows = [row("a", 1, .5, -1), row("b", 2, .5, 0), row("c", 3, .5, 1)]
        model = fit_gain_tree(rows, CONTRACT, min_samples_leaf=1, min_leaf_groups=1)
        report = evaluate_paired_rows(model, rows)
        self.assertEqual(model["root"]["gain"], 0.)
        self.assertEqual(report["candidate_selections"], 0)
        self.assertEqual(report["target_sign_counts"], {"negative": 1, "zero": 1, "positive": 1})
        self.assertEqual(report["selector_mean_gain"], 0.)
        self.assertFalse(report["win_metrics"]["available"])

    def test_outcome_metrics_compare_base_candidate_and_selector_without_changing_target(self):
        rows = [row("a", 1, .2, -2), row("b", 2, .8, 3)]
        rows[0].update(base_outcome="win", candidate_outcome="timeout")
        rows[1].update(base_outcome="loss", candidate_outcome="win")
        model = fit_gain_tree(rows, CONTRACT, min_samples_leaf=1, min_leaf_groups=1)
        report = evaluate_paired_rows(model, rows)
        metrics = report["win_metrics"]
        self.assertTrue(metrics["available"])
        self.assertEqual(metrics["base_win_rate"], .5)
        self.assertEqual(metrics["always_candidate_win_rate"], .5)
        self.assertEqual(metrics["selector_win_rate"], 1.)
        self.assertEqual(metrics["selector_win_rate_delta"], .5)
        self.assertEqual(model["training"]["target"], "candidate_return_minus_base_return")
        del rows[1]["candidate_outcome"]
        self.assertFalse(evaluate_paired_rows(model, rows)["win_metrics"]["available"])

    def test_group_split_is_deterministic_and_never_splits_a_scenario_seeds(self):
        rows = [row(f"group{i % 12}", i, .5, i % 3 - 1) for i in range(36)]
        train, validation = group_split(rows)
        train_groups = {r["group_id"] for r in train}
        validation_groups = {r["group_id"] for r in validation}
        self.assertFalse(train_groups & validation_groups)
        self.assertEqual(len(validation_groups), 3)
        reversed_train, reversed_validation = group_split(list(reversed(rows)))
        self.assertEqual(train_groups, {r["group_id"] for r in reversed_train})
        self.assertEqual(validation_groups, {r["group_id"] for r in reversed_validation})
        with self.assertRaisesRegex(ValueError, "not enough independent"):
            group_split(rows[:4])

    def test_training_rejects_duplicated_pairs_and_untrustworthy_labels(self):
        good = row("a", 1, .5, 2)
        mutations = [
            ("target_gain", 1.), ("candidate_return", float("nan")),
            ("evidence_kind", "offline_replay"), ("comparison_contract_hash", "b" * 64),
            ("sc2_seed", True), ("group_id", ""),
        ]
        for name, value in mutations:
            with self.subTest(name=name):
                bad = deepcopy(good)
                bad[name] = value
                with self.assertRaises(ValueError):
                    validate_rows([bad], CONTRACT)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_rows([good, deepcopy(good)], CONTRACT)
        for name, value in [("final_win", 1), ("health_ratio", float("inf")), ("health_ratio", "low")]:
            with self.subTest(feature=name, value=value):
                bad = deepcopy(good)
                bad["features"][name] = value
                with self.assertRaises(ValueError):
                    validate_rows([bad], CONTRACT)

    def test_depth_and_minimum_coverage_cannot_silently_escape_protocol(self):
        rows = [row(f"g{i}", i, i / 10, i % 2) for i in range(8)]
        for config in ({"max_depth": 3}, {"max_depth": True}, {"min_samples_leaf": 0}, {"min_leaf_groups": False}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                fit_gain_tree(rows, CONTRACT, **config)
        constant = fit_gain_tree(rows, CONTRACT, max_depth=0)
        self.assertIn("gain", constant["root"])

    def test_complete_cli_exports_loadable_policy_and_refuses_overwrite(self):
        policy = envelope()
        contract = comparison_contract_hash(policy["base_tree"], policy["candidate"])
        rows = [row(f"g{i}", i, .2 if i < 8 else .8, -2 if i < 8 else 3, contract)
                for i in range(16)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path, dataset_path, output = root / "base.json", root / "dataset.json", root / "trained"
            base_path.write_text(json.dumps(policy), encoding="utf-8")
            dataset_path.write_text(json.dumps(data(rows, contract)), encoding="utf-8")
            args = ["--dataset", str(dataset_path), "--policy", str(base_path), "--output-dir", str(output)]
            self.assertEqual(main(args), 0)
            deployed = json.loads((output / "policy.json").read_text(encoding="utf-8"))
            model = json.loads((output / "model.json").read_text(encoding="utf-8"))
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            validate_policy(deployed)
            self.assertEqual(deployed["base_tree"], policy["base_tree"])
            self.assertEqual(deployed["candidate"], policy["candidate"])
            self.assertEqual(deployed["selection"], {"mode": "model", "model": model})
            self.assertEqual(report["status"], "trained_not_game_validated")
            self.assertFalse(set(report["split"]["training_groups"]) & set(report["split"]["validation_groups"]))
            before = (output / "policy.json").read_bytes()
            with self.assertRaises(SystemExit):
                main(args)
            self.assertEqual((output / "policy.json").read_bytes(), before)

    def test_explicit_validation_rejects_group_overlap_even_with_different_seeds(self):
        policy = envelope()
        contract = comparison_contract_hash(policy["base_tree"], policy["candidate"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.json"
            train = root / "train.json"
            validation = root / "validation.json"
            base.write_text(json.dumps(policy), encoding="utf-8")
            train.write_text(json.dumps(data([row(f"g{i}", i, .5, 1, contract) for i in range(4)], contract)), encoding="utf-8")
            validation.write_text(json.dumps(data([row("g0", 99, .5, -1, contract)], contract)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                train_policy([train], base, validation_paths=[validation])
            validation.write_text(json.dumps(data([row("heldout", 99, .5, -1, contract)], contract)), encoding="utf-8")
            _, _, report = train_policy([train], base, validation_paths=[validation])
            self.assertEqual(report["split"]["method"], "explicit_validation_files")
            self.assertEqual(report["validation"]["selector_mean_gain"], -1.)

    def test_dataset_load_rejects_schema_drift_and_cross_file_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            contents = data([row("g0", 1, .2, 1)])
            path.write_text(json.dumps(contents), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_datasets([path, path], CONTRACT)
            contents["feature_schema"]["names"].append("return_after_action")
            path.write_text(json.dumps(contents), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                load_datasets([path], CONTRACT)

    def test_context_excludes_only_policy_and_scenario_manifest(self):
        base, candidate = execution_metadata(), execution_metadata("candidate.json")
        candidate["source_hashes"]["C:/synthetic/candidate.json"] = "e" * 64
        candidate["manifest_sha256"] = "f" * 64
        candidate["scenario_count"] = 20
        self.assertEqual(experiment_context(base), experiment_context(candidate))
        candidate["source_hashes"]["some_other.json"] = "f" * 64
        self.assertIn("some_other.json", experiment_context(candidate)["source_hashes"])
        self.assertNotEqual(experiment_context(base), experiment_context(candidate))
        base["skill_source_hashes"] = {"skills\\ground.py": "b" * 64}
        self.assertEqual(experiment_context(base)["skill_source_hashes"], {"skills/ground.py": "b" * 64})
        base["source_hashes"].pop("C:/synthetic/base.json")
        base["source_hashes"]["base.json"] = "c" * 64  # Same basename is not the declared file.
        with self.assertRaisesRegex(ValueError, "exactly one"):
            experiment_context(base)

    def test_cross_dataset_context_drift_is_rejected_even_with_same_medical_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first.json", Path(directory) / "second.json"
            first.write_text(json.dumps(data([row("first", 1, .5, 1)])), encoding="utf-8")
            changed = experiment_context(execution_metadata())
            changed["skill_parameters"]["posture"] = "defensive"
            second.write_text(json.dumps(data([row("second", 2, .5, 1)], context=changed)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different experiment contexts"):
                load_datasets([first, second], CONTRACT)
            broken = data([row("second", 2, .5, 1)])
            broken["experiment_context_hash"] = "f" * 64
            second.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "context hash mismatch"):
                load_datasets([second], CONTRACT)

    def test_deployment_binds_ground_sources_environment_versions_and_settings(self):
        base = execution_metadata()
        context = experiment_context(base)
        model = {"training": {"experiment_context": context, "experiment_context_hash": context_hash(context)}}
        selector = execution_metadata("learned.json")
        self.assertTrue(validate_deployment(selector, model)["matched"])
        for field, value in [("env_args", {"difficulty": 8}), ("versions", {"simulator": "v2"}),
                             ("unit_value_mode", "off"), ("medical_patient_coordination", True)]:
            with self.subTest(field=field):
                changed = deepcopy(selector)
                changed[field] = value
                with self.assertRaisesRegex(ValueError, "context differs"):
                    validate_deployment(changed, model)
        changed = deepcopy(selector)
        changed["source_hashes"]["skills/ground.py"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "context differs"):
            validate_deployment(changed, model)
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_deployment(selector, {"training": {}})

    def test_runtime_generated_paired_dataset_is_accepted_by_training_loader(self):
        from medivac_tree.learning.dataset import build_dataset
        from medivac_tree.tests.test_return_dataset import make_run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_run(root / "base", "base", 2.)
            make_run(root / "candidate", "candidate", 3.)
            paired = build_dataset(root / "base", root / "candidate")
            self.assertEqual(paired["rejected"], [])
            path = root / "dataset.json"
            path.write_text(json.dumps(paired), encoding="utf-8")
            rows, inputs, context = load_datasets([path], paired["comparison_contract_hash"])
            self.assertEqual(rows[0]["target_gain"], 1.)
            self.assertEqual(inputs[0]["accepted_rows"], 1)
            self.assertEqual(context_hash(context), paired["experiment_context_hash"])


if __name__ == "__main__":
    unittest.main()
