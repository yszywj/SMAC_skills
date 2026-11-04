"""Regression cases for uninformative choices and paired-run attribution."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tree_skills.training.train_tree import (
    ConstantZeroScorer, evaluate_ranking, main as train_main)
from tree_skills.evaluation.paired_report import (
    build_report, load_episodes, paired_bootstrap, paired_stats, validate_pairs)


def group(loop, labels, predictions=None, healer="u5", episode="ep1"):
    predictions = predictions or {t: 0.0 for t in labels}
    return {"episode_id": episode, "game_loop": loop, "healer_id": healer,
            "labels": labels, "features": {t: [predictions[t]] for t in labels}}


class FeatureScorer:
    def predict(self, rows):
        return [row[0] for row in rows]


class TestRankingMetrics(unittest.TestCase):
    def test_single_candidate_success_cannot_mask_no_ranking_skill(self):
        groups = [group(i, {"u0": 0.0}) for i in range(99)]
        groups.append(group(99, {"u0": -1.0, "u1": 1.0}))
        metrics = evaluate_ranking(ConstantZeroScorer(), groups)
        self.assertEqual(metrics["all_top1_agreement"], 0.99)
        self.assertEqual(metrics["multi_candidate_top1_agreement"], 0.0)
        self.assertEqual(metrics["constant_zero_baseline"]["multi_candidate_top1_agreement"], 0.0)
        self.assertEqual(metrics["nontrivial_joint_assignment_agreement"], 0.0)
        self.assertEqual(metrics["n_multi_candidate_groups"], 1)

    def test_joint_metric_keeps_single_candidate_healer_reservation(self):
        # u3 reserves u0. Both matchers must allocate u1 to u2, even though
        # their independent best choices for u2 disagree.
        groups = [group(1, {"u0": 10.0, "u1": 0.0},
                        {"u0": -1.0, "u1": 1.0}, healer="u2"),
                  group(1, {"u0": 0.0}, healer="u3")]
        metrics = evaluate_ranking(FeatureScorer(), groups)
        self.assertEqual(metrics["multi_candidate_top1_agreement"], 0.0)
        self.assertEqual(metrics["nontrivial_joint_assignment_agreement"], 1.0)
        self.assertEqual(metrics["nontrivial_n_participating_healers"], 2)

    def test_score_ties_use_deployment_lexical_target_order(self):
        metrics = evaluate_ranking(ConstantZeroScorer(),
                                   [group(1, {"u2": -1.0, "u10": 1.0})])
        self.assertEqual(metrics["multi_candidate_top1_agreement"], 1.0)
        self.assertEqual(metrics["nontrivial_joint_assignment_agreement"], 1.0)

    def test_infeasible_joint_assignment_does_not_count_as_agreement(self):
        metrics = evaluate_ranking(ConstantZeroScorer(),
                                   [group(1, {"u0": 0.0}, healer="u2"),
                                    group(1, {"u0": 0.0}, healer="u3")])
        self.assertEqual(metrics["all_n_infeasible_joint_steps"], 1)
        self.assertIsNone(metrics["all_joint_assignment_agreement"])

    def test_teacher_ties_have_zero_regret_but_no_unique_top1_denominator(self):
        metrics = evaluate_ranking(ConstantZeroScorer(),
                                   [group(1, {"u0": 0.0, "u1": 0.0})])
        self.assertIsNone(metrics["multi_candidate_top1_agreement"])
        self.assertEqual(metrics["multi_candidate_n_tied_groups"], 1)
        self.assertEqual(metrics["multi_candidate_mean_teacher_regret"], 0.0)

    def test_training_refuses_selection_on_only_trivial_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "shard.jsonl").write_text("\n".join(json.dumps(g) for g in
                [group(1, {"u0": 0.0}, episode="train"),
                 group(1, {"u0": 0.0}, episode="valid")]), encoding="utf-8")
            (path / "valid.json").write_text('["valid"]', encoding="utf-8")
            with patch("tree_skills.training.train_tree.fit") as fit:
                with self.assertRaisesRegex(SystemExit, "no multi-candidate"):
                    train_main(["--train-shard", str(path / "shard.jsonl"),
                                "--valid-episodes", str(path / "valid.json"),
                                "--output-dir", str(path / "model")])
                fit.assert_not_called()


def episode(win, index=1):
    return {"win": win, "scenario": f"scenario{index}", "initial_state": f"initial{index}"}


class TestPairedReports(unittest.TestCase):
    def test_secondary_means_use_only_complete_finite_pairs(self):
        a = {k: episode(True, k) for k in range(1, 5)}
        b = {k: episode(True, k) for k in range(1, 5)}
        for k, va, vb in ((1, 0.2, 0.4), (2, 0.8, None),
                          (3, float("nan"), 0.9), (4, 0.5, 1.2)):
            a[k]["enemy_elimination_fraction"] = va
            b[k]["enemy_elimination_fraction"] = vb
        result = paired_stats(a, b, list(a), rounds=200)["secondary_outcomes"]
        enemy = result["enemy_elimination_fraction"]
        self.assertEqual(enemy["n_pairs"], 1)
        self.assertEqual(enemy["n_missing_or_invalid_pairs"], 3)
        self.assertEqual(enemy["mean_a"], 0.2)
        self.assertEqual(enemy["mean_b"], 0.4)
        self.assertEqual(enemy["paired_mean_delta"], 0.2)
        self.assertTrue(enemy["paired_diff"]["bootstrap_degenerate"])
        survival = result["allied_survival_fraction"]
        self.assertEqual(survival["n_pairs"], 0)
        self.assertIsNone(survival["mean_a"])
        self.assertIsNone(survival["paired_mean_delta"])
        self.assertIsNone(survival["paired_diff"])

    def test_step_differences_are_descriptive_without_fraction_bounds(self):
        a, b = {1: episode(True)}, {1: episode(True)}
        a[1]["steps"], b[1]["steps"] = 20, 120
        result = paired_stats(a, b, [1], rounds=200)["secondary_outcomes"]["steps"]
        self.assertEqual(result["n_pairs"], 1)
        self.assertEqual(result["mean_a"], 20)
        self.assertEqual(result["mean_b"], 120)
        self.assertEqual(result["paired_mean_delta"], 100)
        self.assertNotIn("paired_diff", result)
        self.assertNotIn("conservative_ci95", result)

    def test_all_zero_bootstrap_reports_uncertainty(self):
        result = paired_bootstrap([0] * 40, rounds=200)
        self.assertEqual((result["ci95_low"], result["ci95_high"]), (0, 0))
        self.assertTrue(result["bootstrap_degenerate"])
        self.assertTrue(result["warning"])
        self.assertGreater(result["conservative_ci95"]["high"], 0)
        self.assertAlmostEqual(result["zero_discordance_bound"]["discordance_probability_upper"],
                               1 - 0.05 ** (1 / 40))

    def test_equal_total_win_rate_does_not_hide_discordant_pairs(self):
        a = {1: episode(True, 1), 2: episode(False, 2)}
        b = {1: episode(False, 1), 2: episode(True, 2)}
        result = paired_stats(a, b, [1, 2], rounds=200)
        self.assertEqual(result["paired_diff"]["point"], 0)
        self.assertEqual(result["n_discordant"], 2)
        self.assertFalse(result["paired_diff"]["bootstrap_degenerate"])
        self.assertEqual(result["paired_outcomes"]["a_win_b_loss"], 1)
        self.assertEqual(result["paired_outcomes"]["a_loss_b_win"], 1)

    def test_same_episode_number_is_not_sufficient_for_pairing(self):
        a, b = {1: episode(True)}, {1: episode(True)}
        b[1]["initial_state"] = "different"
        with self.assertRaisesRegex(ValueError, "initial_state hash mismatch"):
            validate_pairs(a, b, {1})
        b[1]["initial_state"] = None
        with self.assertRaisesRegex(ValueError, "missing initial_state"):
            validate_pairs(a, b, {1})

    def test_missing_episodes_require_explicit_incomplete_reporting(self):
        a, b = {1: episode(True), 2: episode(False, 2)}, {1: episode(True)}
        with self.assertRaisesRegex(ValueError, "incomplete runs"):
            validate_pairs(a, b, {1, 2})
        common, audit = validate_pairs(a, b, {1, 2}, allow_incomplete=True)
        self.assertEqual(common, [1])
        self.assertEqual(audit["missing_completed_in_b"], [2])
        self.assertFalse(audit["complete"])

    def test_duplicate_completion_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            event = {"event": "episode_completed", "episode": 1}
            Path(directory, "events.jsonl").write_text((json.dumps(event) + "\n") * 2,
                                                      encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_episodes(directory)

    def test_different_manifest_is_rejected_before_comparing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory, "a"), Path(directory, "b")
            a.mkdir()
            b.mkdir()
            (a / "manifest.json").write_text('{"seed": 1, "episodes": []}', encoding="utf-8")
            (b / "manifest.json").write_text('{"seed": 2, "episodes": []}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifests differ"):
                build_report(a, b)


if __name__ == "__main__":
    unittest.main()
