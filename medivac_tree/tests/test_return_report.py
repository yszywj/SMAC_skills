"""Whole-episode comparisons must include no-opportunities and actual failures."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest

from medivac_tree.learning.dataset import build_dataset
from medivac_tree.learning.report import build_report, main
from medivac_tree.tests.test_return_dataset import make_run, read, write


class ReturnReportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.base, self.other = self.root / "base", self.root / "other"
        make_run(self.base, "base", 2.)
        make_run(self.other, "candidate", 3.)

    def test_completed_pair_reports_actual_reward_win_rate_and_coverage(self):
        result = build_report(self.base, self.other)
        self.assertEqual(result["status"], "complete", result)
        summary = result["summary"]
        self.assertEqual(summary["mean_return_difference"], 1.)
        self.assertEqual(summary["both_win"], 1)
        self.assertEqual(summary["base"]["win_rate"], 1.)
        self.assertEqual(summary["other"]["candidate_coverage_fraction"], 1.)
        self.assertEqual(summary["other"]["selected_counts"], {"CANDIDATE": 1})
        self.assertEqual(result["training_eligibility"]["eligible_pairs"], 1)
        self.assertFalse(result["training_labels_generated"])

    def test_normal_timeout_is_an_outcome_and_changes_paired_win_result(self):
        records = read(self.base, "episodes.json")
        records[0]["outcome"] = "timeout"
        write(self.base, "episodes.json", records)
        events = read(self.base, "events.jsonl")
        events[-1]["outcome"] = "timeout"
        write(self.base, "events.jsonl", events)
        result = build_report(self.base, self.other)
        self.assertEqual(result["summary"]["other_only_win"], 1)
        self.assertEqual(result["summary"]["base"]["win_rate"], 0.)
        self.assertEqual(result["summary"]["outcome_pairs"], {"timeout->win": 1})

    def test_no_opportunity_counts_in_full_episode_comparison_but_never_as_label(self):
        candidate = {"intent": "HEAL_NEW", "base_intents": ["KEEP_HEAL", "HEAL_CURRENT"], "min_game_loop": 100}
        base, other = self.root / "no-base", self.root / "no-other"
        make_run(base, "base", candidate=candidate)
        make_run(other, "candidate", candidate=candidate)
        # These records also exercise diagnostic reporting for old fixture
        # generators that do not themselves emit candidate_audit.
        for path in (base, other):
            events = read(path, "events.jsonl")
            for event in events:
                if event["event"] == "medivac_tree_step":
                    event["decisions"][0]["candidate_audit"] = {"generator": "fixture", "eligible": False,
                                                               "reason": "min_game_loop", "stop_windows": 0}
            write(path, "events.jsonl", events)
        result = build_report(base, other)
        self.assertEqual(result["status"], "complete", result)
        self.assertEqual(result["summary"]["compared_pairs"], 1)
        self.assertEqual(result["summary"]["other"]["episodes_without_opportunity"], 1)
        self.assertEqual(result["summary"]["other"]["candidate_audit_reason_counts"], {"min_game_loop": 3})
        self.assertEqual(result["training_eligibility"]["eligible_pairs"], 0)

    def test_failed_candidate_command_is_kept_in_policy_outcomes(self):
        events = read(self.other, "events.jsonl")
        feedback = next(e for e in events if e["event"] == "step" and e["step"] == 2)["feedback"]["actors"]["u0"]
        feedback.update(submission="rejected", response={"code": 9, "name": "Error"})
        write(self.other, "events.jsonl", events)
        result = build_report(self.base, self.other)
        self.assertEqual(result["status"], "complete", result)
        self.assertEqual(result["summary"]["compared_pairs"], 1)
        self.assertEqual(result["summary"]["other"]["feedback_failure_counts"], {"feedback_failed": 1})
        self.assertEqual(result["training_eligibility"]["eligible_pairs"], 0)
        self.assertIn("feedback_failed", result["training_eligibility"]["reason_counts"])

    def test_model_runs_compare_without_producing_candidate_labels(self):
        for gain in (-1., 1.):
            model = self.root / ("model-" + str(gain))
            make_run(model, "model", model_gain=gain)
            result = build_report(self.base, model)
            self.assertEqual(result["status"], "complete", result)
            self.assertEqual(result["other_mode"], "model")
            self.assertEqual(result["training_eligibility"]["eligible_pairs"], 0)
            expected = "CANDIDATE" if gain > 0 else "BASE"
            self.assertEqual(result["summary"]["other"]["selected_counts"], {expected: 1})
            self.assertEqual(build_dataset(self.base, model)["rows"], [])

    def test_changed_sources_settings_and_incomplete_runs_refuse_comparison(self):
        original = read(self.other, "run.json")
        for key in ("source", "settings"):
            value = deepcopy(original)
            if key == "source":
                value["source_hashes"]["runtime.py"] = "c" * 64
            else:
                value["versions"]["pysc2"] = "changed"
            write(self.other, "run.json", value)
            result = build_report(self.base, self.other)
            self.assertEqual(result["status"], "rejected")
            self.assertEqual(result["pairs"], [])
        write(self.other, "run.json", original)
        summary = read(self.other, "summary.json")
        summary["status"] = "failed"
        write(self.other, "summary.json", summary)
        self.assertEqual(build_report(self.base, self.other)["status"], "rejected")

    def test_learning_support_counts_independent_groups_and_all_return_signs(self):
        data = build_dataset(self.base, self.other)
        path = self.root / "dataset.json"
        write(self.root, path.name, data)
        result = build_report(self.base, self.other, dataset=path)
        support = result["learning_support"]
        self.assertEqual(support["label_count"], 1)
        self.assertEqual(support["independent_groups"], 1)
        self.assertEqual(support["gain_sign_counts"], {"positive": 1, "zero": 0, "negative": 0})
        self.assertFalse(support["meets_default_group_minimum"])
        self.assertIn("insufficient_independent_groups_for_default_train_validation_split", support["warnings"])
        data["experiment_context_hash"] = "0" * 64
        write(self.root, path.name, data)
        self.assertEqual(build_report(self.base, self.other, dataset=path)["status"], "rejected")

    def test_cli_output_does_not_overwrite(self):
        output = self.root / "report.json"
        args = ["--base-run", str(self.base), "--other-run", str(self.other), "--output", str(output)]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        original = output.read_bytes()
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            main(args)
        self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
