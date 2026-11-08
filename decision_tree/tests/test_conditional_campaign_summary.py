import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from decision_tree.scripts import conditional_campaign_summary as report


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root):
    folder = root / "round_1"
    split = {"training_group_ids": ["train"], "validation_group_ids": ["holdout"]}
    save(root / "protocol.json", {"kind": "conditional_v3_bounded_3x40", "rounds": [{"round": 1}, {"round": 2}]})
    save(folder / "split.json", split)
    save(folder / "manifest.json", {"episodes": [{"sha256": "train", "sc2_seed": 11}, {"sha256": "holdout", "sc2_seed": 22}]})
    old, candidate = {"corrections": []}, {"corrections": [{"gain": None, "veto": None}]}
    for name, value in (("incumbent", old), ("candidate", candidate), ("adopted", old)):
        save(folder / f"{name}_policy.json", value)
    paths = {name: folder / name for name in ("adaptive", "incumbent", "candidate_train", "candidate_holdout")}
    rows = {
        "adaptive": [("train", "win", 20), ("holdout", "win", 20)],
        "incumbent": [("train", "loss", 10), ("holdout", "win", 20)],
        "candidate_train": [("train", "win", 20)], "candidate_holdout": [("holdout", "loss", 8)]}
    for name, path in paths.items():
        save(path / "summary.json", {"status": "complete"})
        policy_hash = None if name == "adaptive" else report.digest(old if name == "incumbent" else candidate)
        save(path / "run.json", {"conditional_skills": {"policy_sha256": policy_hash}})
        save(path / "episodes.json", [{"episode": n, "scenario_sha256": g, "initial_state_sha256": g + "-initial", "outcome": o, "reward": r}
                                      for n, (g, o, r) in enumerate(rows[name], 1)])
    save(folder / "run_paths.json", {name: str(path) for name, path in paths.items() if name in ("adaptive", "incumbent")})
    save(folder / "evaluation.json", {"adopted": False, "candidate_train": str(paths["candidate_train"]),
        "candidate_holdout": str(paths["candidate_holdout"]),
        "training": {"complete": True, "matched_scenes": 1, "mean_win_gain": 1, "mean_return_gain": 10},
        "holdout_vs_incumbent": {"complete": True, "matched_scenes": 1, "mean_win_gain": -1, "mean_return_gain": -12},
        "holdout_vs_adaptive": {"complete": True, "matched_scenes": 1, "mean_win_gain": -1, "mean_return_gain": -12}})
    measured = [{"scene_hash": "train", "family": "retarget", "label_kind": "gain"}]
    save(folder / "measured.json", measured)
    save(folder / "branch_plan.json", {"jobs": [{}], "coverage": [{"family": "retarget", "status": "selected"}]})
    save(folder / "freeze.json", {"candidate_sha256": report.digest(candidate), "training_sha256": report.digest(measured),
                                  "split_sha256": report.digest(split)})
    return folder


class CampaignSummaryTests(unittest.TestCase):
    @patch.object(report, "summarize_run", return_value={"counts": {}, "correction_layers": {}})
    def test_partial_actual_results_pair_seeds_and_output(self, mocked_trace):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            # Incomplete round contents must never be opened for results.
            (root / "round_2").mkdir()
            (root / "round_2/evaluation.json").write_text("not-json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Campaign is incomplete"):
                report.build_summary(root)
            summary = report.build_summary(root, partial=True)
            self.assertEqual(summary["completed_rounds"], [1])
            self.assertEqual(summary["unfinished_rounds"][0]["round"], 2)
            record = summary["rounds"][0]
            self.assertEqual(record["partitions"]["all"]["strategies"]["candidate"]["mean_return"], 14)
            self.assertEqual(record["partitions"]["all"]["comparisons"]["candidate_vs_incumbent"]["rescued_seeds"], [11])
            self.assertEqual(record["partitions"]["all"]["comparisons"]["candidate_vs_adaptive"]["lost_seeds"], [22])
            self.assertEqual(record["policy_layers"], {"incumbent": 0, "candidate": 1, "adopted": 0})
            self.assertEqual(record["completed_native_games"], 7)
            self.assertEqual(mocked_trace.call_count, 4)
            report.write_report(root, summary)
            self.assertEqual(report.read(root / "campaign_summary.json")["status"], "partial")
            self.assertIn("最终单一模型", (root / "results.md").read_text(encoding="utf-8"))

    @patch.object(report, "summarize_run", return_value={"counts": {}, "correction_layers": {}})
    def test_saved_evaluation_must_match_actual_episodes(self, _):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = fixture(root)
            value = report.read(folder / "candidate_holdout/episodes.json")
            value[0]["reward"] = 9
            save(folder / "candidate_holdout/episodes.json", value)
            with self.assertRaisesRegex(ValueError, "Saved evaluation differs from episodes"):
                report.build_summary(root, partial=True)

    @patch.object(report, "summarize_run", return_value={"counts": {}, "correction_layers": {}})
    def test_existing_evaluation_does_not_allow_incomplete_native_run(self, _):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = fixture(root)
            save(folder / "candidate_holdout/summary.json", {"status": "running"})
            with self.assertRaisesRegex(ValueError, "Native run is incomplete"):
                report.build_summary(root, partial=True)


if __name__ == "__main__":
    unittest.main()
