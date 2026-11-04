"""Test report semantics without launching SC2 or reading unfinished runs."""

import json
import copy
from pathlib import Path
import tempfile
import unittest

from tree_skills.evaluation.analyze_tokenfix import (action_divergences, analyze_experiment,
                                                    attach_live_provenance,
                                                    load_run_evidence,
                                                    intervention_execution_feedback,
                                                    teacher_intent)
from tree_skills.teacher.action_mapping import SmacActionMap
from tree_skills.tests.test_collection_labels import sample_row
from tree_skills.data.build_shard import build_shard
from tree_skills.scripts.relabel_collection import sha256


class TokenFixAnalysisTest(unittest.TestCase):
    def test_first_divergence_is_not_cascade_count(self):
        a = {(1, loop): {"step": i, "actions": [1, 1]} for i, loop in enumerate((2, 10, 18))}
        b = {(1, loop): {"step": i, "actions": actions} for i, (loop, actions) in
             enumerate(((2, [1, 1]), (10, [1, 7]), (18, [4, 7]), (26, [4, 1])))}
        result = action_divergences(a, b)
        self.assertEqual(result["episodes_with_action_divergence"], 1)
        self.assertEqual(result["divergent_matched_loops_including_cascades"], 2)
        ep = result["episodes"][0]
        self.assertEqual(ep["first_action_divergence"]["game_loop"], 10)
        self.assertEqual(ep["first_action_divergence"]["changed_unit_indices"], [1])
        self.assertEqual(ep["loops_only_in_b"], [26])

    def test_incomplete_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "summary.json").write_text(json.dumps({"status": "running"}))
            with self.assertRaisesRegex(ValueError, "not complete"):
                load_run_evidence(run)

    def test_live_shard_provenance_requires_matching_actions_and_corrected_token(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            for name in ("run.json", "summary.json", "collect.jsonl"):
                (run / name).write_text("{}")
            rows = [sample_row()]
            config = {"env_indicator": 0, "last_token": "on", "failure_policy": "strict",
                      "checkpoint_sha256": "known-sha"}
            steps = {(1, 8): {"actions": rows[0]["final_actions"]}}
            provenance = attach_live_provenance(rows, run, config, [run / "collect.jsonl"], steps)
            self.assertEqual(provenance["teacher_configuration"]["checkpoint_sha256"], "known-sha")
            self.assertEqual(rows[0]["label_provenance"]["kind"], "live_corrected_teacher_policy_trajectory")
            with self.assertRaisesRegex(ValueError, "do not match"):
                attach_live_provenance(rows, run, config, [], {(1, 8): {"actions": [0]}})
            with self.assertRaisesRegex(ValueError, "corrected token"):
                attach_live_provenance(rows, run, {**config, "env_indicator": 1}, [], steps)

    def test_teacher_heal_preference_can_be_excluded_from_candidates(self):
        row = sample_row()
        row["teacher_raw"]["logits"]["0"][8] = 8
        row["healers"]["u0"]["candidates"].pop("u2")
        mapping = SmacActionMap(n_agents=3, n_enemies=3)
        report = teacher_intent([row], mapping)
        self.assertEqual(report["all"]["heal"], 1)
        self.assertEqual(report["all"]["preferred_heal_outside_candidates"], 1)
        self.assertEqual(report["multi_candidate"]["groups"], 0)
        self.assertIsNone(report["multi_candidate"]["stop_probabilities"]["mean"])

    def test_execution_separates_submission_success_already_targeted_and_unknown(self):
        mapping = SmacActionMap(n_agents=3, n_enemies=3)
        rows = [sample_row(loop) for loop in (8, 16, 24, 32, 40, 48, 56)]
        rows[-1]["final_actions"][0] = mapping.keep_action_id()
        def feedback(submission, code, name, target="u1", errors=None):
            return {"actors": {"u0": {"requested_action_id": "7",
                "command": {"kind": "heal", "target_unit": target},
                "submission": submission, "response": {"code": code, "name": name},
                "observation_errors": errors or []}}}
        steps = {
            (1, 8): {"feedback": feedback("accepted", 1, "Success")},
            (1, 16): {"feedback": feedback("rejected", 206, "AlreadyTargeted")},
            (1, 24): {"feedback": feedback("rejected", 20, "NotEnoughEnergy", errors=[{"code": 20}])},
            (1, 32): {"feedback": {}},
            (1, 40): {"feedback": feedback("accepted", 1, "Success", target="u2")},
            (1, 48): {"feedback": feedback("error", None, None)},
        }
        result = intervention_execution_feedback(rows, steps, mapping)
        counts = result["counts"]
        self.assertEqual(counts["recommended_changes"], 7)
        self.assertEqual(counts["submitted"], 6)
        for name in ("accepted_success", "already_targeted", "rejected_other", "unknown",
                     "feedback_mismatch", "feedback_error", "not_submitted"):
            self.assertEqual(counts[name], 1, name)
        self.assertEqual(counts["observation_error_count"], 1)
        self.assertEqual(result["records"][1]["execution_outcome"], "already_targeted")
        self.assertIn("existing heal order", result["interpretation"])

    def test_complete_synthetic_report_preserves_separate_label_datasets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a, b, relabel = (root / name for name in ("a", "b", "relabel"))
            for path in (a, b, relabel):
                path.mkdir()
            manifest = {"episodes": [{"sha256": "scenario", "episode_config": {
                "team_gen": {"ally_team": ["medivac", "marine", "marine"]}}}]}
            meta = {"env_args": {"capability_config": {"n_units": 3, "n_enemies": 3}}}
            config = {"event": "pair_configuration", "env_indicator": 0, "last_token": "on",
                      "failure_policy": "strict", "checkpoint_sha256": "known-sha"}
            old = sample_row()
            corrected = copy.deepcopy(old)
            corrected["teacher_raw"]["logits"]["0"][8] = 9
            step = {"event": "step", "episode": 1, "step": 1, "game_loop": 8,
                    "actions": old["final_actions"]}
            completed = {"event": "episode_completed", "episode": 1, "outcome": "win",
                         "scenario_sha256": "scenario", "initial_state_sha256": "initial", "steps": 1}
            for run in (a, b):
                (run / "summary.json").write_text(json.dumps({"status": "complete"}))
                (run / "manifest.json").write_text(json.dumps(manifest))
                (run / "run.json").write_text(json.dumps(meta))
                events = ([config] if run == b else []) + [step, completed]
                (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            source = root / "old.jsonl"
            source.write_text(json.dumps(old) + "\n")
            corrected_path = relabel / "collect-corrected-labels-on-old-trajectory.jsonl"
            corrected_path.write_text(json.dumps(corrected) + "\n")
            (relabel / "relabel_manifest.json").write_text(json.dumps({
                "status": "complete", "source_sha256": sha256(source),
                "checkpoint_sha256": "known-sha", "collection_sha256": sha256(corrected_path)}))
            build_shard([corrected], SmacActionMap(n_agents=3, n_enemies=3),
                        str(relabel / "corrected_labels_shard.jsonl"))
            live = root / "live.jsonl"
            live.write_text(json.dumps(corrected) + "\n")
            result = analyze_experiment(a, b, source, relabel, live, root / "output", rounds=10)
            self.assertEqual(result["same_observation_label_comparison"]["multi_candidate_top1_changed"], 1)
            self.assertEqual(result["live_shard"]["rows"], 1)
            shard = json.loads(Path(result["live_shard"]["path"]).read_text())
            self.assertEqual(shard["diagnostics"]["label_provenance"]["episode_namespace"], "b")
            self.assertTrue((root / "output/episodes.csv").is_file())
            self.assertEqual(result["intervention_execution_feedback"]["counts"]["unknown"], 1)


if __name__ == "__main__":
    unittest.main()
