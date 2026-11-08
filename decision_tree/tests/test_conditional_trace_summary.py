"""Read-only descriptive counts must not confuse layer choices with commands."""
import json
from pathlib import Path
import tempfile
import unittest

from decision_tree.scripts.conditional_trace_summary import summarize_round, summarize_run


def decision(episode, step, selected, interventions, corrections=()):
    return {"event": "conditional_local_decision", "episode": episode, "step": step,
            "game_loop": 2 + step * 8, "baseline_actions": [6],
            "final_actions": [6 if selected == "BASE" else int(selected.split(":")[-1])],
            "selected_id": selected, "natural_selected_id": selected, "forced": False,
            "catalog": {"effects": {"BASE": {}, "u0:7": {}}, "bindings": []},
            "considered": [], "corrections": list(corrections), "interventions": interventions,
            "conditional_policy": {"actor_cooldown_steps": 2}}


def receipt(event):
    return {"event": "step", "episode": event["episode"], "step": event["step"] + 1,
            "actions": event["final_actions"],
            "feedback": {"actors": {"u0": {"submission": "accepted", "observation_errors": []}}}}


class TraceSummaryTests(unittest.TestCase):
    def test_partition_and_layer_changes_are_distinct_from_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            events = [decision(1, 0, "retarget:u0:7", 1),
                      decision(1, 1, "BASE", 1, [{"before": "retarget:u0:7", "after": "BASE"}]),
                      decision(1, 2, "retarget:u0:8", 2, [{"before": "BASE", "after": "retarget:u0:7"},
                                                        {"before": "retarget:u0:7", "after": "retarget:u0:8"}]),
                      decision(1, 3, "BASE", 2, [{"before": "BASE", "after": "retarget:u0:7"},
                                                 {"before": "retarget:u0:7", "after": "BASE"}]),
                      decision(2, 0, "retarget:u0:9", 1)]
            files = {"summary.json": {"status": "complete"}, "run.json": {"env_args": {"step_mul": 8}},
                     "episodes.json": [{"episode": 1, "scenario_sha256": "train", "outcome": "win", "reward": 20, "steps": 4},
                                       {"episode": 2, "scenario_sha256": "holdout", "outcome": "loss", "reward": 7, "steps": 1}]}
            for name, value in files.items():
                (path / name).write_text(json.dumps(value), encoding="utf-8")
            (path / "events.jsonl").write_text("\n".join(json.dumps(row) for e in events for row in (e, receipt(e))) + "\n", encoding="utf-8")
            report = summarize_run(path, {"train"})
            self.assertEqual(report["scene_count"], 1)
            self.assertEqual(report["outcomes"], {"win": 1})
            self.assertEqual(report["counts"]["changed_command_submissions"], 2)
            self.assertEqual(report["counts"]["changed_command_receipt_accepted"], 2)
            self.assertEqual(report["counts"]["steps_with_multiple_layer_changes"], 2)
            self.assertEqual(report["counts"]["steps_corrections_cancelled_original_nonbase"], 1)
            self.assertEqual(report["counts"]["next_step_same_actor_restored_prior_baseline_action_id"], 2)
            self.assertEqual(report["correction_layers"]["0"]["cancelled_to_BASE"], 1)
            self.assertEqual(report["correction_layers"]["1"]["cancelled_to_BASE"], 1)
            self.assertEqual(report["same_actor_intervention_gap_decision_steps"]["min"], 2)
            self.assertEqual(report["anomaly_examples"], [])

    def test_root_holdout_requires_candidate_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Freeze the candidate"):
                summarize_round(tmp, 1, "holdout")
            with self.assertRaisesRegex(ValueError, "Freeze the candidate"):
                summarize_round(tmp, 1, "all")


if __name__ == "__main__":
    unittest.main()
