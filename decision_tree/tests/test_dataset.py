"""Counterfactual audit tests using synthetic, explicitly non-game evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from decision_tree.experiment import digest, policy_contract
from decision_tree.features import FEATURE_NAMES
from decision_tree.learning.dataset import audit_pair, build_dataset, compare_runs, episode, read_run
from decision_tree.schema import load_policy


IDS = ("focus_weakest", "protect")


def feedback():
    return {"response_mapping": "exact", "unmatched_observation_errors": [],
            "actors": {f"u{i}": {"submission": "accepted", "observation_errors": []} for i in range(2)}}


def synthetic_run(arm="BASE", gain=0., *, gate_step=1, call_end=5):
    policy = load_policy({"schema_version": 1, "kind": "skill_return_policy",
        "supported_races": ["protoss"], "gate": {"phases": ["engage"], "min_game_loop": 0},
        "candidates": [{"id": IDS[0], "skill": "adaptive_focus_fire", "target_rule": "lowest_effective_health"},
                       {"id": IDS[1], "skill": "adaptive_focus_fire", "posture": "defensive"}],
        "selection": {"mode": "base"} if arm == "BASE" else {"mode": "candidate", "candidate_id": arm}})
    scene = {"episode": 1, "episode_config": {"synthetic": 1}, "sc2_seed": 7}
    scene["sha256"] = digest(scene["episode_config"])
    manifest = {"env_args": {"map_name": "10gen_protoss"}, "episodes": [scene], "split": "development"}
    manifest["sha256"] = digest(manifest)
    record = {"episode": 1, "steps": 6, "reward": 10. + gain, "outcome": "win",
              "initial_state_sha256": "initial", "scenario_sha256": scene["sha256"]}
    events = [{"event": "episode_started", "episode": 1, "initial_state_sha256": "initial",
               "scenario_sha256": scene["sha256"], "game_version": "5.0.test"}]
    base_call = {"operation": "start", "skill": "adaptive_focus_fire", "max_steps": 4}
    candidate_calls = [{"id": IDS[0], "call": {**base_call, "priority_target": "e1"}},
                       {"id": IDS[1], "call": {**base_call, "posture": "defensive"}}]
    previews = {IDS[0]: [7, 7], IDS[1]: [2, 3]}
    gate = {"event": "skill_tree_decision", "episode": 1, "step": gate_step, "game_loop": gate_step*8,
            "features": dict.fromkeys(FEATURE_NAMES, 0.), "base_call": base_call,
            "candidate_calls": candidate_calls, "base_actions": [6, 6], "candidate_actions": previews,
            "opportunity": True, "opportunity_index": 1, "call_number": 1,
            "selected_id": arm, "selected_call": (base_call if arm == "BASE" else
                                                    next(x["call"] for x in candidate_calls if x["id"] == arm))}
    events.append(gate)
    events.append({"event": "skill_tree_decision", "episode": 1, "step": call_end,
                   "game_loop": call_end*8, "opportunity": False, "selected_id": "BASE",
                   "selected_call": base_call, "base_call": base_call, "call_number": 2})
    for step in range(6):
        events.append({"event": "skill_observation", "episode": 1, "step": step,
                       "payload": {"state": {"game_loop": step*8, "visible": "same"}}})
        actions = previews[arm] if step == gate_step and arm != "BASE" else [6, 6]
        events.append({"event": "step", "episode": 1, "step": step+1, "game_loop": (step+1)*8,
                       "actions": actions, "reward": 10.+gain if step == 5 else 0.,
                       "feedback": feedback(), "terminated": step == 5})
    events.append({"event": "episode_completed", **record})
    metadata = {"method": "adaptive-skill", "skill_version": "current", "env_args": manifest["env_args"],
                "skill_parameters": {}, "versions": {"game": "fixture"}, "source_hashes": {"fixture": "sha"},
                "observation_protocol": "fixture", "native_teacher_control": "off", "medivac_policy": "legacy",
                "medical_patient_coordination": True, "manifest_sha256": manifest["sha256"],
                "skill_tree": {"policy": policy, "policy_contract": policy_contract(policy)}}
    return {"path": arm, "hashes": {}, "game_version": "5.0.test", "run.json": metadata, "manifest.json": manifest,
            "summary.json": {"status": "complete", "completed_episodes": 1, "expected_episodes": 1,
                             "manifest_sha256": manifest["sha256"], "pair_mode": "off"},
            "episodes.json": [record], "events.jsonl": events}


def event(run, kind, step=None):
    return next(e for e in run["events.jsonl"] if e["event"] == kind and (step is None or e.get("step") == step))


def build(runs):
    with patch("decision_tree.learning.dataset.read_run", side_effect=lambda path: deepcopy(runs[str(path)])):
        return build_dataset("BASE", {cid: cid for cid in IDS})


def write_run(directory, run):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=False)
    for name in ("run.json", "manifest.json", "summary.json", "episodes.json", "events.jsonl"):
        value = run[name]
        text = ("\n".join(json.dumps(e, allow_nan=False) for e in value) + "\n"
                if name.endswith("jsonl") else json.dumps(value, allow_nan=False))
        (path / name).write_text(text, encoding="utf-8")
    return path


class DatasetTests(unittest.TestCase):
    def test_on_disk_reads_bind_file_hashes_and_engine_version(self):
        with tempfile.TemporaryDirectory() as directory:
            base = write_run(Path(directory) / "base", synthetic_run())
            candidates = {cid: write_run(Path(directory) / cid, synthetic_run(cid, gain))
                          for cid, gain in zip(IDS, (2., -1.))}
            loaded = read_run(base)
            self.assertEqual(loaded["game_version"], "5.0.test")
            self.assertEqual(len(loaded["hashes"]), 5)
            self.assertTrue(all(len(value) == 64 for value in loaded["hashes"].values()))
            result = build_dataset(base, candidates)
            self.assertEqual(result["game_version"], "5.0.test")
            self.assertEqual(result["rows"][0]["gains"], {IDS[0]: 2., IDS[1]: -1.})

    def test_on_disk_run_without_actual_engine_version_is_rejected(self):
        run = synthetic_run()
        event(run, "episode_started").pop("game_version")
        with tempfile.TemporaryDirectory() as directory:
            path = write_run(Path(directory) / "bad", run)
            with self.assertRaisesRegex(ValueError, "game_version"):
                read_run(path)

    def test_complete_multioutput_labels_include_positive_and_negative_calls(self):
        runs = {"BASE": synthetic_run(), IDS[0]: synthetic_run(IDS[0], 2.), IDS[1]: synthetic_run(IDS[1], -1.)}
        data = build(runs)
        self.assertEqual(data["rejected"], [])
        self.assertEqual(len(data["rows"]), 1)
        self.assertEqual(data["rows"][0]["gains"], {IDS[0]: 2., IDS[1]: -1.})
        self.assertEqual(data["rows"][0]["gate"][IDS[0]]["changed_actors"], [0, 1])

    def test_missing_arm_is_not_a_zero_gain(self):
        with patch("decision_tree.learning.dataset.read_run", return_value=synthetic_run()):
            with self.assertRaisesRegex(ValueError, "missing_candidate_arms"):
                build_dataset("BASE", {IDS[0]: "one"})

    def test_no_opportunity_kept_in_full_comparison_but_not_labels(self):
        runs = {"BASE": synthetic_run(), **{cid: synthetic_run(cid) for cid in IDS}}
        for run in runs.values():
            event(run, "skill_tree_decision", 1)["opportunity"] = False
            event(run, "skill_tree_decision", 1)["selected_id"] = "BASE"
        data = build(runs)
        self.assertEqual(data["rows"], [])
        self.assertEqual(len(data["rejected"]), 1)
        self.assertIn("not_exactly_one_opportunity", data["rejected"][0]["reason"])
        self.assertTrue(all(c["episodes"] == 1 for c in data["comparisons"].values()))

    def test_a_single_bad_candidate_rejects_entire_multioutput_row(self):
        runs = {"BASE": synthetic_run(), **{cid: synthetic_run(cid) for cid in IDS}}
        event(runs[IDS[1]], "skill_observation", 1)["payload"]["state"]["visible"] = "different"
        data = build(runs)
        self.assertEqual(data["rows"], [])
        self.assertIn("observed_prefix_mismatch", data["rejected"][0]["reason"])

    def test_changed_prefix_actions_and_command_errors_reject(self):
        for which in ("actions", "feedback"):
            base, candidate = synthetic_run(), synthetic_run(IDS[0])
            if which == "actions":
                event(candidate, "step", 1)["actions"] = [1, 1]
            else:
                event(candidate, "step", 3)["feedback"]["actors"]["u0"]["submission"] = "rejected"
            with self.subTest(which=which), self.assertRaises(ValueError):
                audit_pair(episode(base, 1), episode(candidate, 1), IDS[0])

    def test_unchanged_actual_action_is_not_a_skill_intervention(self):
        base, candidate = synthetic_run(), synthetic_run(IDS[0])
        for run in (base, candidate):
            event(run, "skill_tree_decision", 1)["candidate_actions"][IDS[0]] = [6, 6]
        event(candidate, "step", 2)["actions"] = [6, 6]
        with self.assertRaisesRegex(ValueError, "no_actual_intervention"):
            audit_pair(episode(base, 1), episode(candidate, 1), IDS[0])

    def test_context_difference_rejects_comparison(self):
        base, candidate = synthetic_run(), synthetic_run(IDS[0])
        candidate["run.json"]["source_hashes"]["fixture"] = "changed"
        with self.assertRaisesRegex(ValueError, "context_mismatch"):
            compare_runs(base, candidate)

    def test_shorter_candidate_still_audits_full_base_call(self):
        base, candidate = synthetic_run(call_end=5), synthetic_run(IDS[0], call_end=3)
        event(base, "step", 4)["feedback"]["actors"]["u0"]["submission"] = "rejected"
        with self.assertRaisesRegex(ValueError, "command_rejected"):
            audit_pair(episode(base, 1), episode(candidate, 1), IDS[0])

    def test_selected_call_must_match_declared_candidate(self):
        base, candidate = synthetic_run(), synthetic_run(IDS[0])
        event(candidate, "skill_tree_decision", 1)["selected_call"] = {"operation": "start", "skill": "cover_retreat"}
        with self.assertRaises(ValueError):
            audit_pair(episode(base, 1), episode(candidate, 1), IDS[0])

    def test_later_intervention_cannot_hide_behind_nonopportunity_flag(self):
        base, candidate = synthetic_run(), synthetic_run(IDS[0])
        later = event(candidate, "skill_tree_decision", 5)
        later["selected_id"] = IDS[0]
        later["selected_call"] = event(candidate, "skill_tree_decision", 1)["selected_call"]
        with self.assertRaises(ValueError):
            audit_pair(episode(base, 1), episode(candidate, 1), IDS[0])


if __name__ == "__main__":
    unittest.main()
