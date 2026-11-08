import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from decision_tree.scripts.precontact_eval_summary import enrich, read_events, summary


def decision(step, status):
    candidate = {"id": "regroup_step:u1:3", "actors": ["u1"], "duration_steps": 1,
                 "family": "regroup_step", "actions": {"u1": 3},
                 "features": {"nearest_enemy_before": 5.0, "nearest_ally_before": 8.0, "ally_alive_count": 10}}
    proposed = status != "no_proposal"
    return {"episode": 1, "event": "local_decision", "game_loop": 2+8*step, "step": step,
            "baseline_actions": [4, 4], "final_actions": [4, 3 if status == "executed" else 4],
            "proposed_id": candidate["id"] if proposed else "BASE",
            "selected_id": candidate["id"] if status == "executed" else "BASE",
            "vetoed": status == "vetoed", "candidates": [candidate]}


def events(status, rejected=False):
    log = []
    for t in range(2):
        d = decision(t, status if t == 0 else "no_proposal")
        log.extend([
            {"episode": 1, "step": t, "event": "skill_observation", "payload": {"state": {"episode": 1, "t": t}}},
            d,
            {"episode": 1, "step": t+1, "game_loop": d["game_loop"], "event": "step",
             "actions": d["final_actions"], "reward": 0, "terminated": t == 1,
             "feedback": {"actors": {"u1": {"requested_action_id": str(d["final_actions"][1]),
                                             "submission": "rejected" if rejected else "accepted"}}}},
        ])
    return log


def stream(log):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/"events.jsonl"
        raw = b"".join(json.dumps(e).encode()+b"\n" for e in log)
        path.write_bytes(raw)
        return read_events(path, hashlib.sha256(raw).hexdigest())[1]


def pair(status, seed, base_wins, model_wins):
    scene = {"sc2_seed": seed, "sha256": str(seed)}
    def episode(won):
        return {"scenario_sha256": str(seed), "initial_state_sha256": str(seed), "steps": 2,
                "outcome": "win" if won else "loss", "reward": 20 if won else 15}
    b, m = episode(base_wins), episode(model_wins)
    row = {"episode": 1, "group_id": str(seed), "sc2_seed": seed, "base_outcome": b["outcome"],
           "model_outcome": m["outcome"], "base_steps": 2, "model_steps": 2,
           "base_return": b["reward"], "model_return": m["reward"], "gain": m["reward"]-b["reward"],
           "proposals": int(status != "no_proposal"), "interventions": int(status == "executed"),
           "vetoes": int(status == "vetoed")}
    return enrich(row, scene, b, m, stream(events("no_proposal")), stream(events(status)))


class PrecontactEvalSummaryTests(unittest.TestCase):
    def test_census_no_proposal_veto_execute_and_flips(self):
        rows = [pair("no_proposal", 11, True, True), pair("vetoed", 12, False, False),
                pair("executed", 13, False, True), pair("executed", 14, True, False)]
        stats = summary(rows)
        self.assertEqual(stats["rescued_seeds"], [13])
        self.assertEqual(stats["lost_seeds"], [14])
        self.assertEqual(stats["proposals"], 3)
        self.assertEqual(stats["mean_gain"], 0)
        self.assertEqual(stats["by_proposal"]["executed"]["episodes"], 2)
        self.assertEqual(stats["observed_prefix_mismatch_seeds"], [])
        self.assertEqual(rows[2]["proposal"]["native_submission"], "accepted")
        self.assertEqual(rows[2]["base_later_candidate_steps_after_proposal"], 1)
        with self.assertRaisesRegex(ValueError, "repeated"):
            summary(rows + [rows[0]])

    def test_rejected_native_action_and_second_actor_fail(self):
        with self.assertRaisesRegex(ValueError, "native receipt"):
            stream(events("executed", rejected=True))
        log = events("executed")
        log[1]["final_actions"][0] = 3
        with self.assertRaisesRegex(ValueError, "scope mismatch"):
            stream(log)

    def test_intervention_gain_prefix_diff_is_flagged_not_filtered(self):
        row = pair("executed", 10, False, True)
        row["observed_prefix_matches"] = False
        stats = summary([row])
        self.assertEqual(stats["observed_prefix_mismatch_seeds"], [10])
        self.assertEqual(stats["independent_scene_count"], 1)
