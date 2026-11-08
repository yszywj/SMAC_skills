"""Regressions for one-step policy reconstruction and missing evidence."""
import tempfile
import unittest
from pathlib import Path

from decision_tree.phase2.evidence import compare, measured_rows, stats
from skills.local_v1 import FEATURE_NAMES
from decision_tree.local.runtime import digest
from decision_tree.phase2.evidence import load_cohorts, file_sha
import json


def candidate(distance):
    features = dict.fromkeys(FEATURE_NAMES, 0.0)
    features.update(ally_alive_count=10.0, nearest_enemy_before=float(distance))
    return {"id": "regroup_step:u1:3", "family": "regroup_step", "features": features}


def scene(group, measured):
    return {"group_id": group, "sc2_seed": 81, "base_return": 15.0, "base_outcome": "loss",
            "frames": [{"step": step, "team_fired_before": step > 0, "candidates": [candidate(distance)]}
                       for step, distance in enumerate([14, 9])],
            "measured": [{"step": 1, "candidate_id": "regroup_step:u1:3", "outcome": "win",
                          "return": 20.0, "source_run": "fixture"}] if measured else []}


class EvidenceTests(unittest.TestCase):
    def test_wait_executes_later_on_same_base_prefix(self):
        waiting = {"mode": "regroup", "rules": {"far": "wait"}}
        report = compare([scene("g", True)], waiting, {"mode": "regroup"})
        row = report["rows"][0]
        self.assertEqual(row["candidate"]["status"], "forced_measured")
        self.assertEqual(row["candidate"]["step"], 1)
        self.assertEqual(len(row["candidate"]["waits"]), 1)
        self.assertEqual(row["incumbent"]["status"], "decline_base")
        self.assertEqual(report["differences"]["incumbent"]["rescues"], [81])

    def test_never_impute_unmeasured_branch_as_draw(self):
        waiting = {"mode": "regroup", "rules": {"far": "wait"}}
        report = compare([scene("g", False)], waiting, {"mode": "base"})
        self.assertEqual(report["rows"][0]["candidate"]["status"], "unmeasured")
        self.assertIsNone(report["rows"][0]["candidate"]["return"])
        self.assertFalse(report["differences"]["base"]["complete"])
        self.assertEqual(report["differences"]["base"]["groups"], 0)
        self.assertIsNone(report["differences"]["base"]["scene_equal_mean_gain"])

    def test_unreached_forced_anchor_is_not_zero(self):
        forced = {"mode": "forced", "candidate_id": "regroup_step:u1:3", "anchor_step": 6}
        report = compare([scene("g", True)], forced, {"mode": "base"})
        row = report["rows"][0]["candidate"]
        self.assertEqual(row["reason"], "anchor_not_reached")
        self.assertIsNone(row["return"])

    def test_other_bindings_are_not_old_selector_labels(self):
        row, = measured_rows(scene("g", True))
        self.assertEqual(row["bucket"], "near_engaged")
        self.assertFalse(row["old_first_proposal"])

    def test_equal_scene_weight_and_leave_one_out(self):
        rows = [{"group_id": "a", "gain": x} for x in (1, 2, 3)]
        rows.append({"group_id": "b", "gain": -4})
        result = stats(rows)
        self.assertEqual(result["scene_equal_mean_gain"], -1)
        self.assertEqual(result["leave_one_group_out_range"], [-4, 2])

    def test_leakage_across_batch_train_and_holdout_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, (group, holdout) in enumerate([("a", "b"), ("b", "z")]):
                split = Path(tmp)/f"split{i}.json"
                split.write_text(json.dumps({"training_group_ids": [group], "validation_group_ids": [holdout]}))
                data = Path(tmp)/f"data{i}.json"
                data.write_text("{}")
                conf = {"kind":"phase2_training_cohort", "split":str(split), "split_sha":file_sha(split),
                        "dataset":str(data), "dataset_sha":file_sha(data), "context":"same", "game_version":"test",
                        "scenes":[{"group_id":group,"sc2_seed":i}]}
                conf["record_sha256"]=digest(conf)
                path=Path(tmp)/f"prepared{i}.json"
                path.write_text(json.dumps(conf))
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "leaked groups"):
                load_cohorts(paths)


if __name__ == "__main__":
    unittest.main()
