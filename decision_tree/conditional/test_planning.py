"""Sampling tests use synthetic observations and labels, without native games."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from decision_tree.conditional.planning import FAMILIES, PHASES, _select_jobs, check_budget, make_plan
from decision_tree.phase2.policy import REGROUP_POLICY
from skills.local_v3.tests.test_audit import synthetic_run


def opportunity(group, family, phase, index=0, aliases=None):
    action = index + 2
    return {"scene_hash": group, "family": family, "step": PHASES.index(phase) * 10 + index,
            "effect": f"{family}-{phase}-{index}", "phase": phase, "preference": 1,
            "binding": f"{family}:u0:{action}", "effect_families": aliases or [family]}


class BoundedPlanningTests(unittest.TestCase):
    def test_family_coverage_temporal_rotation_and_determinism(self):
        rows = [opportunity(group, family, phase, index)
                for group in ("one", "two") for family in FAMILIES for phase in PHASES for index in range(2)]
        selected, coverage = _select_jobs(rows, {"one", "two"}, 1)
        self.assertEqual(len(selected), 8)
        for group in ("one", "two"):
            scene_rows = [r for r in selected if r["scene_hash"] == group]
            self.assertEqual({r["family"] for r in scene_rows}, set(FAMILIES))
            self.assertEqual({r["phase"] for r in scene_rows}, set(PHASES))
            self.assertTrue(all(r["phase"] == r["requested_phase"] for r in scene_rows))
        self.assertEqual((selected, coverage), _select_jobs(list(reversed(rows)), {"two", "one"}, 1))
        self.assertNotEqual(selected, _select_jobs(rows, {"one", "two"}, 2)[0])

    def test_missing_family_is_explicit_and_not_filled_by_extra_target_games(self):
        rows = [opportunity("one", "retarget", phase, i) for phase in PHASES for i in range(4)]
        selected, coverage = _select_jobs(rows, {"one"}, 0)
        self.assertEqual(len(selected), 1)
        self.assertEqual(sum(r["status"] == "no_distinct_alternative" for r in coverage), 3)
        with self.assertRaisesRegex(ValueError, "Holdout"):
            _select_jobs(rows, {"different"}, 0)

    def test_alias_and_scene_caps(self):
        rows = [opportunity("one", family, "early", i, ["retarget", family])
                for i, family in enumerate(FAMILIES)]
        # Put all effects at the same time so the shared family cap applies.
        for row in rows:
            row["step"] = 0
        selected, _ = _select_jobs(rows, {"one"}, 0)
        self.assertLessEqual(len(selected), 2)
        with self.assertRaisesRegex(ValueError, "two branches"):
            check_budget(rows)
        five = [opportunity("one", FAMILIES[i % 4], "early", i) for i in range(5)]
        with self.assertRaisesRegex(ValueError, "four measured"):
            check_budget(five)

    def test_source_plan_has_features_and_excludes_holdout_jobs_and_outcome_ranking(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, context, events = synthetic_run(root)
            (root / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            training = {manifest["episodes"][0]["sha256"]}
            with patch.dict(REGROUP_POLICY["model"], {"context_hash": context}):
                first = make_plan(root, training, 1)
            self.assertEqual(first["census"]["train"]["decision_steps"], 3)
            self.assertEqual(first["census"]["holdout"]["decision_steps"], 3)
            self.assertEqual(first["planned_branch_games"], 1)
            for row in first["jobs"]:
                self.assertIn(row["scene_hash"], training)
                self.assertEqual(set(row["features"]), set(first["feature_names"]))
                self.assertIn(row["label_kind"], ("gain", "veto"))
            changed = deepcopy(events)
            for row in changed:
                if row["event"] == "step" and row["terminated"]:
                    row["reward"] = 42
                elif row["event"] == "episode_completed":
                    row["reward"] = 42
            records = json.loads((root / "episodes.json").read_text(encoding="utf-8"))
            for row in records:
                row["reward"] = 42
            (root / "episodes.json").write_text(json.dumps(records), encoding="utf-8")
            (root / "events.jsonl").write_text("\n".join(json.dumps(e) for e in changed), encoding="utf-8")
            with patch.dict(REGROUP_POLICY["model"], {"context_hash": context}):
                second = make_plan(root, training, 1)
            self.assertEqual(first["jobs"], second["jobs"])


if __name__ == "__main__":
    unittest.main()
