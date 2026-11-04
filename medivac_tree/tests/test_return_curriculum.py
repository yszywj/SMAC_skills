"""Freeze scenario splits independently of outcomes without launching SC2."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from medivac_tree.scripts.prepare_curriculum import (ROOT, InsufficientCurriculumError,
                                                    prepare, read, previous_scenarios)
from sc2_skill_adapter.bridge.config import digest
from sc2_skill_adapter.bridge.scenarios import generate_manifest, validate_manifest


class ReturnCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = ROOT / "sc2_skill_adapter/configs/terran_g4_screen_n40_seed400.json"
        self.env = read(self.config)[0]["env_args"]

    def pool(self, count=12):
        pool = generate_manifest(self.env, count, 91700)
        for row in pool["episodes"]:
            roster = row["episode_config"]["team_gen"]["ally_team"]
            row["episode_config"]["team_gen"]["ally_team"] = ["medivac", *(["marine"] * (len(roster) - 1))]
            row["sha256"] = digest(row["episode_config"])
        pool["sha256"] = digest({k: v for k, v in pool.items() if k != "sha256"})
        return pool

    def test_real_generation_is_offline_reproducible_and_has_three_disjoint_splits(self):
        with patch("subprocess.run", side_effect=AssertionError("preparation must not launch games")):
            first = prepare(self.root / "first", self.config, run_roots=[])
            second = prepare(self.root / "second", self.config, run_roots=[])
        self.assertEqual(first["stats"], second["stats"])
        self.assertEqual(first["stats"]["started_games"], 0)
        all_groups = set()
        for split, count in (("development", 40), ("validation", 20), ("test", 20)):
            a, b = read(first["manifests"][split])[0], read(second["manifests"][split])[0]
            self.assertEqual(a, b)
            validate_manifest(a)
            self.assertEqual(len(a["episodes"]), count)
            groups = {r["sha256"] for r in a["episodes"]}
            self.assertEqual(len(groups), count)
            self.assertFalse(all_groups & groups)
            all_groups.update(groups)
            self.assertTrue(all(r["medivac_count"] >= 1 for r in a["episodes"]))
        declaration = read(self.root / "first/split.json")[0]
        self.assertEqual(declaration["sha256"], digest({k: v for k, v in declaration.items() if k != "sha256"}))
        self.assertTrue(declaration["splits"]["validation"]["sealed_holdout"])
        self.assertTrue(declaration["splits"]["test"]["sealed_holdout"])
        self.assertTrue(all(not split["automatic_execution"] for split in declaration["splits"].values()))
        with self.assertRaises(FileExistsError):
            prepare(self.root / "first", self.config, run_roots=[])

    def test_only_roster_and_hashes_control_filtering_and_fixed_order(self):
        pool = self.pool()
        pool["episodes"][0]["episode_config"]["team_gen"]["ally_team"] = ["marine"] * self.env["capability_config"]["n_units"]
        pool["episodes"][0]["sha256"] = digest(pool["episodes"][0]["episode_config"])
        pool["episodes"][3] = {**deepcopy(pool["episodes"][1]), "index": 3, "sc2_seed": 91703}
        pool["sha256"] = digest({k: v for k, v in pool.items() if k != "sha256"})
        prior = deepcopy(pool)
        prior["episodes"] = [deepcopy(pool["episodes"][2])]
        prior["sha256"] = digest({k: v for k, v in prior.items() if k != "sha256"})
        prior_root = self.root / "runs/old-run"
        prior_root.mkdir(parents=True)
        (prior_root / "manifest.json").write_text(json.dumps(prior), encoding="utf-8")
        # Deliberately unreadable result files: selection must never read them.
        (prior_root / "summary.json").write_text("NOT JSON; do not inspect outcomes", encoding="utf-8")
        (prior_root / "events.jsonl").write_text("NOT JSON; do not inspect trigger rates", encoding="utf-8")
        with patch("medivac_tree.scripts.prepare_curriculum.generate_manifest", return_value=pool) as generated:
            result = prepare(self.root / "study", self.config, pool_size=12,
                             dev_episodes=2, validation_episodes=2, test_episodes=2,
                             run_roots=[self.root / "runs"])
        self.assertEqual(generated.call_count, 1)
        self.assertEqual(result["stats"]["without_medivac"], 1)
        self.assertEqual(result["stats"]["previously_recorded"], 1)
        self.assertEqual(result["stats"]["duplicate_in_pool"], 1)
        selected = [r["source_index"] for name in ("development", "validation", "test")
                    for r in read(result["manifests"][name])[0]["episodes"]]
        self.assertEqual(selected, [1, 4, 5, 6, 7, 8])
        split = read(self.root / "study/split.json")[0]
        self.assertEqual(split["excluded_scenario_hashes"], [pool["episodes"][2]["sha256"]])
        self.assertEqual(split["exclusion_sources"][0]["sha256"], read(prior_root / "manifest.json")[1])

    def test_shortage_keeps_fixed_pool_and_never_writes_partial_splits(self):
        with patch("medivac_tree.scripts.prepare_curriculum.generate_manifest", return_value=self.pool(2)) as generated:
            with self.assertRaises(InsufficientCurriculumError) as failure:
                prepare(self.root / "missing", self.config, pool_size=2,
                        dev_episodes=1, validation_episodes=1, test_episodes=1, run_roots=[])
        self.assertEqual(generated.call_count, 1)
        self.assertEqual(failure.exception.stats["eligible_unique_scenarios"], 2)
        self.assertEqual(failure.exception.stats["requested_scenarios"], 3)
        self.assertFalse((self.root / "missing").exists())

    def test_existing_sealed_curriculum_can_be_excluded_as_a_whole(self):
        pool = self.pool(8)
        with patch("medivac_tree.scripts.prepare_curriculum.generate_manifest", return_value=pool):
            first = prepare(self.root / "first", self.config, pool_size=8,
                            dev_episodes=1, validation_episodes=1, test_episodes=1, run_roots=[])
            second = prepare(self.root / "second", self.config, pool_size=8,
                             dev_episodes=1, validation_episodes=1, test_episodes=1,
                             run_roots=[self.root / "first"])
        first_groups = {r["sha256"] for p in first["manifests"].values() for r in read(p)[0]["episodes"]}
        second_groups = {r["sha256"] for p in second["manifests"].values() for r in read(p)[0]["episodes"]}
        self.assertFalse(first_groups & second_groups)
        self.assertEqual(second["stats"]["excluded_prior_unique_scenarios"], 3)

    def test_invalid_counts_and_tampered_exclusion_manifest_fail_before_writes(self):
        for config in ({"dev_episodes": 41}, {"validation_episodes": 0}, {"test_episodes": True},
                       {"pool_size": 0}, {"seed": -1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                prepare(self.root / "invalid", self.config, run_roots=[], **config)
        prior = self.root / "tampered"
        prior.mkdir()
        pool = self.pool()
        pool["episodes"][0]["sc2_seed"] += 1
        (prior / "manifest.json").write_text(json.dumps(pool), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash"):
            previous_scenarios([prior])
        self.assertFalse((self.root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
