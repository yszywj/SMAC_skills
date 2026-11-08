"""Study preflight tests: real manifests and metadata, never an SC2 process."""

import asyncio
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from decision_tree.experiment import digest, execution_context, expected_context
from decision_tree.features import FEATURE_NAMES
from decision_tree.schema import load_policy
from decision_tree.scripts import study
from sc2_skill_adapter.bridge import runner


class NoGameEnvironment(RuntimeError):
    """Raised instead of constructing the game's environment."""


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source_manifest = study.ROOT / "sc2_skill_adapter/configs/protoss_5_vs_5_smoke.json"
        self.manifest = study.read(source_manifest)
        self.manifest["episodes"] = self.manifest["episodes"][:1]
        self.manifest["sha256"] = digest({k: v for k, v in self.manifest.items() if k != "sha256"})
        self.manifest_path = self.root / "input_manifest.json"
        self.write(self.manifest_path, self.manifest)
        self.policy = load_policy(study.ROOT / "decision_tree/configs/policy_v1.json")
        self.policy_path = self.root / "input_policy.json"
        self.write(self.policy_path, self.policy)
        # Freeze an owned fixture, not mutable files being edited by other work.
        # The actual SC2 map is still included by prepare itself.
        self.source = self.root / "fixture_source.py"
        self.source.write_text("FIXTURE = 1\n", encoding="utf-8")

    def write(self, path, value):
        Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")

    def prepare(self, name="study", **kwargs):
        frozen = {str(self.source): hashlib.sha256(self.source.read_bytes()).hexdigest()}
        destination = self.root / name
        with patch.object(study, "sources", return_value=frozen):
            result = study.prepare(self.manifest_path, self.policy_path, destination, **kwargs)
        self.assertEqual(result["started_games"], 0)
        return destination

    def args(self, prepared, output="unused", flags=None):
        return runner.parser().parse_args([
            "run", *(study.DEFAULT_FLAGS if flags is None else flags),
            "--manifest", str(prepared / "manifest.json"),
            "--skill-tree", str(prepared / "base.json"),
            "--output-dir", str(self.root / output)])

    def model_policy(self, prepared, *, overlap=False, wrong_context=False):
        policy = load_policy(prepared / "base.json")
        manifest = study.read(prepared / "manifest.json")
        context = expected_context(self.args(prepared), manifest, policy)
        if wrong_context:
            context["skill_parameters"]["max_steps"] += 1
        groups = [manifest["episodes"][0]["sha256"]] if overlap else [digest({"training": "another scenario"})]
        model = {"schema_version": 1, "kind": "skill_gain_tree",
                 "game_version": "fixture-not-launched",
                 "candidate_ids": [c["id"] for c in policy["candidates"]],
                 "feature_names": list(FEATURE_NAMES),
                 "root": {"gains": {c["id"]: 0.0 for c in policy["candidates"]}},
                 "context": context, "context_hash": digest(context),
                 "training": {"train_groups": groups, "validation_groups": []}}
        policy["selection"] = {"mode": "model", "model": model,
                               "max_interventions_per_episode": 1, "min_gain": 0}
        path = self.root / "fitted_policy.json"
        self.write(path, load_policy(policy))
        return path

    def test_prepare_and_collect_dryrun_create_no_games_or_output(self):
        with patch.object(study.subprocess, "run") as launch:
            prepared = self.prepare()
            destination = self.root / "collection"
            result = study.execute(prepared, destination, dry_run=True)
        launch.assert_not_called()
        self.assertFalse(destination.exists())
        self.assertEqual(result["started_games"], 0)
        self.assertEqual(result["planned_games"], 3)
        self.assertEqual(set(result["commands"]), {"base", "focus_weakest", "protect"})
        for arm, command in result["commands"].items():
            self.assertEqual(command[command.index("--skill-tree") + 1], str(prepared / (arm + ".json")))
        self.assertEqual(study.read(prepared / "manifest.json")["split"], "development")

    def test_source_change_prevents_any_subprocess(self):
        prepared = self.prepare()
        self.source.write_text("FIXTURE = 2\n", encoding="utf-8")
        destination = self.root / "bad_source"
        with patch.object(study.subprocess, "run") as launch:
            with self.assertRaisesRegex(ValueError, "Frozen source changed"):
                study.execute(prepared, destination)
        launch.assert_not_called()
        self.assertFalse(destination.exists())

    def test_arm_tamper_prevents_launching_even_base(self):
        prepared = self.prepare()
        candidate = study.read(prepared / "protect.json")
        candidate["selection"]["candidate_id"] = "focus_weakest"
        self.write(prepared / "protect.json", candidate)
        with patch.object(study.subprocess, "run") as launch:
            with self.assertRaisesRegex(ValueError, "selection changed"):
                study.execute(prepared, self.root / "bad_arm")
        launch.assert_not_called()

    def test_heldout_manifest_cannot_be_relabelled_or_collected(self):
        self.manifest["split"] = "test"
        self.manifest["sha256"] = digest({k: v for k, v in self.manifest.items() if k != "sha256"})
        self.write(self.manifest_path, self.manifest)
        with self.assertRaisesRegex(ValueError, "relabel"):
            self.prepare()
        prepared = self.prepare(split="test")
        with patch.object(study.subprocess, "run") as launch:
            with self.assertRaisesRegex(ValueError, "development"):
                study.execute(prepared, self.root / "test_collection")
        launch.assert_not_called()

    def test_evaluation_overlap_is_rejected_before_any_subprocess(self):
        prepared = self.prepare(split="validation")
        model = self.model_policy(prepared, overlap=True)
        with patch.object(study.subprocess, "run") as launch:
            with self.assertRaisesRegex(ValueError, "overlaps"):
                study.execute(prepared, self.root / "overlap", mode="evaluate", policy=model)
        launch.assert_not_called()

    def test_context_mismatch_is_rejected_before_any_subprocess(self):
        prepared = self.prepare(split="validation")
        model = self.model_policy(prepared, wrong_context=True)
        destination = self.root / "bad_context"
        with patch.object(study.subprocess, "run") as launch:
            with self.assertRaisesRegex(ValueError, "another execution context"):
                study.execute(prepared, destination, mode="evaluate", policy=model)
        launch.assert_not_called()
        self.assertFalse(destination.exists())

    def test_missing_or_invalid_game_version_rejected_before_any_subprocess(self):
        prepared = self.prepare(split="validation")
        model_path = self.model_policy(prepared)
        original = study.read(model_path)
        for number, value in enumerate((None, 5, "", "   ")):
            with self.subTest(game_version=value):
                changed = deepcopy(original)
                if value is None:
                    changed["selection"]["model"].pop("game_version")
                else:
                    changed["selection"]["model"]["game_version"] = value
                self.write(model_path, changed)
                destination = self.root / f"bad_game_version_{number}"
                with patch.object(study.subprocess, "run") as launch:
                    with self.assertRaisesRegex(ValueError, "game_version provenance"):
                        study.execute(prepared, destination, mode="evaluate", policy=model_path)
                launch.assert_not_called()
                self.assertFalse(destination.exists())

    def test_valid_model_evaluation_dryrun_preflights_both_arms(self):
        prepared = self.prepare(split="validation")
        model = self.model_policy(prepared)
        with patch.object(study.subprocess, "run") as launch:
            result = study.execute(prepared, self.root / "evaluation", mode="evaluate", policy=model, dry_run=True)
        launch.assert_not_called()
        self.assertEqual(set(result["commands"]), {"base", "model"})
        self.assertEqual(result["planned_games"], 2)
        self.assertEqual(result["started_games"], 0)

    def test_flags_rejected_before_study_creation(self):
        variants = [
            ["--method", "nearest"],
            [*study.DEFAULT_FLAGS, "--native-teacher-control", "all"],
            [*study.DEFAULT_FLAGS, "--skill-max-steps", "0"],
            [*study.DEFAULT_FLAGS, "--manifest", str(self.manifest_path)],
            [*study.DEFAULT_FLAGS, "--output-dir=somewhere"],
            [*study.DEFAULT_FLAGS, "--skill-tree", str(self.policy_path)],
            [*study.DEFAULT_FLAGS, "--sc2-path", "somewhere"],
            [*study.DEFAULT_FLAGS, "--race-profile", "terran"],
            [*study.DEFAULT_FLAGS, "--terran-keep-heal-order"],
            [*study.DEFAULT_FLAGS, "--sc2-p", "somewhere"],
            [*study.DEFAULT_FLAGS, 123],
            "--method adaptive-skill",
        ]
        for number, flags in enumerate(variants):
            with self.subTest(flags=flags), redirect_stderr(io.StringIO()):
                with self.assertRaises((ValueError, SystemExit)):
                    self.prepare(name=f"flags_{number}", flags=flags)
                self.assertFalse((self.root / f"flags_{number}").exists())

    def test_expected_context_matches_actual_runner_metadata_without_environment(self):
        # Exercise real runner metadata construction, then intercept environment
        # construction. No subprocess/game/network operation is reached.
        prepared = self.prepare()
        manifest = study.read(prepared / "manifest.json")
        policy = load_policy(prepared / "base.json")
        args = self.args(prepared, "metadata_only")
        want = expected_context(args, manifest, policy)
        from sc2_skill_adapter.bridge import environment
        with patch.object(environment, "ReplayableEnv", side_effect=NoGameEnvironment("metadata only")) as create_env:
            with patch.dict(os.environ, {}, clear=False):
                with self.assertRaises(NoGameEnvironment):
                    asyncio.run(runner.evaluate(args, manifest, "adaptive-skill", self.root / "metadata_only"))
        create_env.assert_called_once()
        metadata = study.read(self.root / "metadata_only/run.json")
        actual = execution_context(metadata)
        self.assertEqual(want, actual)
        self.assertEqual(set(want["source_hashes"]), set(metadata["source_hashes"]))
        self.assertTrue(any("decision_tree" in path for path in actual["source_hashes"]))


if __name__ == "__main__":
    unittest.main()
