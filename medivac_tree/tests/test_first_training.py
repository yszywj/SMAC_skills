from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from medivac_tree.scripts.first_training import execute, plan
from medivac_tree.scripts.return_study import ROOT, prepare, read, write


class FirstTrainingWorkflowTests(unittest.TestCase):
    def test_dry_run_validates_two_arms_without_launching_or_creating_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study, output = root / "study", root / "output"
            prepare(study, ROOT / "sc2_skill_adapter/configs/terran_g4_screen_n40_seed400.json",
                    ROOT / "medivac_tree/tree_v0.json", profile="idle-recovery")
            with patch("subprocess.run", side_effect=AssertionError("dry-run launched a process")):
                result = execute(study, output, dry_run=True)
            self.assertEqual(result["planned_games"], 80)
            self.assertEqual(result["started_games"], 0)
            self.assertEqual(len(result["commands"]), 2)
            self.assertFalse(output.exists())

    def test_failed_first_arm_stops_collection_before_second_arm_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("medivac_tree.scripts.first_training.plan", return_value=[["first"], ["second"]]), \
                    patch("medivac_tree.scripts.first_training.read", return_value={"episodes_per_arm": 4}), \
                    patch("medivac_tree.scripts.first_training.check_workspace_inputs"), \
                    patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, ["first"])) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    execute(root / "study", root / "output")
            self.assertEqual(run.call_count, 1)
            self.assertFalse((root / "output/paired.json").exists())

    def test_sealed_holdout_cannot_be_collected_and_used_groups_cannot_be_evaluated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("validation", "test"):
                with patch("medivac_tree.scripts.first_training.read", return_value={"split": split}):
                    with self.assertRaisesRegex(ValueError, "Sealed"):
                        plan(root, root / "output")
            manifest = {"episodes": [{"sha256": "seen"}]}
            policy = {"selection": {"model": {"training": {"split": {
                "training_groups": ["seen"], "validation_groups": ["other"]}}}}}
            with patch("medivac_tree.scripts.first_training.read", side_effect=[manifest, policy]):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    plan(root, root / "output", mode="evaluate", policy=root / "model.json")

    def test_outside_workspace_is_rejected_before_any_arm_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("medivac_tree.scripts.first_training.plan", return_value=[["first"], ["second"]]), \
                    patch("medivac_tree.scripts.first_training.read", return_value={"episodes_per_arm": 4}), \
                    patch("subprocess.run") as run:
                with self.assertRaisesRegex(ValueError, "workspace"):
                    execute(root, root / "output")
                run.assert_not_called()
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
