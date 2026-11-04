"""Study preparation and actual adapter wiring without launching the game."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from medivac_tree import MedivacTree
from medivac_tree.scripts.return_study import ROOT, command, prepare, read, write
from medivac_tree.tests.fixtures import observe_heal, payload, scene, unit
from sc2_skill_adapter.bridge.runner import parser
from sc2_skill_adapter.bridge.skill_runtime import load_manager, manager_configuration


class ReturnStudyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.study = self.root / "study"
        self.manifest = ROOT / "sc2_skill_adapter/configs/terran_g4_screen_n40_seed400.json"
        self.tree = ROOT / "medivac_tree/tree_v0.json"

    def prepare(self):
        return prepare(self.study, self.manifest, self.tree,
                       "HEAL_NEW", ["KEEP_HEAL", "HEAL_CURRENT"])

    def test_prepare_is_offline_and_preserves_base_tree(self):
        with patch("subprocess.run", side_effect=AssertionError("prepare must not launch processes")):
            result = self.prepare()
        self.assertEqual(result["started_games"], 0)
        base, candidate = read(self.study / "base.json"), read(self.study / "candidate.json")
        self.assertEqual(base["base_tree"], read(self.tree))
        candidate["selection"]["mode"] = "base"
        self.assertEqual(base, candidate)
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_prepared_command_reaches_actual_manager_and_candidate(self):
        self.prepare()
        actions_by_arm = {}
        for arm in ("base", "candidate"):
            argv = command(self.study, arm, self.root / arm, "C:/StarCraft II")
            args = parser().parse_args(argv[2:])
            config = manager_configuration(args)
            runtime = config["manager_kwargs"]["medivac_tree"]
            self.assertEqual(runtime.metadata["selection_mode"], arm)
            manager = load_manager()(**config["manager_kwargs"])
            manager.select("adaptive_focus_fire", posture="balanced")
            state = scene([unit("Medivac"), unit("Marine", (1, 0), health=30),
                           unit("Marine", (1, 1), health=30)])
            observe_heal(state, "u0", "u1", keep=False)
            actions, trace = manager.step(payload(state), 3)
            decision = trace["race_telemetry"]["medivac_tree"]["decisions"]["u0"]
            self.assertEqual(decision["return_selection"]["selected"], arm.upper())
            self.assertEqual(decision["action"], actions[0])
            actions_by_arm[arm] = actions
        self.assertEqual(actions_by_arm["base"][1:], actions_by_arm["candidate"][1:])
        self.assertNotEqual(actions_by_arm["base"][0], actions_by_arm["candidate"][0])

    def test_launcher_rejects_changed_policy_and_source_and_existing_output(self):
        self.prepare()
        with patch("medivac_tree.scripts.return_study.frozen_sources", return_value={"changed": "yes"}):
            with self.assertRaisesRegex(ValueError, "sources changed"):
                command(self.study, "base", self.root / "run", "C:/StarCraft II")
        (self.root / "run").mkdir()
        with self.assertRaises(FileExistsError):
            command(self.study, "base", self.root / "run", "C:/StarCraft II")
        value = read(self.study / "candidate.json")
        value["candidate"]["min_game_loop"] = 80
        write(self.study / "candidate.json", value)
        with self.assertRaisesRegex(ValueError, "policy changed"):
            command(self.study, "candidate", self.root / "new-run", "C:/StarCraft II")

    def test_repeat_base_reuses_policy_and_selector_requires_trained_policy(self):
        self.prepare()
        argv = command(self.study, "base-repeat", self.root / "repeat", "C:/StarCraft II")
        self.assertIn(str(self.study / "base.json"), argv)
        with self.assertRaisesRegex(ValueError, "requires --policy"):
            command(self.study, "selector", self.root / "selector", "C:/StarCraft II")
        with self.assertRaisesRegex(ValueError, "mode"):
            command(self.study, "selector", self.root / "selector", "C:/StarCraft II",
                    self.study / "base.json")

    def test_idle_profile_is_frozen_and_can_be_loaded_by_adapter(self):
        prepare(self.study, self.manifest, self.tree, profile="idle-recovery")
        document = read(self.study / "base.json")
        self.assertEqual(document["candidate"]["generator"], "idle_recovery")
        self.assertEqual(document["candidate"]["base_intents"], ["STOP"])
        self.assertEqual(document["candidate"]["event"]["min_stop_windows"], 3)
        argv = command(self.study, "candidate", self.root / "candidate", "C:/StarCraft II")
        runtime = manager_configuration(parser().parse_args(argv[2:]))["manager_kwargs"]["medivac_tree"]
        self.assertEqual(runtime.metadata["candidate"], document["candidate"])
        self.assertEqual(runtime.metadata["feature_schema"]["version"], 2)

    def test_idle_profile_rejects_unrelated_intent_configuration(self):
        with self.assertRaisesRegex(ValueError, "STOP"):
            prepare(self.study, self.manifest, self.tree, "HEAL_NEW", ["HEAL_CURRENT"],
                    profile="idle-recovery")
        self.assertFalse(self.study.exists())


if __name__ == "__main__":
    unittest.main()
