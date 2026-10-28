"""Exercise the public CLI through evaluate without SC2 or model requests."""
import asyncio
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from bridge import runner, skill_runtime
from bridge.config import load_config
from bridge.scenarios import generate_manifest
from test_bridge import DATA, FakeEnv as SnapshotEnv, unit
from bridge.policy import build_request


class SkillConfigurationTests(unittest.TestCase):
    def arguments(self, directory, extra=()):
        return runner.parser().parse_args([
            "run", "--method", "adaptive-skill", "--manifest", "unused.json",
            "--output-dir", str(directory), *extra])

    def test_default_and_old_namespaces_remain_off(self):
        args = self.arguments("unused")
        self.assertEqual(skill_runtime.unit_value_parameters(args)["mode"], "off")
        self.assertEqual(skill_runtime.unit_value_parameters(NS())["mode"], "off")
        self.assertFalse(args.adaptive_coordinated_approach)
        self.assertFalse(args.adaptive_safe_reentry)
        for key in ("accurate_damage", "critical_escape", "planned_reentry", "engagement_commitment", "approach_progress_guard"):
            self.assertFalse(getattr(args, "adaptive_" + key))

    def test_invalid_numeric_parameters_rejected_by_shared_validation(self):
        for flag, value in (("--unit-value-focus-value-weight", "2"),
                            ("--unit-value-scale", "nan"),
                            ("--unit-value-retreat-value-margin", "-1")):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.arguments("unused", [flag, value])

    def test_hashes_include_new_helper_without_changing_import_path(self):
        before = list(sys.path)
        self.arguments("unused")
        skill_runtime.load_manager()
        self.assertEqual(sys.path, before)
        self.assertIn(skill_runtime.SOURCE / "unit_value.py", skill_runtime.source_files())
        self.assertIn(skill_runtime.SOURCE / "skills/engagement_control.py", skill_runtime.source_files())

    def test_current_and_baseline_load_from_independent_sources_without_path_pollution(self):
        before = list(sys.path)
        current = skill_runtime.load_manager()
        baseline = skill_runtime.load_manager("baseline_v2_70pct")
        self.assertIs(current, skill_runtime.load_manager("current"))
        self.assertIsNot(current, baseline)
        for version, manager in (("current", current), ("baseline_v2_70pct", baseline)):
            module = skill_runtime._load_module("skill_manager", version)
            self.assertEqual(Path(module.__file__).parent, skill_runtime.SOURCES[version])
            self.assertIs(module.SkillManager, manager)
            self.assertTrue(all(path.is_relative_to(skill_runtime.SOURCES[version])
                                for path in skill_runtime.source_files(version)))
            control = manager(combat_params={"window_fire": True, "range_preserving_move": True,
                                              "target_scoring": True})
            payload = build_request(SnapshotEnv(), DATA, "unused", 1, {}, {}, {})
            control.select("adaptive_focus_fire", posture="balanced")
            actions, _ = control.step(payload, 5)
            self.assertEqual(len(actions), 5)
            for actor, question in payload["questions"].items():
                self.assertIn(str(actions[int(actor[1:])]), question["criteria"])
        self.assertEqual(sys.path, before)

    def test_unknown_skill_version_is_rejected(self):
        for load in (lambda: skill_runtime.load_manager("unknown"),
                     lambda: skill_runtime._load_module("skill_manager", "unknown"),
                     lambda: skill_runtime.source_files("unknown")):
            with self.assertRaisesRegex(ValueError, "Unknown skill version"):
                load()

    def test_cli_reaches_manager_and_persists_effective_configuration(self):
        captured = {}

        class CapturingManager(skill_runtime.load_manager()):
            def __init__(self, **kwargs):
                captured.update(kwargs)
                super().__init__(**kwargs)

        class FakeEnv:
            def __init__(self, config):
                self.n_agents = self.n_enemies = 5
                self.agents = {i: unit(100 + i, i) for i in range(5)}
                self.enemies = {i: unit(200 + i, i + 5) for i in range(5)}
                self._obs = NS(observation=NS(game_loop=2))
                self._step_mul = 8
                self.env = self
                self._controller = NS(data=lambda: None,
                                      ping=lambda: NS(game_version="fake"),
                                      save_replay=lambda: b"test-replay")
            def reset_record(self, scenario):
                pass
            def get_avail_agent_actions(self, i):
                return [1] * 11
            def step(self, actions):
                self._obs.observation.game_loop += 8
                return 1, True, {"battle_won": True, "dead_enemies": 5, "dead_allies": 0}
            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "run"
            flags = ["--unit-value-mode", "contextual", "--unit-value-focus-value-weight", "0.31",
                     "--unit-value-w-matchup", "0.3", "--unit-value-w-support", "0.4",
                     "--unit-value-w-scarcity", "0.1", "--unit-value-scale", "3",
                     "--unit-value-focus-switch-margin", "0.07",
                     "--unit-value-retreat-value-margin", "0.09",
                     "--unit-value-retreat-threshold-upper-bound", "0.7",
                     "--unit-value-heal-value-weight", "0.12",
                     "--unit-value-spatial-context", "--adaptive-coordinated-approach",
                     "--adaptive-safe-reentry", "--adaptive-window-fire",
                     "--adaptive-range-preserving-move", "--adaptive-target-scoring",
                     "--adaptive-accurate-damage", "--adaptive-critical-escape",
                     "--adaptive-planned-reentry", "--adaptive-engagement-commitment",
                     "--adaptive-approach-progress-guard", "--enable-regroup",
                     "--enable-cohesive-search", "--enable-combat-positioning", "--auto-spatial-dispatch"]
            args = self.arguments(destination, flags)
            manifest = generate_manifest(load_config(5), 1, 42)
            payload = build_request(SnapshotEnv(), DATA, "unused", 1, {}, {}, {})
            payload["state"]["game_loop"] = 2
            with patch("bridge.skill_runtime.load_manager", return_value=CapturingManager), \
                    patch("bridge.environment.ReplayableEnv", FakeEnv), \
                    patch("bridge.policy.build_request", return_value=payload), \
                    patch("bridge.feedback.action_feedback", return_value={}), \
                    patch("bridge.client.JevClient", side_effect=AssertionError("model forbidden")), \
                    patch("bridge.planner.MapPlanner", side_effect=AssertionError("planner forbidden")), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = asyncio.run(runner.evaluate(args, manifest, "adaptive-skill", destination))
            effective = skill_runtime.unit_value_parameters(args)
            self.assertEqual(effective, {
                "mode": "contextual", "spatial_context": True,
                "w_matchup": .3, "w_support": .4, "w_scarcity": .1, "scale": 3.,
                "focus_value_weight": .31, "focus_switch_margin": .07,
                "retreat_value_margin": .09, "retreat_threshold_upper_bound": .7,
                "heal_value_weight": .12,
            })
            self.assertEqual(captured["unit_value_params"], effective)
            self.assertTrue(effective["spatial_context"])
            self.assertEqual(effective["focus_value_weight"], .31)
            self.assertTrue(captured["combat_params"]["coordinated_approach"])
            self.assertTrue(captured["combat_params"]["safe_reentry"])
            self.assertTrue(captured["combat_params"]["target_scoring"])
            for key in ("accurate_damage", "critical_escape", "planned_reentry", "engagement_commitment", "approach_progress_guard"):
                self.assertTrue(captured["combat_params"][key])
            meta = json.loads((destination / "run.json").read_text(encoding="utf-8"))
            stored = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
            for record in (meta, stored, result):
                self.assertEqual(record["skill_source"], str(skill_runtime.SOURCE))
                self.assertEqual(record["unit_value_parameters"], effective)
                self.assertEqual(record["unit_value_mode"], "contextual")
                self.assertTrue(record["unit_value_version"])
                self.assertEqual(record["skill_parameters"]["unit_value_parameters"], effective)
                self.assertEqual(record["skill_parameters"]["spatial_params"], captured["spatial_params"])
                self.assertTrue(record["skill_parameters"]["spatial_params"]["enable_regroup"])
                self.assertTrue(record["skill_parameters"]["safe_reentry"])
                for key in ("accurate_damage", "critical_escape", "planned_reentry", "engagement_commitment", "approach_progress_guard"):
                    self.assertTrue(record["skill_parameters"][key])
            self.assertIn("skills/current/unit_value.py",
                          {Path(name).as_posix() for name in meta["source_hashes"]})
            self.assertIn("skills/current/skills/engagement_control.py",
                          {Path(name).as_posix() for name in meta["source_hashes"]})
            events = [json.loads(line) for line in (destination / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            configuration = next(e for e in events if e["event"] == "unit_value_configuration")
            self.assertEqual(configuration["unit_value_parameters"], effective)
            step = next(e for e in events if e["event"] == "skill_step")
            self.assertTrue(step["engagement_commitment_enabled"])
            self.assertTrue(step["approach_progress_guard_enabled"])
            self.assertIn("engagement_diagnostics", step)
            self.assertEqual(result["jev_requests"], 0)
            self.assertEqual(result["status"], "complete")


if __name__ == "__main__":
    unittest.main()
