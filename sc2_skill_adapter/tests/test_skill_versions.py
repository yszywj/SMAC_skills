"""Run both frozen skill implementations through the adapter without SC2 or APIs."""
import asyncio
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from bridge import runner, skill_runtime
from bridge.config import load_config
from bridge.policy import build_request
from bridge.scenarios import generate_manifest
from test_bridge import DATA, FakeEnv as SnapshotEnv, unit


LEGACY_FLAGS = ["--adaptive-" + key.replace("_", "-")
                for key in skill_runtime.LEGACY_COMBAT_PARAMETERS]
NEW_FLAGS = ["--adaptive-" + key.replace("_", "-")
             for key in skill_runtime.NEW_COMBAT_PARAMETERS]


class VersionEnv:
    actions = []

    def __init__(self, config):
        self.n_agents = self.n_enemies = 5
        self.agents = {i: unit(100 + i, i) for i in range(5)}
        self.enemies = {i: unit(200 + i, i + 5) for i in range(5)}
        self._obs = NS(observation=NS(game_loop=2))
        self._step_mul = 8
        self.env = self
        self._episode_steps = 0
        self._controller = NS(data=lambda: None, ping=lambda: NS(game_version="offline"),
                              save_replay=lambda: b"offline-fixture")

    def reset_record(self, scenario):
        pass

    def get_avail_agent_actions(self, index):
        return [1] * 11

    def step(self, actions):
        self.actions.append(list(actions))
        self._episode_steps += 1
        self._obs.observation.game_loop += 8
        return 1, self._episode_steps == 2, {
            "battle_won": True, "dead_enemies": 5, "dead_allies": 0}

    def close(self):
        pass


class SkillVersionTests(unittest.TestCase):
    def arguments(self, directory, flags=()):
        return runner.parser().parse_args([
            "run", "--method", "adaptive-skill", "--manifest", "unused.json",
            "--output-dir", str(directory), *flags])

    def test_both_actual_managers_execute_and_persist_only_their_own_configuration(self):
        manifest = generate_manifest(load_config(5), 2, 170)
        template = build_request(SnapshotEnv(), DATA, "unused", 1, {}, {}, {})

        def observation(env, *args):
            payload = copy.deepcopy(template)
            payload["state"]["game_loop"] = env._obs.observation.game_loop
            return payload

        for version in skill_runtime.SOURCES:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temporary:
                captured, resets = [], []
                implementation = skill_runtime.load_manager(version)

                class CapturingManager(implementation):
                    def __init__(self, **kwargs):
                        captured.append(copy.deepcopy(kwargs))
                        super().__init__(**kwargs)

                    def reset(self):
                        resets.append(True)
                        super().reset()

                flags = ["--skill-version", version, *LEGACY_FLAGS]
                if version != "baseline_v2_70pct":
                    supported = [f for f in NEW_FLAGS if version != "baseline_v6_20261024" or
                                 f != "--adaptive-approach-progress-guard"]
                    flags += [*supported, "--unit-value-mode", "contextual",
                              "--unit-value-spatial-context"]
                output = Path(temporary) / "run"
                args = self.arguments(output, flags)
                VersionEnv.actions = []
                with patch("bridge.skill_runtime.load_manager", return_value=CapturingManager) as load, \
                        patch("bridge.environment.ReplayableEnv", VersionEnv), \
                        patch("bridge.policy.build_request", side_effect=observation), \
                        patch("bridge.feedback.action_feedback", return_value={}), \
                        patch("bridge.client.JevClient", side_effect=AssertionError("model forbidden")), \
                        patch("bridge.planner.MapPlanner", side_effect=AssertionError("planner forbidden")), \
                        patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
                    summary = asyncio.run(runner.evaluate(args, manifest, "adaptive-skill", output))
                load.assert_called_once_with(version)
                self.assertEqual(len(captured), 1)
                self.assertEqual(len(resets), 3)  # Construction, then every episode.
                self.assertEqual(captured[0]["max_steps"], 4)
                self.assertEqual(summary["completed_episodes"], 2)
                self.assertEqual(summary["status"], "complete")
                self.assertEqual(summary["jev_requests"], 0)
                self.assertEqual(len(VersionEnv.actions), 4)
                for actions in VersionEnv.actions:
                    for actor, question in template["questions"].items():
                        self.assertIn(str(actions[int(actor[1:])]), question["criteria"])
                metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
                events = [json.loads(line) for line in (output / "events.jsonl")
                          .read_text(encoding="utf-8").splitlines()]
                hashes = {path.relative_to(runner.ROOT.parent).as_posix():
                          hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in skill_runtime.source_files(version)}
                for record in (metadata, summary):
                    self.assertEqual(record["skill_version"], version)
                    self.assertEqual(record["skill_source"], str(skill_runtime.SOURCES[version]))
                    self.assertEqual(record["skill_source_hashes"], hashes)
                self.assertTrue(all(event["skill_version"] == version for event in events))
                configuration = next(event for event in events if event["event"] == "skill_configuration")
                self.assertEqual(configuration["skill_source_hashes"], hashes)
                all_hashes = {Path(key).as_posix(): value for key, value in metadata["source_hashes"].items()}
                self.assertTrue(all(all_hashes[key] == value for key, value in hashes.items()))
                source_prefix = skill_runtime.SOURCES[version].relative_to(runner.ROOT.parent).as_posix() + "/"
                self.assertEqual({key for key in all_hashes if key.startswith(source_prefix)}, set(hashes))
                steps = [event for event in events if event["event"] == "skill_step"]
                if version == "baseline_v2_70pct":
                    self.assertEqual(captured[0], {"max_steps": 4, "combat_params": {
                        key: True for key in skill_runtime.LEGACY_COMBAT_PARAMETERS}})
                    for record in (metadata, summary):
                        self.assertEqual(record["unit_value_mode"], "off")
                        self.assertIsNone(record["unit_value_version"])
                        self.assertIsNone(record["unit_value_parameters"])
                        for key in skill_runtime.NEW_COMBAT_PARAMETERS:
                            self.assertNotIn(key, record["skill_parameters"])
                    self.assertTrue(all("unit_value_summary" not in event for event in steps))
                    provenance = json.loads((skill_runtime.SOURCES[version] / "provenance.json")
                                            .read_text(encoding="utf-8"))
                    self.assertEqual(len(provenance["sources"]), 10)
                    for source in provenance["sources"]:
                        key = "skills/baseline_v2_70pct/" + source["exported_path"]
                        self.assertEqual(hashes[key], source["sha256"])
                else:
                    self.assertEqual(captured[0]["unit_value_params"]["mode"], "contextual")
                    self.assertTrue(all(captured[0]["combat_params"].values()))
                    self.assertTrue(all(event["engagement_commitment_enabled"] for event in steps))

    def test_baseline_rejects_unsupported_or_changed_preset_before_output_and_environment(self):
        invalid = [([flag], "does not support") for flag in NEW_FLAGS]
        invalid += [(["--" + flag.replace("_", "-")], "does not support")
                    for flag in skill_runtime.SPATIAL_FLAGS]
        invalid += [(["--unit-value-mode", mode], "no unit-value module")
                    for mode in ("basic", "contextual")]
        invalid += [(["--unit-value-spatial-context"], "no unit-value module"),
                    (["--unit-value-focus-value-weight", ".31"], "no unit-value module"),
                    (["--skill-max-steps", "5"], "requires --skill-max-steps 4"),
                    (["--adaptive-posture", "defensive"], "requires --skill-max-steps 4")]
        for extra, message in invalid:
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "run"
                args = self.arguments(output, ["--skill-version", "baseline_v2_70pct", *LEGACY_FLAGS, *extra])
                with patch("bridge.environment.ReplayableEnv") as environment, \
                        patch("bridge.skill_runtime.load_manager") as manager:
                    with self.assertRaisesRegex(ValueError, message):
                        asyncio.run(runner.evaluate(args, {}, "adaptive-skill", output))
                    environment.assert_not_called()
                    manager.assert_not_called()
                self.assertFalse(output.exists())
        for missing in LEGACY_FLAGS:
            args = self.arguments("unused", ["--skill-version", "baseline_v2_70pct",
                                             *[flag for flag in LEGACY_FLAGS if flag != missing]])
            with self.assertRaisesRegex(ValueError, "requires its original flags"):
                skill_runtime.manager_configuration(args)

    def test_default_version_and_invalid_suite_fail_before_manifest_access(self):
        self.assertEqual(self.arguments("unused").skill_version, "current")
        with patch("sys.argv", ["run.py", "suite", "--methods", "jev", "adaptive-skill",
                                "--skill-version", "baseline_v2_70pct", *LEGACY_FLAGS,
                                "--adaptive-safe-reentry", "--manifest", "missing.json",
                                "--output-dir", "unused"]):
            with self.assertRaisesRegex(SystemExit, "does not support"):
                runner.main()

    def test_archived_v6_is_immutable_and_rejects_new_guard(self):
        version = "baseline_v6_20261024"
        source = skill_runtime.SOURCES[version]
        record = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(len(record["source_hashes"]), 15)
        for name, expected in record["source_hashes"].items():
            self.assertEqual(hashlib.sha256((source / name).read_bytes()).hexdigest(), expected)
        args = self.arguments("unused", ["--skill-version", version, *LEGACY_FLAGS,
                                         "--adaptive-engagement-commitment"])
        kwargs = skill_runtime.manager_configuration(args)["manager_kwargs"]
        self.assertNotIn("approach_progress_guard", kwargs["combat_params"])
        self.assertTrue(kwargs["combat_params"]["engagement_commitment"])
        manager = skill_runtime.load_manager(version)(**kwargs)
        self.assertIsNot(type(manager), skill_runtime.load_manager("current"))
        args.adaptive_approach_progress_guard = True
        with self.assertRaisesRegex(ValueError, "does not support"):
            skill_runtime.manager_configuration(args)


if __name__ == "__main__":
    unittest.main()
