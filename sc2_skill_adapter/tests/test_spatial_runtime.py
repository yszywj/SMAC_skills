"""Real adapter loop: spatial partial merge, final commit and immediate handoff."""
import asyncio
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bridge import runner, skill_runtime
from bridge.config import load_config
from bridge.scenarios import generate_manifest
from bridge.policy import build_request
from test_bridge import DATA, FakeEnv as SnapshotEnv
from test_skill_versions import VersionEnv


class SpatialAdapterRuntimeTests(unittest.TestCase):
    def test_both_spatial_modes_merge_then_handoff_to_combat_on_next_snapshot(self):
        template = build_request(SnapshotEnv(), DATA, "unused", 1, {}, {}, {})
        for mode in ("regroup", "cohesive_search"):
            committed = []
            implementation = skill_runtime.load_manager()

            class Manager(implementation):
                def commit_final_actions(self, actions):
                    committed.append(list(actions))
                    return super().commit_final_actions(actions)

            def observation(env, *args):
                payload = copy.deepcopy(template)
                state = payload["state"]
                state["game_loop"] = env._obs.observation.game_loop
                if env._episode_steps == 0:
                    state["visible_enemies"] = {}
                    state["relations"] = {a: {} for a in state["allies"]}
                    for i, (actor, unit) in enumerate(state["allies"].items()):
                        unit["position"] = [float(i) if mode == "cohesive_search" or i < 4 else 25., 0.]
                        payload["questions"][actor]["criteria"] = {str(a): "legal" for a in range(1, 6)}
                        state["movement_candidates"][actor] = {
                            str(a): {"destination": [unit["position"][0]+dx, dy],
                                     "distances_to_visible_enemies": {}, "distance_to_search_location": 8.}
                            for a, dx, dy in ((2,0,1),(3,0,-1),(4,1,0),(5,-1,0))}
                return payload

            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)/"run"
                args = runner.parser().parse_args(["run", "--method", "adaptive-skill", "--manifest", "unused",
                    "--output-dir", str(output), "--enable-"+mode.replace("_", "-"), "--auto-spatial-dispatch"])
                VersionEnv.actions = []
                with patch("bridge.skill_runtime.load_manager", return_value=Manager), \
                        patch("bridge.environment.ReplayableEnv", VersionEnv), \
                        patch("bridge.policy.build_request", side_effect=observation), \
                        patch("bridge.feedback.action_feedback", return_value={}), \
                        patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
                    result = asyncio.run(runner.evaluate(args, generate_manifest(load_config(5), 1, 210),
                                                         "adaptive-skill", output))
                self.assertEqual(result["status"], "complete")
                self.assertEqual(VersionEnv.actions, committed)
                self.assertEqual(len(committed), 2)
                self.assertTrue(all(len(a) == 5 and all(x > 0 for x in a) for a in committed))
                events = [json.loads(line) for line in (output/"events.jsonl").read_text().splitlines()]
                self.assertTrue(any(e.get("source") == "spatial_interrupt_handoff" for e in events))
                self.assertTrue(any(e.get("skill") == mode and e.get("status") == "running" for e in events))
