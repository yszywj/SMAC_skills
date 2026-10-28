import copy
import json
import asyncio
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from s2clientprotocol import raw_pb2
from s2clientprotocol import data_pb2

from bridge.config import load_config, public_brief
from bridge.environment import visible_enemies
from bridge.policy import build_request, batch_requests, decode_actions, nearest_actions
from bridge.planner import attach_plan
from bridge.scenarios import generate_manifest, reset_config, validate_manifest
from bridge.semantics import planning_input
from bridge.runner import assert_clock, outcome


def unit(tag, x, unit_type=74, health=80):
    value = raw_pb2.Unit(tag=tag, unit_type=unit_type, health=health, health_max=80,
                         shield=40, shield_max=80, radius=0.5)
    value.pos.x, value.pos.y = x, 5
    return value


class FakeEnv:
    def __init__(self, n=5):
        self.n_agents = self.n_enemies = n
        self.map_name = "10gen_protoss"
        self.conic_fov = False
        self.agents = {i: unit(100 + i, i) for i in range(n)}
        self.enemies = {i: unit(9000 + i, 4 + i) for i in range(n)}
        self._obs = NS(observation=NS(game_loop=8))
        self._step_mul, self._episode_steps, self.episode_limit = 8, 1, 200
        self._move_amount, self.map_x, self.map_y = 2, 32, 32
        self.scenario_brief = public_brief(load_config(n))
        self.seen = {i: {0} for i in range(n)}
        self._controller = NS(observe=lambda: self._obs)

    def get_obs_move_feats_size(self):
        return 4

    def get_obs_enemy_feats_size(self):
        return self.n_enemies, 9

    def get_obs_agent(self, i):
        values = np.zeros((self.n_enemies, 9))
        for enemy in self.seen[i]:
            values[enemy, 4] = 1
        return np.r_[np.ones(4), values.flatten()]

    def get_avail_agent_actions(self, i):
        if self.agents[i].health <= 0:
            return [1] + [0] * (5 + self.n_enemies)
        return [0] + [1] * (5 + self.n_enemies)

    def unit_shoot_range(self, i):
        return {73: 2, 74: 6, 4: 7}[self.agents[i].unit_type]


DATA = NS(units={74: "Stalker", 73: "Zealot", 4: "Colossus"}, unit_stats={})


class BridgeTests(unittest.TestCase):
    def test_portable_skill_uses_bridge_snapshot_without_import_path_pollution(self):
        from bridge.skill_runtime import load_manager
        before = list(sys.path)
        manager = load_manager()(combat_params={
            "window_fire": True, "range_preserving_move": True, "target_scoring": True})
        env = FakeEnv()
        request = build_request(env, DATA, "unused", 1, {}, {}, {})
        manager.select("adaptive_focus_fire", posture="balanced")
        actions, telemetry = manager.step(request, env.n_agents)
        self.assertEqual(len(actions), 5)
        for i, action in enumerate(actions):
            self.assertTrue(env.get_avail_agent_actions(i)[action])
        self.assertEqual(sys.path, before)
        self.assertTrue(telemetry["window_fire_enabled"])
        self.assertTrue(telemetry["range_preserving_move_enabled"])
        self.assertIn("movement_diagnostics", json.loads(json.dumps(telemetry)))

    def test_native_balance_uses_same_type_ids_for_both_owners(self):
        from bridge.environment import RecordedEnv, StarCraft2Env
        env = RecordedEnv.__new__(RecordedEnv)
        env.unit_balance = "native"
        env.ally_unit_map, env.enemy_unit_map = {}, {}
        with patch.object(StarCraft2Env, "_init_ally_unit_types"):
            env._init_ally_unit_types(0)
        self.assertEqual(env.ally_unit_map, env.enemy_unit_map)
        self.assertEqual(env.ally_unit_map, {"stalker": 74, "zealot": 73, "colossus": 4})
        self.assertEqual(env.colossus_id, 4)

    def test_balance_audit_rejects_attributes_upgrades_and_custom_types(self):
        from bridge.environment import audit_unit_balance
        env = FakeEnv()
        stat = data_pb2.UnitTypeData(unit_id=74, name="Stalker")
        data = NS(unit_stats={74: stat})
        self.assertTrue(audit_unit_balance(env, data)["passed"])
        env.enemies[0].shield_max = 79
        with self.assertRaises(RuntimeError):
            audit_unit_balance(env, data)
        env.enemies[0].shield_max = 80
        env.enemies[0].attack_upgrade_level = 1
        with self.assertRaises(RuntimeError):
            audit_unit_balance(env, data)
        env.enemies[0].attack_upgrade_level = 0
        env.agents[0].unit_type = 999
        with self.assertRaises(RuntimeError):
            audit_unit_balance(env, data)

    def test_manifest_deterministic_including_nested_positions(self):
        for size in (5, 10):
            a = generate_manifest(load_config(size), 3, 41)
            b = generate_manifest(load_config(size), 3, 41)
            self.assertEqual(a, b)
            self.assertNotEqual(a["episodes"][0]["sha256"], a["episodes"][1]["sha256"])
            self.assertEqual(reset_config(a["episodes"][0])["ally_start_positions"]["item"].shape, (size, 2))
            validate_manifest(a)

    def test_manifest_tampering_rejected(self):
        manifest = generate_manifest(load_config(5), 1, 1)
        manifest["episodes"][0]["sc2_seed"] = 123
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_hidden_enemy_not_in_payload_despite_available_attack(self):
        env = FakeEnv()
        payload = build_request(env, DATA, "test", 1)
        self.assertEqual(set(payload["state"]["visible_enemies"]), {"e0"})
        self.assertNotIn('"9001"', json.dumps(payload))
        self.assertNotIn("7", payload["questions"]["u0"]["criteria"])

    def test_visibility_union_excludes_dead_observer(self):
        env = FakeEnv()
        env.seen[1] = {2}
        self.assertEqual(set(visible_enemies(env)), {0, 2})
        env.agents[1].health = 0
        self.assertEqual(set(visible_enemies(env)), {0})

    def test_per_type_candidate_ranges(self):
        env = FakeEnv()
        env.agents[0].unit_type = 73
        env.agents[1].unit_type = 4
        payload = build_request(env, DATA, "test", 1)
        self.assertFalse(payload["state"]["relations"]["u0"]["e0"]["smac_center_distance_allowed"])
        self.assertTrue(payload["state"]["relations"]["u1"]["e0"]["smac_center_distance_allowed"])
        self.assertNotIn("cap of 6", json.dumps(payload))

    def test_dead_units_never_get_choice(self):
        env = FakeEnv()
        env.agents[2].health = 0
        payload = build_request(env, DATA, "test", 1)
        self.assertNotIn("u2", payload["questions"])
        self.assertEqual(nearest_actions(payload, 5)[2], 0)

    def test_decoder_rejects_missing_and_illegal_choices(self):
        payload = build_request(FakeEnv(), DATA, "test", 1)
        with self.assertRaises(ValueError):
            decode_actions(payload, {}, 5)
        answers = {key: {"choice": "0"} for key in payload["questions"]}
        with self.assertRaises(ValueError):
            decode_actions(payload, answers, 5)

    def test_plan_preserves_candidates_and_uses_no_episode_snapshot(self):
        env = FakeEnv()
        payload = build_request(env, DATA, "test", 1)
        active = {"plan": {"map": env.map_name}}
        attached = attach_plan(payload, active)
        for key in payload["questions"]:
            self.assertEqual(payload["questions"][key]["criteria"], attached["questions"][key]["criteria"])
        data = planning_input(payload, env)
        self.assertNotIn("9000", json.dumps(data))
        self.assertNotIn("allies", data)
        self.assertEqual(data, planning_input({"hidden": "do not copy"}, env))

    def test_batched_ten_units_retain_all_team_core(self):
        env = FakeEnv(10)
        for i in env.seen:
            env.seen[i] = set(range(10))
        payload = build_request(env, DATA, "test", 1)
        batches = batch_requests(payload, state_question_limit=12500, request_limit=16000)
        self.assertGreater(len(batches), 1)
        self.assertEqual(sorted(k for batch in batches for k in batch["questions"]),
                         sorted(payload["questions"]))
        for batch in batches:
            self.assertEqual(batch["state"]["game_loop"], 8)
            self.assertEqual(len(batch["state"]["team_core"]["allies"]), 10)
        self.assertIn("tag", payload["state"]["allies"]["u0"])

    def test_clock_detects_unexpected_advancement(self):
        env = FakeEnv()
        assert_clock(env, 8)
        with self.assertRaises(RuntimeError):
            assert_clock(env, 0)

    def test_timeout_and_loss_are_distinct(self):
        env = FakeEnv()
        env._episode_steps = env.episode_limit
        self.assertEqual(outcome({"dead_allies": 1, "dead_enemies": 2}, env), "timeout")
        self.assertEqual(outcome({"dead_allies": 5, "dead_enemies": 2}, env), "loss")

    def test_nonstandard_visibility_config_rejected(self):
        from bridge.config import validate_config
        config = load_config(5)
        config["prob_obs_enemy"] = 0.5
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_planner_reuses_one_plan_and_rejects_other_team_size(self):
        from bridge.planner import MapPlanner, PlannerError
        plan = {
            "map": "10gen_protoss", "objective": "Win", "opening": "Observe",
            "target_selection": "Use visible targets", "movement_and_cooldowns": "Use legal moves",
            "priorities": [], "contingencies": [], "assumptions": [],
            "unit_roles": [{"unit_type": "Stalker", "guidance": "Use range"}],
        }

        async def exercise(path):
            planner = MapPlanner(path / "planner", lambda *a, **k: None, executable=sys.executable)
            response = {"stdout": json.dumps({"type": "turn.completed", "usage": {}}),
                        "plan_text": json.dumps(plan), "error": None}
            try:
                with patch.object(planner, "_request", return_value=response) as request:
                    first = await planner.get_plan({"private_snapshot": "SECRET"}, FakeEnv(5))
                    second = await planner.get_plan({}, FakeEnv(5))
                    self.assertEqual(first, second)
                    self.assertEqual(request.call_count, 1)
                    self.assertNotIn("SECRET", request.call_args.args[0])
                    with self.assertRaises(PlannerError):
                        await planner.get_plan({}, FakeEnv(10))
            finally:
                await planner.close()

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(exercise(Path(directory)))


if __name__ == "__main__":
    unittest.main()
