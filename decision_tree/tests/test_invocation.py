"""Actual executor checks for invocation duration, state and target binding."""
import copy
import unittest

from sc2_skill_adapter.bridge.skill_runtime import load_manager
from skills.current.tests.test_group_skills import scene, enemy_scene, payload, advance


def combat_scene():
    state = scene([(0., 0.)], {"e0": enemy_scene((2., 0.)),
                              "e1": enemy_scene((3., 0.))})
    state["visible_enemies"]["e0"].update(health=1., effective_health=1.)
    for relation in state["relations"]["u0"].values():
        relation.update(smac_attack_available=True, in_weapon_range=True, distance=2.)
    return state


class InvocationTests(unittest.TestCase):
    def manager(self, **kwargs):
        return load_manager()(**kwargs)

    def test_default_call_matches_legacy_actions_and_memory(self):
        legacy, managed = self.manager(), self.manager()
        left = combat_scene()
        right = copy.deepcopy(left)
        for index in range(10):
            if not legacy.active:
                legacy.select("adaptive_focus_fire", posture="balanced")
                managed.apply_call({"skill": "adaptive_focus_fire"})
            original = legacy.step(payload(left), 1)
            actual = managed.step(payload(right), 1)
            self.assertEqual(actual, original)
            self.assertEqual(managed.unit_states, legacy.unit_states)
            self.assertEqual(managed.target_assignments, legacy.target_assignments)
            self.assertEqual(managed.all_in, legacy.all_in)
            left["game_loop"] += 8
            right["game_loop"] += 8

    def test_priority_target_changes_real_adaptive_attack(self):
        base, changed = self.manager(), self.manager()
        base.apply_call({"skill": "adaptive_focus_fire"})
        changed.apply_call({"skill": "adaptive_focus_fire", "priority_target": "e1"})
        base_actions, base_log = base.step(payload(combat_scene()), 1)
        other_actions, other_log = changed.step(payload(combat_scene()), 1)
        self.assertEqual(base_log["attack_assignments"], {"u0": "e0"})
        self.assertEqual(other_log["attack_assignments"], {"u0": "e1"})
        self.assertNotEqual(other_actions, base_actions)

    def test_priority_passes_through_race_wrappers(self):
        for race in ("terran", "zerg"):
            with self.subTest(race=race):
                manager = self.manager(race_profile=race)
                manager.apply_call({"skill": "adaptive_focus_fire", "priority_target": "e1"})
                _, telemetry = manager.step(payload(combat_scene()), 1)
                self.assertEqual(telemetry["attack_assignments"], {"u0": "e1"})

    def test_invocation_duration_actually_ends_skill(self):
        manager = self.manager(max_steps=4)
        manager.apply_call({"skill": "adaptive_focus_fire", "max_steps": 1})
        _, log = manager.step(payload(combat_scene()), 1)
        self.assertEqual(log["termination_reason"], "max_steps")
        self.assertFalse(manager.active)
        manager.apply_call({"skill": "adaptive_focus_fire"})
        self.assertEqual(manager.max_steps, 4)
        state = combat_scene()
        state["game_loop"] += 8
        manager.step(payload(state), 1)
        self.assertTrue(manager.active)

    def test_keep_preserves_everything_and_does_not_extend_budget(self):
        manager = self.manager()
        manager.apply_call({"skill": "adaptive_focus_fire", "max_steps": 2})
        state = combat_scene()
        manager.step(payload(state), 1)
        before = copy.deepcopy(manager.__dict__)
        result = manager.apply_call({"operation": "keep"})
        self.assertEqual(result["operation"], "keep")
        self.assertEqual(manager.__dict__, before)
        state["game_loop"] += 8
        manager.step(payload(state), 1)
        self.assertFalse(manager.active)
        with self.assertRaises(ValueError):
            manager.apply_call({"operation": "keep"})

    def test_overrides_reset_but_battle_history_survives(self):
        manager = self.manager(combat_params={"max_retreat_steps": 5})
        manager.apply_call({"skill": "adaptive_focus_fire", "posture": "defensive",
                            "max_steps": 2, "adaptive_params": {"max_retreat_steps": 3}})
        manager.unit_states["u0"] = {"role": "retreat", "retreat_steps": 2}
        manager._last_move["u0"] = 4
        manager.all_in = True
        self.assertNotIn("max_retreat_steps", manager.combat_params)
        manager.apply_call({"skill": "adaptive_focus_fire"})
        self.assertEqual(manager.posture, "balanced")
        self.assertEqual(manager.max_steps, 4)
        self.assertEqual(manager.combat_params["max_retreat_steps"], 5)
        self.assertEqual(manager.unit_states["u0"]["retreat_steps"], 2)
        self.assertEqual(manager._last_move["u0"], 4)
        self.assertTrue(manager.all_in)

    def test_legacy_select_and_episode_reset_restore_constructor_settings(self):
        for restore in ("select", "reset"):
            with self.subTest(restore=restore):
                manager = self.manager()
                baseline = copy.deepcopy(manager.adaptive_params)
                manager.apply_call({"skill": "adaptive_focus_fire", "max_steps": 1,
                                    "posture": "defensive", "adaptive_params": {"disable_retreat": True}})
                if restore == "select":
                    manager.select("adaptive_focus_fire")
                else:
                    manager.reset()
                self.assertEqual(manager.max_steps, 4)
                self.assertEqual(manager.adaptive_params, baseline)
                self.assertEqual(manager.posture, "balanced")
                self.assertIsNone(manager.last_invocation)

    def test_configs_and_cache_invalidation_are_isolated(self):
        settings = {"max_retreat_steps": 3}
        manager = self.manager(spatial_params={"enable_combat_positioning": True})
        before_key = manager._spatial_config_key
        manager._engagement_cached_step = (80, [999], {})
        manager._spatial_cache["old"] = {"action": 999}
        metadata = manager.apply_call({"skill": "adaptive_focus_fire", "adaptive_params": settings,
                                       "movement_profile": {"spacing": "spread", "layout": "front_back"}})
        settings["max_retreat_steps"] = 8
        metadata["effective_config"]["adaptive_params"]["max_retreat_steps"] = 7
        self.assertEqual(manager.adaptive_params["max_retreat_steps"], 3)
        self.assertEqual(manager.last_invocation["adaptive_params"]["max_retreat_steps"], 3)
        self.assertEqual(manager.combat_params["positioning_spacing"], "spread")
        self.assertNotEqual(manager._spatial_config_key, before_key)
        self.assertIsNone(manager._engagement_cached_step)
        self.assertEqual(manager._spatial_cache, {})
        manager.apply_call({"skill": "adaptive_focus_fire"})
        self.assertEqual(manager._spatial_config_key, before_key)

    def test_invalid_calls_fail_before_mutation(self):
        invalid = [
            {"max_steps": 0}, {"max_steps": True}, {"max_steps": 1.5},
            {"max_steps": 65}, {"posture": "wild"}, {"priority_target": "u1"},
            {"actor_scope": ["u0"]}, {"operation": "update"}, {"made_up": 1},
            {"adaptive_params": {"max_retreat_steps": True}},
            {"adaptive_params": {"max_retreat_steps": 9}},
            {"adaptive_params": {"all_in_team_ratio": float("nan")}},
            {"adaptive_params": {"critical_escape_ratio": float("inf")}},
            {"adaptive_params": {"disable_retreat": 1}},
            {"adaptive_params": {"unknown": 1}},
            {"movement_profile": {"spacing": "spread"}},
            {"skill": "regroup"}, {"skill": "heal_assign", "priority_target": "e0"},
            {"skill": "focus_fire_budget", "adaptive_params": {"max_retreat_steps": 3}},
            {"operation": "keep", "max_steps": 8},
        ]
        manager = self.manager()
        manager.apply_call({"skill": "adaptive_focus_fire"})
        before = copy.deepcopy(manager.__dict__)
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    manager.apply_call({"skill": "adaptive_focus_fire", **overrides})
                self.assertEqual(manager.__dict__, before)


if __name__ == "__main__":
    unittest.main()
