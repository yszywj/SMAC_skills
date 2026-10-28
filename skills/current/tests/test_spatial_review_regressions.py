"""Multi-step regressions for the actual spatial allocators and dispatcher."""
import unittest

from .test_group_skills import scene, payload, advance, enemy_scene
from ..skill_manager import SkillManager
from ..skills.cohesive_search import allocate_cohesive_search


class SpatialReviewRegressions(unittest.TestCase):
    def test_positioning_and_v7_progress_guard_finalize_together(self):
        state = scene([(0., 0.), (1., 1.)], {"e0": enemy_scene((18., 0.))})
        manager = SkillManager(4, spatial_params={"enable_combat_positioning": True},
                               combat_params={"approach_progress_guard": True,
                                              "coordinated_approach": True})
        manager.select("adaptive_focus_fire")
        actions, telemetry = manager.step(payload(state), 2)
        self.assertEqual(len(actions), 2)
        rows = telemetry["approach_diagnostics"]
        self.assertTrue(rows)
        for actor, row in rows.items():
            self.assertIn("progress_guard", row)
            self.assertIn("predicted_destinations", row)
            self.assertEqual(row["progress_guard"]["final_action"], actions[int(actor[1:])])

    def test_isolated_unit_cannot_report_instant_regroup_success(self):
        for count in (5, 10):
            with self.subTest(count=count):
                positions = [(float(i % 3), float(i // 3)) for i in range(count - 1)] + [(20., 0.)]
                state = scene(positions)
                manager = SkillManager(4, spatial_params={"enable_regroup": True,
                    "enable_cohesive_search": True, "auto_spatial_dispatch": True})
                for _ in range(3):
                    chosen = manager.spatial_dispatch(payload(state))
                    if not manager.active:
                        manager.select(chosen)
                    actions, telemetry = manager.step(payload(state), count)
                    self.assertEqual(chosen, "regroup")
                    self.assertEqual(telemetry["status"], "running")
                    self.assertIn(count - 1, actions)
                    self.assertIn(actions[count - 1], (2, 3, 4, 5))
                    manager.commit_final_actions([actions.get(i, 1) for i in range(count)])
                    advance(state, {f"u{i}": action for i, action in actions.items()})

    def test_visible_enemy_interrupts_active_regroup_before_attack_range(self):
        manager = SkillManager(4, spatial_params={"enable_regroup": True,
                                                 "auto_spatial_dispatch": True})
        state = scene([(0., 0.), (20., 0.)])
        manager.select("regroup")
        manager.step(payload(state), 2)
        state = scene([(0., 0.), (20., 0.)], {"e0": enemy_scene((35., 0.))}, loop=88)
        self.assertIsNone(manager.spatial_dispatch(payload(state)))
        actions, telemetry = manager.step(payload(state), 2)
        self.assertEqual(actions, {})
        self.assertEqual(telemetry["status"], "interrupted")
        self.assertFalse(manager.active)

    def test_degenerate_waypoint_stays_fixed_until_arrival_then_rotates(self):
        state, memory, waypoints = scene([(0., 0.)]), {}, []
        for _ in range(8):
            result = allocate_cohesive_search(state, ["u0"], memory)
            waypoint = result["diagnostics"]["waypoint"]
            waypoints.append(waypoint)
            memory.update(result["state_updates"])
            if result["reason"] == "waypoint_checked_pick_next":
                self.assertEqual(len(memory["visited_waypoints"]), 1)
                self.assertTrue(all(point == waypoints[0] for point in waypoints))
                state["game_loop"] += 8
                next_result = allocate_cohesive_search(state, ["u0"], memory)
                self.assertNotEqual(next_result["diagnostics"]["waypoint"], waypoint)
                self.assertNotEqual(next_result["actions"]["u0"], 3)
                return
            advance(state, result["actions"])
        self.fail("A fixed eight-unit leg should enter the three-unit arrival radius")


if __name__ == "__main__":
    unittest.main()
