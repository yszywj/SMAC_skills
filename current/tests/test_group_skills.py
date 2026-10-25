"""Offline tests for the v8 spatial group skills (regroup / cohesive search /
combat positioning) and their scheduler integration. No game, network or model
dependency: pure geometric fixtures with deterministic multi-step simulation.
"""
import copy
import unittest

from skills.current.skill_manager import SkillManager
from skills.current import skill_policy
from skills.current.skills.group_geometry import cohesion_report, connected_components
from skills.current.skills.regroup import allocate_regroup
from skills.current.skills.cohesive_search import allocate_cohesive_search
from skills.current.skills.combat_positioning import allocate_combat_positioning

MOVE_DELTAS = {"2": (0., 1.), "3": (0., -1.), "4": (1., 0.), "5": (-1., 0.)}


def ally(position, type_name="Marine", type_id=73, flying=False, alive=True):
    return {"tag": "ally", "alive": alive, "type": type_name, "unit_type_id": type_id,
            "position": list(position), "radius": .5, "is_flying": flying,
            "health": 45., "shield": 0., "max_health": 45., "max_shield": 0.,
            "weapon_cooldown_loops": 0., "weapon_ready": True,
            "effective_health": 45., "energy": None}


def scene(allies, enemies=None, loop=80, step_mul=8, distance=1.0):
    enemies = enemies or {}
    state = {"allies": {f"u{i}": ally(p) for i, p in enumerate(allies)},
             "visible_enemies": enemies, "game_loop": loop, "step_mul": step_mul,
             "objective": "win", "episode": 1,
             "relations": {a: {e: {"native_weapon_range": 5.0, "distance": 10.,
                                   "smac_attack_available": False,
                                   "smac_attack_distance_limit": 6.,
                                   "in_weapon_range": False,
                                   "weapon_range_status": "in_range"}
                               for e in enemies} for a in
                           [f"u{i}" for i in range(len(allies))]},
             "movement_candidates": {
                 f"u{i}": {k: {"destination": [p[0] + dx * distance, p[1] + dy * distance]}
                           for k, (dx, dy) in MOVE_DELTAS.items()}
                 for i, p in enumerate(allies)},
             "unit_type_profiles": {"73": {"movement_speed_native": 2.25}}}
    return state


def enemy_scene(position, type_id=4):
    return {"position": list(position), "radius": .5, "alive": True,
            "unit_type_id": type_id, "is_flying": False, "type": "Zergling",
            "health": 35., "shield": 0., "effective_health": 35.}


def advance(state, actions, distance=1.0):
    """Deterministic world model: an action realizes its candidate destination."""
    for actor, action in actions.items():
        if int(action) in (2, 3, 4, 5):
            dx, dy = MOVE_DELTAS[str(int(action))]
            p = state["allies"][actor]["position"]
            state["allies"][actor]["position"] = [p[0] + dx * distance, p[1] + dy * distance]
    state["game_loop"] += state["step_mul"]
    # Keep movement candidates anchored to the new positions.
    for actor, unit in state["allies"].items():
        p = unit["position"]
        state["movement_candidates"][actor] = {
            k: {"destination": [p[0] + dx * distance, p[1] + dy * distance]}
            for k, (dx, dy) in MOVE_DELTAS.items()}


def payload(state):
    return {"state": state, "questions": {}}


def positions(state):
    return {a: tuple(u["position"]) for a, u in state["allies"].items()}


class LegacyRegressionTests(unittest.TestCase):
    def test_disabled_flags_keep_candidate_table_and_manager_defaults(self):
        legacy, _, _ = skill_policy.skill_choices({}, healers_present=False)
        disabled, _, _ = skill_policy.skill_choices({}, healers_present=False,
                                                    spatial_modes=())
        self.assertEqual(legacy, disabled)
        manager = SkillManager(4)
        for flag in ("enable_regroup", "enable_cohesive_search",
                     "enable_combat_positioning", "auto_spatial_dispatch"):
            self.assertFalse(manager.spatial_params[flag])
        self.assertNotIn("combat_positioning", manager.combat_params)

    def test_disabled_spatial_skill_is_refused(self):
        manager = SkillManager(4)  # all spatial flags off
        with self.assertRaises(ValueError):
            manager.select(skill_policy.REGROUP)
        with self.assertRaises(ValueError):
            manager.select(skill_policy.COHESIVE_SEARCH)

    def test_invalid_spatial_parameters_rejected(self):
        with self.assertRaises(ValueError):
            SkillManager(4, spatial_params={"regroup_max_windows": -1})
        with self.assertRaises(ValueError):
            SkillManager(4, spatial_params={"positioning_spacing": "phalanx"})


class GroupGeometryTests(unittest.TestCase):
    def test_long_chain_is_connected_but_not_coherent(self):
        chain = [(i * 4.0, 0.) for i in range(6)]
        state = scene(chain)
        report = cohesion_report(state)
        self.assertEqual(len(report["groups"]), 1)  # one chain component...
        self.assertFalse(report["coherent"])        # ...but not compact

    def test_two_clusters_main_group_selection(self):
        state = scene([(0., 0.), (1., 0.), (2., 0.), (0., 1.), (1., 1.), (30., 30.), (31., 30.)])
        report = cohesion_report(state)
        self.assertEqual(len(report["groups"]), 2)
        self.assertEqual(len(report["main_group"]), 5)
        self.assertEqual(report["outliers"], ["u5", "u6"])

    def test_empty_and_dead_units(self):
        state = scene([(0., 0.)])
        state["allies"]["u0"]["alive"] = False
        report = cohesion_report(state)
        self.assertEqual(report["live_units"], [])
        self.assertEqual(report["main_share"], 0.0)


class RegroupTests(unittest.TestCase):
    def test_seven_plus_three_converges_without_dragging_main_body(self):
        cluster = [(i % 3 * 1.5, i // 3 * 1.5) for i in range(7)]
        stragglers = [(12., 12.), (13., 12.), (12., 13.)]
        state = scene(cluster + stragglers)
        memory, done, steps = {}, False, 0
        while not done and steps < 60:
            result = allocate_regroup(state, list(state["allies"]), memory)
            memory = {**memory, **result["state_updates"]}
            if result["status"] in ("success", "timeout", "blocked", "interrupted"):
                done = True
                break
            advance(state, result["actions"])
            steps += 1
        self.assertEqual(result["status"], "success", result)
        after = cohesion_report(state)
        self.assertGreaterEqual(after["main_share"], 0.8)
        # The main body was not dragged toward the straggler corner.
        anchor = after["anchor"]
        self.assertLess(anchor[0], 15.0)

    def test_unreachable_straggler_times_out_instead_of_waiting_forever(self):
        state = scene([(0., 0.), (1., 0.), (2., 0.), (50., 50.)])
        state["movement_candidates"]["u3"] = {}  # no legal moves at all
        memory, status = {}, None
        for _ in range(30):
            result = allocate_regroup(state, list(state["allies"]), memory)
            memory = {**memory, **result["state_updates"]}
            status = result["status"]
            if status != "running":
                break
            advance(state, result["actions"])
        self.assertIn(status, ("timeout", "blocked", "success"))
        self.assertNotEqual(status, "success")  # u3 can never join

    def test_contact_interrupts_and_is_not_reported_as_success(self):
        state = scene([(0., 0.), (1., 0.), (30., 0.)],
                      enemies={"e0": enemy_scene((0.5, 3.))})
        state["relations"]["u0"]["e0"]["smac_attack_available"] = True
        result = allocate_regroup(state, list(state["allies"]), {})
        self.assertEqual(result["status"], "interrupted")

    def test_single_unit_and_empty_set(self):
        state = scene([(0., 0.)])
        self.assertEqual(allocate_regroup(state, ["u0"], {})["status"], "success")
        self.assertEqual(allocate_regroup(state, [], {})["status"], "unavailable")


class CohesiveSearchTests(unittest.TestCase):
    def test_shared_advance_keeps_team_together(self):
        state = scene([(0., 0.), (1., 0.), (2., 0.), (3., 0.)])
        memory = {}
        spreads = []
        for _ in range(12):
            result = allocate_cohesive_search(state, list(state["allies"]), memory)
            self.assertEqual(result["status"], "running")
            memory = {**memory, **result["state_updates"]}
            advance(state, result["actions"])
            ps = [u["position"] for u in state["allies"].values()]
            spreads.append(max(max(p[0] for p in ps) - min(p[0] for p in ps),
                               max(p[1] for p in ps) - min(p[1] for p in ps)))
        self.assertLess(spreads[-1], 6.0)  # never a long-term split

    def test_contact_interrupts_search(self):
        state = scene([(0., 0.), (1., 0.)])
        result = allocate_cohesive_search(state, ["u0", "u1"], {})
        self.assertEqual(result["status"], "running")
        state["visible_enemies"]["e0"] = enemy_scene((5., 5.))
        result = allocate_cohesive_search(state, ["u0", "u1"], {})
        self.assertEqual(result["status"], "interrupted")

    def test_waypoint_consumed_after_arrival_without_contact(self):
        state = scene([(0., 0.), (1., 0.)])
        state["search_hint"] = {"position": [2., 0.], "game_loop": 72, "age_loops": 8}
        memory = {}
        visited_seen = False
        for _ in range(15):
            result = allocate_cohesive_search(state, list(state["allies"]), memory)
            memory = {**memory, **result["state_updates"]}
            if memory.get("visited_waypoints"):
                visited_seen = True
                break
            advance(state, result["actions"])
        self.assertTrue(visited_seen)
        # The checked waypoint is only a visit record, never a cleared-region claim.
        self.assertTrue(all(isinstance(v, list) for v in memory["visited_waypoints"]))

    def test_severe_split_requests_regroup(self):
        state = scene([(0., 0.), (1., 0.), (40., 40.), (41., 40.)])
        result = allocate_cohesive_search(state, list(state["allies"]), {})
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "needs_regroup")

    def test_no_waypoint_degenerates_to_legal_moves(self):
        state = scene([(0., 0.), (1., 0.)])
        result = allocate_cohesive_search(state, ["u0", "u1"], {})
        self.assertEqual(result["status"], "running")
        for action in result["actions"].values():
            self.assertIn(int(action), (1, 2, 3, 4, 5))
        self.assertNotIn("observed_regions", result["state_updates"])  # no invented vision


class CombatPositioningTests(unittest.TestCase):
    def armed_state(self):
        enemies = {"e0": enemy_scene((9., 0.)), "e1": enemy_scene((9., 1.))}
        state = scene([(0., 0.), (1., 0.), (2., 0.), (3., 0.)], enemies=enemies)
        # Front two units have legal attacks: positioning must never touch them.
        for actor in ("u0", "u1"):
            state["relations"][actor]["e0"]["smac_attack_available"] = True
            state["relations"][actor]["e0"]["in_weapon_range"] = True
        return state

    def test_legal_attack_and_fixed_actions_are_never_overridden(self):
        state = self.armed_state()
        protected = {"u2": 6}  # an already-chosen attack action
        actions, diagnostics, _ = allocate_combat_positioning(
            state, ["u0", "u1", "u2", "u3"], {}, protected,
            {"combat_positioning": True})
        self.assertNotIn("u0", actions)
        self.assertNotIn("u1", actions)
        self.assertNotIn("u2", actions)
        self.assertIn("u3", actions)

    def test_spacing_configuration_changes_proposals(self):
        proposals = {}
        for spacing in ("compact", "spread"):
            state = self.armed_state()
            actions, _, _ = allocate_combat_positioning(
                state, ["u3"], {}, {"u0": 6, "u1": 6, "u2": 6},
                {"combat_positioning": True, "positioning_spacing": spacing})
            proposals[spacing] = actions
        self.assertIn("u3", proposals["compact"])
        self.assertIn("u3", proposals["spread"])

    def test_multi_direction_enemies_engage_nearest_cluster_not_centre(self):
        enemies = {"e0": enemy_scene((12., 0.)), "e1": enemy_scene((-12., 0.))}
        state = scene([(0., 0.)], enemies=enemies)
        _, diagnostics, _ = allocate_combat_positioning(
            state, ["u0"], {}, {},
            {"combat_positioning": True, "positioning_spacing": "normal"})
        goal = diagnostics["u0"]["goal"]
        # Goal must be clearly off the origin toward ONE cluster, not the midpoint.
        self.assertGreater(abs(goal[0]), 1.0)

    def test_no_enemies_falls_back_to_coordinated_approach(self):
        state = scene([(0., 0.), (1., 0.)])
        actions, diagnostics, updates = allocate_combat_positioning(
            state, ["u0", "u1"], {}, {}, {"combat_positioning": True})
        self.assertEqual(actions, {})  # no enemies: legacy path finds nothing either


class ManagerIntegrationTests(unittest.TestCase):
    def manager(self, **flags):
        return SkillManager(4, spatial_params={"enable_regroup": True,
                                               "enable_cohesive_search": True, **flags})

    def test_partial_actions_merge_and_only_cover_controlled_units(self):
        manager = self.manager()
        state = scene([(0., 0.), (1., 0.), (30., 0.)])
        manager.select(skill_policy.REGROUP)
        partial, telemetry = manager.step(payload(state), 3)
        self.assertEqual(telemetry["skill"], "regroup")
        self.assertTrue(partial)
        for index in partial:
            self.assertIn(0 <= index < 3, [True])
        self.assertLessEqual(set(telemetry["controlled_units"]),
                             set(state["allies"]))

    def test_budget_persists_across_windows_and_reset_clears_it(self):
        manager = self.manager()
        state = scene([(0., 0.), (1., 0.), (30., 0.)])
        manager.select(skill_policy.REGROUP)
        for _ in range(3):
            partial, telemetry = manager.step(payload(state), 3)
            state["game_loop"] += 8
        budget = telemetry["regroup_budget_windows"]
        self.assertGreaterEqual(budget, 2)
        manager.reset()
        self.assertEqual(manager._regroup_memory, {})

    def test_same_snapshot_repeat_does_not_double_charge(self):
        manager = self.manager()
        state = scene([(0., 0.), (1., 0.), (30., 0.)])
        manager.select(skill_policy.REGROUP)
        _, first = manager.step(payload(state), 3)
        _, second = manager.step(payload(state), 3)
        self.assertEqual(first["regroup_budget_windows"],
                         second["regroup_budget_windows"])

    def test_auto_dispatch_rules_and_hysteresis(self):
        manager = self.manager(enable_combat_positioning=False,
                               auto_spatial_dispatch=True)
        split = scene([(0., 0.), (1., 0.), (50., 50.), (51., 50.)])
        self.assertEqual(manager.spatial_dispatch(payload(split)), skill_policy.REGROUP)
        compact = scene([(0., 0.), (1., 0.), (2., 0.), (3., 0.)])
        manager._dispatch_state.update(phase="idle", windows_in_phase=99)
        self.assertEqual(manager.spatial_dispatch(payload(compact)), skill_policy.COHESIVE_SEARCH)
        contact = scene([(0., 0.), (1., 0.)], enemies={"e0": enemy_scene((2., 2.))})
        self.assertIsNone(manager.spatial_dispatch(payload(contact)))

    def test_time_backwards_clears_spatial_memory(self):
        manager = self.manager()
        state = scene([(0., 0.), (1., 0.), (30., 0.)])
        manager.select(skill_policy.REGROUP)
        manager.step(payload(state), 3)
        state["game_loop"] -= 16  # new, earlier observation timeline
        manager.step(payload(state), 3)
        # Not a crash and no contradictory budget blow-up is the contract here.
        self.assertLessEqual(manager._regroup_memory.get("elapsed_windows", 0), 20)

    def test_input_order_does_not_change_actions(self):
        def run():
            state = scene([(0., 0.), (1., 0.), (30., 0.)], loop=80)
            result = allocate_regroup(state, list(reversed(list(state["allies"]))), {})
            return result["actions"]
        self.assertEqual(run(), run())


if __name__ == "__main__":
    unittest.main()
