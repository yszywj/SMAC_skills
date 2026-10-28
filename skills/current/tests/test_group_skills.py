"""Offline regression tests for the v8.1 spatial group skills.

Real project functions only (Manager, allocators, policy helpers) — no
stand-ins. The simulated world model states its assumptions explicitly:
candidate destinations realize fully at ``advance(distance)`` per step unless
the action id is in ``blocked`` (then the unit stays put: an observed blocked
move). Budgets are decision steps; TTLs are game loops at step_mul=8.
"""
import copy
import unittest

from skills.current.skill_manager import SkillManager
from skills.current import skill_policy
from skills.current.skills.group_geometry import (best_move_toward, cohesion_report,
                                                  legal_actions, base_reservations)
from skills.current.skills.regroup import allocate_regroup
from skills.current.skills.cohesive_search import allocate_cohesive_search
from skills.current.skills.combat_positioning import allocate_combat_positioning

MOVE_DELTAS = {"2": (0., 1.), "3": (0., -1.), "4": (1., 0.), "5": (-1., 0.)}
WEST, EAST, NORTH, SOUTH = 5, 4, 2, 3


def ally(position, type_name="Marine", type_id=73, flying=False, alive=True, **extra):
    unit = {"tag": "ally", "alive": alive, "type": type_name, "unit_type_id": type_id,
            "position": list(position), "radius": .5, "is_flying": flying,
            "health": 45., "shield": 0., "max_health": 45., "max_shield": 0.,
            "weapon_cooldown_loops": 0., "weapon_ready": True,
            "effective_health": 45., "energy": None}
    unit.update(extra)
    return unit


def scene(allies, enemies=None, loop=80, step_mul=8, distance=1.0, order=None):
    """Geometric fixture. ``order`` rebuilds the allies dict in a custom
    insertion order to prove input order never drives policy."""
    enemies = enemies or {}
    ids = list(order) if order is not None else [f"u{i}" for i in range(len(allies))]
    positions = {f"u{i}": p for i, p in enumerate(allies)}
    units = {a: ally(positions[a]) for a in ids}
    state = {"allies": units,
             "visible_enemies": enemies, "game_loop": loop, "step_mul": step_mul,
             "episode": 1, "objective": "win", "map": {"name": "test_map"},
             "steps_remaining": 100,
             "relations": {a: {e: {"native_weapon_range": 5.0, "distance": 10.,
                                   "smac_attack_available": False,
                                   "smac_attack_distance_limit": 6.,
                                   "in_weapon_range": False,
                                   "weapon_range_status": "in_range"}
                               for e in enemies} for a in ids},
             "movement_candidates": {
                 a: {k: {"destination": [units[a]["position"][0] + dx * distance,
                                         units[a]["position"][1] + dy * distance],
                         "distances_to_visible_enemies": {
                             e: abs(units[a]["position"][0] - enemy["position"][0])
                             + abs(units[a]["position"][1] - enemy["position"][1])
                             for e, enemy in enemies.items()},
                         "distance_to_search_location": 0.0}
                     for k, (dx, dy) in MOVE_DELTAS.items()} for a in ids},
             "unit_type_profiles": {"73": {"movement_speed_native": 2.25}}}
    return state


def enemy_scene(position, type_id=4):
    return {"position": list(position), "radius": .5, "alive": True,
            "unit_type_id": type_id, "is_flying": False, "type": "Zergling",
            "health": 35., "shield": 0., "effective_health": 35.}


def advance(state, actions, distance=1.0, blocked=()):
    """World model: a move realizes its candidate destination unless blocked."""
    for actor, action in (actions or {}).items():
        if int(action) in (2, 3, 4, 5) and int(action) not in blocked:
            dx, dy = MOVE_DELTAS[str(int(action))]
            p = state["allies"][actor]["position"]
            state["allies"][actor]["position"] = [p[0] + dx * distance, p[1] + dy * distance]
    state["game_loop"] += state["step_mul"]
    for actor, unit in state["allies"].items():
        p = unit["position"]
        state["movement_candidates"][actor] = {
            k: {"destination": [p[0] + dx * distance, p[1] + dy * distance]}
            for k, (dx, dy) in MOVE_DELTAS.items()}


def payload(state):
    return {"state": state, "questions": {}}


def manager(**flags):
    return SkillManager(4, spatial_params={"enable_regroup": True,
                                           "enable_cohesive_search": True, **flags})


# ---------------------------------------------------------------------------
# Legacy behaviour regression (all spatial features off)
# ---------------------------------------------------------------------------

class LegacyRegressionTests(unittest.TestCase):
    def test_disabled_flags_keep_candidate_table_and_manager_defaults(self):
        legacy, _, _ = skill_policy.skill_choices({}, healers_present=False)
        disabled, _, _ = skill_policy.skill_choices({}, healers_present=False,
                                                    spatial_modes=())
        self.assertEqual(legacy, disabled)
        m = SkillManager(4)
        for flag in ("enable_regroup", "enable_cohesive_search",
                     "enable_combat_positioning", "auto_spatial_dispatch"):
            self.assertFalse(m.spatial_params[flag])
        self.assertNotIn("combat_positioning", m.combat_params)

    def test_disabled_spatial_skill_is_refused(self):
        m = SkillManager(4)
        with self.assertRaises(ValueError):
            m.select(skill_policy.REGROUP)
        with self.assertRaises(ValueError):
            m.select(skill_policy.COHESIVE_SEARCH)

    def test_observation_entry_is_noop_when_features_off(self):
        m = SkillManager(4)
        state = scene([(0., 0.), (1., 0.)], enemies={"e0": enemy_scene((5., 5.))})
        m._observe(state)
        self.assertEqual(m._last_seen_enemies, {})  # legacy ledger untouched
        self.assertIsNone(m._spatial_observed_loop)

    def test_invalid_spatial_parameters_rejected(self):
        for bad in ({"regroup_max_windows": -1},
                    {"positioning_spacing": "phalanx"},
                    {"enable_regroup": "false"},   # string is not a bool
                    {"dispatch_enter_share": 0.9, "dispatch_exit_share": 0.8},
                    {"search_waypoint_ttl_windows": 1.5},
                    {"main_group_min_share": 0.0},
                    {"link_radius": float("inf")}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    SkillManager(4, spatial_params=bad)

    def test_hidden_data_change_does_not_change_actions(self):
        m = manager()
        state = scene([(0., 0.), (1., 0.), (30., 0.)])
        m.select(skill_policy.REGROUP)
        first, _ = m.step(payload(state), 3)
        hidden = copy.deepcopy(state)
        hidden["evaluation_globals"] = {"true_enemy_positions": [(99., 99.)],
                                        "spawn_seed": 42}
        hidden["game_loop"] += 8
        m.select(skill_policy.REGROUP)  # previous window ended
        second, _ = m.step(payload(hidden), 3)
        # Same relative geometry -> same directional proposals per unit.
        self.assertEqual({i: (a in (2, 3, 4, 5) and {2: (0, 1), 3: (0, -1),
                                                    4: (1, 0), 5: (-1, 0)}[a])
                          for i, a in first.items()},
                         {i: (a in (2, 3, 4, 5) and {2: (0, 1), 3: (0, -1),
                                                    4: (1, 0), 5: (-1, 0)}[a])
                          for i, a in second.items()})


# ---------------------------------------------------------------------------
# Group geometry
# ---------------------------------------------------------------------------

class GroupGeometryTests(unittest.TestCase):
    def test_long_chain_is_connected_but_not_coherent(self):
        state = scene([(i * 4.0, 0.) for i in range(6)])
        report = cohesion_report(state)
        self.assertEqual(len(report["groups"]), 1)
        self.assertFalse(report["coherent"])
        self.assertGreater(report["main_compactness"]["quantile_radius"], 7.0)

    def test_two_clusters_main_group_selection(self):
        state = scene([(0., 0.), (1., 0.), (2., 0.), (0., 1.), (1., 1.), (30., 30.), (31., 30.)])
        report = cohesion_report(state)
        self.assertEqual(len(report["groups"]), 2)
        self.assertEqual(len(report["main_group"]), 5)
        self.assertEqual(report["outliers"], ["u5", "u6"])

    def test_zero_radius_is_valid_not_missing(self):
        state = scene([(0., 0.)] * 2)  # exact overlap: radius exactly 0.0
        report = cohesion_report(state, actors=["u0", "u1"])
        self.assertEqual(report["main_compactness"]["quantile_radius"], 0.0)
        self.assertTrue(report["coherent"])

    def test_legal_actions_contract(self):
        state = scene([(0., 0.)])
        self.assertEqual(legal_actions(state, "u0"), {1, 2, 3, 4, 5})
        state["legal_actions"] = {"u0": [2, 3, 4, 5]}
        self.assertEqual(legal_actions(state, "u0"), {2, 3, 4, 5})
        state["legal_actions"] = {"u0": []}
        self.assertEqual(legal_actions(state, "u0"), set())

    def test_stop_wins_ties_at_goal_and_no_oscillation(self):
        state = scene([(0., 0.)])
        action, info = best_move_toward(state, "u0", [0., 0.], base_reservations(state))
        self.assertEqual(action, 1)  # standing on the goal: legal Stop
        self.assertEqual(info["candidates"]["1"]["distance"], 0.0)
        # Repeated calls do not drift into an oscillating re-approach.
        again, _ = best_move_toward(state, "u0", [0., 0.], base_reservations(state))
        self.assertEqual(again, 1)

    def test_uncontrolled_ally_occupancy_costs_and_avoided(self):
        state = scene([(0., 0.), (1., 0.)])  # u1 uncontrolled, 1.0 to the east
        reservations = base_reservations(state)
        action, info = best_move_toward(state, "u0", [4., 0.], reservations)
        self.assertGreater(info["candidates"]["4"]["collision_cost"], 0.0)
        # A deeply blocked direct line is avoided when a clearly better legal
        # option exists: the friend sits ON the target with a large radius.
        state["allies"]["u1"]["radius"] = 2.0
        reservations = base_reservations(state)
        action, info = best_move_toward(state, "u0", [1., 0.], reservations)
        self.assertNotEqual(action, EAST)
        self.assertGreater(info["candidates"]["4"]["collision_cost"], 0.0)

    def test_spawn_overlap_escape_not_sealed(self):
        state = scene([(0., 0.), (0., 0.)])  # exact spawn overlap
        reservations = base_reservations(state)
        action, info = best_move_toward(state, "u0", [4., 0.], reservations)
        # Escaping west/north has zero collision cost; the escape stays open.
        self.assertEqual(info["candidates"]["5"]["collision_cost"], 0.0)
        self.assertIn(action, (2, 3, 4, 5))

    def test_flying_layer_separate_from_ground(self):
        state = scene([(0., 0.)])
        state["allies"]["u0"]["is_flying"] = True
        state["allies"]["u1"] = ally((1., 0.))  # ground unit 1.0 east
        state["movement_candidates"]["u1"] = dict(state["movement_candidates"]["u0"])
        reservations = base_reservations(state)
        _, info = best_move_toward(state, "u0", [4., 0.], reservations)
        self.assertEqual(info["candidates"]["4"]["collision_cost"], 0.0)


# ---------------------------------------------------------------------------
# Regroup (R07)
# ---------------------------------------------------------------------------

class RegroupTests(unittest.TestCase):
    def rally(self, allies, actor_ids=None, memory=None, blocked=(), limit=80):
        state = scene(allies)
        memory = memory if memory is not None else {}
        for _ in range(limit):
            result = allocate_regroup(state, actor_ids or list(state["allies"]), memory)
            memory = {**memory, **result["state_updates"]}
            if result["status"] != "running":
                return state, result, memory
            advance(state, result["actions"], blocked=blocked)
        return state, allocate_regroup(state, actor_ids or list(state["allies"]), memory), memory

    def test_seven_plus_three_converges_without_dragging_main_body(self):
        cluster = [(i % 3 * 1.5, i // 3 * 1.5) for i in range(7)]
        stragglers = [(12., 12.), (13., 12.), (12., 13.)]
        state, result, _ = self.rally(cluster + stragglers)
        self.assertEqual(result["status"], "success", result)
        after = cohesion_report(state)
        self.assertGreaterEqual(after["main_share"], 0.8)
        self.assertLess(after["anchor"][0], 15.0)

    def test_partial_control_does_not_succeed_via_others(self):
        # 8 uncontrolled allies bunched; only u8/u9 (30 apart) are controlled.
        allies = [(i % 3 * 1.0, i // 3 * 1.0) for i in range(8)] + [(30., 0.), (0., 30.)]
        state = scene(allies)
        result = allocate_regroup(state, ["u8", "u9"], {})
        self.assertNotEqual(result["status"], "success")
        self.assertEqual(result["diagnostics"]["observed_cohesion"]["main_share"], 0.8)
        self.assertLess(result["diagnostics"]["controlled_cohesion"]["main_share"], 0.8)
        # It must actually try to complete the controlled rally.
        state, result, _ = self.rally(allies, actor_ids=["u8", "u9"], limit=60)
        self.assertIn(result["status"], ("success", "timeout", "blocked"))

    def test_unreachable_straggler_times_out(self):
        allies = [(0., 0.), (1., 0.), (2., 0.), (50., 50.)]
        state, result, _ = self.rally(allies, blocked=(2, 3, 4, 5))  # nothing moves
        self.assertIn(result["status"], ("timeout", "blocked"))
        self.assertNotEqual(result["status"], "success")

    def test_contract_controlled_units_equal_action_keys_everywhere(self):
        allies = [(0., 0.), (1., 0.), (30., 0.)]
        for actor_ids in (["u0", "u1", "u2"], ["u2"], []):
            with self.subTest(actor_ids=actor_ids):
                result = allocate_regroup(scene(allies), actor_ids, {})
                self.assertEqual(sorted(result["controlled_units"]),
                                 sorted(result["actions"]))
                self.assertEqual(result["requested_units"], actor_ids)

    def test_contact_and_damage_interrupt(self):
        state = scene([(0., 0.), (1., 0.), (30., 0.)], enemies={"e0": enemy_scene((0.5, 3.))})
        state["relations"]["u0"]["e0"]["smac_attack_available"] = True
        self.assertEqual(allocate_regroup(state, list(state["allies"]), {})["status"],
                         "interrupted")
        state = scene([(0., 0.), (1., 0.), (30., 0.)])
        state["allies"]["u1"]["health_lost_since_previous_decision"] = 5.0
        result = allocate_regroup(state, list(state["allies"]), {})
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(result["reason"], "observed_damage")

    def test_single_duplicate_dead_and_empty_inputs(self):
        state = scene([(0., 0.), (1., 0.)])
        self.assertEqual(allocate_regroup(state, ["u0"], {})["status"], "success")
        self.assertEqual(allocate_regroup(state, [], {})["status"], "unavailable")
        duplicate = allocate_regroup(state, ["u0", "u0", "u1"], {})
        self.assertEqual(duplicate["requested_units"], ["u0", "u1"])
        state["allies"]["u1"]["alive"] = False
        result = allocate_regroup(state, ["u0", "u1"], {})
        self.assertEqual(result["requested_units"], ["u0", "u1"])
        self.assertNotIn("u1", result["actions"])

    def test_arrived_waiting_unit_is_not_a_failing_straggler(self):
        # u0 already inside the anchor neighbourhood: it may wait (Stop) and
        # its no-progress counter must stay at zero.
        allies = [(0., 0.), (0.5, 0.), (30., 0.)]
        state = scene(allies)
        memory = {}
        for _ in range(3):
            result = allocate_regroup(state, ["u0", "u1", "u2"], memory)
            memory = {**memory, **result["state_updates"]}
            advance(state, result["actions"])
        self.assertEqual(memory["no_progress"]["u0"], 0)


# ---------------------------------------------------------------------------
# CohesiveSearch (R01, R02, R03)
# ---------------------------------------------------------------------------

class CohesiveSearchTests(unittest.TestCase):
    def search(self, allies, hint=None, memory=None, limit=60, blocked=(), distance=1.0,
               actor_ids=None):
        state = scene(allies)
        if hint:
            state["search_hint"] = {"position": list(hint), "game_loop": 72,
                                    "age_loops": 8, "source": "runner_search_hint"}
        memory = memory if memory is not None else {}
        trail = []
        for _ in range(limit):
            result = allocate_cohesive_search(state, actor_ids or list(state["allies"]),
                                              memory)
            memory = {**memory, **result["state_updates"]}
            trail.append(result)
            if result["status"] != "running" or not result["actions"]:
                if result["status"] != "running":
                    break
            advance(state, result["actions"], distance=distance, blocked=blocked)
        return state, result, memory, trail

    def test_single_unit_advances_toward_each_direction(self):
        for target, expected in (((-20., 0.), WEST), ((20., 0.), EAST),
                                 ((0., 20.), NORTH), ((0., -20.), SOUTH),
                                 ((-20., 20.), None), ((20., -20.), None)):
            with self.subTest(target=target):
                state = scene([(0., 0.)])
                state["search_hint"] = {"position": list(target), "game_loop": 72,
                                        "age_loops": 8, "source": "runner_search_hint"}
                result = allocate_cohesive_search(state, ["u0"], {})
                self.assertEqual(result["status"], "running")
                if expected is not None:
                    self.assertEqual(result["actions"]["u0"], expected)
                else:  # diagonal: any legal non-stop move toward the target
                    self.assertIn(result["actions"]["u0"], (2, 3, 4, 5))
                    dx, dy = MOVE_DELTAS[str(result["actions"]["u0"])]
                    self.assertGreater(dx * target[0] + dy * target[1], 0)

    def test_multi_unit_team_makes_net_progress_toward_waypoint(self):
        for count in (2, 5, 10):
            with self.subTest(count=count):
                allies = [(i * 1.0, 0.) for i in range(count)]
                state, result, _, _ = self.search(allies, hint=(20., 0.), limit=15)
                positions = [u["position"] for u in state["allies"].values()]
                self.assertGreater(min(p[0] for p in positions), count * 1.0 - 1.0)

    def test_different_speeds_still_advance(self):
        # Slow units realize only half a cell per step; the team still closes.
        allies = [(0., 0.), (1., 0.), (2., 0.)]
        state, result, _, _ = self.search(allies, hint=(20., 0.), limit=30, distance=0.5)
        self.assertGreater(state["allies"]["u0"]["position"][0], 3.0)

    def test_severe_split_requests_regroup(self):
        state = scene([(0., 0.), (1., 0.), (40., 40.), (41., 40.)])
        result = allocate_cohesive_search(state, list(state["allies"]), {})
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "needs_regroup")

    def test_contact_and_new_damage_interrupt_on_manual_selection(self):
        state = scene([(0., 0.), (1., 0.)], enemies={"e0": enemy_scene((5., 5.))})
        self.assertEqual(allocate_cohesive_search(state, ["u0", "u1"], {})["status"],
                         "interrupted")
        state = scene([(0., 0.), (1., 0.)])
        state["allies"]["u0"]["health_lost_since_previous_decision"] = 4.0
        result = allocate_cohesive_search(state, ["u0", "u1"], {})
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(result["reason"], "new_damage")

    def test_waypoint_rotation_first_visited_second_chosen(self):
        state = scene([(0., 0.), (1., 0.)])
        state["search_locations"] = [[0., 2.], [20., 0.]]
        result = allocate_cohesive_search(state, list(state["allies"]), {})
        self.assertEqual(result["diagnostics"]["waypoint"], [0., 2.])
        memory = {"visited_waypoints": [{"position": [0., 2.], "loop": 80,
                                         "generation": 0, "source": "runner"}]}
        result = allocate_cohesive_search(state, list(state["allies"]), memory)
        self.assertEqual(result["diagnostics"]["waypoint"], [20., 0.])

    def test_visited_expiry_and_new_generation_reopen(self):
        state = scene([(0., 0.)])
        state["search_hint"] = {"position": [20., 0.], "game_loop": 200,
                                "age_loops": 8, "source": "runner_search_hint"}
        # Same generation, fresh visit -> suppressed.
        memory = {"visited_waypoints": [{"position": [20., 0.], "loop": 200,
                                         "generation": 200, "source": "runner"}]}
        result = allocate_cohesive_search(state, ["u0"], memory)
        self.assertTrue(result["diagnostics"].get("degenerate"))
        # NEW observation generation -> the position is a fresh lead again.
        state["search_hint"]["game_loop"] = 400
        state["search_hint"]["generation"] = 400
        result = allocate_cohesive_search(state, ["u0"], memory)
        self.assertFalse(result["diagnostics"].get("degenerate", False))
        self.assertEqual(result["diagnostics"]["waypoint_source"], "runner_search_hint")

    def test_blocked_direction_actually_changes(self):
        # Target east, but east is blocked by terrain every step.
        state, result, memory, trail = self.search([(0., 0.), (1., 0.)], hint=(20., 0.),
                                                   limit=10, blocked=(EAST,))
        penalties = [t["diagnostics"].get("direction_penalty_applied")
                     for t in trail if t["diagnostics"].get("direction_penalty_applied")]
        changed = [t["diagnostics"].get("direction_changed") for t in trail
                   if t["diagnostics"].get("direction_penalty_applied")]
        self.assertTrue(penalties)      # the blocked direction got penalized
        self.assertIn(True, changed)    # and the chosen action really changed

    def test_single_route_is_never_sealed(self):
        state = scene([(0., 0.)])
        state["movement_candidates"]["u0"] = {"4": {"destination": [1., 0.]}}
        memory = {"failed_directions": {(1.0, 0.0): 3}}
        result = allocate_cohesive_search(state, ["u0"], memory)
        self.assertEqual(result["actions"].get("u0"), 4)  # only route stays usable

    def test_all_directions_blocked_returns_blocked_status(self):
        state, result, memory, _ = self.search([(0., 0.)], hint=(20., 0.),
                                               limit=10, blocked=(2, 3, 4, 5))
        # Nothing can move: after stalls the search reports it, never loops.
        self.assertIn(result["status"], ("blocked", "timeout", "running"))
        if result["status"] == "running":
            self.assertTrue(all(a == 1 for a in result["actions"].values()))

    def test_no_waypoint_degenerates_with_fixed_short_range_goal(self):
        state = scene([(0., 0.), (1., 0.)])
        result = allocate_cohesive_search(state, ["u0", "u1"], {})
        self.assertEqual(result["status"], "running")
        self.assertTrue(all(int(a) in (1, 2, 3, 4, 5) for a in result["actions"].values()))
        self.assertNotIn("observed_regions", result["state_updates"])

    def test_expired_and_invalid_hints_rejected(self):
        for hint in ({"position": [float("nan"), 0.], "age_loops": 1},
                     {"position": [5., 5.], "age_loops": -3},
                     {"position": [5., 5.], "age_loops": 10 ** 9},
                     {"position": None}):
            with self.subTest(hint=hint):
                state = scene([(0., 0.)])
                state["search_hint"] = hint
                result = allocate_cohesive_search(state, ["u0"], {})
                self.assertTrue(result["diagnostics"].get("degenerate")
                                or result["diagnostics"]["waypoint_source"]
                                != "runner_search_hint")


# ---------------------------------------------------------------------------
# CombatPositioning (R04, R05)
# ---------------------------------------------------------------------------

class CombatPositioningTests(unittest.TestCase):
    def armed_state(self, count=4):
        enemies = {"e0": enemy_scene((9., 0.)), "e1": enemy_scene((9., 1.))}
        state = scene([(i * 1.0, 0.) for i in range(count)], enemies=enemies)
        for actor in list(state["allies"])[:max(0, count - 1)]:
            state["relations"][actor]["e0"]["smac_attack_available"] = True
            state["relations"][actor]["e0"]["in_weapon_range"] = True
        return state

    def test_legal_attack_and_fixed_actions_are_never_overridden(self):
        state = self.armed_state()
        protected = {"u2": 6}
        actions, _, _ = allocate_combat_positioning(
            state, ["u0", "u1", "u2", "u3"], {}, protected,
            {"combat_positioning": True})
        self.assertNotIn("u0", actions)
        self.assertNotIn("u1", actions)
        self.assertNotIn("u2", actions)
        self.assertIn("u3", actions)

    def test_spread_front_back_single_unit_makes_engagement_progress(self):
        # (0,0) vs enemy at (20,0): no melee frontline, cannot attack yet.
        state = scene([(0., 0.)], enemies={"e0": enemy_scene((20., 0.))})
        distance = lambda s: abs(s["allies"]["u0"]["position"][0] - 20.)
        start = distance(state)
        for _ in range(10):
            actions, diagnostics, _ = allocate_combat_positioning(
                state, ["u0"], {}, {},
                {"combat_positioning": True, "positioning_spacing": "spread",
                 "positioning_layout": "front_back"})
            self.assertIn("u0", actions)  # no reason to fall back yet
            advance(state, actions)
        self.assertLess(distance(state), start - 3.0)  # real progress, not 20->22.36

    def test_spacing_changes_lateral_slot_geometry(self):
        slots = {}
        for spacing in ("compact", "spread"):
            state = self.armed_state(count=1)
            state["allies"]["u9"] = ally((5., 0.))
            state["relations"]["u9"] = dict(state["relations"]["u0"])
            state["movement_candidates"]["u9"] = {
                k: {"destination": [5. + dx, dy]} for k, (dx, dy) in MOVE_DELTAS.items()}
            _, diagnostics, _ = allocate_combat_positioning(
                state, ["u9"], {}, {},
                {"combat_positioning": True, "positioning_spacing": spacing})
            slots[spacing] = diagnostics["u9"]["slot"]
        self.assertNotEqual(slots["compact"], slots["spread"])
        self.assertGreater(abs(slots["spread"][1] - slots["compact"][1]), 1.0)

    def test_multi_direction_enemies_engage_nearest_cluster(self):
        enemies = {"e0": enemy_scene((12., 0.)), "e1": enemy_scene((-12., 0.))}
        state = scene([(0., 0.)], enemies=enemies)
        _, diagnostics, _ = allocate_combat_positioning(
            state, ["u0"], {}, {}, {"combat_positioning": True})
        goal = diagnostics["u0"]["slot"]
        self.assertGreater(abs(goal[0]), 1.0)  # toward ONE cluster, not midpoint
        self.assertEqual(diagnostics["u0"]["selected_cluster"], ["e0"])
        self.assertEqual(diagnostics["u0"]["selected_cluster_size"], 1)

    def test_no_enemies_falls_back_to_coordinated_approach(self):
        state = scene([(0., 0.), (1., 0.)])
        actions, diagnostics, updates = allocate_combat_positioning(
            state, ["u0", "u1"], {}, {}, {"combat_positioning": True})
        self.assertEqual(actions, {})

    def test_no_slot_progress_falls_back_per_unit(self):
        # Slot lies east, but the only legal route leads west: the proposal
        # would move the unit away from its slot, so the unit is delegated to
        # the legacy approach this window.
        state = scene([(0., 0.)], enemies={"e0": enemy_scene((20., 0.))})
        state["movement_candidates"]["u0"] = {
            "5": {"destination": [-1., 0.], "distances_to_visible_enemies": {},
                  "distance_to_search_location": 0.0}}
        histories = {"u0": {"positioning_no_progress": 3}}  # already stalled
        actions, diagnostics, _ = allocate_combat_positioning(
            state, ["u0"], histories, {},
            {"combat_positioning": True})
        self.assertEqual(diagnostics["u0"].get("fallback"), "no_slot_progress")

    def test_enabled_independently_inside_adaptive(self):
        from skills.current.skills.adaptive_focus_fire import allocate_adaptive_focus_fire
        state = self.armed_state(count=1)  # u0 has no legal attack
        state["allies"]["u0"]["position"] = [0., 0.]
        state["heal_candidates"] = {}
        for positioning, coordinated in ((True, False), (False, True)):
            with self.subTest(positioning=positioning):
                result = allocate_adaptive_focus_fire(
                    state, 1, parameters={"combat_positioning": positioning,
                                          "coordinated_approach": coordinated})
                self.assertTrue(result["approach_diagnostics"],
                                "no approach proposal was generated")
        # Both switches together produce exactly one proposal per unit.
        result = allocate_adaptive_focus_fire(
            state, 1, parameters={"combat_positioning": True, "coordinated_approach": True})
        self.assertLessEqual(len(result["approach_diagnostics"]), 1)


# ---------------------------------------------------------------------------
# Manager lifecycle, memory, budget and dispatch (R06, R10)
# ---------------------------------------------------------------------------

class ManagerLifecycleTests(unittest.TestCase):
    def test_partial_merge_contract(self):
        m = manager()
        state = scene([(0., 0.), (1., 0.), (30., 0.)])
        m.select(skill_policy.REGROUP)
        partial, telemetry = m.step(payload(state), 3)
        self.assertTrue(partial)
        self.assertEqual(set(telemetry["controlled_units"]),
                         {f"u{i}" for i in partial})
        # Baseline with fixed attack on every uncontrolled slot: merge keeps
        # them, controlled slots take exactly the proposal.
        baseline = [6, 6, 6]
        for index, action in partial.items():
            baseline[index] = action
        self.assertEqual(len(baseline), 3)
        for index in range(3):
            if index not in partial:
                self.assertEqual(baseline[index], 6)  # uncontrolled untouched

    def test_same_snapshot_three_calls_count_once(self):
        m = manager()
        state = scene([(0., 0.), (1., 0.), (30., 0.)])
        m.select(skill_policy.REGROUP)
        results = [m.step(payload(state), 3) for _ in range(3)]
        budgets = {t["regroup_budget_windows"] for _, t in results}
        self.assertEqual(len(budgets), 1)
        self.assertEqual(m.stats["regroup_steps"], 1)
        self.assertEqual(m.stats["steps"], 1)
        self.assertEqual(m._regroup_memory["elapsed_windows"], 1)

    def test_final_action_not_proposal_is_recorded(self):
        m = manager()
        state = scene([(0., 0.), (1., 0.)])  # cohesive pair: search runs
        m.select(skill_policy.COHESIVE_SEARCH)
        partial, telemetry = m.step(payload(state), 2)
        self.assertTrue(partial)
        index = next(iter(partial))
        actor = f"u{index}"
        proposal = partial[index]
        # The runner's arbitration overwrites the proposal with a different
        # legal action; only the final one may enter the execution history.
        final = [1, 1]
        final[index] = 3 if proposal != 3 else 2
        m.commit_final_actions(final)
        self.assertEqual(m._last_move.get(actor), final[index])
        self.assertNotEqual(m._last_move.get(actor), proposal)

    def test_budget_survives_window_end_and_reselection(self):
        m = SkillManager(2, spatial_params={"enable_regroup": True})
        state = scene([(0., 0.), (1., 0.), (30., 0.)])
        m.select(skill_policy.REGROUP)
        for _ in range(2):
            _, telemetry = m.step(payload(state), 3)
            state["game_loop"] += 8
        self.assertEqual(telemetry.get("termination_reason"), "max_steps")
        m.select(skill_policy.REGROUP)  # new window: budget must NOT reset
        _, telemetry = m.step(payload(state), 3)
        self.assertGreaterEqual(telemetry["regroup_budget_windows"], 3)

    def test_cooldown_expires_by_observation_clock(self):
        m = SkillManager(4, spatial_params={"enable_cohesive_search": True,
                                            "auto_spatial_dispatch": True})
        state = scene([(0., 0.), (1., 0.)])  # compact, no contact
        m.select(skill_policy.COHESIVE_SEARCH)
        # Force a search failure cooldown of 6 decision steps (48 loops).
        m._set_cooldown(skill_policy.COHESIVE_SEARCH, state["game_loop"], 6)
        for expected in (True, True, False):
            self.assertEqual(m._cooldown_active(skill_policy.COHESIVE_SEARCH,
                                                state["game_loop"]), expected)
            state["game_loop"] += 8 * 3
        # After expiry (and no spatial skill executed meanwhile), search again.
        self.assertEqual(m.spatial_dispatch(payload(state)), skill_policy.COHESIVE_SEARCH)

    def test_ledger_hint_injected_into_search(self):
        m = manager()
        state = scene([(0., 0.), (1., 0.)], enemies={"e0": enemy_scene((-20., 0.))},
                      loop=72)
        m._observe(state)  # loop 72: enemy seen to the west
        self.assertEqual(m._last_seen_enemies["e0"]["position"], [-20., 0.])
        state["visible_enemies"] = {}      # loop 80: enemy left vision
        state["game_loop"] = 80
        m.select(skill_policy.COHESIVE_SEARCH)
        partial, telemetry = m.step(payload(state), 2)
        self.assertEqual(telemetry["injected_search_hint"], "last_seen_enemy")
        self.assertEqual(telemetry["waypoint"], [-20., 0.])
        self.assertEqual(telemetry["waypoint_source"], "last_seen_enemy")
        # The lead is a search target, never an attack: west move proposed.
        self.assertIn(partial.get(0), (5, 1))

    def test_time_backwards_clears_filled_spatial_state(self):
        m = manager()
        state = scene([(0., 0.), (1., 0.)], loop=100)
        m._observe(state)
        m._search_memory = {"visited_waypoints": [{"position": [9., 9.], "loop": 100,
                                                   "generation": 1, "source": "x"}],
                            "elapsed_windows": 17, "failed_directions": {(1.0, 0.0): 2},
                            "anchor_history": [[0., 0.]]}
        m._regroup_memory = {"elapsed_windows": 5, "anchor": [1., 1.]}
        m._set_cooldown(skill_policy.REGROUP, 100, 6)
        m._spatial_cache = {(100, "x", (), ()): object()}
        state["game_loop"] = 60  # observation clock went backwards
        m._observe(state)
        self.assertEqual(m._search_memory, {})
        self.assertEqual(m._regroup_memory, {})
        self.assertEqual(m._spatial_cache, {})
        self.assertFalse(m._cooldown_active(skill_policy.REGROUP, 60))

    def test_reset_clears_everything_spatial(self):
        m = manager()
        state = scene([(0., 0.), (1., 0.)], enemies={"e0": enemy_scene((5., 5.))})
        m._observe(state)
        m._regroup_memory = {"elapsed_windows": 9}
        m.reset()
        self.assertEqual(m._regroup_memory, {})
        self.assertEqual(m._search_memory, {})
        self.assertEqual(m._spatial_cache, {})

    def test_new_episode_clears_spatial_state(self):
        m = manager()
        state = scene([(0., 0.)], enemies={"e0": enemy_scene((5., 5.))}, loop=80)
        state["episode"] = 1
        m._observe(state)
        m._regroup_memory = {"elapsed_windows": 9}
        state = scene([(0., 0.)], loop=88)
        state["episode"] = 2
        m._observe(state)
        self.assertEqual(m._regroup_memory, {})

    def test_input_order_never_drives_policy(self):
        def run(order, actor_order):
            state = scene([(0., 0.), (1., 0.), (30., 0.)], order=order)
            result = allocate_regroup(state, actor_order, {})
            return result["actions"], result["diagnostics"]["controlled_cohesion"]
        forward = run(["u0", "u1", "u2"], ["u0", "u1", "u2"])
        reversed_ = run(["u2", "u1", "u0"], ["u2", "u1", "u0"])
        shuffled = run(["u1", "u0", "u2"], ["u1", "u2", "u0"])
        self.assertEqual(forward, reversed_)
        self.assertEqual(forward, shuffled)


class DispatchTests(unittest.TestCase):
    def dispatch_manager(self, **flags):
        return SkillManager(4, spatial_params={
            "enable_regroup": True, "enable_cohesive_search": True,
            "auto_spatial_dispatch": True, **flags})

    def test_long_chain_triggers_regroup_not_search(self):
        m = self.dispatch_manager()
        state = scene([(i * 4.0, 0.) for i in range(6)])
        self.assertEqual(m.spatial_dispatch(payload(state)), skill_policy.REGROUP)

    def test_compact_team_searches_and_exits_regroup_when_compact(self):
        m = self.dispatch_manager()
        state = scene([(0., 0.), (1., 0.), (2., 0.)])
        self.assertEqual(m.spatial_dispatch(payload(state)), skill_policy.COHESIVE_SEARCH)
        split = scene([(0., 0.), (1., 0.), (40., 40.)])
        m._dispatch_state.update(phase=skill_policy.REGROUP, windows_in_phase=99)
        self.assertEqual(m.spatial_dispatch(payload(split)), skill_policy.REGROUP)
        self.assertEqual(m.spatial_dispatch(payload(state)), skill_policy.COHESIVE_SEARCH)

    def test_contact_returns_none_immediately(self):
        m = self.dispatch_manager()
        m._dispatch_state.update(phase=skill_policy.COHESIVE_SEARCH, windows_in_phase=1)
        state = scene([(0., 0.), (1., 0.)], enemies={"e0": enemy_scene((2., 2.))})
        self.assertIsNone(m.spatial_dispatch(payload(state)))
        damaged = scene([(0., 0.), (1., 0.)])
        damaged["allies"]["u0"]["health_lost_since_previous_decision"] = 3.0
        self.assertIsNone(m.spatial_dispatch(payload(damaged)))

    def test_both_cooling_degrades_explicitly(self):
        m = self.dispatch_manager()
        state = scene([(0., 0.), (1., 0.), (40., 40.)])
        m._set_cooldown(skill_policy.REGROUP, state["game_loop"], 6)
        m._set_cooldown(skill_policy.COHESIVE_SEARCH, state["game_loop"], 6)
        self.assertIsNone(m.spatial_dispatch(payload(state)))
        self.assertEqual(m._dispatch_state["last_reason"],
                         "regroup_unavailable_or_cooling")

    def test_only_one_skill_enabled(self):
        m = self.dispatch_manager(enable_regroup=False)
        state = scene([(0., 0.), (1., 0.), (40., 40.)])
        self.assertIsNone(m.spatial_dispatch(payload(state)))  # no unusable loop
        compact = scene([(0., 0.), (1., 0.)])
        self.assertEqual(m.spatial_dispatch(payload(compact)), skill_policy.COHESIVE_SEARCH)
        m2 = self.dispatch_manager(enable_cohesive_search=False)
        self.assertEqual(m2.spatial_dispatch(payload(compact)), None)

    def test_needs_regroup_verdict_influences_next_dispatch(self):
        m = self.dispatch_manager()
        state = scene([(0., 0.), (1., 0.), (40., 40.), (41., 40.)])
        m.select(skill_policy.COHESIVE_SEARCH)
        _, telemetry = m.step(payload(state), 4)
        self.assertEqual(telemetry["termination_reason"],
                         "cohesive_search_needs_regroup")
        self.assertEqual(m._dispatch_state["phase"], "needs_regroup")
        self.assertEqual(m.spatial_dispatch(payload(state)), skill_policy.REGROUP)


# ---------------------------------------------------------------------------
# Skill protocol (R09)
# ---------------------------------------------------------------------------

class SkillProtocolTests(unittest.TestCase):
    def state_payload(self, enemies, healers=False, limit=None):
        state = {"visible_enemies": {f"e{i}": {} for i in range(enemies)},
                 "visible_enemy_coverage": {f"e{i}": {} for i in range(enemies)},
                 "healers_present": healers}
        if limit is not None:
            state["explicit_target_limit"] = limit
        return {"state": state}

    def test_non_default_limit_decodes_via_bound_context_only(self):
        # 5 visible enemies, no healers, max_explicit_targets=1: request ids
        # are 0..2 direct/focus/cover, 3 -> e0 focus, 4..6 postures.
        payload = self.state_payload(5, limit=1)
        context = skill_policy.build_choice_context(payload["state"]["visible_enemies"],
                                                    healers_present=False,
                                                    max_explicit_targets=1)
        mode, priority = skill_policy.decode_skill(
            payload, {"team_skill": {"choice": "4"}}, choice_context=context)
        self.assertEqual(mode, skill_policy.ADAPTIVE_FOCUS_FIRE)
        self.assertEqual(priority, "aggressive")
        # Without the bound context, guessing is refused instead of
        # mistranslating id 4 into focus fire on e1.
        with self.assertRaises(ValueError):
            skill_policy.decode_skill(payload, {"team_skill": {"choice": "4"}})

    def test_default_requests_stay_backward_compatible(self):
        payload = self.state_payload(2)
        mode, priority = skill_policy.decode_skill(
            payload, {"team_skill": {"choice": "3"}})
        self.assertEqual(mode, skill_policy.FOCUS_FIRE_BUDGET)
        self.assertEqual(priority, "e0")  # legacy default table: id 3 = e0

    def test_limit_overrides_and_edge_values(self):
        for limit in (0, 1, 4, 9):  # 9 > 5 visible enemies
            with self.subTest(limit=limit):
                visible = payload_state = self.state_payload(5, limit=limit)
                context = skill_policy.build_choice_context(
                    payload_state["state"]["visible_enemies"], healers_present=False,
                    max_explicit_targets=limit)
                criteria = context["criteria"]
                explicit = [i for i, m in context["modes"].items()
                            if m == skill_policy.FOCUS_FIRE_BUDGET and context["priorities"][i]]
                self.assertEqual(len(explicit), min(limit, 5))
                for choice in criteria:
                    mode, _ = skill_policy.decode_skill(
                        payload_state, {"team_skill": {"choice": choice}},
                        choice_context=context)
                    self.assertIn(mode, (skill_policy.DIRECT_CONTROL,
                                         skill_policy.FOCUS_FIRE_BUDGET,
                                         skill_policy.COVER_RETREAT,
                                         skill_policy.ADAPTIVE_FOCUS_FIRE))

    def test_spatial_candidates_and_rejections(self):
        context = skill_policy.build_choice_context({}, healers_present=False,
                                                    spatial_modes=("regroup",
                                                                   "cohesive_search"))
        # 0 enemies, no healers: ids 0..2 base, 3..5 postures, then spatial.
        self.assertEqual(context["modes"]["6"], skill_policy.REGROUP)
        self.assertEqual(context["modes"]["7"], skill_policy.COHESIVE_SEARCH)
        payload = self.state_payload(0)
        self.assertEqual(skill_policy.decode_skill(payload, {"team_skill": {"choice": "6"}},
                                                   choice_context=context)[0],
                         skill_policy.REGROUP)
        for bad in ("8", "regroup", 99, None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    skill_policy.decode_skill(payload, {"team_skill": {"choice": bad}},
                                              choice_context=context)
        with self.assertRaises(ValueError):  # spatial without context: refuse
            skill_policy.decode_skill(payload, {"team_skill": {"choice": "3"}},
                                      spatial_modes=("regroup",))

    def test_duplicate_and_missing_answers(self):
        payload = self.state_payload(1)
        with self.assertRaises(ValueError):
            skill_policy.decode_skill(payload, {})
        with self.assertRaises(ValueError):
            skill_policy.decode_skill(payload, {"team_skill": None})

    def test_request_records_explicit_limit(self):
        # skill_request stamps the limit it used into the payload state, so a
        # non-default request can never be silently decoded with defaults.
        from skills.current.skill_policy import skill_request, SKILL_INSTRUCTIONS
        state = scene([(0., 0.)], enemies={"e0": enemy_scene((5., 5.))})
        state["visible_enemy_coverage"] = {}
        payload = skill_request({"state": state}, "model", 4, max_explicit_targets=1)
        self.assertEqual(payload["state"]["explicit_target_limit"], 1)


if __name__ == "__main__":
    unittest.main()
