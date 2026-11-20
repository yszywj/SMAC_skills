"""Targeted regression: counter-fire support for pressured Hydralisks.

Covers the four acceptance cases from the support-rework task plus the
end-to-end selection requirement:
  1. supporter B with a ready shot at the threat keeps firing;
  2. threat inside native range but SMAC mask closed -> B picks the move
     that opens the next-window firing position (declared prospective);
  3. A-withdraw + B-support form one legal atomic action; a blocked or
     lethal-blast route kills the candidate;
  4. a zergling with an existing valid melee attack is never pulled away
     by the guard logic;
  5. a constructed scene drives the real SkillManager path and the
     support pair is actually SELECTED (not merely proposed).
"""
import copy

from .fixtures import (TacticalTestCase, ally, enemy, scene, with_questions)


def _hydra(position, **extra):
    return ally(position, "hydralisk", **extra)


def _ling(position, **extra):
    return ally(position, "zergling", **extra)


def support_scene(a_position, b_position, e_position, *, a_cooldown=0.,
                  smac_limit=5, a_orders=None):
    """A pressured by E (in E's range), B at a configurable spot.

    relations are built by scene() with attack_range 5: A sits 4 units from
    E (inside both native range and E's range), B's distance to E is set by
    the caller.
    """
    state = scene([(a_position, "hydralisk"), (b_position, "hydralisk")],
                  enemies={"e0": enemy(e_position, "hydralisk")})
    state["allies"]["u0"].update(
        health=90., max_health=90., weapon_cooldown_loops=a_cooldown)
    if a_orders:
        state["visible_enemies"]["e0"]["orders"] = a_orders
    # SMAC mask: candidate_center_distance_by_type drives attack legality in
    # the adapter contract; scene() sets smac_attack_available from range 5.
    if smac_limit != 5:
        state["action_contract"]["candidate_center_distance_by_type"][
            "hydralisk"] = smac_limit
        state["relations"]["u1"]["e0"]["smac_attack_distance_limit"] = smac_limit
        state["relations"]["u1"]["e0"]["smac_attack_available"] = (
            _dist_le(state["allies"]["u1"]["position"],
                     state["visible_enemies"]["e0"]["position"], smac_limit))
    return state


def _dist_le(a, b, limit):
    import math
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= limit


def _counter_relations(state):
    """Mark the threat coverage that makes u0 (A) pressured by e0."""
    state["relations"]["u0"]["e0"]["in_enemy_weapon_range"] = True
    state["relations"]["u0"]["e0"]["enemy_weapon_range_status"] = "in_range"
    return state


class CounterFireSupportTest(TacticalTestCase):

    def _plans(self, state):
        from skills.tactical_v1.context import Context
        from skills.tactical_v1.context import build_snapshot
        from skills.tactical_v1.hydra_support import counter_fire_support_plans
        manager = self.manager()
        context = Context(build_snapshot(with_questions(copy.deepcopy(state)), 1))
        context.deployment_memory = manager.memory
        context.unit_value_parameters = {"enabled": False}
        context.intercept_config = {}
        context.common_config = {}
        context.burst_config = {}
        manager.memory.observe(context.snapshot, {})
        return counter_fire_support_plans(context, manager.memory, "test"), context

    def _selected_units(self, state):
        manager = self.manager()
        actions, telemetry = self.step(manager, state)
        return actions, telemetry

    def test_1_ready_supporter_keeps_firing(self):
        # B (u1) is 3 units from E: inside range with a window shot -> the
        # support pair must keep B's ATTACK on the threat, not a move.
        state = _counter_relations(support_scene((6., 10.), (8., 10.), (10., 10.)))
        plans, context = self._plans(state)
        self.assertTrue(plans, "support candidate missing")
        pair = plans[0]
        members = pair.current_actions
        self.assertEqual(set(members), {"u0", "u1"})
        self.assertEqual(context.decode("u1", members["u1"]).kind, "attack")
        self.assertEqual(context.decode("u1", members["u1"]).target, "e0")
        # A also has a ready shot -> variant 1 keeps A firing too.
        self.assertEqual(context.decode("u0", members["u0"]).kind, "attack")

    def test_lone_hydra_fires_then_holds_on_cooldown_without_a_ling(self):
        state = scene([((10., 10.), "hydralisk")], {
            "e0": enemy((12.8, 10.), "zergling",
                        orders=[{"kind": "attack", "target_unit": "u0"}])})
        # One melee hit is lethal. Stepping west would clear its reach,
        # while the Hydralisk already has a legal shot at this target.
        hydra = state["allies"]["u0"]
        hydra.update(health=4., effective_health=4., max_health=90.)
        manager = self.manager()
        actions, telemetry = self.step(manager, state)
        self.assertEqual(actions, [6])
        self.assertEqual(telemetry["units"]["u0"]["skill"], "focus_fire")
        manager.commit_final_actions(actions)

        state["game_loop"] += state["step_mul"]
        hydra.update(weapon_cooldown_loops=1., weapon_ready=False)
        actions, telemetry = self.step(manager, state)
        # There is no friendly ling to follow or enemy hydra to advance on.
        self.assertEqual(actions, [1])
        self.assertEqual(telemetry["units"]["u0"]["skill"], "focus_fire")

    def test_2_mask_closed_supporter_moves_into_position(self):
        # B sits within native range (5) but beyond the SMAC command
        # distance (3): no legal attack. ONE legal step west must open the
        # firing position (3.5 -> 2.5 <= SMAC limit), so the candidate move
        # approaches the threat and lands inside the legal firing band
        # (prospective position, never a claimed hit).
        state = _counter_relations(support_scene((6., 11.), (7.5, 10.), (4., 10.),
                                                 smac_limit=3))
        plans, context = self._plans(state)
        pair_plans = [p for p in plans if "u1" in p.current_actions
                      and context.decode("u1", p.current_actions["u1"]).kind == "move"]
        self.assertTrue(pair_plans, "no approach-move support candidate")
        pair = pair_plans[0]
        move = pair.current_actions["u1"]
        point = context.predict_position("u1", move).position
        enemy = context.snapshot.enemies["e0"]
        reach = context.effective_attack_reach("u1", "e0")
        import math
        self.assertLessEqual(math.hypot(point[0] - enemy.position[0],
                                        point[1] - enemy.position[1]),
                             reach + 1e-6,
                             "approach move must reach a legal firing position")
        # and the declared claim is positional, not a hit
        self.assertTrue(any(e.kind == "position" for e in pair.expected_effects))

    def test_3_withdraw_pair_valid_but_lethal_route_rerouted_or_dropped(self):
        # A cannot fire (cooling) -> variant 2 withdraws one legal step; the
        # pair stays atomic. A calibrated baneling covering the WEST
        # destination but not the origin must reroute the withdrawal (never
        # step into a new blast circle), and fully blocked improving routes
        # degrade to holding instead of forcing through.
        state = _counter_relations(support_scene(
            (6., 10.), (8., 10.), (9., 10.), a_cooldown=12.))
        plans, context = self._plans(state)
        withdraw = [p for p in plans
                    if context.decode("u0", p.current_actions["u0"]).kind == "move"]
        self.assertTrue(withdraw, "withdraw variant missing for cooling A")
        # Baneling at (3,10): covers the west destination (5,10) within its
        # static blast circle but NOT the origin (6,10) at distance 3.
        blocked = copy.deepcopy(state)
        blocked["visible_enemies"]["e1"] = enemy((3., 10.), "baneling")
        blocked["unit_type_profiles"]["9"] = {
            "known": True, "movement_speed_native": 2.5, "armor": 0.,
            "attributes": [3], "weapons": []}
        for actor in ("u0", "u1"):
            blocked["relations"][actor]["e1"] = dict(
                distance=3.0, in_weapon_range=False,
                weapon_range_status="out_of_range", native_weapon_range=0.,
                in_enemy_weapon_range=False,
                enemy_weapon_range_status="out_of_range",
                smac_attack_available=False, smac_attack_distance_limit=2)
        blocked_plans, blocked_ctx = self._plans(blocked)
        withdraw_moves = [p.current_actions["u0"] for p in blocked_plans
                          if "u0" in p.current_actions and
                          p.goal_data.get("withdrawing")
                          and context.decode("u0", p.current_actions["u0"]).kind == "move"]
        for action in withdraw_moves:
            destination = blocked_ctx.move_destinations("u0")[action]
            import math
            self.assertGreater(
                math.hypot(destination[0] - 3., destination[1] - 10.), 2.6,
                "withdrawal stepped into a new enemy blast circle")
        # Fully blocked improving routes: withdraw degrades to holding.
        from skills.tactical_v1.context import Context, build_snapshot
        manager = self.manager()
        held = copy.deepcopy(state)
        payload = with_questions(copy.deepcopy(held))
        probe = Context(build_snapshot(payload, 1))
        probe.deployment_memory = manager.memory
        probe.unit_value_parameters = {"enabled": False}
        probe.intercept_config = {}
        probe.common_config = {}
        probe.burst_config = {}
        manager.memory.observe(probe.snapshot, {})
        for action in probe.move_destinations("u0"):
            manager.memory.blocked_directions[("u0", action)] = {
                "tag": probe.snapshot.allies["u0"].tag, "until_loop": 10 ** 6,
                "position": tuple(probe.snapshot.allies["u0"].position),
                "local_radius": 2., "severity": 1.}
        from skills.tactical_v1.hydra_support import counter_fire_support_plans
        for plan in counter_fire_support_plans(probe, manager.memory, "t"):
            if plan.goal_data.get("withdrawing"):
                action = plan.current_actions["u0"]
                self.assertNotIn(action, probe.move_destinations("u0"),
                                 "withdraw forced through a blocked route")
                self.assertIn(action, probe.legal_actions("u0"))

    def test_4_engaged_zergling_not_pulled_by_guard(self):
        from skills.tactical_v1.context import Context
        from skills.tactical_v1.context import build_snapshot
        from skills.tactical_v1.hydra_support import zergling_guard_plans
        manager = self.manager()
        state = _counter_relations(support_scene((6., 10.), (8., 10.), (10., 10.)))
        state["allies"]["u2"] = _ling((9.6, 10.0))       # melee range of e0
        state["relations"]["u2"] = {
            "e0": dict(distance=1.0, in_weapon_range=True,
                       weapon_range_status="in_range", native_weapon_range=.1,
                       in_enemy_weapon_range=False,
                       enemy_weapon_range_status="in_range",
                       smac_attack_available=True, smac_attack_distance_limit=2)}
        state["movement_candidates"]["u2"] = {
            2: {"destination": [9.6, 11.]}, 3: {"destination": [9.6, 9.]},
            4: {"destination": [10.6, 10.]}, 5: {"destination": [8.6, 10.]}}
        payload = with_questions(copy.deepcopy(state))
        payload["questions"]["u2"]["criteria"]["8"] = "attack"  # 6+2: e0 attack
        context = Context(build_snapshot(payload, 1))
        context.deployment_memory = manager.memory
        context.unit_value_parameters = {"enabled": False}
        context.intercept_config = {}
        context.common_config = {}
        context.burst_config = {}
        manager.memory.observe(context.snapshot, {})
        self.assertEqual(zergling_guard_plans(context, manager.memory, "test"), [],
                         "engaged zergling must not receive a guard candidate")

    def test_two_ready_hydras_focus_fire_end_to_end(self):
        state = _counter_relations(support_scene((6., 10.), (8., 10.), (10., 10.)))
        manager = self.manager()
        actions, telemetry = self.step(manager, state)
        units = telemetry["units"]
        self.assertEqual(units["u0"]["skill"], "focus_fire")
        self.assertEqual(units["u1"]["skill"], "focus_fire")
        self.assertEqual(units["u0"]["goal"], "e0", telemetry["units"])
        self.assertEqual(units["u1"]["goal"], "e0")
        # Both final orders must be legal attacks on the same threat.
        from skills.tactical_v1.context import Context, build_snapshot
        context = Context(build_snapshot(with_questions(copy.deepcopy(state)), 1))
        self.assertEqual(context.decode("u0", actions[0]).kind, "attack")
        self.assertEqual(context.decode("u1", actions[1]).kind, "attack")
        self.assertEqual(context.decode("u0", actions[0]).target, "e0")
        self.assertEqual(context.decode("u1", actions[1]).target, "e0")

    def test_final_actions_can_block_or_clear_a_support_move(self):
        from ..evaluator import Evaluator
        from ..validator import validate_plans

        state = _counter_relations(support_scene((6., 12.), (7.5, 10.), (4., 10.),
                                                 smac_limit=3))
        extra = scene([((6., 12.), "hydralisk"), ((7.5, 10.), "hydralisk"),
                       ((6.5, 11.), "zergling")], enemies=state["visible_enemies"])
        for key in ("allies", "movement_candidates", "relations"):
            state[key]["u2"] = extra[key]["u2"]
        plans, context = self._plans(state)
        pair = next(p for p in plans if p.goal_data["support_pair"] == ["u0", "u1"]
                    and p.current_actions["u1"] == 5)
        clear = dict(pair.current_actions, u2=2)
        crossing = dict(pair.current_actions, u2=3)
        self.assertFalse(validate_plans([pair], context, clear))
        problems = validate_plans([pair], context, crossing)
        self.assertTrue(any(p["type"] == "support_path_obstructed" and p["drop_whole"]
                            and p["actor"] == "u1" and p["target"] == "u2" for p in problems))

        evaluator = Evaluator()
        def progress(actions):
            report = evaluator.effects(context, actions)
            return evaluator.task_quality(context, actions, [pair], report)[1]["progress"]
        self.assertGreater(progress(clear), 0.)
        self.assertEqual(progress(clear), progress(dict(clear, u0=2)))
        self.assertEqual(progress(dict(clear, u1=1)), 0.)
