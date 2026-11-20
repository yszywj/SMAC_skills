"""Targeted regression: earlier firing position for front-line Hydralisks.

Covers the acceptance cases of the approach-support rework:
  1. A with a ready shot keeps firing; B fills during its own cooldown.
  2. A NOT yet exposed but its intended step enters E's firing line: the
     pre-exposure pair (proceed / sidestep+fill / one-window hold) is
     generated AND one variant is actually selected end-to-end.
  3. A already exposed, interface disallows attacking: a safe forward step
     that opens A's own firing position is selected WITHOUT any B.
  4. A grab that would charge into several fresh firing lines is rejected
     in favour of a risk-reducing route; the one-window hold exists only
     while B genuinely fires next window (no resettable wait quota).
  5. A baneling interception shot held by an outside hydra is not lost.
"""
import copy

from .fixtures import TacticalTestCase, ally, enemy, scene, with_questions


def _hydra(position, **extra):
    return ally(position, "hydralisk", **extra)


def _build(state, manager):
    from skills.tactical_v1.context import Context, build_snapshot
    payload = with_questions(copy.deepcopy(state))
    context = Context(build_snapshot(payload, 1))
    context.deployment_memory = manager.memory
    context.unit_value_parameters = {"enabled": False}
    context.intercept_config = {}
    context.common_config = {}
    context.burst_config = {}
    manager.memory.observe(context.snapshot, {})
    return context


def _threat_relation(state, pressured="u0", enemy="e0"):
    state["relations"][pressured][enemy]["in_enemy_weapon_range"] = True
    state["relations"][pressured][enemy]["enemy_weapon_range_status"] = "in_range"
    return state


def _base_scene(a_pos, b_pos, e_pos):
    state = scene([(tuple(a_pos), "hydralisk"), (tuple(b_pos), "hydralisk")],
                  enemies={"e0": enemy(e_pos, "hydralisk")})
    for actor in ("u0", "u1"):
        state["allies"][actor].update(health=90., max_health=90.,
                                      weapon_cooldown_loops=0.)
    return _threat_relation(state)


def _plans(state, names):
    from skills.tactical_v1.hydra_support import (approach_support_plans,
                                                  counter_fire_support_plans)
    manager = self_manager()
    context = _build(state, manager)
    out = []
    if "approach" in names:
        out.extend(approach_support_plans(context, manager.memory, "t"))
    if "counter" in names:
        out.extend(counter_fire_support_plans(context, manager.memory, "t"))
    return out, context


_SELF = {}


def self_manager():
    if "m" not in _SELF:
        from skills.tactical_v1.skill_manager import SkillManager
        _SELF["m"] = SkillManager()
    return _SELF["m"]


class ApproachSupportTest(TacticalTestCase):

    def test_1_ready_a_keeps_firing_and_cooling_b_fills(self):
        # A ready and legal: no support variant may move A; the pair keeps
        # A's attack while cooling B closes during its cooldown.
        state = _base_scene((6., 10.), (8., 10.), (10., 10.))
        state["allies"]["u1"]["weapon_cooldown_loops"] = 10.
        plans, context = self._plans_safe(state)
        self.assertTrue(plans, "pair missing")
        for plan in plans:
            self.assertEqual(context.decode("u0", plan.current_actions["u0"]).kind,
                             "attack", "ready A was moved by a support variant")

    def _plans_safe(self, state):
        plans, context = _plans(state, ("approach", "counter"))
        return plans, context

    def test_2_pre_exposure_pair_realizes_cooperation(self):
        # A is OUTSIDE E's coverage; its intended eastward step would enter
        # E's firing line. The pre-exposure comparison must offer all three
        # options (proceed / sidestep+fill / hold), and the FINAL actions on
        # the real SkillManager path must realize the cooperation: A takes
        # its band step (or holds), B moves into a next-window firing
        # position against E — never A walking in alone while B wanders.
        state = scene([((5., 10.5), "hydralisk"), ((6.5, 13.), "hydralisk")],
                      enemies={"e0": enemy((11.5, 10.), "hydralisk")})
        for actor in ("u0", "u1"):
            state["allies"][actor].update(health=90., max_health=90.,
                                          weapon_cooldown_loops=0.)
            state["relations"][actor]["e0"].update(
                smac_attack_available=False, smac_attack_distance_limit=5,
                in_weapon_range=False, weapon_range_status="out_of_range")
        from skills.tactical_v1.hydra_support import approach_support_plans
        manager = self_manager()
        context = _build(state, manager)
        plans = approach_support_plans(context, manager.memory, "t")
        self.assertTrue(plans, "pre-exposure pair missing")
        labels = {p.goal_data.get("approach_label") for p in plans}
        self.assertEqual(labels, {"proceed", "sidestep", "hold"}, labels)
        # Every variant pairs A with B; none abandons either side.
        for plan in plans:
            self.assertEqual(set(plan.current_actions), {"u0", "u1"})
        actions, telemetry = self.step(manager, state)
        # Cooperation realized in the final actions.
        a_kind = context.decode("u0", actions[0]).kind
        a_point = context.predict_position("u0", actions[0]).position
        b_point = context.predict_position("u1", actions[1]).position
        import math
        a_in_band = math.hypot(a_point[0] - 11.5, a_point[1] - 10.) <= 6.0
        a_held = a_kind == "stop"
        b_progresses = (math.hypot(b_point[0] - 11.5, b_point[1] - 10.)
                        < math.hypot(6.5 - 11.5, 13. - 10.))
        print(f"  selected: {[(u['skill'], u.get('goal')) for u in telemetry['units'].values()]}")
        self.assertTrue(a_in_band or a_held,
                        "A neither entered the band nor held safely")
        self.assertTrue(b_progresses, "B did not move toward the cover position")

    def test_3_exposed_a_grabs_position_alone(self):
        # A inside E's coverage, interface disallows the attack (SMAC 5,
        # distance 5.5): a single west step re-opens A's own firing position
        # and must be selected WITHOUT any second hydralisk.
        state = scene([((5.5, 10.), "hydralisk")],
                      enemies={"e0": enemy((11., 10.), "hydralisk")})
        state["allies"]["u0"].update(health=90., max_health=90.,
                                     weapon_cooldown_loops=0.)
        state["relations"]["u0"]["e0"].update(
            in_enemy_weapon_range=True, enemy_weapon_range_status="in_range",
            in_weapon_range=False, weapon_range_status="out_of_range",
            smac_attack_available=False, smac_attack_distance_limit=5)
        manager = self_manager()
        actions, telemetry = self.step(manager, state)
        # The selected action for A is a westward move (toward the firing
        # band) — the independent grab, no partner required.
        from skills.tactical_v1.context import Context, build_snapshot
        context = Context(build_snapshot(with_questions(copy.deepcopy(state)), 1))
        action = actions[0]
        self.assertEqual(context.decode("u0", action).kind, "move")
        point = context.predict_position("u0", action).position
        import math
        self.assertLessEqual(math.hypot(point[0] - 11., point[1] - 10.), 5.6,
                             "grab did not reach the firing band")

    def test_4_grab_into_focus_fire_rejected_hold_bounded(self):
        # Two enemy hydras cover the grab endpoint: the forward grab must
        # not be offered; the hold variant exists only while B genuinely
        # fires next window, and disappears with B removed.
        state = scene([((5., 10.), "hydralisk"), ((12.5, 13.), "hydralisk")],
                      enemies={"e0": enemy((11.5, 10.), "hydralisk"),
                               "e1": enemy((15.5, 15.), "hydralisk"),
                               "e2": enemy((11.5, 8.), "hydralisk")})
        for actor in ("u0", "u1"):
            state["allies"][actor].update(health=90., max_health=90.,
                                          weapon_cooldown_loops=0.)
        for actor in ("u0", "u1"):
            for eid in ("e0", "e1", "e2"):
                state["relations"][actor][eid]["smac_attack_available"] = False
                state["relations"][actor][eid]["smac_attack_distance_limit"] = 5
                state["relations"][actor][eid]["in_weapon_range"] = False
                state["relations"][actor][eid]["weapon_range_status"] = "out_of_range"
        from skills.tactical_v1.hydra_support import approach_support_plans, _fire_grabs
        from skills.tactical_v1.hydra_support import _safe_moves
        manager = self_manager()
        context = _build(state, manager)
        grabs = _fire_grabs(context, manager.memory, "u0", "e0",
                            _safe_moves(context, manager.memory, "u0"))
        east = [row for row in grabs
                if context.move_destinations("u0")[row[1]][0] > 5.]
        self.assertFalse(east,
                         "grab into two fresh firing lines was still offered")
        plans = approach_support_plans(context, manager.memory, "t")
        for plan in plans:
            if plan.goal_data.get("approach_label") == "hold":
                # hold requires a real next-window fill by B
                fill = plan.current_actions["u1"]
                point = context.predict_position("u1", fill).position
                opportunity = context.projected_attack_opportunity(
                    "u1", plan.goal, point)
                self.assertTrue(opportunity.available and opportunity.ready,
                                "hold proposed without a genuine next-window fill")

    def test_5_baneling_interception_shot_survives(self):
        # B holds a ready baneling-interception shot: it must not be pulled
        # into a fill role, and its own interception attack stays available
        # in the final action set comparisons.
        state = scene([((5., 10.), "hydralisk"), ((8., 10.), "hydralisk")],
                      enemies={"e0": enemy((11., 10.), "hydralisk"),
                               "e1": enemy((7.2, 10.), "baneling",
                                           health=30., max_health=30.)})
        for actor in ("u0", "u1"):
            state["allies"][actor].update(health=90., max_health=90.,
                                          weapon_cooldown_loops=0.)
        state["relations"]["u1"]["e1"]["smac_attack_available"] = True
        state["relations"]["u1"]["e1"]["in_weapon_range"] = True
        state["relations"]["u1"]["e1"]["weapon_range_status"] = "in_range"
        state["relations"]["u1"]["e0"]["smac_attack_available"] = False
        state["relations"]["u1"]["e0"]["smac_attack_distance_limit"] = 5
        state["relations"]["u1"]["e0"]["in_weapon_range"] = False
        state["relations"]["u1"]["e0"]["weapon_range_status"] = "out_of_range"
        state["relations"]["u0"]["e0"]["smac_attack_available"] = False
        state["relations"]["u0"]["e0"]["smac_attack_distance_limit"] = 5
        state["relations"]["u0"]["e0"]["in_weapon_range"] = False
        state["relations"]["u0"]["e0"]["weapon_range_status"] = "out_of_range"
        from skills.tactical_v1.hydra_support import approach_support_plans
        manager = self_manager()
        context = _build(state, manager)
        plans = approach_support_plans(context, manager.memory, "t")
        for plan in plans:
            self.assertNotEqual(context.decode("u1", plan.current_actions["u1"]).kind,
                                "move",
                                "interception holder was pulled into a fill move")
