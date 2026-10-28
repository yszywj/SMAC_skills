"""AdaptiveFocusFire: focus fire by default, brief local retreats when they
pay off, and a sticky All-in phase when retreating no longer helps.

Composition (no duplicated algorithms):

- Danger detection and retreat move selection reuse CoverRetreat's
  ``detect_retreat_candidates`` / ``select_retreat_action``.
- All attacking units — even when NOBODY retreats — are assigned by
  FocusFireBudget's ``allocate_focus_fire`` restricted to the attacker set.
  The controller never degrades to per-unit nearest fallbacks.
- Healers (Medivacs) are Support: legal heal candidates or safe movement,
  never attack orders.

Model assumptions and scope (first version):

- Retreat value is an explainable heuristic over observed facts only (health
  ratio, recent observed loss, weapon readiness, legal targets, kill
  opportunity, existence of a safe legal move). It is NOT a survival-rate or
  future-damage prediction.
- "Combat progress" is measured only through observable visible-enemy health
  changes; hidden enemies are never assumed dead or damaged. Contact must
  have happened before the stall counter can trigger All-in, so a fresh
  scenario with no visible enemies never mis-triggers.
- All-in is sticky once entered (per episode, in manager state) and always
  recorded with its trigger reasons.
- Planned damage from FocusFireBudget is prediction, never actual damage.
"""

from math import dist

from .combat_timing import can_fire_within_window
from .cover_retreat import (PARAMS as COVER_PARAMS, detect_retreat_candidates,
                            select_retreat_action, effective_entry_threshold,
                            retreat_value_diagnostics)
from ..unit_value import enabled_values
from .focus_fire import allocate_focus_fire, expected_damage, fallback_action
from .movement_scoring import (predicted_move, threat_count, score_movement_candidate,
                               select_planned_reentry, visible_volley_pressure)
from .combat_timing import window_loops
from .coordinated_approach import coordinate_approaches
from .engagement_control import prepare_engagement, retreat_override, finalize_engagement

PARAMS = {
    "max_retreat_steps": 2,       # per unit, per retreat episode
    "stall_steps": 6,             # contact made + no observable enemy damage this long -> reassess
    "all_in_team_ratio": 0.35,    # remaining-force ratio seen as severe disadvantage
    "max_retreat_cycles": 3,      # per unit: unproductive retreat episodes before calling it a loop
    "disable_retreat": False,     # ablation: no local retreats at all
    "disable_all_in": False,      # ablation: never enter All-in
    "max_recover_steps": 2,       # brief cooldown recovery after leaving attack candidates
    "search_memory_steps": 4,     # remembered positions are search hints, never attack targets
    "coordinated_approach": False, # visible-only route reservations and measured stall recovery
    "safe_reentry": False,        # preserve next-window fire unless escape reduces danger
    "critical_escape": False,     # ready critical units may take a truly safer legal move
    "planned_reentry": False,     # jointly consider cooldown, travel and visible pressure
    "critical_escape_ratio": 0.30, # matches the observation's critical health classification
    "engagement_commitment": False, # all-role personal progress and bounded shot commitment
}

# Postures only tune the explainable thresholds; one template, three configs.
POSTURE_PARAMS = {
    "aggressive": {"critical_ratio": 0.15, "recent_loss_needed": 0.25},
    "balanced": {"critical_ratio": 0.25, "recent_loss_needed": 0.20},
    "defensive": {"critical_ratio": 0.35, "recent_loss_needed": 0.15},
}

DEFAULT_UNIT_STATE = {"role": "attack", "retreat_steps": 0, "retreat_cycles": 0}


def _is_healer(unit):
    return str(unit.get("type", "")).lower().startswith("medivac")


def _legal_attack_targets(state, actor):
    return [enemy_id for enemy_id, relation in state.get("relations", {}).get(actor, {}).items()
            if relation["smac_attack_available"]]


def _has_kill_opportunity(state, actor, parameters=None):
    """True when this unit can very likely kill a visible target this step.

    Uses the same window fire-timing as FocusFireBudget (shared helper), so
    the two skills can never disagree about whether a cooling-down unit can
    still shoot this control window.
    """
    unit = state["allies"][actor]
    fires = (lambda target: can_fire_within_window(state, actor, target,
                                                    parameters=parameters))         if (parameters or {}).get("window_fire") else (lambda target: unit["weapon_ready"])
    if (not unit["weapon_ready"] and not (parameters or {}).get("window_fire")):
        return False
    for enemy_id in _legal_attack_targets(state, actor):
        relation = state["relations"][actor][enemy_id]
        if relation["in_weapon_range"] is not True:
            continue
        if not fires(enemy_id):
            continue
        damage = expected_damage(state, actor, enemy_id, parameters=parameters)
        enemy = state["visible_enemies"][enemy_id]
        if damage is not None and damage >= enemy["health"] + enemy["shield"]:
            return True
    return False


def critical_escape_conditions(state, actor, retreat_move, parameters=None):
    """Observed injury plus strictly reduced visible geometry/volley pressure."""
    unit = state['allies'][actor]
    maximum = (unit.get('max_health') or 0) + (unit.get('max_shield') or 0) or 1
    ratio = (unit.get('health', 0) + unit.get('shield', 0)) / maximum
    origin = unit['position']
    info = {'applied': False, 'eligible': False, 'survival_ratio': ratio,
            'critical_threshold': (parameters or {}).get('critical_escape_ratio', .30),
            'standing_threat_count': threat_count(state, actor, origin, parameters),
            'standing_visible_base_volley_pressure': visible_volley_pressure(state, actor, origin, parameters),
            'proposed_action': retreat_move}
    if str(retreat_move) not in state.get('movement_candidates', {}).get(actor, {}):
        return info
    destination = predicted_move(state, actor, retreat_move)['destination']
    exposure = threat_count(state, actor, destination, parameters)
    pressure = visible_volley_pressure(state, actor, destination, parameters)
    info.update(predicted_threat_count=exposure, predicted_visible_base_volley_pressure=pressure,
                eligible=(ratio <= info['critical_threshold']
                          and exposure < info['standing_threat_count']
                          and pressure < info['standing_visible_base_volley_pressure']))
    return info


def evaluate_retreat_value(state, actor, retreat_move, unit_state, posture="balanced",
                           parameters=None, unit_values=None):
    """Should this endangered unit retreat this step? (bool, reason).

    Heuristic rules, in priority order:
    - A unit that can immediately kill a target keeps attacking.
    - A ready, in-range weapon with a legal target normally keeps attacking.
      Opt-in critical_escape permits a truly safer move for a critical unit
      without requiring one unusually large recent damage window.
    - A unit on cooldown / out of range retreats only when a safe legal move
      exists, the retreat cap is not exhausted, and re-engagement is possible
      (it still has some legal attack relation to come back to).

    The per-unit retreat cap applies to EVERY retreat-returning branch —
    including the critical-unit branch. There is deliberately no emergency
    exception: a capped unit stays in the fire plan (or repositions via the
    normal attack path) instead of retreating forever.
    """
    params = {**PARAMS, **(parameters or {})}
    posture_params = POSTURE_PARAMS[posture]
    unit = state["allies"][actor]
    if params.get('engagement_commitment'):
        if _has_kill_opportunity(state, actor, parameters):
            return False, 'kill_opportunity'
        override = retreat_override(state, actor, retreat_move, unit_state, params)
        if override is not None:
            return override
    if unit_state.get("retreat_steps", 0) >= params["max_retreat_steps"]:
        if _has_kill_opportunity(state, actor, parameters):
            return False, "kill_opportunity"
        return False, "retreat_cap_reached"
    if _has_kill_opportunity(state, actor, parameters):
        return False, "kill_opportunity"
    if (parameters or {}).get("window_fire"):
        can_attack_now = any(can_fire_within_window(state, actor, e, parameters=parameters)
                             for e in _legal_attack_targets(state, actor))
    else:
        can_attack_now = (unit["weapon_ready"]
                          and any(state["relations"][actor][e]["in_weapon_range"] is not False
                                  for e in _legal_attack_targets(state, actor)))
    maximum = (unit["max_health"] or 0) + (unit["max_shield"] or 0) or 1
    ratio = (unit["health"] + unit["shield"]) / maximum
    recent_loss = (unit.get("health_lost_since_previous_decision", 0)
                   + unit.get("shield_lost_since_previous_decision", 0)) / maximum
    if can_attack_now:
        if params.get('critical_escape') and critical_escape_conditions(
                state, actor, retreat_move, params)['eligible']:
            return True, 'critical_safe_escape'
        if ratio < effective_entry_threshold(posture_params["critical_ratio"], actor, unit_values) \
                and recent_loss >= posture_params["recent_loss_needed"]:
            return (retreat_move is not None), "critical_despite_weapon"
        return False, "weapon_engaged"
    if retreat_move is None:
        return False, "no_safe_move"
    if not _legal_attack_targets(state, actor):
        if params.get("range_preserving_move"):
            # Hand off to bounded recovery/re-entry instead of starting a
            # fresh retreat or immediately reversing while still on cooldown.
            return False, "cooldown_recovery" if (
                unit_state.get("role") in ("retreat", "recover")
                and unit["weapon_cooldown_loops"] > window_loops(state, parameters)
            ) else "reentry_required"
        return False, "cannot_reengage"
    return True, "cooldown_or_out_of_range"


def recovery_action(state, actor, history, previous_move, parameters):
    """Brief non-reversing recovery outside the attack mask, then re-enter.

    A deliberate Stop can hold position while the weapon recovers. It is not
    a no-op and is only used when no attack is currently legal.
    """
    if (history.get("role") not in ("retreat", "recover")
            or history.get("recover_steps", 0) >= parameters["max_recover_steps"]
            or _legal_attack_targets(state, actor)
            or not state["visible_enemies"]
            or state["allies"][actor]["weapon_cooldown_loops"] <= window_loops(state, parameters)):
        return None
    opposite = {2: 3, 3: 2, 4: 5, 5: 4}.get(previous_move)
    origin = state["allies"][actor]["position"]
    exposure = threat_count(state, actor, origin, parameters)
    target = min(state["relations"].get(actor, {}),
                 key=lambda e: state["relations"][actor][e]["distance"], default=None)
    options = []
    for action in state.get("movement_candidates", {}).get(actor, {}):
        # Lateral moves avoid both immediate reversal and extending the same
        # retreat past its cap. Holding is preferable to a worse exposure.
        if int(action) in (previous_move, opposite):
            continue
        move = predicted_move(state, actor, action)
        if threat_count(state, actor, move["destination"], parameters) <= exposure:
            options.append((score_movement_candidate(state, actor, action, target,
                                                      "tactical", parameters, previous_move),
                            int(action)))
    return min(options)[1] if options else 1


def should_enter_all_in(state, stall, contacted, unit_states, heal_capacity,
                        parameters=None):
    """Sticky All-in trigger. Returns (enter, reasons).

    Normal short retreats (threat -> brief move -> re-engage -> damage) are
    NOT penalized: ``retreat_cycles`` counts only a single unit's
    *unproductive* retreat episodes (retreating again without any observable
    enemy damage in between), maintained by the SkillManager. Two separate
    survival metrics are reported and never conflated:

    - ``alive_ratio``: average health state of currently living units.
    - ``remaining_force_ratio``: surviving effective health relative to the
      full roster's maximum (8 full-HP marines down to 1 -> 0.125 even
      though that survivor is at 100%).

    A stall alone never triggers (a healthy team may legitimately kite); it
    must combine with a real disadvantage, or one unit must be stuck in a
    genuinely unproductive retreat loop.
    """
    params = {**PARAMS, **(parameters or {})}
    if not contacted or not state["visible_enemies"]:
        return False, None  # Never mis-trigger before contact / with no visible enemy.
    alive = [u for u in state["allies"].values() if u["alive"]]
    if not alive:
        return False, None
    alive_max = sum((u["max_health"] or 0) + (u["max_shield"] or 0) for u in alive) or 1
    alive_ratio = sum(u["health"] + u["shield"] for u in alive) / alive_max
    roster_max = sum((u["max_health"] or 0) + (u["max_shield"] or 0)
                     for u in state["allies"].values()) or 1
    remaining_force_ratio = sum(u["health"] + u["shield"] for u in alive) / roster_max
    worst_cycles = max((u.get("retreat_cycles", 0) for u in unit_states.values()),
                       default=0)
    reasons = []
    if stall >= params["stall_steps"]:
        reasons.append("no_damage_progress")
    if remaining_force_ratio < params["all_in_team_ratio"] and not heal_capacity:
        reasons.append(f"low_remaining_force({remaining_force_ratio:.2f};"
                       f"alive_ratio={alive_ratio:.2f})_no_heal")
    if worst_cycles >= params["max_retreat_cycles"] and not params.get('engagement_commitment'):
        reasons.append(f"unproductive_retreat_loop(unit_cycles={worst_cycles})")
    if any(r.startswith("unproductive_retreat_loop") for r in reasons):
        return True, ";".join(reasons)
    if reasons and "no_damage_progress" in reasons and len(reasons) > 1:
        return True, ";".join(reasons)
    return False, None


def allocate_adaptive_focus_fire(state, n_agents, unit_states=None, all_in=False,
                                 stall=0, contacted=False, posture="balanced",
                                 parameters=None, previous_attack_assignments=None, move_history=None,
                                 unit_values=None):
    """One AdaptiveFocusFire decision step on a fresh snapshot.

    ``unit_states`` carries per-unit role history (retreat step counts and
    cycles) owned by the SkillManager and persisted across 4-step skill
    windows within the episode, so re-activation cannot reset retreat caps.
    """
    params = {**PARAMS, **(parameters or {})}
    unit_states = unit_states or {}
    original_unit_states = unit_states
    engagement_diagnostics = {}
    if params.get('engagement_commitment'):
        unit_states, engagement_diagnostics = prepare_engagement(state, unit_states, params)
    allies = state["allies"]

    heal_capacity = any(candidates for candidates in state.get("heal_candidates", {}).values())
    if all_in and not params["disable_all_in"]:
        all_in_now, all_in_reason = True, "sticky"
    elif params["disable_all_in"]:
        all_in_now, all_in_reason = False, None  # Ablation: never All-in, no back door.
    else:
        all_in_now, all_in_reason = should_enter_all_in(state, stall, contacted,
                                                        unit_states, heal_capacity, params)

    retreat_units, retreat_actions, retreat_reasons, movement_diagnostics = {}, {}, {}, {}
    critical_diagnostics, planned_diagnostics = {}, {}
    value_thresholds = {}
    if not all_in_now and not params["disable_retreat"]:
        # Danger thresholds follow the same risk_threshold CLI knob as
        # CoverRetreat when the caller supplies one.
        cover_parameters = {**COVER_PARAMS, **params}
        if params.get("safe_reentry") or params.get('critical_escape') or params.get('planned_reentry'):
            cover_parameters["range_preserving_move"] = True
        endangered = detect_retreat_candidates(state, cover_parameters, unit_values=unit_values)
        value_thresholds = retreat_value_diagnostics(state, cover_parameters, unit_values)
        for actor in sorted(endangered, key=lambda a: int(a[1:])):
            unit = state["allies"][actor]
            maximum = (unit["max_health"] or 0) + (unit["max_shield"] or 0) or 1
            recent_loss = (unit.get("health_lost_since_previous_decision", 0)
                           + unit.get("shield_lost_since_previous_decision", 0)) / maximum
            movement_mode = ("emergency"
                             if (unit["health"] + unit["shield"]) / maximum < params.get("emergency_ratio", 0.15)
                             or recent_loss >= params.get("emergency_recent_loss", 0.5)
                             else "tactical")
            move, move_reason, detail = select_retreat_action(
                state, actor, {actor}, cover_parameters, movement_mode=movement_mode,
                previous_action=(move_history or {}).get(actor))
            value, reason = evaluate_retreat_value(
                state, actor, move, unit_states.get(actor, DEFAULT_UNIT_STATE),
                posture, params, unit_values=unit_values)
            if actor in engagement_diagnostics and reason.startswith('engagement_'):
                unguarded, _ = evaluate_retreat_value(
                    state, actor, move, original_unit_states.get(actor, DEFAULT_UNIT_STATE), posture,
                    {**params, 'engagement_commitment': False}, unit_values=unit_values)
                engagement_diagnostics[actor].update(
                    retreat_blocked=not value, applied=value != unguarded, reason=reason)
            if params.get('critical_escape'):
                critical = critical_escape_conditions(state, actor, move, params)
                without_critical, _ = evaluate_retreat_value(
                    state, actor, move, unit_states.get(actor, DEFAULT_UNIT_STATE), posture,
                    {**params, 'critical_escape': False}, unit_values=unit_values)
                critical.update(applied=value and not without_critical,
                                accepted=value, reason=reason)
                critical_diagnostics[actor] = critical
            if 'planned_reentry' in detail:
                planned = {**detail['planned_reentry']}
                baseline_value, _ = evaluate_retreat_value(
                    state, actor, planned['baseline_action'], unit_states.get(actor, DEFAULT_UNIT_STATE),
                    posture, {**params, 'planned_reentry': False}, unit_values=unit_values)
                planned.update(applied=(value != baseline_value or
                                        (value and move != planned['baseline_action'])),
                               selected_action=move if value else None, accepted=value)
                planned_diagnostics[actor] = planned
            if actor in value_thresholds:
                row = value_thresholds[actor]
                critical = POSTURE_PARAMS[posture]["critical_ratio"]
                adjusted_critical = effective_entry_threshold(critical, actor, unit_values)
                ratio = (unit["health"] + unit["shield"]) / maximum
                row.update(original_critical_threshold=critical,
                           effective_critical_threshold=adjusted_critical,
                           retreat_reason=reason, retreat_executed=value,
                           selected_action=move if value else None,
                           changed=value and (row["changed"] or (
                               reason == "critical_despite_weapon"
                               and critical <= ratio < adjusted_critical)))
            movement_diagnostics[actor] = {
                **detail, "movement_mode": movement_mode, "reason": move_reason,
                "proposed_action": move, "retreat_accepted": value, "retreat_value_reason": reason}
            if value:
                retreat_units[actor] = True
                retreat_actions[actor] = move
                retreat_reasons[actor] = f"{reason};move={move_reason}"

    retreat_set = set(retreat_units)
    healers = sorted((a for a, u in allies.items() if u["alive"] and _is_healer(u)),
                     key=lambda a: int(a[1:]))
    attackers = sorted((a for a, u in allies.items() if u["alive"] and a not in retreat_set
                        and a not in healers and _legal_attack_targets(state, a)),
                        key=lambda a: int(a[1:]))

    # Everyone able to fight goes through FocusFireBudget together —
    # including the no-retreater case. Retreating units are excluded from
    # the budget's ally set, so their damage is never counted while fleeing.
    attack_result = None
    if attackers:
        restricted = dict(state)
        restricted["allies"] = {a: allies[a] for a in attackers}
        restricted["relations"] = {a: state["relations"][a] for a in attackers}
        attack_result = allocate_focus_fire(
            restricted, n_agents,
            previous_assignments={a: t for a, t in (previous_attack_assignments or {}).items()
                                  if a in attackers},
            parameters={k: v for k, v in (parameters or {}).items()
                        if k in ("window_fire", "target_scoring", "window_tolerance", "accurate_damage")},
            unit_values=unit_values)
        attack_assignments = attack_result["assignments"]
    else:
        attack_assignments = {}

    actions, roles, reentry_decisions = [0] * n_agents, {}, {}
    for actor, unit in allies.items():
        index = int(actor[1:])
        if index >= n_agents or not unit["alive"]:
            continue  # Dead units: SMAC no-op 0 only.
        if actor in retreat_set:
            actions[index] = retreat_actions[actor]
            roles[actor] = "retreat"
        elif actor in attack_assignments or actor in attackers:
            actions[index] = attack_result["actions"][index]
            roles[actor] = "all_in_attack" if all_in_now else "attack"
        elif actor in healers:
            actions[index] = fallback_action(state, actor) or 1  # Support: heal or move.
            roles[actor] = "support"
        else:
            baseline_recover = recovery_action(
                state, actor, unit_states.get(actor, {}), (move_history or {}).get(actor), params
            ) if params.get("range_preserving_move") and not all_in_now else None
            recover = baseline_recover
            if params.get('planned_reentry') and not all_in_now:
                planned_action, planned_info = select_planned_reentry(
                    state, actor, unit_states.get(actor, {}), params)
                if planned_action is not None:
                    baseline_action = (recovery_action(
                        state, actor, unit_states.get(actor, {}), (move_history or {}).get(actor),
                        {**params, 'planned_reentry': False})
                        if params.get('range_preserving_move') else None)
                    if baseline_action is None:
                        baseline_action = fallback_action(state, actor) or 1
                    planned_info.update(applied=planned_action != baseline_action,
                                        phase='recovery_selection', baseline_action=baseline_action)
                    planned_diagnostics[actor] = planned_info
                    recover = planned_action
            if recover is not None:
                actions[index], roles[actor] = recover, "recover"
                reentry_decisions[actor] = {"reason": "bounded_cooldown_recovery", "action": recover}
            else:
                actions[index] = fallback_action(state, actor) or 1
                roles[actor] = "reposition"

    approach_diagnostics, approach_updates = {}, {}
    if params.get("coordinated_approach"):
        approach_actors = [a for a, role in roles.items() if role == "reposition"]
        fixed_actions = {a: actions[int(a[1:])] for a in roles if a not in approach_actors}
        approach_actions, approach_diagnostics, approach_updates = coordinate_approaches(
            state, approach_actors, unit_states, fixed_actions, params)
        for actor, action in approach_actions.items():
            actions[int(actor[1:])] = action

    engagement_updates = {}
    if params.get('engagement_commitment'):
        engagement_updates = finalize_engagement(
            state, unit_states, actions, roles, attack_assignments, retreat_actions, retreat_reasons,
            engagement_diagnostics, params)
        for actor, row in engagement_diagnostics.items():
            if row.get('repeated_move_blocked'):
                actual = actions[int(actor[1:])]
                if actor in approach_updates:
                    approach_updates[actor].update(
                        approach_action=actual,
                        approach_expected_travel=dist(state['allies'][actor]['position'],
                                                     predicted_move(state, actor, actual)['destination']))
                if actor in approach_diagnostics:
                    approach_diagnostics[actor].update(
                        pre_engagement_action=approach_diagnostics[actor]['selected_action'],
                        selected_action=actual, route_changed_by_engagement=True)
                if actor in reentry_decisions:
                    reentry_decisions[actor].update(
                        pre_engagement_action=reentry_decisions[actor]['action'], action=actual,
                        route_changed_by_engagement=True)
            if row.get('repeated_move_blocked') and actor in retreat_actions:
                actual = retreat_actions[actor]
                if actor in critical_diagnostics:
                    before = critical_diagnostics[actor]
                    critical_diagnostics[actor] = {
                        **before, **critical_escape_conditions(state, actor, actual, params),
                        'applied': before['applied'], 'accepted': before['accepted'],
                        'reason': before['reason'], 'route_changed_by_engagement': True}
                if actor in movement_diagnostics:
                    movement_diagnostics[actor].update(
                                                      pre_engagement_action=movement_diagnostics[actor].get('proposed_action'),
                                                      selected_action=actual, proposed_action=actual,
                                                      route_changed_by_engagement=True)

    reengaged = sorted((a for a in roles
                        if roles[a] in ("attack", "all_in_attack")
                        and unit_states.get(a, DEFAULT_UNIT_STATE).get("role") in ("retreat", "recover")),
                       key=lambda a: int(a[1:]))
    reason = None
    if not any(u["alive"] for u in allies.values()):
        reason = "all_units_dead"

    value_diagnostics = {}
    if enabled_values(unit_values):
        for actor, row in value_thresholds.items():
            index = int(actor[1:])
            row["selected_action"] = actions[index] if 0 <= index < n_agents else None
            row["retreat_executed"] = 0 <= index < n_agents and actor in retreat_set
            row["changed"] = row["changed"] and row["retreat_executed"]
        value_diagnostics = {**(attack_result or {}).get("unit_value_diagnostics", {}),
                             "adaptive_retreat": {"thresholds": value_thresholds,
                                                  "all_in": all_in_now,
                                                  "influenced_decisions": sum(r["changed"] for r in value_thresholds.values()),
                                                  "changed": any(r["changed"] for r in value_thresholds.values())}}

    if params.get('planned_reentry') and planned_diagnostics:
        # Attribute actual final action changes, including interaction with
        # approach coordination. Comparing only proposed retreat candidates
        # would overcount when the later allocator selects the same action.
        baseline_result = allocate_adaptive_focus_fire(
            state, n_agents, unit_states=original_unit_states, all_in=all_in, stall=stall,
            contacted=contacted, posture=posture,
            parameters={**params, 'planned_reentry': False},
            previous_attack_assignments=previous_attack_assignments,
            move_history=move_history, unit_values=unit_values)
        for actor, diagnostic in planned_diagnostics.items():
            index = int(actor[1:])
            diagnostic.update(applied=actions[index] != baseline_result['actions'][index],
                              baseline_final_action=baseline_result['actions'][index],
                              selected_final_action=actions[index],
                              attribution='same_snapshot_final_action_with_planned_reentry_disabled')

    return {
        "actions": actions,
        "phase": "all_in" if all_in_now else ("mixed" if retreat_set else "engage"),
        "all_in": all_in_now,
        "all_in_reason": all_in_reason,
        "attack_units": sorted((a for a in roles if roles[a] in ("attack", "all_in_attack")),
                               key=lambda a: int(a[1:])),
        "retreat_units": sorted(retreat_set, key=lambda a: int(a[1:])),
        "support_units": sorted((a for a in roles if roles[a] == "support"),
                                key=lambda a: int(a[1:])),
        "attack_assignments": attack_assignments,
        "retreat_actions": retreat_actions,
        "retreat_reasons": retreat_reasons,
        "roles": roles,
        "reengaged": reengaged,
        "predicted_damage": (attack_result or {}).get("predicted_damage"),
        "estimated_overkill": (attack_result or {}).get("estimated_overkill"),
        "immediate_damage": (attack_result or {}).get("immediate_damage", {}),
        "window_only_damage": (attack_result or {}).get("window_only_damage", {}),
        "window_fire_enabled": bool(params.get("window_fire")),
        "target_scoring_enabled": bool(params.get("target_scoring")),
        "range_preserving_move_enabled": bool(params.get("range_preserving_move")),
        "movement_diagnostics": movement_diagnostics,
        "approach_diagnostics": approach_diagnostics,
        "approach_state_updates": approach_updates,
        "safe_reentry_diagnostics": {a: row["safe_reentry"] for a, row in movement_diagnostics.items()
                                     if "safe_reentry" in row},
        "critical_escape_diagnostics": critical_diagnostics,
        "planned_reentry_diagnostics": planned_diagnostics,
        "engagement_diagnostics": engagement_diagnostics,
        "engagement_state_updates": engagement_updates,
        "engagement_commitment_enabled": bool(params.get('engagement_commitment')),
        "damage_model": (attack_result or {}).get('damage_model', {}),
        "unit_value_diagnostics": value_diagnostics,
        "reentry_decisions": reentry_decisions,
        "focus_fire_terminated": (attack_result or {}).get("termination_reason"),
        "terminated": reason is not None,
        "termination_reason": reason,
    }
