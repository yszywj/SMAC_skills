"""Shared movement scoring: reduce danger while preserving fire opportunity.

Inspired by Sharpy's generic_micro.py / pathing_manager.py idea that moving
away from enemies is not intrinsically better — the best retreat position is
one that is safer AND lets the unit keep (or quickly regain) its next attack.
Re-designed for SMAC's four legal moves; no new coordinates, no pathing, no
hidden enemy data. Scores are heuristic estimates of safety/attack value and
are never claims about damage avoided or dealt.

Two internal movement intents (NOT top-level skills):

- ``tactical`` (TacticalReposition): reposition during weapon cooldown while
  staying at fighting distance from the focus target.
- ``emergency`` (EmergencyRetreat): clear mortality risk; leaving attack
  range is acceptable, re-engagement must stay possible.
"""

import math

from .combat_timing import can_fire_within_window, window_loops
from .focus_fire import _compatible_weapon

PARAMS = {
    "threat_weight": 1.0,          # per world unit of distance to closest visible enemy
    "danger_radius": 5.5,          # enemy closer than this after the move is dangerous
    "danger_penalty": 2.0,         # per dangerous enemy after the move
    "fire_opportunity_weight": 0.8,  # value of keeping/holding the next shot
    "support_weight": 0.15,        # preference for moving toward allied support
    "reentry_cost": 1.5,           # cost of leaving own weapon range of the target
    "direction_change_cost": 0.25, # penalty for flipping direction (anti ping-pong)
    "range_buffer": 0.75,          # stay this far inside own weapon range when tactical
    "melee_range": 1.5,            # own native range below this counts as melee
    "approach_weight": 1.2,        # melee/shorter-range: closing distance is valuable
    "movement_improvement_min": 0.0,  # ties keep the attack; smaller scores are better
}


def predicted_move(state, actor, action_id):
    """Estimate travel in one control window, not the full command distance.

    SC2 catalog movement_speed is in simulation units per 16 game loops.
    Acceleration, collision and moving enemies remain unmodelled.
    """
    nominal = state["movement_candidates"][actor][str(action_id)]
    origin = state["allies"][actor]["position"]
    unit_id = str(state["allies"][actor]["unit_type_id"])
    speed = state.get("unit_type_profiles", {}).get(unit_id, {}).get("movement_speed_native")
    length = math.dist(origin, nominal["destination"])
    fraction = min(1.0, speed * state["step_mul"] / 16.0 / length) \
        if speed is not None and "step_mul" in state and length > 0 else 1.0
    destination = [a + fraction * (b - a) for a, b in zip(origin, nominal["destination"])]
    return {**nominal, "destination": destination,
            "distances_to_visible_enemies": {
                e: math.dist(destination, u["position"]) for e, u in state["visible_enemies"].items()}}


def engagement_reach(state, actor, target_id):
    """Intersection of estimated native reach and the command candidate limit."""
    relation = state.get("relations", {}).get(actor, {}).get(target_id, {})
    native_range = relation.get("native_weapon_range")
    target = state["visible_enemies"].get(target_id)
    if native_range is None or target is None:
        return None
    unit = state["allies"][actor]
    reach = native_range + unit["radius"] + target["radius"]
    mask_range = relation.get("smac_attack_distance_limit")
    return min(reach, mask_range) if mask_range is not None else reach


def threat_count(state, actor, position, parameters=None):
    """Visible enemy weapon envelopes, with a heuristic fallback if unknown."""
    params = {**PARAMS, **(parameters or {})}
    unit = state["allies"][actor]
    count = 0
    for enemy in state["visible_enemies"].values():
        profile = state.get("unit_type_profiles", {}).get(str(enemy["unit_type_id"]), {})
        weapon = _compatible_weapon(profile, unit)
        reach = (weapon["range"] + unit["radius"] + enemy["radius"]
                 if weapon and weapon.get("range") is not None else params["danger_radius"])
        count += math.dist(position, enemy["position"]) <= reach
    return count


def _finite_number(value, fallback):
    try:
        return float(value) if math.isfinite(float(value)) else fallback
    except (TypeError, ValueError):
        return fallback


def visible_volley_pressure(state, actor, position, parameters=None):
    """Geometry-weighted visible base volleys, not expected incoming damage.

    Enemy cooldown, intent, bonuses and projectiles are not observed here.
    A fallback is only a conservative scoring proxy, never a claimed volley.
    """
    params = {**PARAMS, **(parameters or {})}
    unit = state['allies'][actor]
    pressure = 0.0
    for enemy in state['visible_enemies'].values():
        profile = state.get('unit_type_profiles', {}).get(str(enemy['unit_type_id']), {})
        weapon = _compatible_weapon(profile, unit)
        native_range = _finite_number(weapon.get('range'), None) if weapon else None
        reach = (native_range + unit['radius'] + enemy['radius']
                 if native_range is not None else params['danger_radius'])
        if math.dist(position, enemy['position']) <= reach:
            base = _finite_number(weapon.get('base_damage'), 12) if weapon else 12
            hits = _finite_number(weapon.get('hits_per_attack'), 1) if weapon else 1
            pressure += max(0.0, base) * max(0.0, hits)
    return pressure


def estimated_attack_timing(state, actor, position, target_id, first_move=False, parameters=None):
    """Static-target earliest-shot estimate after an optional movement window.

    Reach is intersected with the SMAC command distance. Future pathing and
    masks are unknown; this never creates an attack action or legal target.
    """
    reach = engagement_reach(state, actor, target_id)
    target = state['visible_enemies'].get(target_id)
    if reach is None or target is None:
        return None
    unit = state['allies'][actor]
    travel = max((math.dist(unit['position'], predicted_move(state, actor, a)['destination'])
                  for a in state.get('movement_candidates', {}).get(actor, {})), default=0.0)
    gap = max(0.0, math.dist(position, target['position']) - reach)
    if gap > 0 and travel <= 0:
        return None
    windows = math.ceil(max(0.0, gap - 1e-6) / travel) if travel else 0
    # Travel consumes real control loops; firing tolerance does not make the
    # unit move for an extra loop.
    loops = max(1.0, _finite_number(state.get('step_mul'), 8))
    movement_loops = (int(first_move) + windows) * loops
    cooldown = max(0.0, _finite_number(unit.get('weapon_cooldown_loops'), 0))
    return {'estimated_shot_loops': round(max(cooldown, movement_loops), 3),
            'estimated_remaining_move_windows': windows,
            'distance_outside_attack_envelope': round(gap, 3),
            'estimated_movement_loops': movement_loops,
            'extra_delay_after_cooldown': round(max(0.0, movement_loops - cooldown), 3)}


def select_planned_reentry(state, actor, history, parameters=None):
    """Plan bounded recovery jointly with readiness instead of fixed laterals.

    Only follows a retreat/recovery with no legal attack. The first action
    is a supplied legal move, or SMAC Stop for a living unit. Later travel is
    an estimate; the actual attack allocator checks a fresh mask next step.
    """
    params = parameters or {}
    info = {'applied': False, 'candidates': {}, 'meaning':
            'Visible static geometry and base-volley pressure; not damage or a path guarantee.'}
    if (history.get('role') not in ('retreat', 'recover')
            or history.get('recover_steps', 0) >= params.get('max_recover_steps', 2)
            or any(r.get('smac_attack_available') for r in state.get('relations', {}).get(actor, {}).values())
            or not state['visible_enemies']):
        return None, info
    unit = state['allies'][actor]
    origin = unit['position']
    targets = [e for e, r in state.get('relations', {}).get(actor, {}).items()
               if e in state['visible_enemies'] and r.get('weapon_range_status') != 'no_compatible_weapon'
               and engagement_reach(state, actor, e) is not None]
    if not targets:
        return None, info
    cooldown = max(0.0, _finite_number(unit.get('weapon_cooldown_loops'), 0))
    loops = max(1.0, _finite_number(state.get('step_mul'), 8))
    current_pressure = visible_volley_pressure(state, actor, origin, params)
    current_exposure = threat_count(state, actor, origin, params)
    health = max(1.0, unit.get('health', 0) + unit.get('shield', 0))
    candidates = [(int(a), predicted_move(state, actor, a)['destination'])
                  for a in state.get('movement_candidates', {}).get(actor, {})]
    # Waiting is bounded by existing recover_steps and considered only while
    # cooling; a ready unit always takes a legal route back if one exists.
    if cooldown > window_loops(state, params):
        candidates.append((1, origin))
    ranked = []
    for action, destination in candidates:
        timing_options = [(estimated_attack_timing(state, actor, destination, e, True, params), e)
                          for e in targets]
        timing_options = [(t, e) for t, e in timing_options if t is not None]
        if not timing_options:
            continue
        timing, target = min(timing_options, key=lambda pair: (pair[0]['estimated_shot_loops'],
                                                               pair[0]['distance_outside_attack_envelope'], pair[1]))
        exposure = threat_count(state, actor, destination, params)
        pressure = visible_volley_pressure(state, actor, destination, params)
        # Do not enter extra enemy envelopes while too early to shoot in the
        # following window. Approaching safely outside range remains allowed.
        rejected = cooldown > 2 * loops and exposure > current_exposure and pressure > current_pressure
        score = (timing['estimated_shot_loops'] / loops
                 + 1.5 * min(3.0, pressure / health)
                 + 0.05 * timing['distance_outside_attack_envelope'])
        info['candidates'][str(action)] = {**timing, 'target': target,
            'predicted_destination': list(destination), 'threat_count': exposure,
            'visible_base_volley_pressure': pressure, 'score': round(score, 4),
            'rejected': 'additional_exposure_before_ready' if rejected else None}
        if not rejected:
            ranked.append((score, action))
    if not ranked:
        info['reason'] = 'no_safe_reentry_candidate'
        return None, info
    action = min(ranked)[1]
    info.update(selected_action=action, reason='timed_reentry', cooldown_loops=cooldown,
                standing_threat_count=current_exposure, standing_visible_base_volley_pressure=current_pressure)
    return action, info


def score_movement_candidate(state, actor, action_id, target_id, movement_mode,
                             parameters=None, previous_action=None):
    """Score one legal movement candidate (smaller is better).

    Only legal snapshot moves are scored (the caller passes candidates from
    ``movement_candidates``); the nominal destination is a prediction — SC2
    may not fully realize it, so no hard guarantees are derived from it.
    """
    move = predicted_move(state, actor, action_id)
    return _score_position(state, actor, move, target_id, movement_mode,
                           parameters, action_id, previous_action)


def _score_position(state, actor, move, target_id, movement_mode,
                    parameters=None, action_id=None, previous_action=None):
    """Same objective for moving and staying; staying is not a Stop command."""
    params = {**PARAMS, **(parameters or {})}
    relations = state.get("relations", {}).get(actor, {})
    destination = move["destination"]
    distances = list(move["distances_to_visible_enemies"].values())
    min_enemy = min(distances) if distances else params["danger_radius"] + 1
    danger_count = threat_count(state, actor, destination, parameters)

    # Tactical mode optimizes the fighting position, not raw distance: the
    # threat term is scaled down so a comfortable in-range spot can beat a
    # farther but range-breaking one. Emergency mode keeps full weight.
    effective_threat = params["threat_weight"] * (0.4 if movement_mode == "tactical" else 1.0)
    score = -min_enemy * effective_threat + danger_count * params["danger_penalty"]

    # Fire opportunity: staying inside own native range of the focus target
    # matters most when the weapon can fire within this window. Three zones:
    # comfortably inside (bonus), at the edge without buffer (small cost),
    # out of range (re-entry cost). Melee units in tactical mode weight
    # threat ~0: stepping away from a melee target saves nothing, closing in
    # restores the next attack.
    fires_soon = can_fire_within_window(state, actor, target_id,
                                        parameters=parameters) \
        if target_id in relations else False
    relation = relations.get(target_id)
    native_range = (relation or {}).get("native_weapon_range")
    melee = native_range is not None and native_range <= params["melee_range"]
    if melee and movement_mode == "tactical":
        score += min_enemy * effective_threat  # Cancel the raw distance term.
    if native_range is not None:
        target_distance = ((destination[0] - state["visible_enemies"][target_id]["position"][0]) ** 2
                           + (destination[1] - state["visible_enemies"][target_id]["position"][1]) ** 2) ** 0.5
        reach = engagement_reach(state, actor, target_id)
        if movement_mode == "tactical":
            if target_distance <= reach - params["range_buffer"]:
                score -= params["fire_opportunity_weight"]  # Keeps the next shot.
            elif target_distance <= reach:
                score += 0.5 * params["reentry_cost"]  # At the edge: fragile position.
            else:
                score += params["reentry_cost"]  # Must close back in later.
        else:  # emergency: leaving range is accepted, but re-entry gets no bonus
            if fires_soon and target_distance > reach:
                score += params["reentry_cost"] * 0.5
    elif fires_soon:
        score -= params["fire_opportunity_weight"] * 0.5  # Unknown range: mild bonus.

    # Allied support: closer to living non-retreating allies is better.
    supporters = [u["position"] for a, u in state["allies"].items()
                  if a != actor and u.get("alive") and not u.get("__retreating__")]
    if supporters:
        support = min((destination[0] - p[0]) ** 2 + (destination[1] - p[1]) ** 2
                      for p in supporters) ** 0.5
        score += support * params["support_weight"]

    if action_id is not None and previous_action is not None and str(previous_action) != str(action_id):
        score += params["direction_change_cost"]
    if params.get('planned_reentry'):
        timing = estimated_attack_timing(state, actor, destination, target_id,
                                          action_id is not None, params)
        if timing is not None:
            score += timing['extra_delay_after_cooldown'] / window_loops(state, params)
        # Small pressure term distinguishes envelopes with different base
        # volleys without pretending that every enemy will actually fire.
        score += 0.02 * visible_volley_pressure(state, actor, destination, params)
    return round(score, 3)


def _fire_opportunity(state, actor, destination, target_id, parameters=None):
    relation = state.get("relations", {}).get(actor, {}).get(target_id, {})
    target = state["visible_enemies"].get(target_id)
    reach = engagement_reach(state, actor, target_id)
    if target is None or reach is None:
        return None
    unit = state["allies"][actor]
    distance = sum((a - b) ** 2 for a, b in zip(destination, target["position"])) ** 0.5
    return (distance <= reach
            and unit.get("weapon_cooldown_loops", 0) <= window_loops(state, parameters))


def select_movement(state, actor, target_id, movement_mode, previous_action=None,
                    parameters=None):
    """Best legal move under the given intent. Returns (action, reason, info).

    Tactical mode keeps standing fire when no move improves its score.
    Emergency mode ordinarily moves; opt-in safe_reentry also compares
    standing and preserves a near-ready shot unless coverage decreases or
    recent severe damage to a nearly dead unit justifies the escape.
    """
    planned = bool((parameters or {}).get('planned_reentry'))
    baseline = (select_movement(state, actor, target_id, movement_mode, previous_action,
                                {**parameters, 'planned_reentry': False}) if planned else None)
    moves = state.get("movement_candidates", {}).get(actor, {})
    position = state["allies"][actor]["position"]
    standing = {"destination": position, "distances_to_visible_enemies": {
        enemy_id: sum((a - b) ** 2 for a, b in zip(position, enemy["position"])) ** 0.5
        for enemy_id, enemy in state["visible_enemies"].items()}}
    standing_score = _score_position(state, actor, standing, target_id, movement_mode, parameters)
    info = {"movement_mode": movement_mode, "standing_score": standing_score,
            "movement_candidate_scores": {}, "selected_action": None,
            "predicted_destinations": {}, "rejected_candidates": {},
            "standing_threat_count": threat_count(state, actor, position, parameters),
            "standing_fire_opportunity": _fire_opportunity(state, actor, position, target_id, parameters),
            "meaning": "Heuristic position scores and geometric fire estimates; not observed damage. "
                       "No move means return to the attack allocator, not issue Stop."}
    if planned:
        info['planned_reentry'] = {'applied': False, 'phase': 'retreat_selection',
            'baseline_action': baseline[0], 'baseline_reason': baseline[1],
            'standing_timing': estimated_attack_timing(state, actor, position, target_id, False, parameters),
            'candidate_timing': {}}
    if not moves:
        return None, "no_legal_moves", info
    scored = []
    reach = engagement_reach(state, actor, target_id)
    target = state["visible_enemies"].get(target_id)
    currently_engaged = reach is not None and math.dist(position, target["position"]) <= reach
    safe_reentry = bool((parameters or {}).get("safe_reentry"))
    unit = state["allies"][actor]
    maximum = unit.get("max_health", 0) + unit.get("max_shield", 0) or 1
    survival = (unit.get("health", 0) + unit.get("shield", 0)) / maximum
    recent_loss = (unit.get("health_lost_since_previous_decision", 0)
                   + unit.get("shield_lost_since_previous_decision", 0)) / maximum
    # This is a narrow exception, not permission for every low-health unit
    # to surrender its next shot. Genuine danger reduction always qualifies.
    extreme = survival <= 0.08 and recent_loss >= 0.20
    if safe_reentry:
        info["safe_reentry"] = {"enabled": True, "critical_exception": extreme,
                                 "cooldown_loops": unit.get("weapon_cooldown_loops", 0),
                                 "standing_next_window_attack": currently_engaged and
                                     unit.get("weapon_cooldown_loops", 0) <= 2 * window_loops(state, parameters),
                                 "candidate_checks": {}}
    for action_id in moves:
        prediction = predicted_move(state, actor, action_id)
        if planned:
            info['planned_reentry']['candidate_timing'][str(action_id)] = {
                **(estimated_attack_timing(state, actor, prediction['destination'], target_id, True, parameters) or {}),
                'visible_base_volley_pressure': visible_volley_pressure(state, actor, prediction['destination'], parameters)}
        info["predicted_destinations"][str(action_id)] = prediction["destination"]
        next_exposure = threat_count(state, actor, prediction["destination"], parameters)
        exits_attack = (currently_engaged and
                        math.dist(prediction["destination"], target["position"]) > reach)
        fires_next_window = unit.get("weapon_cooldown_loops", 0) <= 2 * window_loops(state, parameters)
        reduces_exposure = next_exposure < info["standing_threat_count"]
        if safe_reentry:
            travel = math.dist(position, prediction["destination"])
            distance_outside_reach = (max(0.0, math.dist(prediction["destination"], target["position"]) - reach)
                                      if target is not None and reach is not None else None)
            info["safe_reentry"]["candidate_checks"][str(action_id)] = {
                "exits_attack_envelope": exits_attack,
                "predicted_threat_count": next_exposure,
                "reduces_exposure": reduces_exposure,
                "weapon_ready_by_next_window": fires_next_window,
                "estimated_reentry_windows": (round(distance_outside_reach / travel, 3)
                                               if travel > 0 and distance_outside_reach is not None else None),
                "predicted_next_window_attack": (
                    fires_next_window and math.dist(prediction["destination"], target["position"]) <= reach
                    if reach is not None and target is not None else None)}
            if exits_attack and fires_next_window and not reduces_exposure and not extreme:
                info["rejected_candidates"][str(action_id)] = "loses_next_attack_without_reducing_exposure"
                continue
        # Do not give up the ability to issue attacks for a tactical move that
        # still leaves the unit in as many enemy weapon envelopes.
        if (movement_mode == "tactical" and currently_engaged
                and math.dist(prediction["destination"], target["position"]) > reach
                and threat_count(state, actor, prediction["destination"], parameters)
                >= info["standing_threat_count"]):
            info["rejected_candidates"][str(action_id)] = "loses_attack_without_reducing_exposure"
            continue
        score = score_movement_candidate(state, actor, action_id, target_id,
                                         movement_mode, parameters, previous_action)
        scored.append((score, int(action_id)))
    scored.sort()
    if not scored:
        return None, "keep_attack_position", info
    best_score, best_action = scored[0]
    info.update(movement_candidate_scores={str(a): s for s, a in scored},
                best_move_action=best_action, best_move_score=best_score,
                improvement=round(standing_score - best_score, 3),
                keeps_fire_opportunity=_fire_opportunity(
                    state, actor, predicted_move(state, actor, best_action)["destination"],
                    target_id, parameters),
                selected_threat_count=threat_count(
                    state, actor, predicted_move(state, actor, best_action)["destination"], parameters))
    margin = {**PARAMS, **(parameters or {})}["movement_improvement_min"]
    if movement_mode == "tactical" and best_score >= standing_score - margin:
        return None, "keep_attack_position", info
    if (safe_reentry and movement_mode == "emergency" and not extreme
            and info["selected_threat_count"] >= info["standing_threat_count"]
            and best_score >= standing_score - margin):
        return None, "keep_attack_position", info
    info["selected_action"] = best_action
    return best_action, "scored_move", info
