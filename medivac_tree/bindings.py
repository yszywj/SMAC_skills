"""Explicit filtering and lexicographic binding for parameterized tree leaves.

Every comparison key is returned in the log. There are no weighted sums,
teacher logits, global matching, or calls to the old controller.
"""

import math
from .facts import (actor_index, clearance, enough_energy, exposure, finite, heal_candidates,
                    hp_ratio, loss_ratio, moves, observed_target, position, wounded_patient)


def critical(unit, parameters):
    return hp_ratio(unit) <= parameters["critical_hp_ratio"] or (
        hp_ratio(unit) <= parameters["rapid_loss_hp_ratio"] and loss_ratio(unit) >= parameters["rapid_loss_ratio"])


def build_bindings(state, actor, legal, old, reservations, threat_rows, parameters):
    allies = state["allies"]
    unit = allies[actor]
    origin = position(unit)
    loop, step = state["game_loop"], state.get("step_mul", 8)
    current = observed_target(state, actor)
    current_tag = allies[current].get("tag") if current else None
    same_order = current is not None and old.get("order_target") == current and old.get("order_target_tag") == current_tag
    destinations = moves(state, actor, legal)
    all_candidates = heal_candidates(state, actor, legal)
    free = {t: a for t, a in all_candidates.items() if reservations.get(t, actor) == actor}
    blocked = sorted(set(all_candidates) - set(free), key=actor_index)
    bindings = {}
    if 1 in legal:
        bindings["STOP"] = {"action": 1, "target": None, "reason": "explicit_stop_cancels_order", "selection_key": []}

    def target_key(target):
        patient = allies[target]
        return (not critical(patient, parameters), hp_ratio(patient), -loss_ratio(patient),
                math.dist(origin, position(patient)), actor_index(target))

    candidate_log = [{"target": t, "action": free[t], "selection_key": list(target_key(t))}
                     for t in sorted(free, key=target_key)]
    if current in free:
        bindings["HEAL_CURRENT"] = {"action": free[current], "target": current,
                                    "reason": "reissue_legal_observed_target", "selection_key": []}
    others = [t for t in free if t != current]
    if others:
        target = min(others, key=target_key)
        bindings["HEAL_NEW"] = {"action": free[target], "target": target,
                                "reason": "critical_then_hp_loss_distance_id", "selection_key": list(target_key(target))}
    cooldown = old.get("last_rescue_loop")
    cooldown_ready = cooldown is None or loop - cooldown >= parameters["rescue_cooldown_windows"] * step
    stable = current is not None and hp_ratio(allies[current]) >= parameters["stable_hp_ratio"] and loss_ratio(allies[current]) < parameters["stable_loss_ratio"]
    rescue = [t for t in others if critical(allies[t], parameters)]
    if stable and cooldown_ready and rescue:
        target = min(rescue, key=target_key)
        bindings["HEAL_RESCUE"] = {"action": free[target], "target": target,
                                   "reason": "stable_patient_to_critical_unoccupied_patient", "selection_key": list(target_key(target))}

    # Self-preservation and support use disclosed lexicographic geometric keys.
    movement_log = []
    retreat_choices = []
    for action, point in destinations.items():
        key = (exposure(threat_rows, point), -clearance(threat_rows, point), action)
        retreat_choices.append((key, action))
        movement_log.append({"action": action, "destination": list(point), "retreat_key": list(key)})
    if retreat_choices:
        key, action = min(retreat_choices)
        bindings["RETREAT"] = {"action": action, "target": None, "reason": "least_exposure_then_clearance_direction", "selection_key": list(key)}
    elif 1 in legal:
        bindings["RETREAT"] = {"action": 1, "target": None, "reason": "no_legal_retreat_move_stop", "selection_key": []}

    # KEEP only retains an actual order; its safety/expiry tests are visible facts.
    keep_old = old.get("keep", {}) if same_order else {}
    keep_record = {}
    keep_rejections = []
    contract = state.get("action_contract", {}).get("keep_heal_order", {})
    candidate = state.get("keep_heal_order_candidates", {}).get(actor, {})
    keep_action = contract.get("action_id")
    keep_eligible = (current is not None and current not in all_candidates and enough_energy(unit)
                     and wounded_patient(allies[current]) and reservations.get(current, actor) == actor
                     and contract.get("enabled") is True and isinstance(keep_action, int) and not isinstance(keep_action, bool)
                     and keep_action >= 6 and keep_action in legal and candidate.get("action_id") == keep_action
                     and candidate.get("target") == current)
    if current:
        gap = math.dist(origin, position(allies[current]))
        energy = unit.get("energy") if finite(unit.get("energy")) else None
        productive = bool(keep_old and energy is not None and finite(keep_old.get("energy")) and keep_old["energy"] - energy > parameters["energy_drop"])
        progress = keep_old.get("gap", gap) - gap
        following = bool(keep_old and progress >= parameters["approach_min_progress"])
        stalled = 0 if productive or following else keep_old.get("stalled", 0) + int(bool(keep_old))
        unconfirmed = 0 if productive else keep_old.get("unconfirmed", 0) + 1
        keep_record = {"target": current, "target_tag": current_tag, "gap": gap, "energy": energy,
                       "stalled": stalled, "unconfirmed": unconfirmed, "progress": progress,
                       "productive": productive, "following": following}
        if stalled >= parameters["keep_stall_windows"]:
            keep_rejections.append("keep_stalled")
        if unconfirmed > parameters["keep_max_unconfirmed_windows"]:
            keep_rejections.append("keep_unconfirmed_limit")
        # Automatic following cannot be precisely predicted. Require both
        # endpoints outside visible threat envelopes; this is conservative.
        if exposure(threat_rows, origin) or exposure(threat_rows, position(allies[current])):
            keep_rejections.append("visible_threat_along_follow_endpoints")
    if keep_eligible and not keep_rejections:
        bindings["KEEP_HEAL"] = {"action": keep_action, "target": current,
                                 "reason": "retain_observed_order_with_progress_budget", "selection_key": []}

    # Approach is bounded across real observations and never represented as healing.
    approach_old = old.get("approach", {})
    approach_record = {}
    approach_rejections = []
    previous_target = approach_old.get("target")
    valid_previous = previous_target in allies and allies[previous_target].get("tag") == approach_old.get("target_tag") and wounded_patient(allies[previous_target])
    if not valid_previous:
        approach_old = {}
    cooldown_until = approach_old.get("cooldown_until", -1)
    if loop < cooldown_until:
        approach_rejections.append("approach_cooldown")
        approach_record = dict(approach_old)
    elif approach_old and "cooldown_until" not in approach_old:
        gap = math.dist(origin, position(allies[previous_target]))
        if approach_old.get("windows", 0) >= parameters["approach_max_windows"] or approach_old.get("gap", gap) - gap < parameters["approach_min_progress"]:
            approach_rejections.append("approach_expired_or_no_observed_progress")
            approach_record = {**approach_old, "cooldown_until": loop + step * parameters["approach_cooldown_windows"]}
    elif "cooldown_until" in approach_old:
        approach_old = {}
    options = []
    if enough_energy(unit) and not current and not approach_rejections:
        for target, patient in allies.items():
            if target in all_candidates or target in reservations or not wounded_patient(patient):
                continue
            gap = math.dist(origin, position(patient))
            for action, point in destinations.items():
                next_gap = math.dist(point, position(patient))
                if gap - next_gap < parameters["approach_min_progress"] or exposure(threat_rows, point) > exposure(threat_rows, origin):
                    continue
                key = (target != approach_old.get("target"), not critical(patient, parameters), hp_ratio(patient),
                       -loss_ratio(patient), next_gap, actor_index(target), action)
                options.append((key, target, action, gap, next_gap))
    if options:
        key, target, action, gap, next_gap = min(options)
        bindings["APPROACH"] = {"action": action, "target": target, "reason": "bounded_safe_approach_no_heal_claim", "selection_key": list(key), "gap": gap, "next_gap": next_gap}
        approach_record = {"target": target, "target_tag": allies[target].get("tag"), "gap": gap,
                           "windows": approach_old.get("windows", 0) + 1 if target == approach_old.get("target") else 1}

    friends = [position(u) for u in allies.values() if u.get("alive") and not str(u.get("type", "")).lower().startswith("medivac") and u.get("unit_type_id") != 54]
    anchor = [sum(p[i] for p in friends) / len(friends) for i in (0, 1)] if friends else list(origin)
    support = []
    for action, point in {**destinations, **({1: origin} if 1 in legal else {})}.items():
        key = (exposure(threat_rows, point), max(0., math.dist(point, anchor) - 2.), -clearance(threat_rows, point), action != 1, action)
        support.append((key, action))
    if support:
        key, action = min(support)
        bindings["SUPPORT"] = {"action": action, "target": None, "reason": "least_exposure_then_friendly_support", "selection_key": list(key), "anchor": anchor}
    details = {"candidate_count": len(free), "candidate_bindings": candidate_log, "blocked_targets": blocked,
               "movement_bindings": movement_log, "current_target": current, "current_target_tag": current_tag,
               "rescue_cooldown_ready": cooldown_ready, "current_patient_stable": stable,
               "keep": keep_record, "keep_eligible": keep_eligible, "keep_rejections": keep_rejections,
               "approach": approach_record, "approach_rejections": approach_rejections}
    return bindings, details
