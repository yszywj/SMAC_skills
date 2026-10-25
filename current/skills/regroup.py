"""Regroup: budgeted re-concentration of a scattered controlled ally set.

A top-level spatial skill. It never fires weapons and never claims a path:
legal actions only, Stop (1) as a legal wait. Returns a partial action
proposal (actor-keyed); the scheduler merges it into the final action array.

Interrupt rules: any visible legal attack edge (contact) or fresh observed
damage interrupts the rally and hands control back at the decision boundary.
``interrupted`` / ``blocked`` / ``timeout`` are never reported as success.
"""

import math

from .group_geometry import (PARAMS as GEOMETRY_PARAMS, best_move_toward,
                             cohesion_report, compactness, ring_slots, unit_role)

PARAMS = {
    **GEOMETRY_PARAMS,
    "regroup_entry_share": 0.5,      # main share below this (or many groups) = clearly split
    "regroup_target_share": 0.8,     # completion: main share of controlled units
    "regroup_target_radius": 5.0,    # completion: main-group quantile radius
    "regroup_max_windows": 24,       # total rally budget across execution windows (decision steps)
    "regroup_max_wait_windows": 3,   # main-group wait budget per straggler
    "regroup_no_progress_limit": 4,  # windows without measurable compaction -> blocked straggler
    "regroup_retry_cooldown_windows": 6,  # after timeout, wait before re-entering the same rally
    "regroup_anchor_reselect_distance": 4.0,  # anchor only moves when it improves by this much
    "regroup_spacing": 1.5,          # metres between assigned local goals
}


def _contact_or_danger(state):
    """Visible legal attack edge or fresh observed damage ends the rally."""
    contact = any(relation.get("smac_attack_available")
                  for row in state.get("relations", {}).values()
                  for relation in row.values())
    damaged = any((u.get("health_lost_since_previous_decision", 0)
                   + u.get("shield_lost_since_previous_decision", 0)) > 0
                  for u in state["allies"].values() if u.get("alive"))
    return contact, damaged


def allocate_regroup(state, actor_ids, memory, parameters=None, previous_moves=None):
    """One Regroup decision step. ``memory`` is episode-scoped manager state.

    Contract: {"actions": {actor: legal_action_id}, "controlled_units": [...],
    "status": running|success|interrupted|blocked|timeout|unavailable,
    "reason", "state_updates", "diagnostics"}.
    """
    params = {**PARAMS, **(parameters or {})}
    memory = memory if memory is not None else {}
    live = [a for a in actor_ids if state["allies"].get(a, {}).get("alive")]
    # Observation uses the FULL ally set; control stays within actor_ids.
    report = cohesion_report(state, params)
    contact, damaged = _contact_or_danger(state)
    result = {"actions": {}, "controlled_units": [], "status": "running",
              "reason": "", "state_updates": {}, "diagnostics": {}}
    diagnostics = {"cohesion": report, "contact": contact, "damaged_observed": damaged}
    result["diagnostics"] = diagnostics

    if contact or damaged:
        result.update(status="interrupted", reason="contact" if contact else "observed_damage")
        result["state_updates"] = {"phase": "idle", "elapsed_windows": 0}
        return result
    if not live:
        result.update(status="unavailable", reason="no_controlled_units")
        return result
    if len(live) == 1:
        result.update(status="success", reason="single_unit_nothing_to_group",
                      controlled_units=live)
        return result

    elapsed = memory.get("elapsed_windows", 0)
    anchor = memory.get("anchor")
    anchor_candidates = [a for a in report["main_group"] if a in live]
    anchor_units = anchor_candidates or report["main_group"] or live
    new_anchor = None
    positions = [state["allies"][a]["position"] for a in anchor_units]
    if positions:
        new_anchor = [sum(p[i] for p in positions) / len(positions) for i in range(2)]
    # Anchor stability: keep the remembered anchor unless the new one is clearly
    # better (shorter) or the old one no longer sits near any live ally.
    if anchor is not None:
        near_old = any(math.dist(state["allies"][a]["position"], anchor) <= params["link_radius"]
                       for a in live)
        moved = math.dist(new_anchor, anchor) if new_anchor else None
        if near_old and (moved is None or moved < params["regroup_anchor_reselect_distance"]):
            new_anchor = anchor
    diagnostics["anchor"] = [round(c, 3) for c in new_anchor] if new_anchor else None
    diagnostics["anchor_kept"] = anchor is not None and new_anchor == anchor

    controlled_live = [a for a in live if a in set(actor_ids)]
    main_compact = compactness(state, [a for a in controlled_live
                                       if a in set(report["main_group"])])
    diagnostics["controlled_compactness"] = main_compact
    done = (report["main_share"] >= params["regroup_target_share"]
            and (report["main_compactness"].get("quantile_radius") or 1e9)
            <= params["regroup_target_radius"])
    if done:
        result.update(status="success",
                      reason=f"main_share={report['main_share']};"
                             f"radius={report['main_compactness'].get('quantile_radius')}",
                      controlled_units=controlled_live)
        result["diagnostics"]["remaining_outliers"] = report["outliers"]
        result["state_updates"] = {"phase": "done", "elapsed_windows": 0, "anchor": None}
        return result
    if elapsed >= params["regroup_max_windows"]:
        result.update(status="timeout",
                      reason=f"budget_exhausted({elapsed})",
                      controlled_units=controlled_live)
        result["state_updates"] = {"phase": "cooldown",
                                   "cooldown_windows": params["regroup_retry_cooldown_windows"],
                                   "elapsed_windows": 0, "anchor": None}
        return result

    roles = {a: unit_role(state, a) for a in controlled_live}
    spacing = params["regroup_spacing"]
    ordered = sorted(controlled_live,
                     key=lambda a: (roles[a] != "melee",
                                    math.dist(state["allies"][a]["position"], new_anchor)
                                    if new_anchor else 0.0, a))
    slots = ring_slots(len(ordered), spacing, [roles[a] for a in ordered])
    goals = {a: [new_anchor[0] + s[0], new_anchor[1] + s[1]]
             for a, s in zip(ordered, slots)} if new_anchor else {}

    reservations, actions, blocked = {}, {}, {}
    waits = dict(memory.get("main_waits", {}))
    no_progress = dict(memory.get("no_progress", {}))
    last_positions = memory.get("last_positions", {})
    # Progress is measured on OBSERVED positions only (actual arrival check),
    # never on predicted destinations.
    for a in controlled_live:
        if a in last_positions:
            if math.dist(state["allies"][a]["position"], last_positions[a]) < 0.2:
                no_progress[a] = no_progress.get(a, 0) + 1
            else:
                no_progress[a] = 0
    for a in ordered:
        unit = state["allies"][a]
        goal = goals.get(a)
        in_main = a in set(report["main_group"])
        distance = math.dist(unit["position"], goal) if goal else 0.0
        if goal is None:
            continue
        if in_main and distance <= spacing:
            # Main-group unit already at its slot: legal Stop, bounded wait.
            if waits.get(a, 0) >= params["regroup_max_wait_windows"]:
                actions[a] = None  # release: fall through to scheduler default
                reservations[a] = unit["position"]
                continue
            waits[a] = waits.get(a, 0) + 1
            actions[a] = 1
            reservations[a] = unit["position"]
            continue
        if no_progress.get(a, 0) >= params["regroup_no_progress_limit"]:
            # Long-stuck straggler: do not hold the team forever.
            blocked[a] = no_progress[a]
            actions[a] = None
            reservations[a] = unit["position"]
            continue
        action, info = best_move_toward(state, a, goal, reservations, params,
                                        (previous_moves or {}).get(a))
        if action is None:
            blocked[a] = "no_legal_moves"
            reservations[a] = unit["position"]
            continue
        actions[a] = action
        from .movement_scoring import predicted_move
        reservations[a] = predicted_move(state, a, action)["destination"]

    emitted = {a: act for a, act in actions.items() if act is not None}
    if blocked and len(blocked) == len(actions):
        result.update(status="blocked", reason=f"all_controlled_blocked:{sorted(blocked)}",
                      actions=emitted, controlled_units=list(emitted))
        result["state_updates"] = {"phase": "cooldown",
                                   "cooldown_windows": params["regroup_retry_cooldown_windows"],
                                   "elapsed_windows": 0, "anchor": None}
        return result
    result.update(actions=emitted, controlled_units=list(emitted), status="running",
                  reason="rally")
    result["state_updates"] = {
        "phase": "rallying", "anchor": [round(c, 4) for c in new_anchor] if new_anchor else None,
        "elapsed_windows": elapsed + 1, "main_waits": waits, "no_progress": no_progress,
        "last_positions": {a: list(state["allies"][a]["position"]) for a in controlled_live},
        "blocked": blocked,
    }
    result["diagnostics"]["blocked_stragglers"] = blocked
    return result
