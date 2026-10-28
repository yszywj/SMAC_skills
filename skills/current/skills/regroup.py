"""Regroup: budgeted re-concentration of a scattered controlled ally set.

A top-level spatial skill. It never fires weapons and never claims a path:
legal actions only, Stop (1) as a legal wait. Returns a partial action
proposal (actor-keyed); the scheduler merges it into the final action array.

Cohesion contract (v8.1): the FULL live roster is observed for collisions,
support and anchor choice (``observed_cohesion``), but completion is judged on
the CONTROLLED set only (``controlled_cohesion``): other allies already being
bunched must never make a split controlled set "succeed".

Interrupt rules: any visible legal attack edge (contact) or fresh observed
damage interrupts the rally. ``interrupted`` / ``blocked`` / ``timeout`` are
never reported as success.
"""

import math

from .group_geometry import (PARAMS as GEOMETRY_PARAMS, best_move_toward,
                             cohesion_report, compactness, ring_slots, unit_role)

PARAMS = {
    **GEOMETRY_PARAMS,
    "regroup_entry_share": 0.5,      # main share below this (or many groups) = clearly split
    "regroup_target_share": 0.8,     # completion: main share WITHIN the controlled set
    "regroup_target_radius": 5.0,    # completion: controlled-set quantile radius (0.0 valid)
    "regroup_max_windows": 24,       # total rally budget across execution windows (decision steps)
    "regroup_max_wait_windows": 3,   # main-group wait budget per unit
    "regroup_no_progress_limit": 4,  # MOVE windows without displacement -> blocked straggler
    "regroup_retry_cooldown_windows": 6,  # after timeout, wait before re-entering the same rally
    "regroup_anchor_reselect_distance": 4.0,  # anchor only moves when it improves by this much
    "regroup_spacing": 1.5,          # metres between assigned local goals
}


def _contact_or_damage(state, actor_ids):
    """Visible legal attack edge (any ally) or fresh damage on a controlled unit."""
    contact = bool(state["visible_enemies"]) or any(relation.get("smac_attack_available")
                  for row in state.get("relations", {}).values()
                  for relation in row.values())
    damaged = [a for a in actor_ids
               if (state["allies"][a].get("health_lost_since_previous_decision", 0)
                   + state["allies"][a].get("shield_lost_since_previous_decision", 0)) > 0]
    return contact, damaged


def _rally_anchor(state, controlled, observed, params):
    """Anchor near the main body of the CONTROLLED set (observed allies may
    inform the neighbourhood, but a few outliers can never drag it)."""
    anchor_pool = [a for a in observed["main_group"] if a in controlled] \
        or observed["main_group"] or controlled
    positions = [state["allies"][a]["position"] for a in anchor_pool]
    if not positions:
        return None
    return [sum(p[i] for p in positions) / len(positions) for i in range(2)]


def allocate_regroup(state, actor_ids, memory, parameters=None, previous_moves=None):
    """One Regroup decision step. ``memory`` is episode-scoped manager state.

    Contract: {"actions": {actor: legal_action_id}, "controlled_units": [...],
    "status": running|success|interrupted|blocked|timeout|unavailable,
    "reason", "state_updates", "diagnostics"}.
    ``controlled_units`` ALWAYS equals the key set of ``actions``; the full
    input membership (deduplicated, dead-filtered) is reported separately as
    ``requested_units``.
    """
    params = {**PARAMS, **(parameters or {})}
    memory = memory if memory is not None else {}
    # Deduplicate while keeping a deterministic order; dead members are not
    # controlled (their actions are the environment's no-op domain).
    requested = sorted({a for a in actor_ids if a in state["allies"]},
                       key=lambda a: (len(a), a))
    controlled = [a for a in requested if state["allies"][a].get("alive")]
    observed = cohesion_report(state, params)          # full live roster
    controlled_cohesion = cohesion_report(state, params, actors=controlled)
    contact, damaged = _contact_or_damage(state, controlled)
    result = {"actions": {}, "controlled_units": [], "requested_units": requested,
              "status": "running", "reason": "", "state_updates": {}, "diagnostics": {}}
    diagnostics = {"observed_cohesion": observed,
                   "controlled_cohesion": controlled_cohesion,
                   "contact": contact, "damaged_units": damaged}
    result["diagnostics"] = diagnostics

    if contact or damaged:
        result.update(status="interrupted",
                      reason="contact" if contact else "observed_damage")
        result["state_updates"] = {"phase": "idle", "elapsed_windows": 0, "anchor": None}
        return result
    if not controlled:
        result.update(status="unavailable", reason="no_controlled_units")
        return result
    if len(controlled) == 1:
        # A single unit has nobody to group with: succeed immediately, no move.
        result.update(status="success", reason="single_unit_nothing_to_group")
        return result

    elapsed = memory.get("elapsed_windows", 0)
    anchor = memory.get("anchor")
    new_anchor = _rally_anchor(state, controlled, observed, params)
    if new_anchor is None:
        result.update(status="unavailable", reason="no_anchor_candidates")
        return result
    # Anchor stability: keep the remembered anchor unless no controlled unit is
    # near it any more, or the new one is clearly better.
    if anchor is not None:
        near_old = any(math.dist(state["allies"][a]["position"], anchor) <= params["link_radius"]
                       for a in controlled)
        if near_old and math.dist(new_anchor, anchor) < params["regroup_anchor_reselect_distance"]:
            new_anchor = anchor
    diagnostics["anchor"] = [round(c, 3) for c in new_anchor]
    diagnostics["anchor_kept"] = anchor is not None and new_anchor == anchor

    # Completion is judged on the CONTROLLED set only. A quantile radius of
    # 0.0 is a valid "stacked" value and must not be read as missing.
    radius = controlled_cohesion["main_compactness"]["quantile_radius"]
    done = (controlled_cohesion["main_share"] >= params["regroup_target_share"]
            and radius is not None and radius <= params["regroup_target_radius"]
            and not controlled_cohesion["isolated"])
    if done:
        result.update(status="success",
                      reason=f"controlled_share={controlled_cohesion['main_share']};"
                             f"radius={radius}",
                      state_updates={"phase": "done", "elapsed_windows": 0, "anchor": None})
        diagnostics["remaining_outliers"] = controlled_cohesion["outliers"]
        return result
    if elapsed >= params["regroup_max_windows"]:
        result.update(status="timeout", reason=f"budget_exhausted({elapsed})",
                      state_updates={"phase": "cooldown",
                                     "cooldown_windows": params["regroup_retry_cooldown_windows"],
                                     "elapsed_windows": 0, "anchor": None})
        return result

    roles = {a: unit_role(state, a) for a in controlled}
    spacing = params["regroup_spacing"]
    heading = (1.0, 0.0)  # rally slots are isotropic; heading irrelevant here
    ordered = sorted(controlled,
                     key=lambda a: (roles[a] != "melee",
                                    math.dist(state["allies"][a]["position"], new_anchor), a))
    slots = ring_slots(len(ordered), spacing, [roles[a] for a in ordered], heading)
    goals = {a: [new_anchor[0] + s[0], new_anchor[1] + s[1]]
             for a, s in zip(ordered, slots)}

    # Reservations: every living ally occupies space by default; decided
    # actions overwrite with predicted endpoints.
    from .group_geometry import base_reservations, legal_actions, predicted_destination
    reservations = base_reservations(state)
    actions, blocked, waits = {}, {}, dict(memory.get("main_waits", {}))
    no_progress = dict(memory.get("no_progress", {}))
    last_positions = memory.get("last_positions", {})
    last_final_moves = memory.get("last_final_moves", {})
    # No-progress is measured on OBSERVED positions against the PREVIOUS FINAL
    # MOVE only: a unit that legally waited (Stop) at its slot is not a
    # failing straggler, and predicted endpoints never masquerade as arrivals.
    for a in controlled:
        previous_move = last_final_moves.get(a)
        if (a in last_positions and previous_move in (2, 3, 4, 5)
                and math.dist(state["allies"][a]["position"], last_positions[a]) < 0.2):
            no_progress[a] = no_progress.get(a, 0) + 1
        else:
            no_progress[a] = 0
    for a in ordered:
        unit = state["allies"][a]
        goal = goals[a]
        distance = math.dist(unit["position"], goal)
        at_goal = distance <= spacing * 0.75
        if at_goal:
            # Already at its slot: legal Stop (bounded wait), never a failing
            # straggler and never an oscillating re-approach. If Stop is not
            # legal for this unit, hold via the scheduler's base policy
            # instead of inventing an illegal action.
            if 1 in legal_actions(state, a):
                if waits.get(a, 0) >= params["regroup_max_wait_windows"]:
                    continue  # release to the scheduler's base policy
                waits[a] = waits.get(a, 0) + 1
                actions[a] = 1
                continue
            continue
        if no_progress.get(a, 0) >= params["regroup_no_progress_limit"]:
            blocked[a] = no_progress[a]  # long-stuck: do not hold the team forever
            continue
        action, info = best_move_toward(state, a, goal, reservations, params,
                                        (previous_moves or {}).get(a))
        if action is None:
            blocked[a] = "no_legal_actions"
            continue
        actions[a] = action
        reservations[a] = (predicted_destination(state, a, action), unit.get("is_flying", False))

    if blocked and not actions:
        result.update(status="blocked", reason=f"all_controlled_blocked:{sorted(blocked)}",
                      state_updates={"phase": "cooldown",
                                     "cooldown_windows": params["regroup_retry_cooldown_windows"],
                                     "elapsed_windows": 0, "anchor": None})
        return result
    result.update(actions=actions, controlled_units=sorted(actions, key=lambda a: (len(a), a)),
                  status="running", reason="rally")
    result["state_updates"] = {
        "phase": "rallying", "anchor": [round(c, 4) for c in new_anchor],
        "elapsed_windows": elapsed + 1, "main_waits": waits, "no_progress": no_progress,
        "last_positions": {a: list(state["allies"][a]["position"]) for a in controlled},
        "last_final_moves": {a: actions.get(a) for a in controlled},
        "blocked": blocked,
    }
    result["diagnostics"]["blocked_stragglers"] = blocked
    return result
