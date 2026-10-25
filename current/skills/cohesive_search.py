"""CohesiveSearch: advance as one group toward a shared search target.

Top-level spatial skill for snapshots with NO current enemy position. Target
priority: (1) a valid runner-provided last-seen hint (``state['search_hint']``
or ``state['search_locations']``), (2) remembered waypoints, (3) conservative
local degenerate mode from legal moves. Shared heading/anchor plus per-role
offsets; visited waypoints expire by TTL; observed regions are recorded only
with real vision evidence (none is invented here).
"""

import math

from .group_geometry import (PARAMS as GEOMETRY_PARAMS, best_move_toward,
                             cohesion_report, ring_slots, unit_role)

PARAMS = {
    **GEOMETRY_PARAMS,
    "search_arrive_radius": 3.0,       # team anchor within this of waypoint = checked
    "search_waypoint_ttl_windows": 6,  # a checked/old waypoint keeps this priority
    "search_max_windows": 40,          # total search budget per episode phase
    "search_no_progress_limit": 5,     # windows without anchor movement -> rotate direction
    "search_leader_wait_limit": 2,     # leader wait/rejoin budget per window
    "search_severe_split_share": 0.5,  # below this main share -> needs_regroup
    "search_spacing": 2.0,
    "search_memory_ttl_loops": 1500,   # last-seen hint validity in game loops
}


def _hint_waypoint(state, params):
    """Best-evidence shared waypoint or None; source and validity recorded."""
    loop = state.get("game_loop", 0)
    hint = state.get("search_hint")
    if hint and hint.get("position") is not None:
        age = hint.get("age_loops")
        if age is None or age <= params["search_memory_ttl_loops"]:
            return list(hint["position"]), {"source": "last_seen_enemy", "age_loops": age}
    locations = state.get("search_locations")
    if locations:
        first = sorted(locations, key=lambda p: (p[0], p[1]))[0]
        return list(first), {"source": "runner_search_locations"}
    return None, None


def allocate_cohesive_search(state, actor_ids, memory, parameters=None, previous_moves=None):
    """One CohesiveSearch decision step (partial action proposal)."""
    params = {**PARAMS, **(parameters or {})}
    memory = memory if memory is not None else {}
    live = [a for a in actor_ids if state["allies"].get(a, {}).get("alive")]
    result = {"actions": {}, "controlled_units": [], "status": "running",
              "reason": "", "state_updates": {}, "diagnostics": {}}
    if not live:
        result.update(status="unavailable", reason="no_controlled_units")
        return result
    if state["visible_enemies"]:
        # Contact: stop proposing search moves; combat dispatcher takes over.
        result.update(status="interrupted", reason="enemy_visible")
        result["state_updates"] = {"phase": "idle", "elapsed_windows": 0}
        return result

    report = cohesion_report(state, params)
    result["diagnostics"]["cohesion"] = report
    elapsed = memory.get("elapsed_windows", 0)
    if elapsed >= params["search_max_windows"]:
        result.update(status="timeout", reason=f"budget_exhausted({elapsed})")
        result["state_updates"] = {"phase": "cooldown", "cooldown_windows":
                                   params["search_waypoint_ttl_windows"], "elapsed_windows": 0}
        return result
    severe = (len(report["groups"]) > 1
              and (report["main_share"] < params["search_severe_split_share"]
                   or len(report["outliers"]) >= len(report["main_group"])))
    if live and severe:
        result.update(status="blocked", reason="needs_regroup")
        result["state_updates"] = {"phase": "needs_regroup"}
        return result

    anchor_unit = report["main_group"] or live
    positions = [state["allies"][a]["position"] for a in anchor_unit]
    anchor = [sum(p[i] for p in positions) / len(positions) for i in range(2)]
    waypoint, source = _hint_waypoint(state, params)
    visited = list(memory.get("visited_waypoints", []))
    diagnostics = {"waypoint_source": source and source["source"], "anchor": anchor,
                   "elapsed_windows": elapsed}

    if waypoint is not None:
        fresh = all(math.dist(waypoint, v) > params["search_arrive_radius"] for v in visited)
        if not fresh:
            waypoint = None  # already checked: consume it, pick a new direction
            diagnostics["waypoint_consumed"] = True
    if waypoint is None:
        # Conservative local degenerate mode: no map bounds, no pathing — pick
        # a short-range shared heading from the legal move set, rotating when
        # progress stalls or the team hugs an edge (A-B-A style oscillation).
        direction = memory.get("degenerate_direction")
        no_progress = memory.get("no_progress", 0)
        last_anchor = memory.get("last_anchor")
        if last_anchor is not None and math.dist(anchor, last_anchor) < 0.2:
            no_progress += 1
        else:
            no_progress = 0
        if direction is None or no_progress >= params["search_no_progress_limit"]:
            from .movement_scoring import predicted_move
            sample = live[0]
            moves = state.get("movement_candidates", {}).get(sample, {})
            if moves:
                direction = max(
                    (predicted_move(state, sample, a)["destination"] for a in moves),
                    key=lambda d: (round(d[0], 3), round(d[1], 3)))
                direction = [direction[0] - state["allies"][sample]["position"][0],
                             direction[1] - state["allies"][sample]["position"][1]]
                length = math.hypot(*direction) or 1.0
                direction = [direction[0] / length, direction[1] / length]
            else:
                direction = [1.0, 0.0]
            no_progress = 0
            diagnostics["direction_rotated"] = True
        distance = 8.0  # short-range internal goal only
        waypoint = [anchor[0] + direction[0] * distance, anchor[1] + direction[1] * distance]
        diagnostics["degenerate"] = True
        diagnostics["no_progress_windows"] = no_progress
        memory = {**memory, "degenerate_direction": direction, "no_progress": no_progress}

    diagnostics["waypoint"] = [round(w, 3) for w in waypoint]
    if math.dist(anchor, waypoint) <= params["search_arrive_radius"]:
        # Arrived without contact: consume the waypoint, remember the visit.
        visited.append([round(w, 3) for w in waypoint])
        diagnostics["waypoint_arrived"] = True
        memory = {**memory, "degenerate_direction": None}

    toward = [waypoint[0] - anchor[0], waypoint[1] - anchor[1]]
    length = math.hypot(*toward) or 1.0
    heading = [toward[0] / length, toward[1] / length]
    roles = {a: unit_role(state, a) for a in live}
    ordered = sorted(live, key=lambda a: (roles[a] != "melee", a))
    slots = ring_slots(len(ordered), params["search_spacing"], [roles[a] for a in ordered])
    reservations, actions = {}, {}
    waits = dict(memory.get("leader_waits", {}))
    for actor, slot in zip(ordered, slots):
        unit = state["allies"][actor]
        goal = [anchor[0] + slot[0] + heading[0] * 1.0, anchor[1] + slot[1] + heading[1] * 1.0]
        ahead = ((unit["position"][0] - anchor[0]) * heading[0]
                 + (unit["position"][1] - anchor[1]) * heading[1])
        distance = math.dist(unit["position"], goal)
        if ahead > params["link_radius"] and distance > params["search_spacing"] * 1.5:
            # Leader too far ahead: bounded wait so stragglers catch up.
            if waits.get(actor, 0) < params["search_leader_wait_limit"]:
                waits[actor] = waits.get(actor, 0) + 1
                actions[actor] = 1
                reservations[actor] = unit["position"]
                continue
        waits[actor] = 0
        action, info = best_move_toward(state, actor, goal, reservations, params,
                                        (previous_moves or {}).get(actor))
        if action is None:
            continue
        actions[actor] = action
        from .movement_scoring import predicted_move
        reservations[actor] = predicted_move(state, actor, action)["destination"]

    result.update(actions=actions, controlled_units=list(actions), status="running",
                  reason="shared_advance")
    result["state_updates"] = {
        "phase": "searching", "elapsed_windows": elapsed + 1,
        "visited_waypoints": visited[-20:],
        # observed_regions is only ever filled from real vision evidence; the
        # search itself never marks a region clear.
        "leader_waits": waits, "last_anchor": anchor,
        "degenerate_direction": memory.get("degenerate_direction"),
        "no_progress": memory.get("no_progress", 0),
    }
    result["diagnostics"] = diagnostics
    return result
