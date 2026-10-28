"""CohesiveSearch: advance as one group toward a shared search target.

Top-level spatial skill for snapshots with NO current enemy position. Target
priority: (1) a valid runner-provided hint (``state['search_hint']`` /
``state['search_locations']``), (2) the manager's shared last-seen-enemy
ledger (injected as a derived ``search_hint`` with position, observation
loop, age and source), (3) conservative local degenerate mode from legal
moves with bounded failure-direction memory.

Formation reference frame: one shared waypoint and team anchor define the
heading; member slots are (forward, lateral) offsets rotated into that frame.
A single controlled unit gets the zero slot and simply advances. Waypoint
visits carry timestamps and clue generations, so a re-observed enemy position
becomes a fresh lead instead of being suppressed by an old visit.
"""

import math

from .group_geometry import (PARAMS as GEOMETRY_PARAMS, best_move_toward,
                             base_reservations, cohesion_report, predicted_destination,
                             ring_slots, unit_role)

PARAMS = {
    **GEOMETRY_PARAMS,
    "search_arrive_radius": 3.0,       # team anchor within this of waypoint = checked
    "search_waypoint_ttl_windows": 6,  # a visited waypoint stays suppressed this long
    "search_max_windows": 40,          # total search budget per episode phase
    "search_no_progress_limit": 3,     # stalled windows before the direction is failed
    "search_leader_wait_limit": 2,     # leader wait/rejoin budget per window
    "search_severe_split_share": 0.5,  # hard floor below which a split is always severe
    "regroup_target_share": 0.8,       # main share below this counts as split
    "search_spacing": 2.0,
    "search_memory_ttl_loops": 1500,   # last-seen hint validity in game loops
    "search_visited_capacity": 20,     # bounded visit memory
    "search_direction_penalty_windows": 4,  # failed direction ban duration
}


def _valid_hint(hint, loop, params):
    """A hint is usable when its coordinates are finite and it is not expired.

    Negative age (a hint from the future) is rejected; only a trusted known
    death would clear a target, and this module never infers death.
    """
    if not hint or hint.get("position") is None:
        return False
    position = hint["position"]
    if not all(_finite(c) is not None for c in position):
        return False
    age = hint.get("age_loops")
    if age is not None and (age < 0 or age > params["search_memory_ttl_loops"]):
        return False
    return True


def _finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _visited_suppresses(waypoint, visited, params, loop=None, generation=None):
    """A visit suppresses a waypoint only while its record is unexpired.

    Records carry (loop, generation); a newer clue generation does not
    suppress, so a re-observed enemy position can be searched again.
    """
    ttl_loops = params["search_waypoint_ttl_windows"] * 8  # windows -> loops at step_mul 8
    for record in visited or []:
        if math.dist(waypoint, record["position"]) > params["search_arrive_radius"]:
            continue
        if loop is not None and loop - record.get("loop", 0) > ttl_loops:
            continue  # expired visit: the waypoint is a candidate again
        if generation is not None and record.get("generation", 0) < generation:
            continue  # newer clue: re-observed position is a fresh lead
        return record
    return None


def _waypoint_candidates(state, params, visited):
    """Ordered candidate waypoints: runner hint, public search locations, then
    nothing (the caller falls back to the local degenerate mode)."""
    loop = state.get("game_loop", 0)
    candidates = []
    hint = state.get("search_hint")
    if _valid_hint(hint, loop, params):
        candidates.append((list(hint["position"]),
                           {"source": hint.get("source", "runner_search_hint"),
                            "generation": hint.get("generation", hint.get("game_loop", 0))}))
    locations = state.get("search_locations") or []
    for location in sorted(locations, key=lambda p: (p[0], p[1])):
        if all(_finite(c) is not None for c in location):
            candidates.append((list(location), {"source": "runner_search_locations",
                                                "generation": 0}))
    return [(waypoint, source) for waypoint, source in candidates
            if not _visited_suppresses(waypoint, visited, params, loop,
                                       source.get("generation", 0))]


def _unit_heading_vector(state, actor, memory, params):
    """Local degenerate direction from legal moves, avoiding failed directions.

    Deterministic rotation over the legal move set (ordered by angle) instead
    of a repeated max() rule; directions penalized by observed failure are
    skipped while alternates exist. Returns (unit_vector, info).
    """
    moves = state.get("movement_candidates", {}).get(actor, {})
    if not moves:
        return None, {"reason": "no_legal_moves"}
    origin = state["allies"][actor]["position"]
    penalties = memory.get("failed_directions", {})  # vector tuple -> remaining windows
    def direction_key(destination):
        dx, dy = destination[0] - origin[0], destination[1] - origin[1]
        length = math.hypot(dx, dy) or 1.0
        return (round(dx / length, 3), round(dy / length, 3))
    options = []
    for action in moves:
        from .movement_scoring import predicted_move
        destination = predicted_move(state, actor, action)["destination"]
        vector = direction_key(destination)
        options.append((vector, int(action)))
    options.sort(key=lambda pair: (math.atan2(pair[0][1], pair[0][0]), pair[1]))
    fresh = [(vector, action) for vector, action in options
             if penalties.get(vector, 0) <= 0]
    pool = fresh or options  # never permanently forbid every direction
    # Rotate through the pool using a stable counter so retries actually
    # change direction instead of re-picking the same first entry.
    index = memory.get("degenerate_index", 0) % len(pool)
    chosen = pool[index]
    return chosen[0], {"action_hint": chosen[1], "rotated": bool(penalties),
                       "options": [list(v) for v, _ in options]}


def allocate_cohesive_search(state, actor_ids, memory, parameters=None, previous_moves=None):
    """One CohesiveSearch decision step (partial action proposal)."""
    params = {**PARAMS, **(parameters or {})}
    memory = memory if memory is not None else {}
    requested = sorted({a for a in actor_ids if a in state["allies"]},
                       key=lambda a: (len(a), a))
    live = [a for a in requested if state["allies"][a].get("alive")]
    result = {"actions": {}, "controlled_units": [], "requested_units": requested,
              "status": "running", "reason": "", "state_updates": {}, "diagnostics": {}}
    if not live:
        result.update(status="unavailable", reason="no_controlled_units")
        return result
    # Contact OR fresh damage interrupts the search on every call path
    # (manual selection included), not only under the auto dispatcher.
    if state["visible_enemies"]:
        result.update(status="interrupted", reason="enemy_visible",
                      state_updates={"phase": "idle", "elapsed_windows": 0,
                                     "degenerate_waypoint": None})
        return result
    damaged = [a for a in live
               if (state["allies"][a].get("health_lost_since_previous_decision", 0)
                   + state["allies"][a].get("shield_lost_since_previous_decision", 0)) > 0]
    if damaged:
        result.update(status="interrupted", reason="new_damage",
                      state_updates={"phase": "idle", "elapsed_windows": 0,
                                     "degenerate_waypoint": None})
        result["diagnostics"]["damaged_units"] = damaged
        return result

    observed = cohesion_report(state, params)   # full roster informs split judgement
    controlled_cohesion = cohesion_report(state, params, actors=live)
    result["diagnostics"]["observed_cohesion"] = observed
    result["diagnostics"]["controlled_cohesion"] = controlled_cohesion
    elapsed = memory.get("elapsed_windows", 0)
    if elapsed >= params["search_max_windows"]:
        result.update(status="timeout", reason=f"budget_exhausted({elapsed})",
                      state_updates={"phase": "cooldown",
                                     "cooldown_windows": params["search_waypoint_ttl_windows"],
                                      "elapsed_windows": 0, "degenerate_waypoint": None})
        return result
    # A split is severe when the main group holds less than the regroup
    # completion share, or when an isolated unit is far outside support
    # distance. Multiple groups alone (e.g. 8+2 bunched pairs) do not stop
    # the search.
    severe = (len(observed["groups"]) > 1
              and (observed["main_share"] < params["regroup_target_share"]
                   or observed["isolated"]
                   or len(observed["outliers"]) >= len(observed["main_group"])))
    if live and severe:
        result.update(status="blocked", reason="needs_regroup",
                      state_updates={"phase": "needs_regroup"})
        return result

    anchor_units = live
    positions = [state["allies"][a]["position"] for a in anchor_units]
    anchor = [sum(p[i] for p in positions) / len(positions) for i in range(2)]
    visited = list(memory.get("visited_waypoints", []))
    waypoint, waypoint_source = None, None

    candidates = _waypoint_candidates(state, params, visited)
    degenerate_waypoint = None
    if candidates:
        waypoint, waypoint_source = candidates[0]
    if waypoint is None:
        # Conservative local degenerate mode. The short-range goal is FIXED
        # for this leg (anchor + 8 in the chosen direction), so it cannot
        # slide forward every step and become unreachable.
        direction, direction_info = _unit_heading_vector(state, live[0], memory, params)
        if direction is None:
            result.update(status="blocked", reason="no_legal_moves",
                          state_updates={"phase": "cooldown",
                                         "cooldown_windows": params["search_waypoint_ttl_windows"],
                                         "elapsed_windows": 0})
            return result
        saved = memory.get("degenerate_waypoint")
        if saved:
            delta = [saved["position"][i] - anchor[i] for i in range(2)]
            norm = math.hypot(*delta) or 1.0
            if any(remaining > 0 and sum(delta[i] / norm * heading[i] for i in range(2)) > .95
                   for heading, remaining in memory.get("failed_directions", {}).items()):
                saved = None
        degenerate_waypoint = saved or {
            "position": [anchor[0] + direction[0] * 8.0, anchor[1] + direction[1] * 8.0],
            "generation": state.get("game_loop", 0)}
        waypoint = list(degenerate_waypoint["position"])
        waypoint_source = {"source": "degenerate_local", "generation": degenerate_waypoint["generation"]}
        result["diagnostics"]["degenerate"] = True
        result["diagnostics"]["direction_info"] = direction_info

    result["diagnostics"]["waypoint"] = [round(w, 3) for w in waypoint]
    result["diagnostics"]["waypoint_source"] = waypoint_source["source"]
    result["diagnostics"]["anchor"] = [round(c, 3) for c in anchor]
    result["diagnostics"]["elapsed_windows"] = elapsed

    if math.dist(anchor, waypoint) <= params["search_arrive_radius"]:
        # Arrived without contact: consume the waypoint as a VISIT (timestamp
        # + generation). A visit is never a cleared-region claim.
        visited.append({"position": [round(w, 3) for w in waypoint],
                        "loop": state.get("game_loop", 0),
                        "generation": waypoint_source.get("generation", 0),
                        "source": waypoint_source["source"]})
        visited = visited[-params["search_visited_capacity"]:]
        result["diagnostics"]["waypoint_arrived"] = True
        result.update(actions={}, controlled_units=[], status="running",
                      reason="waypoint_checked_pick_next")
        result["state_updates"] = {
            "phase": "searching", "elapsed_windows": elapsed + 1,
            "visited_waypoints": visited, "last_anchor": anchor,
            "degenerate_waypoint": None,
            "degenerate_index": (memory.get("degenerate_index", 0) + 1),
            "failed_directions": {v: n for v, n in memory.get("failed_directions", {}).items()
                                  if n > 1},
        }
        return result

    toward = [waypoint[0] - anchor[0], waypoint[1] - anchor[1]]
    length = math.hypot(*toward) or 1.0
    heading = (toward[0] / length, toward[1] / length)
    roles = {a: unit_role(state, a) for a in live}
    ordered = sorted(live, key=lambda a: (roles[a] != "melee", a))
    slots = ring_slots(len(ordered), params["search_spacing"],
                       [roles[a] for a in ordered], heading, forward_stagger=0.0)
    # March centre: a memory-owned formation reference that advances toward
    # the waypoint every window but never outruns the real team by more than
    # link_radius. Anchoring goals to the live centroid alone would deadlock
    # (everyone reaches "centroid + one step" and the centroid stops moving).
    march = list(memory.get("march_center") or anchor)
    step = 1.0  # one typical control-window move
    march = [march[0] + heading[0] * step, march[1] + heading[1] * step]
    if math.dist(march, anchor) > params["link_radius"]:
        pull = [anchor[0] - march[0], anchor[1] - march[1]]
        norm = math.hypot(*pull) or 1.0
        march = [anchor[0] - pull[0] / norm * params["link_radius"],
                 anchor[1] - pull[1] / norm * params["link_radius"]]
    result["diagnostics"]["march_center"] = [round(c, 3) for c in march]
    # Observed-failed directions become a bounded penalty on the matching
    # legal action ids (never a removal: a single remaining route stays usable).
    failed_actions = set()
    failed_headings = [key for key, remaining in memory.get("failed_directions", {}).items()
                       if remaining > 0]
    for actor in live:
        origin = state["allies"][actor]["position"]
        for action in state.get("movement_candidates", {}).get(actor, {}):
            from .movement_scoring import predicted_move
            destination = predicted_move(state, actor, action)["destination"]
            dx, dy = destination[0] - origin[0], destination[1] - origin[1]
            norm = math.hypot(dx, dy) or 1.0
            unit = (dx / norm, dy / norm)
            # Heading drifts with the anchor; match by direction, not exact tuple.
            if any(unit[0] * h[0] + unit[1] * h[1] > 0.95 for h in failed_headings):
                failed_actions.add((actor, int(action)))
    actions, waits = {}, dict(memory.get("leader_waits", {}))
    from .group_geometry import legal_actions
    reservations = base_reservations(state)
    previous_move_for = {a: (previous_moves or {}).get(a) for a in live}
    for actor, slot in zip(ordered, slots):
        unit = state["allies"][actor]
        goal = [march[0] + slot[0], march[1] + slot[1]]
        ahead = ((unit["position"][0] - anchor[0]) * heading[0]
                 + (unit["position"][1] - anchor[1]) * heading[1])
        slot_distance = math.dist(unit["position"], goal)
        legal = legal_actions(state, actor)
        if len(live) > 1 and ahead > params["link_radius"] and \
                slot_distance > params["search_spacing"] * 1.5:
            # Leader too far ahead of the shared anchor: bounded wait so the
            # stragglers catch up (single-unit teams never wait; waiting uses
            # the legal Stop only, else the unit keeps moving).
            if waits.get(actor, 0) < params["search_leader_wait_limit"] \
                    and 1 in legal:
                waits[actor] = waits.get(actor, 0) + 1
                actions[actor] = 1
                continue
        waits[actor] = 0
        action, info = best_move_toward(state, actor, goal, reservations, params,
                                        previous_move_for.get(actor),
                                        penalize={a for who, a in failed_actions
                                                  if who == actor})
        if action is None:
            continue
        actions[actor] = action
        reservations[actor] = (predicted_destination(state, actor, action),
                               unit.get("is_flying", False))

    if failed_actions:
        # "direction_changed" is claimed ONLY when the actually chosen action
        # differs from the previously executed one under a live penalty.
        result["diagnostics"]["direction_penalty_applied"] = [
            [actor, action] for actor, action in sorted(failed_actions)]
        result["diagnostics"]["direction_changed"] = bool(any(
            actions.get(actor) not in (None, previous_move_for.get(actor))
            for actor, _ in failed_actions))

    # Team progress check on the OBSERVED anchor: consecutive stalls or an
    # observed A-B-A of the anchor mark the current direction as failed for a
    # bounded window; alternates are tried before any degradation.
    last_anchor = memory.get("last_anchor")
    anchor_history = list(memory.get("anchor_history", []))[-2:]
    anchor_history.append([round(anchor[0], 3), round(anchor[1], 3)])
    # A single sub-threshold step is normal shuffle (lateral slot adjustment,
    # waiting for a straggler): the direction only counts as failed after the
    # no-progress budget of consecutive stalled windows, or an observed A-B-A.
    stall_count = memory.get("anchor_stall", 0)
    if last_anchor is not None and math.dist(anchor, last_anchor) < 0.2 and actions:
        stall_count += 1
    else:
        stall_count = 0
    failed = dict(memory.get("failed_directions", {}))
    aba = (len(anchor_history) == 3
           and math.dist(anchor_history[0], anchor_history[2]) < 0.3
           and math.dist(anchor_history[0], anchor_history[1]) > 0.4)
    if (stall_count >= params["search_no_progress_limit"] and actions) or aba:
        heading_key = (round(heading[0], 3), round(heading[1], 3))
        failed[heading_key] = params["search_direction_penalty_windows"]
        stall_count = 0
        result["diagnostics"]["direction_failed"] = {"heading": list(heading_key),
                                                     "aba": aba}
    failed = {v: n - 1 for v, n in failed.items() if n - 1 > 0}

    result.update(actions=actions, controlled_units=sorted(actions, key=lambda a: (len(a), a)),
                  status="running", reason="shared_advance")
    result["state_updates"] = {
        "phase": "searching", "elapsed_windows": elapsed + 1,
        "visited_waypoints": visited,
        # observed_regions is only ever filled from real vision evidence; the
        # search itself never marks a region clear.
        "leader_waits": waits, "last_anchor": anchor, "anchor_history": anchor_history,
        "march_center": [round(c, 4) for c in march],
        "degenerate_index": memory.get("degenerate_index", 0),
        "degenerate_waypoint": (None if result["diagnostics"].get("direction_failed")
                                else degenerate_waypoint),
        "failed_directions": failed, "anchor_stall": stall_count,
    }
    return result
