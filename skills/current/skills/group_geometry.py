"""Shared spatial utilities for regroup / cohesive search / combat positioning.

Pure geometric helpers over the snapshot's public ally data only. Distances
are Euclidean approximations on observed positions: without terrain or a
pathing grid they never claim true reachability or supportability. Ideas
follow Sharpy's group combat manager and Ares squad behaviors (grouping by
proximity, main-group selection, explainable cohesion metrics), re-implemented
for SMAC's discrete action space with no external dependency.

Complexity note: pairwise grouping, span computation and collision comparison
contain O(n^2) terms over the live roster; cost grows quadratically, not
linearly, in unit count (fine for SMAC rosters, never claimed linear).
"""

import math

PARAMS = {
    "link_radius": 6.0,          # allies closer than this (Euclidean) share an edge
    "main_group_min_share": 0.6, # main group must hold this share of live allies
    "target_radius": 7.0,        # quantile radius below which the team counts as compact
    "outlier_distance": 8.0,     # a unit farther than this from the main-group anchor is isolated
    "spacing_factor": 1.6,       # ring slot spacing in multiples of combined radius
    "melee_range": 1.5,          # native weapon range below this counts as melee
}

STOP_ACTION = 1  # SMAC stop; legal for living units unless a stricter mask says otherwise


def _finite(value, fallback=None):
    try:
        value = float(value)
        return value if math.isfinite(value) else fallback
    except (TypeError, ValueError):
        return fallback


def living_allies(state):
    """Alive allies sorted by id; ids are opaque strings."""
    return sorted((a for a, u in state["allies"].items() if u.get("alive")), key=lambda a: a)


def unit_role(state, actor):
    """Recognize melee / ranged / healer from public data; conservative fallback.

    Healers are identified by the same public type convention as the combat
    skills (Medivac). Melee/ranged comes from own native weapon range when a
    relation exposes it, else from the unit-type profile; unknown types are
    treated as generic ranged (conservative: keep with the group, no special
    forward role).
    """
    unit = state["allies"][actor]
    if str(unit.get("type", "")).lower().startswith("medivac"):
        return "healer"
    ranges = [r.get("native_weapon_range") for r in state.get("relations", {}).get(actor, {}).values()
              if r.get("native_weapon_range") is not None]
    if not ranges:
        profile = state.get("unit_type_profiles", {}).get(str(unit.get("unit_type_id")), {})
        ranges = [w.get("range") for w in profile.get("weapons", [])
                  if w.get("range") is not None]
    if ranges:
        return "melee" if min(ranges) <= PARAMS["melee_range"] else "ranged"
    return "ranged"  # unknown type: conservative generic role


def connected_components(state, actors=None, link_radius=None):
    """Union-find grouping of the given allies by Euclidean proximity.

    A long single-file chain is one component but is NOT compact; use
    compactness metrics separately before declaring the team concentrated.
    """
    radius = link_radius if link_radius is not None else PARAMS["link_radius"]
    members = actors if actors is not None else living_allies(state)
    positions = {a: state["allies"][a]["position"] for a in members}
    parent = {a: a for a in members}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(members):
        for b in members[i + 1:]:
            if math.dist(positions[a], positions[b]) <= radius:
                parent[find(a)] = find(b)
    groups = {}
    for a in members:
        groups.setdefault(find(a), []).append(a)
    return sorted(groups.values(), key=lambda g: (-len(g), min(g)))


def centroid(positions):
    points = [p for p in positions if p is not None]
    if not points:
        return None
    return [sum(p[i] for p in points) / len(points) for i in range(len(points[0]))]


def compactness(state, actors):
    """Explainable cohesion metrics for one set of actors.

    ``quantile_radius`` of 0.0 is a valid value (everyone stacked) and is
    never treated as missing. Geometry only.
    """
    positions = [state["allies"][a]["position"] for a in actors]
    if not positions:
        return {"count": 0, "centroid": None, "span": None,
                "median_radius": None, "quantile_radius": None}
    center = centroid(positions)
    distances = sorted(math.dist(p, center) for p in positions)
    span = max((math.dist(p, q) for p in positions for q in positions), default=0.0)
    quantile = distances[max(0, math.ceil(0.9 * len(distances)) - 1)]
    return {"count": len(actors), "centroid": [round(c, 3) for c in center],
            "span": round(span, 3),
            "median_radius": round(distances[len(distances) // 2], 3),
            "quantile_radius": round(quantile, 3)}


def main_group(state, actors=None):
    """Largest proximity component; ties broken by lowest member id."""
    groups = connected_components(state, actors)
    return list(groups[0]) if groups else []


def cohesion_report(state, parameters=None, actors=None):
    """Cohesion snapshot; over ALL live allies by default (full observation),
    or over a restricted actor set when asked (controlled-cohesion contract).
    """
    params = {**PARAMS, **(parameters or {})}
    live = living_allies(state) if actors is None else [a for a in actors
                                                        if state["allies"][a].get("alive")]
    groups = connected_components(state, live, params["link_radius"])
    main = list(groups[0]) if groups else []
    anchor = centroid(state["allies"][a]["position"] for a in main) if main else None
    others = [a for a in live if a not in main]
    isolated = [a for a in others
                if anchor is not None
                and math.dist(state["allies"][a]["position"], anchor) >= params["outlier_distance"]]
    main_metrics = compactness(state, main) if main else compactness(state, [])
    # One long chain is a single component but NOT concentrated: coherence
    # requires BOTH the main-share threshold and a compact main group.
    coherent = bool(live
                    and len(main) / len(live) >= params["main_group_min_share"]
                    and main_metrics["quantile_radius"] is not None
                    and main_metrics["quantile_radius"] <= params["target_radius"])
    return {
        "live_units": live,
        "groups": [list(g) for g in groups],
        "main_group": main,
        "main_share": round(len(main) / len(live), 3) if live else 0.0,
        "anchor": [round(c, 3) for c in anchor] if anchor else None,
        "main_compactness": main_metrics,
        "outliers": others,
        "isolated": isolated,
        "team_compactness": compactness(state, live),
        "coherent": coherent,
    }


def ring_slots(count, spacing, roles=None, heading=(1.0, 0.0), forward_stagger=0.5):
    """Deterministic spaced slot offsets in the HEADING frame (forward/lateral).

    Slot 0 is on the point (no offset); further members alternate left/right
    at increasing lateral distance with a slight backward stagger. Melee roles
    are sorted to the inner slots. Offsets rotate with ``heading`` so the same
    formation follows the search direction instead of pushing units sideways
    in world coordinates.
    """
    pairs = [(0.0, 0.0)]
    for index in range(1, max(count, 1)):
        side = 1 if index % 2 == 1 else -1
        ring = min((index + 1) // 2, 1)  # tight column: lateral spread stays within one spacing
        pairs.append((-forward_stagger * spacing * ring, side * spacing * ring))
    if roles and count:
        order = sorted(range(count), key=lambda i: (roles[i] != "melee", i))
        pairs = [pairs[i] for i in order]
    fx, fy = heading
    lx, ly = -fy, fx
    return [(f * fx + l * lx, f * fy + l * ly) for f, l in pairs[:count]]


def legal_actions(state, actor):
    """Action ids usable by this living unit, under an explicit adapter contract.

    ``state['legal_actions'][actor]`` (optional, set of int action ids) is the
    authoritative mask when present. Otherwise the contract is: every key of
    ``movement_candidates[actor]`` is a legal move, and SMAC Stop (1) is legal
    for a living unit. This function never invents attack ids or legality.
    """
    explicit = (state.get("legal_actions") or {}).get(actor)
    if explicit is not None:
        return {int(a) for a in explicit}
    moves = set(int(a) for a in state.get("movement_candidates", {}).get(actor, {}))
    if state["allies"][actor].get("alive"):
        moves.add(STOP_ACTION)
    return moves


def _stop_predicted(unit):
    return {"destination": list(unit["position"]),
            "distances_to_visible_enemies": {}}


def predicted_destination(state, actor, action):
    """Predicted endpoint of one action; never raises on non-move ids.

    Stop and non-movement actions (attack/heal) keep the unit at its current
    position; Stop never enters predicted_move() (whose keys are move ids
    only). A move id absent from movement_candidates is treated the same way
    rather than inventing a destination.
    """
    if int(action) == STOP_ACTION or \
            str(action) not in state.get("movement_candidates", {}).get(actor, {}):
        return list(state["allies"][actor]["position"])
    from .movement_scoring import predicted_move
    return predicted_move(state, actor, action)["destination"]


def base_reservations(state, skip=()):
    """Every living ally's CURRENT position as the default space reservation.

    Uncontrolled allies, not-yet-processed controlled units and attacking or
    healing units all occupy space; decided actions overwrite their entry with
    the predicted destination. Air and ground are separate collision layers.
    """
    reservations = {}
    for actor, unit in state["allies"].items():
        if not unit.get("alive") or actor in skip:
            continue
        reservations[actor] = (unit["position"], unit.get("is_flying", False))
    return reservations


def best_move_toward(state, actor, target, reservations=None, parameters=None,
                     previous_action=None, allow_stop=True, penalize=()):
    """Pick the legal action whose predicted endpoint is closest to ``target``.

    Scoring: distance + collision + direction-flip penalty. Collisions compare
    within the same air/ground layer using public radii; a unit ALREADY
    overlapping a friend is not penalized for moves that increase the gap
    (escape from spawn overlap stays legal). Legal Stop competes with moves
    and wins ties (deterministic: stop sorts first), so a unit standing on its
    goal does not oscillate. ``penalize`` adds a bounded score penalty to
    observed-failed action ids instead of removing them: with a single legal
    route the route is still taken. Returns (action or None, info); None only
    when no legal action exists at all.
    """
    params = {**PARAMS, **(parameters or {})}
    unit = state["allies"][actor]
    origin = unit["position"]
    actions = legal_actions(state, actor)
    if not allow_stop:
        actions = actions - {STOP_ACTION}
    if not actions:
        return None, {"reason": "no_legal_actions", "candidates": {}}
    penalized = {int(a) for a in penalize}
    move_ids = {int(a) for a in actions if int(a) != STOP_ACTION}
    if move_ids and move_ids <= penalized:
        penalized = set()  # single remaining route is never sealed by a penalty
    layered = {}
    for other, (position, flying) in (reservations or {}).items():
        if other != actor and flying == unit.get("is_flying", False):
            layered[other] = position
    info = {"candidates": {}, "rejected": {}}
    scored = []
    for action in sorted(actions):
        if int(action) == STOP_ACTION:
            destination = list(origin)
        else:
            if str(action) not in state.get("movement_candidates", {}).get(actor, {}):
                info["rejected"][str(action)] = "not_a_movement_candidate"
                continue
            from .movement_scoring import predicted_move
            destination = predicted_move(state, actor, action)["destination"]
        distance = math.dist(destination, target)
        collision = 0.0
        for other, other_position in layered.items():
            friend = state["allies"].get(other)
            if friend is None:
                continue
            clearance = unit.get("radius", .5) + friend.get("radius", .5) + 0.1
            gap = math.dist(destination, other_position)
            old_gap = math.dist(origin, other_position)
            # Entering (or staying in) an overlap is costly only when it does
            # not improve an existing overlap: escape directions stay open.
            if gap < clearance and gap < old_gap + 0.1:
                collision += (clearance - gap) / clearance
                if gap < clearance * 0.3:
                    # Near-coincidence with a friend is heavily discouraged:
                    # stacking on one cell is never worth a shorter distance.
                    collision += 2.0 * (clearance - gap) / clearance
        flip = 0.25 if previous_action is not None and int(action) != int(previous_action) else 0.0
        block_penalty = 3.0 if int(action) in penalized else 0.0
        # Collision weight stays below one step's travel gain so a normal
        # side-by-side march is never paralysis; deep stacking still loses.
        score = distance + 0.8 * collision + flip + block_penalty
        info["candidates"][str(action)] = {
            "destination": [round(d, 3) for d in destination],
            "distance": round(distance, 3), "collision_cost": round(collision, 3),
            "flip_penalty": flip, "blocked_penalty": block_penalty,
            "score": round(score, 3)}
        scored.append((round(score, 6), int(action) != STOP_ACTION, int(action)))
    if not scored:
        return None, {"reason": "no_usable_actions", **info}
    scored.sort()
    info["selected_action"] = scored[0][2]
    return scored[0][2], info
