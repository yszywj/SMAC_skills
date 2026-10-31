"""HealAssign: joint, one-to-one healer/target assignment with sustained healing.

Model assumptions and scope (first version):

- Works on one ``build_request`` snapshot (allies / visible_enemies /
  relations / heal_candidates / movement_candidates). Healer legality comes
  exclusively from the snapshot's ``heal_candidates``, which the environment
  already gates on aliveness, energy, target type, flight limits, missing
  health and the SMAC legality mask — this module never bypasses
  ``get_avail_agent_actions()``.
- Heal demand is health-only: ``missing_health = max_health - health``.
  Shield loss never makes a target healable by a Medivac (the environment's
  heal candidates already enforce this; we keep the same semantics here).
- The urgency score is an explainable heuristic over observed facts only
  (health ratio, recent observed health loss, visible threats, distance).
  Future damage, heal rates and arrival times are not predicted; urgency is
  an estimate, never a claim about future health.
- Assignment is a deterministic maximum-weight one-to-one matching between
  healers and legal targets. The Terran composition additionally reserves
  targets occupied by observed heal orders until a later observation confirms
  release: this project's native feedback rejects conflicting orders with
  AlreadyTargeted. Do not infer simultaneous healing from candidate legality.
- Planned healing is never counted as restored health. Observed health
  changes are net of enemy damage; this module records assignments only.
- An endangered healer self-preserves: it is removed from the matching, its
  reservation is released, and it moves via CoverRetreat's legal retreat
  scoring (reused, not reimplemented). One unit, one final action per step.
- Non-healer units are NOT controlled here; HealAssign returns actions only
  for healer slots and the caller merges a declared base policy for the rest.
"""

from .cover_retreat import PARAMS as COVER_PARAMS, select_retreat_action
from ..unit_value import enabled_values, normalized_value, validate_parameters

import math

PARAMS = {
    "switch_margin": 0.15,      # score bonus for keeping the previous target (hysteresis)
    "threat_distance": COVER_PARAMS["threat_distance"],
    "healer_danger_ratio": 0.35,  # healer survival ratio below which it may self-preserve
    "distance_cost": 0.02,       # per world unit of healer-target distance
}

HEAL_ORDER_ABILITY = 386  # Medivac heal, as used by the existing executor.


def is_healer(unit):
    """Medivac-style unit identified by its public type name in the snapshot."""
    return str(unit.get("type", "")).lower().startswith("medivac")


def health_ratio(unit):
    return unit["health"] / unit["max_health"] if unit["max_health"] else 1.0


def _threatened(state, actor):
    return any(r["distance"] <= PARAMS["threat_distance"]
               or r.get("in_enemy_weapon_range") is True
               for r in state.get("relations", {}).get(actor, {}).values())


def score_heal_need(state, target_id, parameters=None):
    """Explainable urgency of healing one target (healer-independent part).

    Combines missing health share, low health ratio, recent observed health
    loss and visible threats. Deterministic; ties are broken by target ID at
    the caller.
    """
    params = {**PARAMS, **(parameters or {})}
    target = state["allies"][target_id]
    maximum = target["max_health"] or 1
    missing = max(0.0, maximum - target["health"]) / maximum
    ratio = health_ratio(target)
    recent_loss = min(target.get("health_lost_since_previous_decision", 0) / maximum, 0.5)
    threatened = 1.0 if _threatened(state, target_id) else 0.0
    return round(0.6 * (1 - ratio) + 0.2 * missing + 0.25 * recent_loss + 0.15 * threatened, 3)


def healer_endangered(state, actor, parameters=None):
    params = {**PARAMS, **(parameters or {})}
    unit = state["allies"][actor]
    return health_ratio(unit) < params["healer_danger_ratio"] and _threatened(state, actor)


def _edge_score(state, healer, target_id, entry, urgency, previous_assignments, parameters):
    params = {**PARAMS, **(parameters or {})}
    score = urgency + 0.2 * entry["missing_health_factor"] \
        - params["distance_cost"] * entry["distance"]
    if (previous_assignments or {}).get(healer) == target_id:
        score += params["switch_margin"]  # keep a still-valid relationship stable
    return score


def _best_matching(healers, edges):
    """Deterministic maximum-weight one-to-one matching (healers -> targets).

    Uses ``scipy.optimize.linear_sum_assignment`` (SciPy is already a project
    dependency) on a reward matrix padded with one dummy "stay idle" column
    per healer, so idle healers are free and no illegal edge can become an
    assignment. Falls back to the exhaustive search for small inputs when
    SciPy is unavailable. Ties resolve deterministically for identical input.
    """
    targets = sorted({target for rows in edges.values() for _, target in rows})
    if not healers or not targets:
        return {}
    if len(healers) <= _EXACT_FALLBACK_HEALERS:
        return _exact_matching(healers, edges)
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return _exact_matching(healers, edges)  # Small enough or accept the cost.
    forbidden = -1e9
    reward = np.full((len(healers), len(targets) + len(healers)), forbidden)
    for i in range(len(healers)):
        reward[i, len(targets) + i] = 0.0  # Idle option: always available.
    healer_index = {healer: i for i, healer in enumerate(healers)}
    target_index = {target: j for j, target in enumerate(targets)}
    for healer, rows in edges.items():
        for score, target in rows:
            reward[healer_index[healer], target_index[target]] = score
    rows_ind, cols_ind = linear_sum_assignment(-reward)  # Minimize -reward.
    assignments = {}
    for i, j in zip(rows_ind, cols_ind):
        if j < len(targets) and reward[i, j] > 0:
            assignments[healers[i]] = targets[j]
    return assignments


_EXACT_FALLBACK_HEALERS = 8


def _exact_matching(healers, edges):
    """Exhaustive deterministic optimum for small healer counts."""
    best = {"score": 0.0, "assign": {}}

    def recurse(index, used, total, assign):
        if index == len(healers):
            if total > best["score"] or (total == best["score"] and not best["assign"]):
                best["score"], best["assign"] = total, dict(assign)
            return
        healer = healers[index]
        recurse(index + 1, used, total, assign)  # leaving this healer idle
        for score, target in edges[healer]:
            if target in used or score <= 0:
                continue
            assign[healer] = target
            recurse(index + 1, used | {target}, total + score, assign)
            del assign[healer]

    recurse(0, frozenset(), 0.0, {})
    return best["assign"]


def healer_safe_move(state, healer, parameters=None):
    """Best legal move away from visible enemies for an endangered healer.

    Unlike ``select_retreat_action`` this always returns the safest available
    move even when no candidate strictly "improves" the situation, so a
    threatened healer never falls back into approaching a wounded ally (which
    could move it closer to the enemy). Returns an int action ID or None when
    no movement candidate exists at all.
    """
    moves = state.get("movement_candidates", {}).get(healer, {})
    relations = state.get("relations", {}).get(healer, {})
    rows = []
    for action_id, entry in moves.items():
        distances = list(entry["distances_to_visible_enemies"].values())
        min_enemy = min(distances) if distances else PARAMS["threat_distance"] + 1
        rows.append((-min_enemy, int(action_id)))  # Max distance, then lowest ID.
    return min(rows)[1] if rows else None


def healer_idle_action(state, healer, target_pool, urgency, parameters=None):
    """Legal idle behavior for a healer with no assignment: approach the most
    urgent wounded target along a legal move, else take a safe positioning
    move. Returns an integer action ID or None (base policy keeps control).

    Medivacs never get attack fallbacks: their SMAC attack slots are heal
    commands and attacking enemies is not a legal action for them.
    """
    params = {**PARAMS, **(parameters or {})}
    moves = state.get("movement_candidates", {}).get(healer, {})
    if target_pool:
        best_target = max(sorted(target_pool), key=lambda t: (urgency.get(t, 0), -int(t[1:])))
        goal = state["allies"][best_target]["position"]
        rows = [(math.dist(entry["destination"], goal), int(action_id))
                for action_id, entry in moves.items()]
        if rows:
            return min(rows)[1]  # Approach: never counted as provided healing.
    return healer_safe_move(state, healer, params)


def _forced_matching(healers, edges):
    """Deterministic maximum-weight matching where every healer MUST be
    assigned exactly one target (no idle column, negative edges allowed).

    Only used by the optional injected pair-scorer channel; the default
    path never calls this. Exhaustive search with the same option ordering
    as ``_exact_matching`` (score desc, then target id), strict-improvement
    updates, so identical input resolves ties identistically.
    """
    if len(healers) > _EXACT_FALLBACK_HEALERS:
        raise ValueError("forced matching is only defined for up to "
                         f"{_EXACT_FALLBACK_HEALERS} participating healers")
    best = {"score": None, "assign": None}

    def recurse(index, used, total, assign):
        if index == len(healers):
            if best["score"] is None or total > best["score"]:
                best["score"], best["assign"] = total, dict(assign)
            return
        for score, target in edges[healers[index]]:
            if target in used:
                continue
            assign[healers[index]] = target
            recurse(index + 1, used | {target}, total + score, assign)
            del assign[healers[index]]

    recurse(0, frozenset(), 0.0, {})
    return best["assign"]


def _apply_pair_scorer(state, assignments, edges, target_pool, urgency,
                       previous_assignments, pair_scorer, active):
    """Optional injected re-scoring of the joint heal matching.

    Contract (tree_skills medivac channel, v1):
    - Participation set H is FIXED to the healers the original logic already
      assigned this step; healers the original logic left idle, endangered or
      keep-locked are never added or freed here.
    - Legal pairs are exactly the original ``edges`` rows (environment-gated
      candidates after order reservation) — no filtered target is reopened.
    - Scores come entirely from ``pair_scorer(healer, target_id, context)``;
      negative scores are meaningful (participation is fixed), so no constant
      shift, abs or clipping may be applied by either side.
    - ``active=False`` is shadow mode: only record the proposed matching.

    Returns (assignments, diagnostics). On any infeasibility (a participating
    healer with no legal edge, non-finite score, scorer exception) the
    original matching is kept and the failure is recorded per-healer.
    """
    healers = sorted(assignments, key=lambda h: int(h[1:]))
    context = {"state": state, "urgency": urgency, "target_pool": target_pool,
               "previous_assignments": previous_assignments or {},
               "original_edges": edges}
    new_edges, failures = {}, {}
    for healer in healers:
        rows = []
        for _, target in edges.get(healer, []):
            try:
                score = float(pair_scorer(healer, target, context))
            except Exception as exc:  # diagnostics only; never crash the step
                failures[healer] = f"scorer_error:{type(exc).__name__}"
                rows = None
                break
            if score != score or score in (float("inf"), float("-inf")):
                failures[healer] = f"non_finite_score_for:{target}"
                rows = None
                break
            rows.append((score, target))
        if rows is None:
            continue
        new_edges[healer] = sorted(rows, key=lambda r: (-r[0], r[1]))
    proposed, kept = dict(assignments), dict(assignments)
    if not failures and healers:
        proposed = _forced_matching(healers, new_edges) or dict(assignments)
    changed = {h: {"from": assignments[h], "to": proposed[h]}
               for h in healers if proposed[h] != assignments[h]}
    diag = {"participating_healers": healers,
            "proposed_assignments": proposed,
            "changed": changed,
            "model_influence_rate": (len(changed) / len(healers)) if healers else 0.0,
            "failures": failures,
            "mode": "active" if active else "shadow"}
    return (proposed if (active and not failures) else kept), diag


def assign_healing(state, n_agents, previous_assignments=None, parameters=None,
                   unit_values=None, pair_scorer=None, pair_scoring="off"):
    """One HealAssign decision step on a fresh snapshot.

    Returns a partial action map (healer slots only), full-length filler is
    the caller's responsibility, plus diagnostics. Non-healer allies are
    never assigned here.

    ``pair_scorer``/``pair_scoring`` are an optional, default-off injection
    point for the tree_skills medivac preference channel: ``"off"`` keeps the
    original behaviour bit-for-bit; ``"shadow"`` records a proposed
    re-matching without changing it; ``"on"`` re-ranks targets for the
    already-decided participating healers only (see ``_apply_pair_scorer``).
    """
    params = {**PARAMS, **(parameters or {})}
    use_values = enabled_values(unit_values)
    value_params = validate_parameters(unit_values.get("parameters")) if use_values else None
    allies = state["allies"]
    healers = sorted((a for a, u in allies.items() if u["alive"] and is_healer(u)),
                     key=lambda a: int(a[1:]))

    if not any(is_healer(u) for u in allies.values()):
        return {"available": False, "termination_reason": "no_healers",
                "terminated": True, "assignments": {}, "healer_actions": {},
                "healers": [], "self_preserving": {}, "urgency": {},
                "unassigned_healers": [], "unassigned_reasons": {},
                "retained_orders": {}}

    endangered = {h: healer_endangered(state, h, params) for h in healers}
    dead_healers = [a for a, u in allies.items() if is_healer(u) and not u["alive"]]

    # Candidate targets and urgency come from each healer's legal heal set.
    target_pool, urgency = {}, {}
    for healer in healers:
        if endangered[healer]:
            continue
        for action_id, entry in state.get("heal_candidates", {}).get(healer, {}).items():
            target_id = entry["target"]
            if target_id not in urgency:
                urgency[target_id] = score_heal_need(state, target_id, params)
            target_pool.setdefault(target_id, {})[healer] = {
                "action_id": int(action_id),
                "distance": entry["distance"],
                "missing_health": entry["missing_health"],
                "missing_health_factor": round(
                    entry["missing_health"] / (allies[target_id]["max_health"] or 1), 3)}

    # Wounded, healable-looking allies outside heal range: observation facts
    # only. They are never heal ASSIGNMENTS — idle healers may approach them
    # along legal moves so the next step can have legal edges at all.
    approach_pool = {}
    for target_id, unit in allies.items():
        if (target_id not in target_pool and unit["alive"] and not unit["is_flying"]
                and not is_healer(unit) and unit["health"] < unit["max_health"]):
            urgency.setdefault(target_id, score_heal_need(state, target_id, params))
            approach_pool[target_id] = True

    edges, value_edges = {}, {}
    for healer in healers:
        if endangered[healer]:
            edges[healer] = []
            continue
        rows = []
        for target_id, info in target_pool.items():
            if healer not in info:
                continue  # Not a legal healer-target pair in this snapshot.
            score = _edge_score(state, healer, target_id, info[healer],
                                urgency[target_id], previous_assignments, params)
            if use_values:
                bonus = (value_params["heal_value_weight"]
                         * max(0.0, min(1.0, urgency[target_id]))
                         * normalized_value(unit_values, "allies", target_id))
                value_edges.setdefault(healer, {})[target_id] = {
                    "urgency": urgency[target_id],
                    "original_edge_score": round(score, 4),
                    "value_bonus": bonus,
                    "final_edge_score": round(score + bonus, 4),
                }
                score += bonus
            rows.append((round(score, 4), target_id))
        edges[healer] = sorted(rows, key=lambda r: (-r[0], r[1]))

    assignments = _best_matching([h for h in healers if not endangered[h]], edges) \
        if any(not endangered[h] for h in healers) else {}

    pair_scoring_diag = None
    if pair_scorer is not None and pair_scoring in ("on", "shadow"):
        assignments, pair_scoring_diag = _apply_pair_scorer(
            state, assignments, edges, target_pool, urgency,
            previous_assignments, pair_scorer, active=pair_scoring == "on")

    healer_actions, retained_orders, released = {}, {}, {}
    for healer in healers:
        if endangered[healer]:
            # Self-preservation first: safest available move away from visible
            # enemies. Approaching wounded allies is NOT a fallback here — a
            # failed retreat must not drift the healer toward the threat.
            action = healer_safe_move(state, healer, params)
            healer_actions[healer] = action  # None -> base policy keeps the slot.
            if (previous_assignments or {}).get(healer):
                released[healer] = previous_assignments[healer]
            continue
        target_id = assignments.get(healer)
        if target_id is None:
            # Legal idle behavior: approach a legally healable target first;
            # otherwise head toward the most urgent out-of-range wounded ally
            # (approach only — never counted as provided healing). None
            # leaves the slot to the declared base policy.
            pool = target_pool or approach_pool
            healer_actions[healer] = healer_idle_action(state, healer, pool,
                                                        urgency, params)
            continue
        info = target_pool[target_id][healer]
        healer_actions[healer] = info["action_id"]
        retained_orders[healer] = any(
            order.get("ability_id") == HEAL_ORDER_ABILITY
            and order.get("target_unit") == target_id
            for order in allies[healer].get("orders", []))
        if (previous_assignments or {}).get(healer) and previous_assignments[healer] != target_id:
            released[healer] = previous_assignments[healer]

    reason = None
    live_healers = [h for h in healers]
    if not live_healers:
        reason = "all_healers_dead"
    elif not target_pool and not approach_pool:
        reason = "no_heal_demand"
    elif not assignments and all(endangered.values()):
        reason = "all_healers_endangered"

    result = {
        "available": True,
        "healers": healers,
        "dead_healers": dead_healers,
        "wounded_targets": sorted(target_pool),
        "approach_targets": sorted(approach_pool),
        "assignments": assignments,
        "healer_actions": healer_actions,
        "urgency": urgency,
        "missing_health": {t: target_pool[t][next(iter(target_pool[t]))]["missing_health"]
                           for t in target_pool},
        "self_preserving": {h: True for h, flag in endangered.items() if flag},
        "retained_orders": retained_orders,
        "released": released,
        "unassigned_healers": [h for h in healers if h not in assignments and not endangered[h]],
        "unassigned_reasons": {h: "no_legal_target" for h in healers
                               if h not in assignments and not endangered[h]},
        "terminated": reason is not None,
        "termination_reason": reason,
    }
    if pair_scoring_diag is not None:
        result["pair_scoring"] = pair_scoring_diag
    if use_values:
        occupied = set(assignments.values())
        witnesses = []
        for healer, target in assignments.items():
            selected = value_edges[healer][target]
            for alternative, row in value_edges[healer].items():
                if alternative in occupied:
                    continue
                # Conservative evidence of actual decision influence: the
                # selected edge beats a feasible, unoccupied alternative
                # only after its value bonus. This never reruns matching;
                # changes requiring multi-healer exchanges are not counted.
                original_preferred = (
                    row["original_edge_score"] > selected["original_edge_score"]
                    or (row["original_edge_score"] == selected["original_edge_score"]
                        and alternative < target))
                if (original_preferred and selected["value_bonus"] > row["value_bonus"]
                        and selected["final_edge_score"] >= row["final_edge_score"]):
                    witnesses.append({"healer": healer, "selected": target,
                                      "alternative_without_value": alternative})
                    break
        result["unit_value_diagnostics"] = {"heal_assign": {
            "edges": value_edges, "assignments": dict(assignments),
            "changed": bool(witnesses), "influenced_decisions": len(witnesses),
            "influence_witnesses": witnesses,
            "attribution": "conservative_feasible_single_healer_preference_change",
        }}
    return result
