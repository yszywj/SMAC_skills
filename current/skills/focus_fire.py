"""FocusFireBudget: a formation-size-independent focus-fire allocator.

Model assumptions and scope (first version, deliberately conservative):

- The allocator works on one snapshot of a ``build_request`` payload state
  (allies / visible_enemies / relations / unit_type_profiles /
  movement_candidates / heal_candidates). It never reads hidden enemy state:
  only enemies present in ``state["visible_enemies"]`` can ever be attacked.
- Predicted damage covers exactly one pending attack per attacker: a unit
  whose weapon is ready, whose SMAC attack action is legal and whose weapon
  range is not known to be exceeded contributes its expected single-attack
  damage to the budget. Attackers on cooldown or out of range may still be
  assigned (attack orders persist through cooldown) but contribute nothing to
  the budget this step, so the budget under-counts rather than over-counts.
- Expected damage per attack uses the public combat profile weapons
  (base_damage * hits_per_attack - target armor, floor 0.5 per SC2 rules for
  known damage). If any required datum is unknown (effect-based weapons,
  missing API weapon lists, unknown range), that attacker's damage is treated
  as unknown: it is assigned but never accumulated into the budget.
- The budget is needs-based, recomputed from the current observation every
  step: a target's need is its observed effective health. Earlier versions
  subtracted a cross-step ``pending_damage`` ledger, but the observation
  cannot reliably tell a landed attack from an in-flight one, so that ledger
  could double-subtract damage already reflected in observed health. The
  ledger is therefore disabled (option A): each step re-derives the budget
  from fresh observations only. Consequence, documented: attacks ordered but
  not yet resolved are not credited, so the budget can overestimate the
  firepower still needed and briefly over-assign attackers (visible as
  estimated overkill). That is deliberately preferred over double-counting.
- SC2 ResponseAction success is never interpreted as damage dealt.
"""

import math

from .combat_timing import can_fire_within_window, window_loops
from .damage_model import VERSION as DAMAGE_MODEL_VERSION, expected_attack_damage_details
from ..unit_value import enabled_values, normalized_value, validate_parameters

NO_ATTACK_FALLBACK = None  # A living unit with no legal action at all.


def effective_health(enemy):
    return enemy["health"] + enemy["shield"]


def _compatible_weapon(profile, enemy):
    """Return the profile weapon usable against this target, or None."""
    target_types = {2, 3} if enemy["is_flying"] else {1, 3}
    if enemy["unit_type_id"] == 4:  # Colossus: also hittable by anti-air.
        target_types.add(2)
    for weapon in profile.get("weapons", []):
        kind = {"ground": 1, "air": 2, "any": 3}.get(weapon.get("target_type"), 3)
        if kind in target_types:
            return weapon
    return None


def expected_damage(state, actor, enemy_id, parameters=None):
    """Damage one attack from actor is expected to deal, or None if unknown.

    Armor is applied per hit (SC2 semantics): two hits of 5 damage against 2
    armor deal (5-2)*2 = 6, not 10-2 = 8. Unknown weapon data OR unknown
    target armor yields None — never a fabricated number, and never a
    zero-armor assumption that would overestimate the budget.

    ``accurate_damage=True`` adds catalog attribute bonuses and separates shield
    handling; uncertain shield spillover returns a documented lower bound.
    Default calls preserve the legacy formula exactly.
    """
    if (parameters or {}).get("accurate_damage"):
        return expected_attack_damage_details(state, actor, enemy_id)["estimate"]
    ally = state["allies"][actor]
    enemy = state["visible_enemies"][enemy_id]
    profile = state.get("unit_type_profiles", {}).get(str(ally["unit_type_id"]), {})
    enemy_profile = state.get("unit_type_profiles", {}).get(str(enemy["unit_type_id"]), {})
    weapon = _compatible_weapon(profile, enemy)
    if weapon is None or not profile.get("known", False):
        return None
    base = weapon.get("base_damage")
    hits = weapon.get("hits_per_attack") or 1
    if base is None:  # Effect-based weapons omitted from the API.
        return None
    if not enemy_profile.get("known", False):
        return None  # Target armor unknown: refuse to guess zero armor.
    armor = enemy_profile.get("armor")
    if armor is None:
        return None
    per_hit = max(base - armor, 0.5)  # SC2 floors each hit at 0.5.
    return round(per_hit * hits, 3)


def _countable(state, actor, enemy_id, relation, parameters=None):
    """True when this attacker plausibly contributes damage in this window.

    Default (window_fire off) keeps the original semantics: weapon ready now.
    With window_fire on — inspired by Sharpy's ready_to_shoot(), redesigned
    for SMAC's fixed step_mul — a weapon whose cooldown expires within the
    current control window also counts, so a 3-loop cooldown under an 8-loop
    step no longer reads as "cannot fight". Either way this is a *budget
    prediction*, never an observed hit.
    """
    if not relation["smac_attack_available"] or relation["in_weapon_range"] is False:
        return False
    if parameters and parameters.get("window_fire"):
        return can_fire_within_window(state, actor, enemy_id, parameters=parameters)
    return bool(state["allies"][actor]["weapon_ready"])


def _attacker_order(state, actor, enemy_id, relation, damage, previous_assignments,
                    parameters=None):
    """Deterministic attacker preference for one target (smaller is better).

    With either new combat flag on: window fire opportunity first, then actual range and
    position, then expected damage contribution, then the (demoted) bonus for
    keeping the previous target, then stable IDs. Keeping an old target must
    not mask a clearly better kill opportunity, so hysteresis no longer
    dominates.
    """
    if not any((parameters or {}).get(key) for key in ("window_fire", "target_scoring")):
        return (
            0 if (previous_assignments or {}).get(actor) == enemy_id else 1,
            0 if relation["in_weapon_range"] is True else 1,
            0 if state["allies"][actor]["weapon_ready"] else 1,
            -(damage if damage is not None else 0.0),
            int(actor[1:]),
        )
    fires_this_window = (can_fire_within_window(state, actor, enemy_id,
                                                 parameters=parameters)
                         if (parameters or {}).get("window_fire")
                         else bool(state["allies"][actor]["weapon_ready"]))
    return (
        0 if fires_this_window else 1,
        0 if relation["in_weapon_range"] is True else 1,
        -(damage if damage is not None else 0.0),
        0 if (previous_assignments or {}).get(actor) == enemy_id else 1,  # hysteresis
        int(actor[1:]),
    )


def _soft_target_score(state, enemy_id, damage_sum, parameters):
    """Existing bounded target score; smaller is better, no value term yet."""
    if not (parameters or {}).get("target_scoring"):
        return 0.0
    enemy = state["visible_enemies"][enemy_id]
    need = effective_health(enemy)
    damage_taken = 1.0 - min(1.0, need / max(1.0, enemy["max_health"] + enemy["max_shield"]))
    distances = [row[enemy_id]["distance"]
                 for row in state.get("relations", {}).values()
                 if enemy_id in row and row[enemy_id].get("smac_attack_available")]
    proximity = 1.0 - min(1.0, (min(distances) / 12.0)) if distances else 0.0
    overkill = min(1.0, max(0.0, damage_sum - need) / max(1.0, need))
    return round(-(0.5 * damage_taken + 0.3 * proximity - 0.2 * overkill), 3)


def _target_order(state, enemy_id, priority_target, legal_attackers, damage_sum,
                  parameters=None, one_shot=False):
    """Deterministic target priority (smaller is better).

    Base order: priority target, most attackers able to hit it, cheapest to
    kill, lowest effective health, enemy ID. With target_scoring on, a
    bounded Sharpy-style value is added AFTER the kill-cost constraint: target
    damage taken (already hurt targets are closer to dying), proximity, and an
    overkill penalty. Soft score never overrides legality or kill economics.
    """
    enemy = state["visible_enemies"][enemy_id]
    need = effective_health(enemy)
    attacks_to_kill = math.ceil(need / damage_sum) if damage_sum > 0 else float("inf")
    if not (parameters or {}).get("target_scoring"):
        return (0 if enemy_id == priority_target else 1, -legal_attackers,
                attacks_to_kill, need, int(enemy_id[1:]))
    score = _soft_target_score(state, enemy_id, damage_sum, parameters)
    return (
        0 if enemy_id == priority_target else 1,
        0 if one_shot else 1,  # A ready finisher beats merely having more attackers.
        attacks_to_kill,
        -legal_attackers,
        score,
        need,
        int(enemy_id[1:]),
    )


def _retain_focus(ordered, facts, previous_assignments, parameters, value_params=None):
    """Apply legacy retention, or bounded value retention within the same hard tier."""
    ordered = list(ordered)
    if not previous_assignments or len(ordered) < 2:
        return ordered, None
    if value_params is None and not (parameters or {}).get("target_scoring"):
        return ordered, None
    counts = {enemy: sum(target == enemy for target in previous_assignments.values())
              for enemy in ordered}
    old = max(ordered, key=lambda e: (counts[e], -int(e[1:])))
    best = ordered[0]
    a, b = facts[old], facts[best]
    same_tier = all(a[field] == b[field] for field in (
        "explicit_priority", "one_shot", "estimated_volleys"))
    if value_params is not None:
        # These are exactly the original sorting constraints for the selected
        # target_scoring mode; enabling values must not invent another order.
        same_tier = a["hard_tier"] == b["hard_tier"]
        same_tier = same_tier and (
            a["final_soft_score"] - b["final_soft_score"]
            <= value_params["focus_switch_margin"])
    if (counts[old] and old != best and same_tier and a["legal_attackers"] > 0
            and b["effective_health"] >= a["effective_health"] * 0.8):
        ordered.remove(old)
        ordered.insert(0, old)
        return ordered, old
    return ordered, None


def target_priority(state, priority_target=None, previous_assignments=None, parameters=None,
                    unit_values=None, diagnostics=None):
    """Add a bounded soft value term without replacing the legacy sort skeleton."""
    relations = state.get("relations", {})
    priority_target = priority_target if priority_target in state["visible_enemies"] else None
    use_values = enabled_values(unit_values)
    value_params = validate_parameters(unit_values.get("parameters")) if use_values else None
    value_active = use_values and (value_params["focus_value_weight"] > 0
                                  or value_params["focus_switch_margin"] > 0)
    scoring = bool((parameters or {}).get("target_scoring"))
    rows, baseline_rows, facts = [], [], {}
    for enemy_id, enemy in state["visible_enemies"].items():
        # Live snapshots already exclude deaths. Keep the old behavior exactly
        # when the new consumer is disabled, including adversarial fixtures.
        if value_active and (enemy.get("alive") is False or effective_health(enemy) <= 0):
            continue
        legal = ready_damage = 0
        one_shot = False
        for actor, actor_relations in relations.items():
            relation = actor_relations.get(enemy_id)
            if relation is None:
                continue
            if relation["smac_attack_available"]:
                legal += 1
                if _countable(state, actor, enemy_id, relation, parameters):
                    damage = expected_damage(state, actor, enemy_id, parameters)
                    if damage is not None:
                        ready_damage += damage
                        one_shot = one_shot or damage >= effective_health(enemy)
        baseline_key = _target_order(state, enemy_id, priority_target, legal, ready_damage,
                                     parameters, one_shot)
        original_score = _soft_target_score(state, enemy_id, ready_damage, parameters)
        # With scoring off the original final tie-breaks are health and ID;
        # insert a soft slot there. With it on reuse its existing score slot.
        hard_tier = baseline_key[:-3] if scoring else baseline_key[:-2]
        bonus = (value_params["focus_value_weight"] * normalized_value(
            unit_values, "visible_enemies", enemy_id)) if use_values else 0.0
        facts[enemy_id] = {
            "explicit_priority": enemy_id == priority_target,
            "one_shot": one_shot,
            "estimated_volleys": math.ceil(effective_health(enemy) / ready_damage)
                                 if ready_damage > 0 else None,
            "legal_attackers": legal, "effective_health": effective_health(enemy),
            "hard_tier": tuple(None if value == float("inf") else value for value in hard_tier),
            "original_soft_score": original_score,
            "value_bonus": bonus, "final_soft_score": original_score - bonus,
        }
        key = (*hard_tier, original_score - bonus, *baseline_key[-2:]) if value_active else baseline_key
        rows.append((key, enemy_id))
        baseline_rows.append((baseline_key, enemy_id))
    before, _ = _retain_focus([enemy for _, enemy in sorted(baseline_rows)], facts,
                               previous_assignments, parameters)
    ordered = [enemy for _, enemy in sorted(rows)]
    unretained_best = ordered[0] if ordered else None
    if value_active:
        ordered, retained = _retain_focus(ordered, facts, previous_assignments,
                                          parameters, value_params)
    else:
        ordered, retained = _retain_focus(ordered, facts, previous_assignments, parameters)
    if use_values and diagnostics is not None:
        changed = bool(ordered and before and ordered[0] != before[0]
                       and facts[ordered[0]]["value_bonus"] > facts[before[0]]["value_bonus"]
                       and retained is None)
        diagnostics.update(targets=facts, ordered_before_value=before,
                           ordered_after_value=ordered, unretained_best=unretained_best,
                           retained_target=retained, changed=changed,
                           attribution="legacy_baseline_first_target_comparison_after_hysteresis")
    return ordered


def allocate_focus_fire(state, n_agents, priority_target=None,
                        pending_damage=None, previous_assignments=None, parameters=None,
                        unit_values=None):
    """Assign attackers to visible targets under a per-target damage budget.

    ``pending_damage`` is accepted for interface compatibility but ignored
    (see module docstring: cross-step pending ledgers can double-count damage
    already reflected in observed health). The budget is recomputed from the
    current observation each step.

    Returns a plain dict with the variable-length assignment map, the
    environment-length action list, budget telemetry and a termination hint.
    Every emitted attack action is legal in the snapshot; units that cannot
    attack anything receive an explicit deterministic fallback action.
    """
    previous_assignments = previous_assignments or {}
    allies, relations = state["allies"], state.get("relations", {})
    value_diagnostics = {}
    targets = target_priority(state, priority_target, previous_assignments, parameters,
                              unit_values, value_diagnostics)
    value_params = validate_parameters(unit_values.get("parameters")) if enabled_values(unit_values) else None
    restrict_fallback_targets = value_params is not None and (
        value_params["focus_value_weight"] > 0 or value_params["focus_switch_margin"] > 0)
    attackers = sorted((actor for actor, unit in allies.items() if unit["alive"]), key=lambda a: int(a[1:]))

    damage = {(actor, enemy_id): expected_damage(state, actor, enemy_id, parameters)
              for actor in attackers for enemy_id in targets}
    need = {enemy_id: max(0.0, effective_health(state["visible_enemies"][enemy_id])) for enemy_id in targets}
    predicted = {enemy_id: 0.0 for enemy_id in targets}
    immediate = {enemy_id: 0.0 for enemy_id in targets}
    window_only = {enemy_id: 0.0 for enemy_id in targets}
    assignments, unknown_damage = {}, {}

    free = list(attackers)
    for enemy_id in targets:
        ordered = sorted(
            (actor for actor in free
             if relations.get(actor, {}).get(enemy_id, {}).get("smac_attack_available")),
            key=lambda actor: _attacker_order(state, actor, enemy_id,
                                              relations[actor][enemy_id], damage[(actor, enemy_id)],
                                              previous_assignments, parameters))
        for actor in ordered:
            if predicted[enemy_id] >= need[enemy_id]:
                break  # Budget met; keep remaining attackers for later targets.
            assignments[actor] = enemy_id
            free.remove(actor)
            pair = damage[(actor, enemy_id)]
            if _countable(state, actor, enemy_id, relations[actor][enemy_id], parameters):
                if pair is None:
                    unknown_damage[actor] = enemy_id  # Assigned but not budgeted.
                else:
                    predicted[enemy_id] += pair
                    if state["allies"][actor]["weapon_ready"]:
                        immediate[enemy_id] += pair
                    else:
                        window_only[enemy_id] += pair
    # Attackers with no budget slot left still fire somewhere useful: their
    # best legal target in priority order. Their damage becomes overkill.
    overflow = {}
    for actor in free:
        options = [enemy_id for enemy_id in targets
                   if relations.get(actor, {}).get(enemy_id, {}).get("smac_attack_available")]
        if options:
            assignments[actor] = options[0]
            overflow[actor] = options[0]
            if _countable(state, actor, options[0], relations[actor][options[0]], parameters)                     and damage[(actor, options[0])] is not None:
                predicted[options[0]] += damage[(actor, options[0])]
                if state["allies"][actor]["weapon_ready"]:
                    immediate[options[0]] += damage[(actor, options[0])]
                else:
                    window_only[options[0]] += damage[(actor, options[0])]

    overkill = {enemy_id: round(max(0.0, predicted[enemy_id]
                                    - effective_health(state["visible_enemies"][enemy_id])), 2)
                for enemy_id in targets}
    # Predicted shortfall against each target's survival: 0 once the budget
    # covers it. Positive after all attackers are spent means the budget is
    # insufficient this step (prediction only, never a kill guarantee).
    budget_deficit = {enemy_id: round(max(0.0, need[enemy_id] - predicted[enemy_id]), 2)
                      for enemy_id in targets}

    actions, fallbacks, unexecutable = [0] * n_agents, {}, []
    for actor in attackers:
        index = int(actor[1:])
        if index >= n_agents:
            continue
        enemy_id = assignments.get(actor)
        if enemy_id is not None:
            actions[index] = 6 + int(enemy_id[1:])
            continue
        action = fallback_action(state, actor, attack_targets=set(targets)
                                 if restrict_fallback_targets else None)
        if action is NO_ATTACK_FALLBACK:
            unexecutable.append(actor)
        else:
            actions[index] = action
            fallbacks[actor] = action

    reason = None
    if not state["visible_enemies"]:
        reason = "no_visible_targets"
    elif not assignments:
        reason = "no_legal_attacks"
    elif unexecutable:
        reason = "units_without_legal_actions"

    result = {
        "assignments": assignments,
        "actions": actions,
        "target_priority": targets,
        "predicted_damage": {enemy_id: round(value, 2) for enemy_id, value in predicted.items()},
        # Diagnostic split: damage from weapons ready NOW vs damage that only
        # becomes possible within this control window (window_fire on).
        # Both are predictions; neither is observed/actual damage.
        "immediate_damage": {enemy_id: round(value, 2) for enemy_id, value in immediate.items()},
        "window_only_damage": {enemy_id: round(value, 2) for enemy_id, value in window_only.items()},
        "window_fire_enabled": bool((parameters or {}).get("window_fire")),
        "target_scoring_enabled": bool((parameters or {}).get("target_scoring")),
        "pending_damage_applied": {},  # Cross-step ledger disabled; see docstring.
        "estimated_overkill": overkill,
        "budget_deficit": budget_deficit,
        "unknown_damage_attackers": unknown_damage,
        "overflow_attackers": overflow,
        "unassigned_units": sorted(free, key=lambda a: int(a[1:])),
        "fallback_actions": fallbacks,
        "participating_units": sorted(assignments, key=lambda a: int(a[1:])),
        "unexecutable_units": unexecutable,
        "terminated": reason is not None,
        "termination_reason": reason,
    }
    if (parameters or {}).get("accurate_damage"):
        result["damage_model"] = {
            "version": DAMAGE_MODEL_VERSION, "enabled": True,
            "budget_uses": "lower_bound; unknown_pairs_not_budgeted",
            "pairs": {actor: {enemy_id: expected_attack_damage_details(state, actor, enemy_id)
                              for enemy_id in targets} for actor in attackers},
        }
    if enabled_values(unit_values):
        value_diagnostics["assignments"] = dict(assignments)
        witnessed_actor = None
        before = value_diagnostics.get("ordered_before_value", [])
        if value_diagnostics.get("changed") and before and need[before[0]] > 0:
            original_first = before[0]
            original_candidates = [actor for actor in attackers
                                   if relations.get(actor, {}).get(original_first, {}).get(
                                       "smac_attack_available")]
            if original_candidates:
                first_actor = min(original_candidates, key=lambda actor: _attacker_order(
                    state, actor, original_first, relations[actor][original_first],
                    damage[(actor, original_first)], previous_assignments, parameters))
                if assignments.get(first_actor) != original_first:
                    witnessed_actor = first_actor
        value_diagnostics["changed"] = witnessed_actor is not None
        value_diagnostics["influenced_decisions"] = int(witnessed_actor is not None)
        value_diagnostics["first_assignment_witness"] = witnessed_actor
        value_diagnostics["attribution"] = "conservative_first_assignment_witness_from_same_scores"
        result["unit_value_diagnostics"] = {"focus_fire": value_diagnostics}
    return result


def fallback_action(state, actor, attack_targets=None):
    """Explicit deterministic legal fallback: attack, heal, then move.

    Returns an action ID valid for this snapshot, or None when the unit has
    no legal non-stop action at all. Stop is never used as a no-op.
    """
    relations = state.get("relations", {}).get(actor, {})
    attacks = sorted((enemy_id for enemy_id, relation in relations.items()
                      if relation["smac_attack_available"]
                      and (attack_targets is None or enemy_id in attack_targets)),
                     key=lambda enemy_id: (relations[enemy_id]["distance"], int(enemy_id[1:])))
    if attacks:
        return 6 + int(attacks[0][1:])
    heals = state.get("heal_candidates", {}).get(actor, {})
    if heals:
        return int(min(heals, key=lambda a: (heals[a]["distance"], int(a))))
    moves = state.get("movement_candidates", {}).get(actor, {})
    if moves:
        def move_key(action_id):
            entry = moves[action_id]
            distances = list(entry["distances_to_visible_enemies"].values())
            hint = state.get("search_hint")
            if not distances and hint:
                return (math.dist(entry["destination"], hint["position"]), int(action_id))
            return (min(distances) if distances else entry["distance_to_search_location"],
                    int(action_id))
        return int(min(moves, key=move_key))
    return NO_ATTACK_FALLBACK
