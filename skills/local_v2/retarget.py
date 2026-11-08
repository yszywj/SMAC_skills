"""Compare the opportunity cost of leaving the current attack assignment.

Scores are one-volley estimates from the public catalogue and action menus.
All visible legal target bindings remain available for counterfactual runs.
"""
import math

from skills.current.skills.damage_model import expected_attack_damage_details
from skills.local_v1.library import ATTACK_ROLES, _attack_targets, _healer, _id


def _ready_volley(payload, actor, target):
    state = payload["state"]
    unit = state["allies"][actor]
    relation = state.get("relations", {}).get(actor, {}).get(target, {})
    ready = unit.get("weapon_ready")
    in_range = relation.get("in_weapon_range")
    if ready is False or in_range is False:
        return (0.0, 0.0)
    if ready is not True or in_range is not True or target not in _attack_targets(payload, actor):
        return None
    damage = expected_attack_damage_details(state, actor, target)
    if damage["lower_bound"] is None:
        return None
    return (damage["lower_bound"], damage["upper_bound"])


def _target_option(payload, baseline, actor, target):
    state = payload["state"]
    action = 6 + _id(target, "e")
    assigned = []
    unknown = []
    other_lower = other_upper = 0.0
    for ally, unit in state["allies"].items():
        if ally == actor or not unit.get("alive") or _healer(unit):
            continue
        slot = _id(ally, "u")
        if slot >= len(baseline["actions"]) or baseline["actions"][slot] != action:
            continue
        assigned.append(ally)
        bounds = _ready_volley(payload, ally, target)
        if bounds is None:
            unknown.append(ally)
        else:
            other_lower += bounds[0]
            other_upper += bounds[1]

    enemy = state["visible_enemies"][target]
    hp_parts = (enemy.get("health"), enemy.get("shield"))
    hp_known = all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in hp_parts)
    hp = sum(hp_parts) if hp_known else None
    own = _ready_volley(payload, actor, target)
    row = {
        "action": action, "target": target, "target_effective_health": hp,
        "other_assigned": sorted(assigned), "unknown_volley_actors": sorted(unknown),
        "other_ready_damage": None if unknown else [other_lower, other_upper],
        "actor_ready_damage": list(own) if own is not None else None,
        "adds_kill_certain": None, "adds_kill_possible": None,
        "useful_damage": None, "overkill_damage": None,
    }
    if own is None or unknown or hp is None:
        return row
    # Marginal damage is capped at what the other assigned attackers leave.
    need_min = max(0.0, hp - other_upper)
    need_max = max(0.0, hp - other_lower)
    row.update(
        adds_kill_certain=other_upper < hp <= other_lower + own[0],
        adds_kill_possible=own[1] > 0 and other_lower < hp <= other_upper + own[1],
        useful_damage=[min(own[0], need_min), min(own[1], need_max)],
        overkill_damage=[max(0.0, own[0] - need_max), max(0.0, own[1] - need_min)],
    )
    return row


def analyze(payload, baseline, actor):
    """List the original order and every alternate target; ties retain BASE."""
    unit = payload["state"]["allies"].get(actor)
    if (not unit or not unit.get("alive") or _healer(unit)
            or baseline.get("roles", {}).get(actor) not in ATTACK_ROLES):
        return None
    slot = _id(actor, "u")
    if slot >= len(baseline["actions"]):
        return None
    base = baseline["actions"][slot]
    targets = _attack_targets(payload, actor)
    if not any(6 + _id(e, "e") != base for e in targets):
        return None

    options = [_target_option(payload, baseline, actor, target) for target in targets]
    original = next((option for option in options if option["action"] == base), None)
    if original is None:
        options.insert(0, {"action": base, "target": None, "useful_damage": None})
    best = base
    # A MOVE/STOP order or an uncertain source volley cannot be priced as
    # a free move away from the old target. Preserve that order in exploitation.
    if (unit.get("weapon_ready") is True and original and original["useful_damage"] is not None
            and not original["adds_kill_possible"]):
        alternatives = [
            opt for opt in options if opt["action"] != base and opt["useful_damage"] is not None and
            (opt["adds_kill_certain"] or opt["useful_damage"][0] > original["useful_damage"][1] + 1e-9)
        ]
        if alternatives:
            winner = max(alternatives, key=lambda o: (o["adds_kill_certain"], o["useful_damage"][0], -o["action"]))
            best = winner["action"]
    return {
        "family": "retarget", "actor": actor, "base_action": base, "recommended_action": best,
        "options": options,
    }
