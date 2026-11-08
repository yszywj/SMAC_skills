"""Conservative one-step retargeting with an explicit cost for leaving the source.

The estimates reuse v2's visible, ready-volley accounting. They describe the
current volley only. They do not predict later volleys, splash or projectiles.
The scheduler owns timing and repeat limits; this module is a pure action body.
"""
from skills.local_v2.retarget import analyze as analyze_v2


EPSILON = 1e-9


def _guard_reason(option, source_upper):
    if option["useful_damage"] is None:
        return "wait_target_information"
    if option["other_ready_damage"][0] >= option["target_effective_health"]:
        return "target_already_covered"
    useful_lower = option["useful_damage"][0]
    if useful_lower <= EPSILON:
        return "no_marginal_damage"
    if useful_lower + EPSILON < source_upper:
        return "source_damage_cost_exceeds_target_gain"
    if not option["adds_kill_certain"] and useful_lower <= source_upper + EPSILON:
        return "no_strict_improvement"
    return "supported_volley_improvement"


def analyze(payload, baseline, actor):
    """Describe every legal target and recommend an individually supported change.

    A new target must preserve the source's upper-bound useful damage while
    strictly improving guaranteed useful damage or the estimated certain kill.
    An ally's assignment is recorded as a research signal; only known ready
    damage contributes to the coverage check.
    """
    result = analyze_v2(payload, baseline, actor)
    if result is None:
        return None
    base_action = result["base_action"]
    result["recommended_action"] = base_action
    source = next((option for option in result["options"] if option["action"] == base_action), None)
    source_damage = source.get("useful_damage") if source else None
    result.update(source_target=source.get("target") if source else None,
                  source_useful_damage=source_damage, reason="keep_without_supported_improvement")

    for option in result["options"]:
        other = option.get("other_ready_damage")
        hp = option.get("target_effective_health")
        remaining = None if other is None or hp is None else [max(0.0, hp - other[1]),
                                                             max(0.0, hp - other[0])]
        useful = option.get("useful_damage")
        gain = None if useful is None or source_damage is None else useful[0] - source_damage[1]
        option.update(destination_remaining_health=remaining,
                      own_marginal_gain_lower=gain,
                      other_assignment_present=bool(option.get("other_assigned")),
                      eligible=False,
                      guard_reason="current_order" if option["action"] == base_action else "not_evaluated")

    unit = payload["state"]["allies"][actor]
    if not source or source.get("target") is None:
        result["reason"] = "preserve_nonattack_order"
    elif unit.get("weapon_ready") is not True:
        result["reason"] = "wait_weapon_ready"
    elif source_damage is None:
        result["reason"] = "wait_source_information"
    elif source["adds_kill_possible"]:
        result["reason"] = "preserve_source_finish"
    else:
        for option in result["options"]:
            if option["action"] != base_action:
                reason = _guard_reason(option, source_damage[1])
                option.update(guard_reason=reason, eligible=reason == "supported_volley_improvement")
        alternatives = [option for option in result["options"] if option["eligible"]]
        if alternatives:
            winner = max(alternatives, key=lambda option: (option["adds_kill_certain"],
                         option["own_marginal_gain_lower"], -option["action"]))
            result.update(recommended_action=winner["action"], reason="supported_volley_improvement")

    if result["reason"] != "keep_without_supported_improvement" and result["recommended_action"] == base_action:
        for option in result["options"]:
            if option["action"] != base_action:
                option["guard_reason"] = result["reason"]
    return result
