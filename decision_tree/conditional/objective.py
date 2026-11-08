"""Win-first objectives for audited terminal pairs, with explicit evidence scope.

Each row contains group_id, pair_id, scope, candidate and reference. Endpoints
are {"return": float, "outcome": "win" | "loss" | "draw" | "timeout"}, or None
when a terminal result is missing. The input layer owns native audit/provenance.
single_intervention rows describe a frozen prefix followed by one intervention.
whole_policy rows describe a full policy rollout, including all its interventions.
"""
from collections import defaultdict
import math


SCOPES = frozenset(("single_intervention", "whole_policy"))
OUTCOMES = frozenset(("win", "loss", "draw", "timeout"))
EPSILON = 1e-9


def partition_rows(rows, *, training_groups, validation_groups, partition):
    """Filter scene IDs before inspecting outcomes or fitting any condition."""
    train, heldout = set(training_groups), set(validation_groups)
    if not train or not heldout or train & heldout:
        raise ValueError("Training and validation scene sets must be nonempty and disjoint")
    if partition not in ("train", "validation"):
        raise ValueError("Unknown scene partition")
    selected = train if partition == "train" else heldout
    kept = []
    for row in rows:
        if row["group_id"] not in train | heldout:
            raise ValueError("Row belongs to an unregistered scene")
        if row["group_id"] in selected:
            kept.append(row)
    return kept


def _endpoint(value):
    if value is None:
        return None
    if (not isinstance(value, dict) or value.get("outcome") not in OUTCOMES
            or type(value.get("return")) not in (int, float)
            or not math.isfinite(value["return"])):
        raise ValueError("A terminal label needs a finite return and a known outcome")
    return value


def pair_labels(row):
    """Return the two supervised targets, or None for an incomplete pair."""
    candidate, reference = _endpoint(row["candidate"]), _endpoint(row["reference"])
    if candidate is None or reference is None:
        return None
    return {
        "win_gain": int(candidate["outcome"] == "win") - int(reference["outcome"] == "win"),
        "return_gain": candidate["return"] - reference["return"],
    }


def summarize_pairs(rows, *, scope, min_scenes=4, expected_groups=None):
    """Average matched pairs within each scene, then give scenes equal weight.

The caller supplies every planned pair, including pairs with a missing endpoint.
expected_groups binds a whole-policy cohort, including scenes with no row yet.
Partial summaries can describe training leaves but cannot approve deployment.
"""
    if scope not in SCOPES:
        raise ValueError("Unknown evidence scope")
    if type(min_scenes) is not int or min_scenes < 4:
        raise ValueError("A supported leaf requires at least four independent scenes")
    expected = None if expected_groups is None else set(expected_groups)
    seen, eligible_groups, incomplete_groups = set(), set(), set()
    by_scene = defaultdict(list)
    missing_pairs = []
    rescued, lost = [], []
    for row in rows:
        if row["scope"] != scope:
            raise ValueError("Single-intervention and whole-policy evidence cannot be pooled")
        group, pair = row["group_id"], row["pair_id"]
        if not isinstance(group, str) or not group or not isinstance(pair, str) or not pair:
            raise ValueError("Scene and pair IDs must be nonempty strings")
        if expected is not None and group not in expected:
            raise ValueError("Pair belongs to a different cohort")
        key = group, pair
        if key in seen or (scope == "whole_policy" and group in eligible_groups):
            raise ValueError("Duplicate pair or multiple whole-policy endpoints for one scene")
        seen.add(key)
        eligible_groups.add(group)
        labels = pair_labels(row)
        if labels is None:
            incomplete_groups.add(group)
            missing_pairs.append(list(key))
            continue
        by_scene[group].append(labels)
        if labels["win_gain"] > 0:
            rescued.append(list(key))
        elif labels["win_gain"] < 0:
            lost.append(list(key))

    registered = eligible_groups if expected is None else expected
    absent = registered - eligible_groups
    scene_values = {
        group: {label: math.fsum(p[label] for p in pairs) / len(pairs)
                for label in ("win_gain", "return_gain")}
        for group, pairs in sorted(by_scene.items())
    }
    count = len(scene_values)
    means = {label: math.fsum(v[label] for v in scene_values.values()) / count if count else None
             for label in ("win_gain", "return_gain")}
    return {
        "scope": scope,
        "cohort_declared": expected is not None,
        "eligible_pairs": len(seen),
        "matched_pairs": sum(map(len, by_scene.values())),
        "missing_pairs": missing_pairs,
        "eligible_scenes": len(registered),
        "matched_scenes": count,
        "missing_scene_rows": sorted(absent),
        "incomplete_scenes": sorted(incomplete_groups | absent),
        "complete": bool(registered) and not incomplete_groups and not absent,
        "minimum_scenes": min_scenes,
        "supported": count >= min_scenes,
        "mean_win_gain": means["win_gain"],
        "mean_return_gain": means["return_gain"],
        "rescued_pairs": rescued,
        "lost_pairs": lost,
        "scene_values": scene_values,
    }


def training_preference(summary):
    """Choose the direction of a leaf using win change before return change.

This describes the matched evidence. It does not approve a policy for deployment.
"""
    if not summary["supported"]:
        return {"choice": "insufficient", "reason": "fewer_than_minimum_matched_scenes"}
    wins, reward = summary["mean_win_gain"], summary["mean_return_gain"]
    if wins > EPSILON:
        return {"choice": "candidate", "reason": "higher_win_rate"}
    if wins < -EPSILON:
        return {"choice": "reference", "reason": "lower_win_rate"}
    if reward > EPSILON:
        return {"choice": "candidate", "reason": "equal_wins_higher_return"}
    if reward < -EPSILON:
        return {"choice": "reference", "reason": "equal_wins_lower_return"}
    return {"choice": "tie", "reason": "equal_wins_and_return"}


def adoption_verdict(summary):
    """Apply the objective to an entire declared, matched whole-policy cohort.

Callers must freeze the candidate before collecting validation outcomes. This
function enforces the evidence shape; its verdict is an empirical comparison.
"""
    if summary["scope"] != "whole_policy":
        return {"adopt": False, "reason": "requires_whole_policy_outcomes"}
    if not summary["cohort_declared"] or not summary["complete"]:
        return {"adopt": False, "reason": "requires_complete_declared_cohort"}
    preference = training_preference(summary)
    return {"adopt": preference["choice"] == "candidate", "reason": preference["reason"]}
