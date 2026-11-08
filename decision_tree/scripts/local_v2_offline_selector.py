"""Reconstruct one-shot selectors on existing BASE and audited branch evidence."""
from collections import Counter
from math import fsum
from pathlib import Path

from decision_tree.local.runtime import digest
from decision_tree.phase2.policy import normalize_policy
from decision_tree.scripts.local_v2_training_report import EPS, first_selections, read
from skills.local_v2.explore import checksum, require
from skills.local_v2.run import parse_binding


def _endpoint(step, effect, label):
    """A non-BASE effect key describes an action, so it also needs its BASE step."""
    if label is not None:
        require(label["effect"] == effect, "Outcome belongs to a different effect")
    return {"step": None if effect == "BASE" else step, "effect": effect, "label": label}


def _compare(records, left_name, right_name):
    """Known identical actions have zero relative gain even without an absolute return."""
    deltas = []
    absolute = []
    differing = []
    changed_measured = []
    equal_unobserved = []
    unknown = []
    rescued, lost = [], []
    for scene in records:
        seed = scene["seed"]
        left, right = scene["arms"][left_name], scene["arms"][right_name]
        same = (left["step"], left["effect"]) == (right["step"], right["effect"])
        if not same:
            differing.append(seed)
        lhs, rhs = left["label"], right["label"]
        if lhs is not None and rhs is not None:
            require(not same or lhs == rhs, "Inconsistent terminal return for the same action")
            gain = lhs["return"] - rhs["return"]
            deltas.append(gain)
            absolute.append((lhs, rhs))
            if not same:
                changed_measured.append((seed, gain))
            if lhs["outcome"] == "win" and rhs["outcome"] != "win":
                rescued.append(seed)
            if rhs["outcome"] == "win" and lhs["outcome"] != "win":
                lost.append(seed)
        elif same:
            # Both policies followed the exact same trajectory. Its terminal
            # outcome is missing, so credit only the known relative difference.
            deltas.append(0.0)
            equal_unobserved.append(seed)
        else:
            unknown.append(seed)
    return {
        "eligible_scenes": len(records),
        "paired_terminal_scenes": len(absolute),
        "left_wins_on_paired_terminal_scenes": sum(lhs["outcome"] == "win" for lhs, _ in absolute),
        "right_wins_on_paired_terminal_scenes": sum(rhs["outcome"] == "win" for _, rhs in absolute),
        "relative_known_scenes": len(deltas),
        "same_effect_without_terminal_seeds": equal_unobserved,
        "relative_unknown_seeds": unknown,
        "changed_effect_scenes": len(differing),
        "changed_matched_scenes": len(changed_measured),
        "changed_matched_gains": [{"seed": seed, "gain": value} for seed, value in changed_measured],
        "changed_matched_mean_gain": fsum(v for _, v in changed_measured) / len(changed_measured)
        if changed_measured else None,
        "relative_known_mean_gain": fsum(deltas) / len(deltas) if deltas else None,
        "positive_scenes": sum(value > EPS for value in deltas),
        "negative_scenes": sum(value < -EPS for value in deltas),
        "rescued_seeds": rescued,
        "lost_seeds": lost,
    }


def _absolute(records, name):
    known = [s["arms"][name]["label"] for s in records if s["arms"][name]["label"] is not None]
    return {
        "eligible_scenes": len(records),
        "terminal_known_scenes": len(known),
        "wins_on_known_scenes": sum(r["outcome"] == "win" for r in known),
        "mean_return_on_known_scenes": fsum(r["return"] for r in known) / len(known) if known else None,
    }


def evaluate_selectors(campaign, partition, choose_v2, rows, meta):
    """Evaluate a frozen body switch on exact first interventions, one row per scene.

    choose_v2(row) returns True to use the recommended v2 action and False to
    keep the old selected action. The two existing selectors supply the time,
    actor and frozen target. The ready-retarget selector is a research probe.
    """
    campaign = Path(campaign)
    groups_key = {"train": "training_group_ids", "holdout": "validation_group_ids"}[partition]
    split = read(campaign / "split.json")
    groups = set(split[groups_key])
    require(groups == set(meta["groups"]) == set(meta["base_by_group"]), "Wrong evaluation partition")
    require(not set(split["training_group_ids"]) & set(split["validation_group_ids"]), "Overlapping split")
    summary = read(campaign / "plan" / "summary.json")
    base = Path(summary["base"])
    manifest_path = base / "manifest.json"
    require(checksum(manifest_path) == summary["base_hashes"]["manifest.json"], "Changed BASE manifest")
    manifest = read(manifest_path)
    ordinals = {n: s for n, s in enumerate(manifest["episodes"], 1) if s["sha256"] in groups}
    require(len(ordinals) == len(groups), "Missing BASE scenes")
    policies = {k: normalize_policy(read(campaign / (k + ".json"))) for k in ("old", "ready_retarget")}
    require(policies["old"] == normalize_policy({"mode": "regroup"}), "Unexpected incumbent")
    probe = normalize_policy({"mode": "regroup", "fallback": {"family": "retarget"}})
    require(policies["ready_retarget"] == probe, "Unexpected ready-retarget probe")
    selections = first_selections(base, ordinals, policies, summary["base_hashes"]["events.jsonl"])
    index = {}
    for row in rows:
        require(row["group_id"] in groups, "Pair from a different partition")
        key = (row["group_id"], row["step"], row["family"], row["actor"], row["old_action"])
        require(key not in index, "Duplicate frozen-action pair")
        index[key] = row

    records = []
    dispatch = {name: Counter() for name in policies}
    for n, scene in ordinals.items():
        group, seed = scene["sha256"], scene["sc2_seed"]
        baseline = _endpoint(None, "BASE", meta["base_by_group"][group])
        arms = {"adaptive": baseline}
        choices = {}
        for name, v1_arm, candidate_arm in (("old", "incumbent", "candidate_old"),
                                            ("ready_retarget", "probe_v1", "candidate_probe")):
            choice = selections[n].get(name)
            choices[name] = choice
            stage = choice["action"] if choice else "no_call"
            dispatch[name][stage] += 1
            if stage != "execute":
                arms[v1_arm] = arms[candidate_arm] = baseline
                continue
            step = choice["step"]
            family, actor, old_action = parse_binding(choice["selected_id"])
            row = index.get((group, step, family, actor, old_action))
            require(row is not None and row["in_frozen_menu"], "Frozen executed binding missing from pairs")
            require(row["base"] == baseline["label"] and row["seed"] == seed, "Mismatched pair/scene")
            old = _endpoint(step, row["old_effect"], row["old"])
            use_v2 = choose_v2(row)
            require(type(use_v2) is bool, "The frozen switch must return a boolean")
            selected = _endpoint(step, row["new_effect"], row["new"]) if use_v2 else old
            choices[name] = {**choice, "family": family, "actor": actor, "old_action": old_action,
                             "candidate_action": row["new_action"] if use_v2 else old_action, "use_v2": use_v2}
            dispatch[name]["execute_" + family] += 1
            if use_v2:
                dispatch[name]["switch_to_v2_" + family] += 1
                if selected["effect"] != old["effect"]:
                    dispatch[name]["effective_change_" + family] += 1
            arms[v1_arm], arms[candidate_arm] = old, selected
        records.append({"seed": seed, "group_id": group, "choices": choices, "arms": arms})
    records.sort(key=lambda row: row["seed"])
    contrasts = (("candidate_old", "incumbent"), ("candidate_old", "adaptive"), ("incumbent", "adaptive"),
                 ("candidate_probe", "probe_v1"), ("candidate_probe", "incumbent"), ("candidate_probe", "adaptive"))
    return {
        "partition": partition,
        "reconstruction": "Frozen first consumed action only. BASE supplies all other commands. "
                          "Whole-policy endpoints require the exact effect on the matching BASE prefix. "
                          "Identical actions prove zero relative gain even when absolute returns are missing.",
        "selector_sha256": {n: digest(p) for n, p in policies.items()},
        "dispatch": {n: dict(v) for n, v in dispatch.items()},
        "comparisons": {f"{l}_vs_{r}": _compare(records, l, r) for l, r in contrasts},
        "arms": {name: _absolute(records, name) for name in records[0]["arms"]},
        "by_scene": records,
    }
