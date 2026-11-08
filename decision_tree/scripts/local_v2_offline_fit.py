"""Fit and validate shallow action-version rules offline on the interrupted 40-scene cohort."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import math
from pathlib import Path
from statistics import mean

from .local_v2_training_report import read, save
from skills.local_v2.explore import checksum, require


EPS = 1e-9
PREDICATES = {
    "contact_unknown": ("known_contact", "false"),
    "support_can_attack": ("support_can_attack", "true"),
    "team_has_fired": ("team_fired", "true"),
    "team_yet_to_fire": ("team_fired", "false"),
    "old_delays_contact": ("old_contact_cost", "positive"),
    "old_margin_nonpositive": ("old_margin", "nonpositive"),
    "new_keeps_adaptive": ("new_is_base", "true"),
    "weapon_ready": ("weapon_ready", "true"),
    "weapon_cooling": ("weapon_ready", "false"),
    "original_possible_kill": ("base_kill_possible", "true"),
    "old_no_extra_useful_damage": ("old_useful_delta", "nonpositive"),
    "ally_also_on_old_alternative": ("old_other_assigned", "positive"),
    "ally_also_on_original": ("base_other_assigned", "positive"),
}


def applies(rule, row, last_step):
    if row["step"] > last_step or rule == "never":
        return False
    if rule == "all":
        return True
    key, op = PREDICATES[rule]
    value = row["features"][key]
    if value is None:
        return False
    if op == "true":
        return value == 1
    if op == "false":
        return value == 0
    if op == "positive":
        return value > EPS
    return value <= EPS


def contrast(row, left, right):
    a, b = row[left], row[right]
    return a["return"] - b["return"], int(a["outcome"] == "win") - int(b["outcome"] == "win")


def summarize(observations):
    # Several v1 targets can be offered to one actor. Each actor-step gets one vote.
    by_scene = defaultdict(lambda: defaultdict(list))
    rescued, lost = Counter(), Counter()
    for row, reward, wins in observations:
        by_scene[row["group_id"]][row["step"], row["actor"]].append((reward, wins))
        rescued[row["seed"]] += int(wins > 0)
        lost[row["seed"]] += int(wins < 0)
    scene_deltas = []
    for anchors in by_scene.values():
        scene_deltas.append([mean(mean(p[i] for p in targets) for targets in anchors.values())
                             for i in (0, 1)])
    returns = [delta[0] for delta in scene_deltas]
    return {
        "bindings": len(observations), "actor_steps": sum(map(len, by_scene.values())),
        "scenes": len(by_scene), "mean_return_gain": mean(returns) if returns else None,
        "mean_win_gain": mean(d[1] for d in scene_deltas) if scene_deltas else None,
        "return_gain_without_one_scene_min": ((math.fsum(returns) - max(returns)) / (len(returns) - 1)
                                              if len(returns) > 1 else None),
        "scene_gain_signs": {"positive": sum(x > EPS for x in returns),
                             "negative": sum(x < -EPS for x in returns),
                             "tie": sum(abs(x) <= EPS for x in returns)},
        "rescued_binding_count": sum(rescued.values()), "lost_binding_count": sum(lost.values()),
        "seeds_with_rescued_binding": sorted(k for k, v in rescued.items() if v),
        "seeds_with_lost_binding": sorted(k for k, v in lost.items() if v),
    }


def changed(rows, family, last_step):
    return [r for r in rows if r["family"] == family and r["step"] <= last_step
            and r["matched"] and not r["same_effect"]]


def evaluate_rule(pairs, rule, last_step):
    active = [r for r in pairs if applies(rule, r, last_step)]
    conditional = summarize([(r, *contrast(r, "new", "old")) for r in active])
    total = summarize([(r, *(contrast(r, "new", "old") if applies(rule, r, last_step) else (0, 0)))
                       for r in pairs])
    inactive_scenes = len({r["group_id"] for r in pairs if not applies(rule, r, last_step)})
    return {"rule": rule, "on_active": conditional, "all_matched_changes": total,
            "inactive_scenes": inactive_scenes}


def eligible(stats, minimum):
    leaf = stats["on_active"]
    return (leaf["scenes"] >= minimum and leaf["return_gain_without_one_scene_min"] > EPS
            and leaf["mean_win_gain"] >= -EPS and stats["all_matched_changes"]["mean_win_gain"] >= -EPS
            and stats["all_matched_changes"]["mean_return_gain"] > EPS
            and (stats["rule"] == "all" or stats["inactive_scenes"] >= minimum))


def fit_one(pairs, rules, spec):
    evaluations = [evaluate_rule(pairs, rule, spec["training_max_step"]) for rule in rules]
    best = "never"
    best_gain = 0.0
    for ev in evaluations:
        if eligible(ev, spec["min_changed_scenes"]):
            gain = ev["all_matched_changes"]["mean_return_gain"]
            if gain > best_gain + EPS:
                best, best_gain = ev["rule"], gain
    return best, evaluations


def cross_validate(pairs, rules, spec, groups):
    predictions, votes = [], Counter()
    for group in sorted(groups):
        rule, _ = fit_one([r for r in pairs if r["group_id"] != group], rules, spec)
        votes[rule] += 1
        for r in pairs:
            if r["group_id"] == group:
                use_new = applies(rule, r, spec["training_max_step"])
                predictions.append((r, *(contrast(r, "new", "old") if use_new else (0, 0))))
    return {"held_one_scene_out": summarize(predictions), "rules_per_fold": dict(votes)}


def branch_report(rows, family, rule, spec):
    last_step = spec["training_max_step"]
    early = [r for r in rows if r["family"] == family and r["step"] <= last_step]
    in_range = [r for r in early if r["old_action"] is not None]
    differing = [r for r in in_range if not r["same_effect"]]
    paired = changed(in_range, family, last_step)
    active = [r for r in paired if applies(rule, r, last_step)]
    requested = [r for r in differing if applies(rule, r, last_step)]

    def partition_by_action(keep):
        comparable = [r for r in active if (r["new"]["effect"] == "BASE") == keep]
        return summarize([(r, *contrast(r, "new", "old")) for r in comparable])

    return {
        "early_registered_bindings_with_v1": len(in_range),
        "early_actor_steps_without_v1": len(early) - len(in_range), "different": len(differing),
        "matched_different": len(paired), "missing_different": len(differing) - len(paired),
        "identical_effect": len(in_range) - len(differing),
        "gate_changed_registered": len(requested), "gate_changed_with_returns": len(active),
        **evaluate_rule(paired, rule, last_step),
        "on_active_vs_adaptive": summarize([(r, *contrast(r, "new", "base")) for r in active]),
        "by_new_action_vs_old": {"adaptive": partition_by_action(True),
                                 "other": partition_by_action(False)},
    }


def code_hashes():
    directory = Path(__file__).parent
    names = ("local_v2_offline_fit.py", "local_v2_offline_pairs.py", "local_v2_offline_selector.py")
    return {name: checksum(directory/name) for name in names}


def run(campaign, out, mode):
    from .local_v2_offline_pairs import load_pairs
    from .local_v2_offline_selector import evaluate_selectors

    protocol_file, freeze_file = out/"partial_protocol.json", out/"candidate_frozen.json"
    spec = read(protocol_file)
    partition = "train" if mode == "fit" else "holdout"
    split = read(campaign/"split.json")
    groups = set(split["training_group_ids" if partition == "train" else "validation_group_ids"])
    stamps = {"partial_protocol": checksum(protocol_file), "scene_split": checksum(campaign/"split.json"),
              "plan": checksum(campaign/"plan"/"summary.json"), "code": code_hashes()}
    if mode == "validate":
        frozen = read(freeze_file)
        require(frozen["stamps"] == stamps and frozen["training_evidence"] == checksum(out/"train.json")
                and set(frozen["training_groups"]) == set(split["training_group_ids"])
                and groups.isdisjoint(frozen["training_groups"]), "Validation input differs from freeze")
        require(not (out/"holdout.json").exists(), "Holdout already evaluated")
    else:
        require(not freeze_file.exists() and not (out/"holdout.json").exists(), "Study already frozen")
    rows, meta = load_pairs(campaign, partition)
    require(meta["labels_accepted"] == (779 if mode == "fit" else 260), "Stopped checkpoint changed")
    require(all(r["group_id"] in groups for r in rows), "Cross-partition row")
    fits, rules, analyses = {}, {}, {}
    for family, gates in spec["rules"].items():
        if mode == "fit":
            pairs = changed(rows, family, spec["training_max_step"])
            rules[family], candidates = fit_one(pairs, gates, spec)
            fits[family] = {"candidate_fits": candidates, **cross_validate(pairs, gates, spec, groups)}
        else:
            rules[family] = frozen["rules"][family]
        analyses[family] = branch_report(rows, family, rules[family], spec)

    def choose_v2(row):
        return applies(rules.get(row["family"], "never"), row, spec["training_max_step"])

    selector_report = evaluate_selectors(campaign, partition, choose_v2, rows, meta)
    report = {"partition": partition, "meta": meta, "rules": rules, "family_results": analyses,
              "selector_replay": selector_report, "stamps": stamps}
    if mode == "fit":
        report["fit_diagnostics"] = fits
        save(out/"train.json", report)
        save(freeze_file, {"kind": "partial_offline_action_version_rules", "stamps": stamps,
                           "training_evidence": checksum(out/"train.json"), "rules": rules,
                           "training_groups": sorted(groups), "frozen_utc": datetime.now(timezone.utc).isoformat()})
    else:
        report["frozen_sha256"] = checksum(freeze_file)
        save(out/"holdout.json", report)
    print(partition, rules)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "validate"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.campaign, args.output, args.mode)
