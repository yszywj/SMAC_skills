"""Offline analysis + training-shard construction from real collections.

Reads collect-*.jsonl (tree_skills/data/collect_real.py), reports the
three-tier intervention metrics (P0 §1.5) and emits a training shard where
each row is one sample group (episode, game_loop, healer):

  labels:   y(H,U) = z_T(heal(U)) - mean_V z_T(heal(V)) over the group's
            ORIGINAL-rule-legal candidates (finite native heal slots only;
            keep extension and padding slots never enter)
  features: the recorded 33-dim rows
  diagnostics: full raw logits/mask kept on the collect file; the shard
            keeps the group's heal-slot map and teacher argmax-overall for
            "does the teacher even prefer healing" diagnostics.

Usage:
  python -m tree_skills.data.build_shard --collect <dir|file> --shard <out.jsonl>
"""

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tree_skills.data.labeling import pair_labels  # noqa: E402
from tree_skills.teacher.action_mapping import SmacActionMap  # noqa: E402

def original_assignments(diag):
    """Recover legacy before-scoring assignments without confusing proposals.

    Older collections stored proposals plus every changed edge. Together
    those contain the exact original matching; missing proposals stay unknown.
    """
    if diag.get("original_assignments") is not None:
        return dict(diag["original_assignments"])
    proposed = diag.get("proposed_assignments")
    if proposed is None:
        return None
    original = dict(proposed)
    for healer, change in (diag.get("changed") or {}).items():
        original[healer] = change["from"]
    return original


def native_teacher_diagnostics(row, healer, mapping):
    """Use ONLY legal native actions, never keep or padded model-head slots."""
    agent = int(healer[1:])
    teacher = row.get("teacher_raw") or {}
    entries = {int(e["agent"]): e for e in teacher.get("agents", [])}
    avail = entries[agent]["avail"]
    n = mapping.native_n_actions
    if len(avail) not in (n, n + 1) or any(v not in (0, 1) for v in avail):
        raise ValueError("invalid native availability mask")
    logits = teacher["logits"].get(str(agent), teacher["logits"].get(agent))
    if logits is None or len(logits) < n:
        raise ValueError("missing native teacher logits")
    legal = [i for i in range(n) if avail[i]]
    if not legal or any(not math.isfinite(logits[i]) for i in legal):
        raise ValueError("empty legal action set or non-finite legal logits")
    best = max(legal, key=lambda i: logits[i])
    weights = {i: math.exp(logits[i] - logits[best]) for i in legal}
    total = sum(weights.values())
    return {"teacher_native_argmax": best,
            "teacher_stop_probability": weights.get(1, 0.0) / total,
            "teacher_legal_heal_probability": sum(v for i, v in weights.items() if i >= 6) / total,
            "teacher_native_legal_slots": legal}


def iter_rows(target):
    files = ([target] if os.path.isfile(target)
             else sorted(glob.glob(os.path.join(target, "collect-*.jsonl"))))
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row["_source"] = os.path.basename(path)
                    yield row


def analyze(rows, mapping):
    stats = {"steps": 0, "steps_with_healers": 0, "groups": 0,
             "multi_candidate_groups": 0, "single_candidate_groups": 0,
             "recommended_changes": 0, "scored_groups": 0,
             "recommendations_submitted": 0, "recommendations_kept": 0,
             "recommendations_other_action": 0, "recommendations_action_unknown": 0,
             "teacher_masked_argmax": {}, "teacher_diagnostic_errors": 0,
             "distribution": {"0": 0, "1": 0, "2+": 0}}
    episodes, multi_episodes = set(), set()
    for row in rows:
        stats["steps"] += 1
        healers = row.get("healers") or {}
        if healers:
            stats["steps_with_healers"] += 1
        stats["distribution"]["0" if not healers else
                              ("1" if len(healers) == 1 else "2+")] += 1
        diag = row.get("pair_scoring") or {}
        stats["recommended_changes"] += len(diag.get("changed") or {})
        for healer, change in (diag.get("changed") or {}).items():
            actions = row.get("final_actions") or []
            index = int(healer[1:])
            if index >= len(actions):
                stats["recommendations_action_unknown"] += 1
            elif actions[index] == mapping.heal_slot(int(change["to"][1:])):
                stats["recommendations_submitted"] += 1
            elif actions[index] == mapping.keep_action_id():
                stats["recommendations_kept"] += 1
            else:
                stats["recommendations_other_action"] += 1
        for healer, entry in healers.items():
            episodes.add(row["episode"])
            stats["groups"] += 1
            n = len(entry.get("candidates") or {})
            stats["scored_groups"] += 1
            if n > 1:
                stats["multi_candidate_groups"] += 1
                multi_episodes.add(row["episode"])
            elif n == 1:
                stats["single_candidate_groups"] += 1
            try:
                action = str(native_teacher_diagnostics(row, healer, mapping)["teacher_native_argmax"])
                stats["teacher_masked_argmax"][action] = stats["teacher_masked_argmax"].get(action, 0) + 1
            except (KeyError, ValueError, TypeError):
                stats["teacher_diagnostic_errors"] += 1
    stats["episodes_with_groups"] = len(episodes)
    stats["episodes_with_multi_candidate_groups"] = len(multi_episodes)
    stats["recommendation_metrics_scope"] = "recorded trajectory policy, not counterfactual relabelled policy"
    return stats


def build_shard(rows, mapping, shard_path):
    written = 0
    skipped = {"no_teacher_logits": 0, "label_error": 0, "no_candidates": 0,
               "invalid_mask_or_illegal_candidate": 0}
    with open(shard_path, "x", encoding="utf-8") as out:
        for row in rows:
            teacher = row.get("teacher_raw") or {}
            logits_by_agent = {int(a): v for a, v in (teacher.get("logits") or {}).items()}
            for healer, entry in (row.get("healers") or {}).items():
                agent = int(healer[1:])
                candidates = entry.get("candidates") or {}
                if not candidates:
                    skipped["no_candidates"] += 1
                    continue
                if agent not in logits_by_agent:
                    skipped["no_teacher_logits"] += 1
                    continue
                slot_of = {}
                for target in candidates:
                    ally = int(target[1:])
                    slot = mapping.heal_slot(ally)  # raises on out-of-range
                    mapping.validate_teacher_slots([slot])
                    slot_of[target] = slot
                try:
                    native = native_teacher_diagnostics(row, healer, mapping)
                    if any(s not in native["teacher_native_legal_slots"] for s in slot_of.values()):
                        raise ValueError("candidate is not a legal native action")
                except (KeyError, ValueError, TypeError):
                    skipped["invalid_mask_or_illegal_candidate"] += 1
                    continue
                try:
                    labels, diag = pair_labels(
                        {healer: {str(slot_of[t]): logits_by_agent[agent][slot_of[t]]
                                  for t in candidates}}, slot_of)
                except (KeyError, ValueError):
                    skipped["label_error"] += 1
                    continue
                pair_diag = row.get("pair_scoring") or {}
                out.write(json.dumps({
                    "episode_id": row["episode"], "game_loop": row["game_loop"],
                    "healer_id": healer, "labels": labels,
                    "features": {t: candidates[t]["features"] for t in candidates},
                    "diagnostics": {
                        "group_mean": diag["group_mean"],
                        **native,
                        "heal_slots": {t: int(s) for t, s in slot_of.items()},
                        "source": row.get("_source"),
                        "original_assignments": original_assignments(pair_diag),
                        "proposed_assignments": pair_diag.get("proposed_assignments"),
                        "final_actions": row.get("final_actions"),
                        "keep_decisions": (row.get("healing") or {}).get("keep_decisions"),
                        "label_provenance": row.get("label_provenance"),
                        "trajectory_pair_diagnostics_are_counterfactual": False},
                }) + "\n")
                written += 1
    return written, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", required=True)
    parser.add_argument("--shard", required=True)
    args = parser.parse_args()
    if os.path.exists(args.shard):
        raise SystemExit(f"refusing to overwrite {args.shard}")
    rows = list(iter_rows(args.collect))
    mapping = SmacActionMap()
    stats = analyze(rows, mapping)
    written, skipped = build_shard(rows, mapping, args.shard)
    report = {"stats": stats, "shard_rows": written, "skipped": skipped,
              "shard": args.shard}
    print(json.dumps(report, indent=1))
    with open(args.shard + ".report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
