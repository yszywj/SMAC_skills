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
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tree_skills.data.labeling import pair_labels  # noqa: E402
from tree_skills.teacher.action_mapping import SmacActionMap  # noqa: E402

NATIVE_N_ACTIONS = 16


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
             "distribution": {"0": 0, "1": 0, "2+": 0}}
    for row in rows:
        stats["steps"] += 1
        healers = row.get("healers") or {}
        if healers:
            stats["steps_with_healers"] += 1
        stats["distribution"]["0" if not healers else
                              ("1" if len(healers) == 1 else "2+")] += 1
        diag = row.get("pair_scoring") or {}
        stats["recommended_changes"] += len(diag.get("changed") or {})
        for healer, entry in healers.items():
            stats["groups"] += 1
            n = len(entry.get("candidates") or {})
            stats["scored_groups"] += 1
            if n > 1:
                stats["multi_candidate_groups"] += 1
            elif n == 1:
                stats["single_candidate_groups"] += 1
    return stats


def build_shard(rows, mapping, shard_path):
    written = 0
    skipped = {"no_teacher_logits": 0, "label_error": 0, "no_candidates": 0}
    with open(shard_path, "w", encoding="utf-8") as out:
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
                    labels, diag = pair_labels(
                        {healer: {str(slot_of[t]): logits_by_agent[agent][slot_of[t]]
                                  for t in candidates}}, slot_of)
                except (KeyError, ValueError):
                    skipped["label_error"] += 1
                    continue
                full = logits_by_agent[agent]
                native_argmax = max(range(NATIVE_N_ACTIONS), key=lambda i: full[i])
                out.write(json.dumps({
                    "episode_id": row["episode"], "game_loop": row["game_loop"],
                    "healer_id": healer, "labels": labels,
                    "features": {t: candidates[t]["features"] for t in candidates},
                    "diagnostics": {
                        "group_mean": diag["group_mean"],
                        "teacher_native_argmax": native_argmax,
                        "heal_slots": {t: int(s) for t, s in slot_of.items()},
                        "source": row.get("_source"),
                        "original_assignments":
                            (row.get("pair_scoring") or {}).get("proposed_assignments"),
                        "final_actions": row.get("final_actions")},
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
