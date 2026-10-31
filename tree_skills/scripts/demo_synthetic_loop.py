"""Synthetic end-to-end smoke of the distillation pipeline.

PROOF OF MACHINERY ONLY: labels here come from a synthetic stand-in, not
the real MARL-GPT teacher. Passing this script is NEVER evidence of
distillation quality or win-rate; it only proves the closed loop
features -> labels -> group-weighted training -> export -> tree-score
injection runs. Requires the TRAINING python (scikit-learn).

Usage: python tree_skills/scripts/demo_synthetic_loop.py --output-dir tree_skills/artifacts/demo
"""

import argparse
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, r"C:\Users\ry\skill\SMAC_agent")

from tree_skills.data.labeling import pair_labels, write_sample_group
from tree_skills.export.exporter import export_gbr
from tree_skills.medivac.features import FEATURE_NAMES, SCHEMA_VERSION, FeatureSchema, build_pair_features
from tree_skills.training.train_tree import fit, evaluate_ranking, split_by_episode


def synthetic_group(rng, episode, loop, healer="u4"):
    """One sample group: 2-4 candidates, features from a random snapshot,
    labels from a synthetic 'teacher' logit function of feature[0]."""
    n = rng.randrange(2, 5)
    context = {"original_edges": {healer: [(0.0, f"u{i}") for i in range(n)]}}
    # minimal state shape used by build_pair_features
    context["state"] = {"allies": {
        healer: {"health": 150, "max_health": 150, "type": "medivac", "energy": 100,
                 "alive": True, "orders": []},
        **{f"u{i}": {"health": 45 - rng.randrange(20), "max_health": 45,
                     "type": "marine", "alive": True} for i in range(n)}},
        "relations": {}}
    context["urgency"] = {f"u{i}": rng.random() for i in range(n)}
    context["target_pool"] = {f"u{i}": {healer: {"distance": rng.uniform(1, 5)}}
                              for i in range(n)}
    feats = {t: build_pair_features(healer, t, context) for t in context["urgency"]}
    logits = {healer: {str(6 + i): 2.0 * feats[f"u{i}"][0] + 0.5 * rng.gauss(0, 1)
                       for i in range(n)}}
    slot_of = {f"u{i}": str(6 + i) for i in range(n)}
    labels, diag = pair_labels(logits, slot_of)
    return {"episode_id": episode, "game_loop": loop, "healer_id": healer,
            "labels": labels, "features": feats, "diagnostics": diag}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = args.output_dir
    if os.path.isdir(out) and os.listdir(out):
        raise SystemExit(f"refusing to overwrite non-empty {out}")
    os.makedirs(out, exist_ok=True)
    rng = random.Random(7)

    shard = os.path.join(out, "synthetic_shard.jsonl")
    valid = []
    with open(shard, "w", encoding="utf-8") as fh:  # truncate then append via helper
        pass
    groups = []
    for episode in range(30):
        if episode % 3 == 0:
            valid.append(episode)
        for loop in range(4):
            groups.append(synthetic_group(rng, f"ep{episode}", loop))
    for g in groups:
        write_sample_group(shard, g["episode_id"], g["game_loop"], g["healer_id"],
                           g["labels"], g["diagnostics"], g["features"])

    train, valid_groups = split_by_episode(groups, {f"ep{e}" for e in valid})
    model = fit(train)
    metrics = evaluate_ranking(model, valid_groups)
    export_gbr(model, FEATURE_NAMES, SCHEMA_VERSION, os.path.join(out, "model.json"),
               meta_extra={"synthetic_smoke": True,
                           "disclaimer": "labels from synthetic stand-in, NOT the real teacher"})
    print(json.dumps({"valid_metrics": metrics}, indent=1))

    # part B: the exported model must run the tree-score channel in the
    # SMACv2 adapter venv (stdlib-only rule evaluation).
    check = subprocess.run(
        [r"C:\Users\ry\skill\SMAC_agent\sc2_skill_adapter\.venv\Scripts\python.exe",
         os.path.join(os.path.dirname(__file__), "demo_channel_check.py"),
         "--model-dir", out],
        capture_output=True, text=True)
    print(check.stdout, check.stderr)
    if check.returncode != 0:
        raise SystemExit("channel check failed")
    print("SYNTHETIC LOOP OK (machinery proof only, not evidence of distillation)")


if __name__ == "__main__":
    main()
