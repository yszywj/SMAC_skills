#!/usr/bin/env python3
"""Export a trained zerg scorer model as a deployable JSON package.

Binds a model record to the CURRENT execution context (resolved source
fingerprint + tactical config + runtime policy contract + feature schema)
and writes a single self-describing JSON file that the runtime can load.
Loading is refused on any context mismatch, so re-export after any source
or config change.

Usage (from the repository root):
    python decision_tree/scripts/linux/export_zerg_model.py \
        --model runs/<run>/models/joint_scorer.json \
        --scorer joint_scorer \
        --out decision_tree/zerg/models/joint_scorer.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision_tree.zerg.features import DEFAULT_SCHEMA
from decision_tree.zerg.fingerprints import execution_context, \
    execution_context_hash, source_hash
from decision_tree.zerg.runtime import DEFAULT_SCORER_SLOTS, \
    runtime_policy_for_scorers
from decision_tree.zerg.schema import MODEL_KIND, digest, dumps, load_json, \
    validate_scorer_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="trained zerg_scorer_model JSON (or '-' for a "
                             "training run output)")
    parser.add_argument("--scorer", required=True, choices=DEFAULT_SCORER_SLOTS)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workspace", default=str(ROOT))
    parser.add_argument("--base-dir", default="skills/tactical_v1")
    parser.add_argument("--overlay-dir",
                        default=("runs/contact_all_attack_20261122_230107_"
                                 "662982/marginal_near/skills/tactical_v1"))
    parser.add_argument("--tactical-config", default=None,
                        help="effective tactical_v1 config JSON the model "
                             "was collected under")
    parser.add_argument("--mode", default="shadow",
                        choices=("shadow", "tree_score"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model = load_json(args.model)
    model["scorer"] = args.scorer
    model["kind"] = MODEL_KIND
    if model.get("schema_version") != 1:
        model["schema_version"] = 1

    tactical_config = (load_json(args.tactical_config)
                       if args.tactical_config else {})
    policy = runtime_policy_for_scorers(args.mode, [args.scorer])
    context = execution_context(
        source_hash(args.workspace, args.base_dir, args.overlay_dir),
        tactical_config, policy, DEFAULT_SCHEMA)
    model["execution_context_hash"] = execution_context_hash(context)
    model["context"] = context
    model["feature_schema"] = {"name": DEFAULT_SCHEMA.name,
                               "version": DEFAULT_SCHEMA.version,
                               "digest": DEFAULT_SCHEMA.digest()}
    validate_scorer_model(model)

    fingerprint = {
        "model_sha256": digest(model),
        "scorer": args.scorer,
        "execution_context_hash": model["execution_context_hash"],
        "source_fingerprint": context["source_fingerprint"],
    }
    print(json.dumps(fingerprint, indent=2))
    if args.dry_run:
        return 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps(model), encoding="utf-8")
    print(f"deployable model written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
