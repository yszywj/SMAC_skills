#!/usr/bin/env python3
"""Validate zerg runtime / training configs and scorer model files.

Linux entry for the schema contracts in decision_tree/zerg/schema.py.
Checks:

- runtime config structure, mode, scorer slots and (when enabled and present)
  that each model file matches its slot, its feature schema and — when the
  runtime context is available — its execution-context hash;
- model files (kind zerg_scorer_model_v1): tree depth, feature allowlist,
  capacity declaration;
- training configs (kind zerg_training_config).

Usage (from the repository root):
    python decision_tree/scripts/linux/validate_zerg_config.py
    python decision_tree/scripts/linux/validate_zerg_config.py \
        --config decision_tree/zerg/configs/runtime.json \
        --models decision_tree/zerg/models
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
from decision_tree.zerg.runtime import validate_runtime_config
from decision_tree.zerg.schema import MODEL_KIND, load_json, \
    validate_scorer_model, validate_training_config


def validate_models_dir(models_dir: Path, schema_digest: str) -> list:
    problems = []
    if not models_dir.is_dir():
        return [f"models dir missing: {models_dir}"]
    for path in sorted(models_dir.glob("*.json")):
        try:
            record = load_json(str(path))
        except Exception as exc:
            problems.append(f"{path.name}: unreadable JSON ({exc})")
            continue
        if record.get("kind") != MODEL_KIND:
            problems.append(f"{path.name}: kind is "
                            f"{record.get('kind')!r}, expected {MODEL_KIND!r}")
            continue
        try:
            validate_scorer_model(record)
            embedded = record.get("feature_schema", {}).get("digest")
            if embedded != schema_digest:
                problems.append(f"{path.name}: feature schema digest mismatch")
        except ValueError as exc:
            problems.append(f"{path.name}: {exc}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",
                        default="decision_tree/zerg/configs/runtime.json")
    parser.add_argument("--models", default="decision_tree/zerg/models")
    parser.add_argument("--training-config", default=None,
                        help="additional zerg_training_config JSON to check")
    parser.add_argument("--workspace", default=str(ROOT))
    args = parser.parse_args()

    workspace = Path(args.workspace)
    problems: list = []
    warnings: list = []

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = workspace / config_path
    if not config_path.is_file():
        problems.append(f"runtime config missing: {config_path}")
    else:
        config = load_json(str(config_path))
        try:
            validate_runtime_config(config)
            for name, slot in config.get("scorers", {}).items():
                if slot.get("model_path") and not slot.get("enabled", False):
                    warnings.append(
                        f"{name}: model_path set but slot disabled")
                if slot.get("enabled", False) and not slot.get("model_path"):
                    warnings.append(
                        f"{name}: enabled without model_path; "
                        "the slot runs as a pure rule fallback")
        except ValueError as exc:
            problems.append(f"runtime config: {exc}")
        models_dir = Path(args.models)
        if not models_dir.is_absolute():
            models_dir = workspace / models_dir
        problems.extend(validate_models_dir(models_dir,
                                            DEFAULT_SCHEMA.digest()))

    if args.training_config:
        tc_path = Path(args.training_config)
        if not tc_path.is_absolute():
            tc_path = workspace / tc_path
        try:
            validate_training_config(load_json(str(tc_path)))
        except Exception as exc:
            problems.append(f"training config {tc_path.name}: {exc}")

    status = "ok" if not problems else "problems"
    print(json.dumps({"status": status, "problems": problems,
                      "warnings": warnings}, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
