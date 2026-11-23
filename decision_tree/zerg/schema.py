"""Zerg data-record and model-file contracts.

All artefacts are canonical JSON (sorted keys, no NaN) — the same
serialisation discipline as the rest of decision_tree. Three record kinds:

- decision record (JSONL, one per decision point, candidates nested);
- scorer model (one JSON file per scorer slot, embedded trees);
- training config (capacity + task declaration).

``episode_id`` / ``scene_id`` / actor ids are association keys only: they
appear in records for joining, and are rejected if ever offered as model
features (see FEATURE_FORBIDDEN).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .features import DEFAULT_SCHEMA, ZergFeatureSchema

DECISION_RECORD_SCHEMA = "zerg_decision_record_v1"
SCORER_MODEL_SCHEMA = "zerg_scorer_model_v1"
MODEL_KIND = "zerg_scorer_model"
TARGETS = ("imitation", "gain", "correction", "ranking")
EXECUTION_STATUSES = ("pending", "committed", "arbitrated_changed",
                      "skipped", "unknown")
OUTCOMES = ("win", "loss", "draw", "timeout", "unknown")

FEATURE_FORBIDDEN = frozenset({
    "episode_id", "episode", "scene_id", "seed", "sc2_seed", "unit_tag",
    "actor", "actor_id", "tag", "snapshot_id", "group_id", "game_version"})

IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,128}")


def digest(value: Any) -> str:
    """Canonical JSON SHA256 (matches decision_tree.experiment.digest)."""
    from decision_tree.experiment import digest as framework_digest
    return framework_digest(value)


def _sanitize(value: Any) -> Any:
    """NaN/inf never enter JSON: missing stays explicit null on disk."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


def dumps(record: Dict[str, Any]) -> str:
    return json.dumps(_sanitize(record), sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False) + "\n"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid {name}")
    return value


# -- decision records -------------------------------------------------------

def validate_decision_record(record: Dict[str, Any],
                             schema: ZergFeatureSchema = DEFAULT_SCHEMA
                             ) -> Dict[str, Any]:
    if record.get("schema") != DECISION_RECORD_SCHEMA:
        raise ValueError("decision record requires "
                         f"{DECISION_RECORD_SCHEMA}")
    for field in ("episode_id", "scene_id", "source_hash",
                  "execution_context_hash"):
        _identifier(record.get(field), field)
    if type(record.get("game_loop")) is not int:
        raise ValueError("game_loop must be an integer loop count")
    if type(record.get("snapshot_id")) is not int:
        raise ValueError("snapshot_id must be an integer")
    if record.get("feature_schema_version") != schema.version:
        raise ValueError("feature schema version mismatch")

    for layer in ("team", "skill"):
        names = set(schema.names(layer=layer))
        values = record.get("global_features", {}).get(layer, {})
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError(f"unknown {layer} features: {unknown}")
    units = record.get("global_features", {}).get("units", {})
    allowed_unit = set(schema.names(layer="unit"))
    for actor, values in units.items():
        unknown = sorted(set(values) - allowed_unit)
        if unknown:
            raise ValueError(f"unknown unit features for {actor}: {unknown}")

    candidates = record.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("at least one candidate is required")
    candidate_names = set(schema.names(layer="candidate"))
    teacher = record.get("teacher_choice")
    seen_ids = set()
    for candidate in candidates:
        candidate_id = _identifier(candidate.get("candidate_id"),
                                   "candidate_id")
        if candidate_id in seen_ids:
            raise ValueError("duplicate candidate_id at one decision point")
        seen_ids.add(candidate_id)
        unknown = sorted(set(candidate.get("features", {})) - candidate_names)
        if unknown:
            raise ValueError(f"unknown candidate features: {unknown}")
        if not isinstance(candidate.get("legal"), bool):
            raise ValueError("candidate.legal must be boolean")
    if teacher is not None and teacher not in seen_ids:
        raise ValueError("teacher_choice must reference a listed candidate")
    selected = record.get("selected_candidate")
    if selected is not None and selected not in seen_ids:
        raise ValueError("selected_candidate must reference a listed candidate")
    if record.get("execution_status") not in EXECUTION_STATUSES:
        raise ValueError("unknown execution_status")
    outcome = record.get("outcome")
    if outcome is not None:
        if outcome not in OUTCOMES:
            raise ValueError("unknown outcome")
    if record.get("return") is not None:
        _finite(record["return"], "return")
    provenance = record.get("reward_provenance")
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError("reward_provenance must be an object when present")
    return record


# -- scorer models ----------------------------------------------------------

def scorer_feature_names(schema: ZergFeatureSchema = DEFAULT_SCHEMA,
                         shared_layers: Sequence[str] = ("team", "skill")
                         ) -> List[str]:
    """Flat feature list for scorer rows: candidate-layer features plus the
    declared shared decision-point layers, prefixed by layer."""
    names: List[str] = []
    for layer in shared_layers:
        names.extend(f"{layer}.{name}" for name in schema.names(layer=layer))
    names.extend(f"candidate.{name}"
                 for name in schema.names(layer="candidate"))
    return names


def flatten_row_features(global_features: Dict[str, Any],
                         candidate_features: Dict[str, Any],
                         schema: ZergFeatureSchema = DEFAULT_SCHEMA,
                         shared_layers: Sequence[str] = ("team", "skill")
                         ) -> Dict[str, float]:
    import math as _math

    def as_float(value: Any) -> float:
        try:
            resolved = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return resolved if _math.isfinite(resolved) else float("nan")

    flat: Dict[str, float] = {}
    for layer in shared_layers:
        for name, value in (global_features.get(layer, {}) or {}).items():
            flat[f"{layer}.{name}"] = as_float(value)
    for name, value in (candidate_features or {}).items():
        flat[f"candidate.{name}"] = as_float(value)
    return flat


def validate_tree_node(node: Any, feature_names: Sequence[str],
                       max_depth: int, depth: int = 0) -> None:
    if not isinstance(node, dict):
        raise ValueError("tree node must be an object")
    if set(node) <= {"value", "groups", "samples", "supported"}:
        _finite(node["value"], "leaf value")
        if type(node.get("groups")) is not int or node["groups"] < 1:
            raise ValueError("leaf groups must be a positive integer")
        if type(node.get("samples")) is not int or node["samples"] < node["groups"]:
            raise ValueError("leaf samples must be >= groups")
        return
    if not {"feature", "threshold", "left", "right"} <= set(node):
        raise ValueError("invalid tree node")
    if depth >= max_depth:
        raise ValueError("tree exceeds declared max_depth")
    if node["feature"] not in feature_names:
        raise ValueError(f"tree uses undeclared feature {node['feature']!r}")
    _finite(node["threshold"], "threshold")
    validate_tree_node(node["left"], feature_names, max_depth, depth + 1)
    validate_tree_node(node["right"], feature_names, max_depth, depth + 1)


def validate_scorer_model(model: Dict[str, Any],
                          schema: ZergFeatureSchema = DEFAULT_SCHEMA
                          ) -> Dict[str, Any]:
    if model.get("schema_version") != 1 or model.get("kind") != MODEL_KIND:
        raise ValueError(f"model requires kind={MODEL_KIND} version 1")
    if model.get("scorer") not in ("zergling_scorer", "hydralisk_scorer",
                                   "baneling_scorer", "joint_scorer"):
        raise ValueError("unknown scorer slot")
    if model.get("target") not in TARGETS:
        raise ValueError("unknown model target")
    feature_names = model.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("feature_names required")
    declared = set(scorer_feature_names(schema))
    unknown = sorted(set(feature_names) - declared)
    if unknown:
        raise ValueError(f"features outside the zerg schema: {unknown}")
    forbidden = sorted(set(feature_names) & FEATURE_FORBIDDEN)
    if forbidden:
        raise ValueError(f"association keys cannot be features: {forbidden}")
    embedded = model.get("feature_schema", {})
    if embedded.get("digest") != schema.digest() \
            or embedded.get("version") != schema.version:
        raise ValueError("model feature schema mismatch — refusing to load")
    _identifier(model.get("execution_context_hash"), "execution_context_hash")

    capacity = model.get("capacity", {})
    estimator = capacity.get("estimator_type", "single_threshold_tree")
    if estimator not in ("single_threshold_tree", "bagged_threshold_tree",
                         "boosted_threshold_tree"):
        raise ValueError("unknown estimator_type")
    n_trees = capacity.get("number_of_trees", 1)
    if type(n_trees) is not int or n_trees < 1 or n_trees > 4096:
        raise ValueError("number_of_trees out of range")
    max_depth = capacity.get("max_depth", 2)
    if type(max_depth) is not int or not 0 <= max_depth <= 16:
        raise ValueError("max_depth out of range")
    if type(capacity.get("minimum_leaf_support", 4)) is not int \
            or capacity.get("minimum_leaf_support", 4) < 1:
        raise ValueError("minimum_leaf_support must be a positive integer")
    if type(capacity.get("minimum_leaf_groups", 4)) is not int \
            or capacity.get("minimum_leaf_groups", 4) < 1:
        raise ValueError("minimum_leaf_groups must be a positive integer")
    learning_rate = capacity.get("learning_rate", 1.0)
    _finite(learning_rate, "learning_rate")
    if not 0 < learning_rate <= 1.0:
        raise ValueError("learning_rate must be in (0, 1]")
    if type(capacity.get("random_seed", 0)) is not int:
        raise ValueError("random_seed must be an integer")

    tree_weight_total = 0.0
    trees = model.get("trees", [])
    families = model.get("families", {})
    if families:
        if not isinstance(families, dict) or not families:
            raise ValueError("families must be a non-empty object")
        for family, bundle in families.items():
            _identifier(family, "family")
            _validate_tree_bundle(bundle, feature_names, max_depth, n_trees)
        if trees:
            raise ValueError("declare either trees or families, not both")
    else:
        _validate_tree_bundle({"trees": trees}, feature_names, max_depth,
                              n_trees)
    if model.get("target") == "correction":
        old_hash = model.get("old_model_hash")
        if not isinstance(old_hash, str) or not re.fullmatch(
                r"[0-9a-f]{64}", old_hash):
            raise ValueError("correction model requires frozen old_model_hash")
    _finite(model.get("min_gain", 0.0), "min_gain")
    _finite(model.get("tie_tolerance", 1e-9), "tie_tolerance")
    return model


def _validate_tree_bundle(bundle: Dict[str, Any], feature_names,
                          max_depth: int, n_trees: int) -> None:
    trees = bundle.get("trees")
    if not isinstance(trees, list) or not trees:
        raise ValueError("tree bundle requires trees")
    if len(trees) > n_trees:
        raise ValueError("more trees than declared capacity")
    total_weight = 0.0
    for tree in trees:
        _finite(tree.get("weight", 1.0), "tree weight")
        total_weight += float(tree.get("weight", 1.0))
        validate_tree_node(tree.get("root"), feature_names, max_depth)
    if total_weight <= 0:
        raise ValueError("tree weights must sum positive")


def validate_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if config.get("kind") != "zerg_training_config":
        raise ValueError("training config requires kind=zerg_training_config")
    if config.get("task") not in TARGETS:
        raise ValueError("unknown training task")
    capacity = config.get("capacity", {})
    validate_scorer_model({
        "schema_version": 1, "kind": MODEL_KIND,
        "scorer": config.get("scorer", "hydralisk_scorer"),
        "target": config["task"], "feature_names": scorer_feature_names(),
        "feature_schema": {"digest": DEFAULT_SCHEMA.digest(),
                           "version": DEFAULT_SCHEMA.version},
        "execution_context_hash": "0" * 64,
        "capacity": {**capacity, "number_of_trees":
                     min(int(capacity.get("number_of_trees", 1)), 4096)},
    })
    return config


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))
