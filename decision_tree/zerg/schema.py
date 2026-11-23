"""Zerg data-record and model-file contracts (V1.1).

All artefacts are canonical JSON (sorted keys, NaN -> null on disk). The
single feature-join function :func:`build_model_features` is the ONLY place
where a model-row feature vector is assembled; offline training
(``training.py``/``dataset.py``) and online scoring (``scorer.py``) must both
call it, so names, order, values and missing states are identical on both
sides.

Identity keys (association only, never model features): ``run_id``,
``episode_id``, ``scene_config_id``, ``seed``, ``map_id``, ``snapshot_id``.
Counterfactual pair records (``zerg_counterfactual_pair_v1``) carry their own
audit contract: gain must equal candidate_return - baseline_return, missing
returns stay None, and incomplete pairs never enter gain training.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .features import DEFAULT_SCHEMA, ZergFeatureSchema

DECISION_RECORD_SCHEMA = "zerg_decision_record_v1"
PAIR_RECORD_SCHEMA = "zerg_counterfactual_pair_v1"
SCORER_MODEL_SCHEMA = "zerg_scorer_model_v1"
MODEL_KIND = "zerg_scorer_model"
TARGETS = ("imitation", "gain", "correction", "ranking")
EXECUTION_STATUSES = ("pending", "committed", "arbitrated_changed",
                      "skipped", "unknown")
OUTCOMES = ("win", "loss", "draw", "timeout", "unknown")
INTERVENTION_TYPES = ("single_step", "multi_step_continuous", "unknown")
COMPLETION_STATUSES = ("complete", "prefix_mismatch", "context_mismatch",
                       "candidate_not_executed", "missing_return",
                       "rejected", "unknown")

FEATURE_FORBIDDEN = frozenset({
    "episode_id", "episode", "run_id", "scene_id", "scene_config_id",
    "seed", "sc2_seed", "map_id", "unit_tag", "actor", "actor_id", "tag",
    "snapshot_id", "group_id", "game_version", "candidate_id",
    "proposal_id", "instance_id", "task_key"})

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


def as_feature_float(value: Any) -> float:
    """Value coercion for the join function: non-numeric / non-finite /
    absent -> NaN (missing). Real 0.0 stays 0.0."""
    if value is None or isinstance(value, bool):
        return float("nan")
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return resolved if math.isfinite(resolved) else float("nan")


# -- Task 1.1/1.2: THE single feature join ----------------------------------

class FeatureLayerMissing(ValueError):
    """A whole required feature layer was not passed (distinct from a
    single missing feature, which is imputable)."""

    def __init__(self, layer: str):
        super().__init__(f"feature layer '{layer}' missing entirely")
        self.layer = layer


def build_model_features(global_features: Dict[str, Any],
                         candidate_features: Dict[str, Any],
                         schema: ZergFeatureSchema = DEFAULT_SCHEMA,
                         shared_layers: Sequence[str] = ("team", "skill")
                         ) -> Dict[str, float]:
    """THE single feature-join function (training AND inference).

    Returns a fixed-ORDER dict: ``team.<name>`` then ``skill.<name>`` then
    ``candidate.<name>``, each layer iterated in schema declaration order.
    A single absent value becomes NaN (imputable downstream); a layer that
    was never passed raises :class:`FeatureLayerMissing` — imputation must
    never mask a plumbing error.
    """
    global_features = global_features or {}
    ordered: Dict[str, float] = {}
    for layer in shared_layers:
        layer_values = global_features.get(layer)
        if layer_values is None:
            raise FeatureLayerMissing(layer)
        for name in schema.names(layer=layer):
            ordered[f"{layer}.{name}"] = as_feature_float(
                layer_values.get(name))
    candidate_features = candidate_features or {}
    for name in schema.names(layer="candidate"):
        ordered[f"candidate.{name}"] = as_feature_float(
            candidate_features.get(name))
    return ordered


def missing_report(flat_features: Dict[str, float],
                   feature_names: Sequence[str]) -> Dict[str, Any]:
    """Which model features are missing (NaN) and the ratio."""
    missing = [name for name in feature_names
               if math.isnan(flat_features.get(name, float("nan")))]
    return {
        "missing_features": missing,
        "missing_count": len(missing),
        "missing_ratio": (len(missing) / len(feature_names))
        if feature_names else 0.0,
    }


def scorer_feature_names(schema: ZergFeatureSchema = DEFAULT_SCHEMA,
                         shared_layers: Sequence[str] = ("team", "skill")
                         ) -> List[str]:
    """The exact, ordered feature list every scorer model must declare —
    identical to :func:`build_model_features` key order."""
    return list(build_model_features(
        {layer: {} for layer in shared_layers}, {}, schema,
        shared_layers).keys())


def flatten_row_features(global_features: Dict[str, Any],
                         candidate_features: Dict[str, Any],
                         schema: ZergFeatureSchema = DEFAULT_SCHEMA,
                         shared_layers: Sequence[str] = ("team", "skill")
                         ) -> Dict[str, float]:
    """Deprecated alias; delegates to the single join."""
    return build_model_features(global_features, candidate_features, schema,
                                shared_layers)


# -- Task 3.4: internal (unit-level) scorer feature layout ------------------

UNIT_AGGREGATIONS = ("mean", "min", "max")


def build_internal_model_features(global_features: Dict[str, Any],
                                  candidate_features: Dict[str, Any],
                                  controlled_actors: Sequence[str],
                                  schema: ZergFeatureSchema = DEFAULT_SCHEMA,
                                  shared_layers: Sequence[str] = ("team",
                                                                  "skill")
                                  ) -> Dict[str, float]:
    """Feature join for skill-INTERNAL scorers: team + skill + candidate
    layers plus AGGREGATED unit features over the candidate's controlled
    actors (mean/min/max + count + type fractions). Unit tags never enter
    the vector; aggregation order is schema-fixed, so a missing single
    value never shifts other features' positions."""
    ordered = build_model_features(global_features, candidate_features,
                                   schema, shared_layers)
    units = (global_features or {}).get("units")
    if units is None:
        raise FeatureLayerMissing("units")
    unit_types = (global_features or {}).get("unit_types", {}) or {}
    actors = [actor for actor in (controlled_actors or ())
              if actor in units]

    def unit_values(name: str) -> List[float]:
        values = []
        for actor in actors:
            value = units[actor].get(name)
            resolved = as_feature_float(value)
            if not math.isnan(resolved):
                values.append(resolved)
        return values

    ordered["unit.count"] = float(len(actors))
    for unit_type in schema.unit_types():
        fraction = (sum(1 for actor in actors
                        if unit_types.get(actor) == unit_type)
                    / len(actors)) if actors else 0.0
        ordered[f"unit.type_fraction.{unit_type}"] = float(fraction)
    for name in schema.names(layer="unit"):
        values = unit_values(name)
        if not values:
            for agg in UNIT_AGGREGATIONS:
                ordered[f"unit.{agg}.{name}"] = float("nan")
            continue
        ordered[f"unit.mean.{name}"] = math.fsum(values) / len(values)
        ordered[f"unit.min.{name}"] = float(min(values))
        ordered[f"unit.max.{name}"] = float(max(values))
    return ordered


def internal_scorer_feature_names(schema: ZergFeatureSchema = DEFAULT_SCHEMA,
                                  shared_layers: Sequence[str] = ("team",
                                                                  "skill")
                                  ) -> List[str]:
    return list(build_internal_model_features(
        {layer: {} for layer in shared_layers} | {"units": {}}, {},
        [], schema, shared_layers).keys())


# -- decision records -------------------------------------------------------

def validate_decision_record(record: Dict[str, Any],
                             schema: ZergFeatureSchema = DEFAULT_SCHEMA
                             ) -> Dict[str, Any]:
    if record.get("schema") != DECISION_RECORD_SCHEMA:
        raise ValueError("decision record requires "
                         f"{DECISION_RECORD_SCHEMA}")
    # Task 2.1 identity keys
    for field in ("run_id", "episode_id"):
        _identifier(record.get(field), field)
    for field in ("source_hash", "execution_context_hash"):
        if record.get(field):
            _identifier(record[field], field)
    for field in ("scene_config_id", "map_id"):
        value = record.get(field)
        if value is not None:
            _identifier(value, field)
    if record.get("seed") is not None and type(record.get("seed")) is not int:
        raise ValueError("seed must be an integer when present")
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
    if outcome is not None and outcome not in OUTCOMES:
        raise ValueError("unknown outcome")
    if record.get("return") is not None:
        _finite(record["return"], "return")
    if record.get("baseline_return") is not None:
        _finite(record["baseline_return"], "baseline_return")
    provenance = record.get("reward_provenance")
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError("reward_provenance must be an object when present")
    return record


# -- counterfactual pair records (Task 2.3) ---------------------------------

def validate_pair_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """One BASE-vs-candidate counterfactual comparison."""
    if record.get("schema") != PAIR_RECORD_SCHEMA:
        raise ValueError(f"pair record requires {PAIR_RECORD_SCHEMA}")
    for field in ("experiment_id", "run_id", "episode_id", "anchor_id",
                  "pair_id", "baseline_id", "execution_context_hash"):
        _identifier(record.get(field), field)
    _identifier(record.get("candidate_id"), "candidate_id")
    for field in ("initial_state_hash", "prefix_hash"):
        value = record.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{field} must be a hash string when present")
    if record.get("scene_config_id") is not None:
        _identifier(record["scene_config_id"], "scene_config_id")
    if record.get("intervention_type") not in INTERVENTION_TYPES:
        raise ValueError("intervention_type must be one of "
                         + ", ".join(INTERVENTION_TYPES))
    if type(record.get("intervention_horizon")) is not int \
            or record["intervention_horizon"] < 1:
        raise ValueError("intervention_horizon must be a positive integer "
                         "(decision steps)")
    if record.get("completion_status") not in COMPLETION_STATUSES:
        raise ValueError("unknown completion_status")
    candidate_return = record.get("candidate_return")
    baseline_return = record.get("baseline_return")
    gain = record.get("gain")
    for name, value in (("candidate_return", candidate_return),
                        ("baseline_return", baseline_return),
                        ("gain", gain)):
        if value is not None:
            _finite(value, name)
    outcome = record.get("outcome")
    if outcome is not None and outcome not in OUTCOMES:
        raise ValueError("unknown outcome")
    if gain is not None:
        if candidate_return is None or baseline_return is None:
            raise ValueError("gain requires both returns")
        if abs(gain - (candidate_return - baseline_return)) > 1e-9:
            raise ValueError("gain must equal candidate_return - "
                             "baseline_return")
    return record


def audit_pair(record: Dict[str, Any]) -> List[str]:
    """Pair-integrity audit: problems that disqualify the pair from gain
    training. Missing returns are legitimate (kept None) but the pair is
    then incomplete."""
    problems: List[str] = []
    if record.get("initial_state_hash") is None \
            or record.get("prefix_hash") is None:
        problems.append("missing initial_state_hash or prefix_hash: "
                        "paired start not evidenced")
    if record.get("execution_context_hash") is None:
        problems.append("missing execution_context_hash: contexts may differ")
    if record.get("completion_status") != "complete":
        problems.append(f"completion_status="
                        f"{record.get('completion_status')!r}")
    if record.get("candidate_return") is None \
            or record.get("baseline_return") is None:
        problems.append("missing return (kept None, never fabricated)")
    try:
        validate_pair_record(record)
    except ValueError as exc:
        problems.append(str(exc))
    return problems


def pair_is_trainable(record: Dict[str, Any]) -> bool:
    """A pair may produce a gain label only when fully evidenced."""
    return not audit_pair(record)


# -- scorer models ----------------------------------------------------------

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
    layout = model.get("feature_layout", "standard")
    if layout not in ("standard", "internal"):
        raise ValueError("feature_layout must be 'standard' or 'internal'")
    # Task 1.3 / Task 3.4: names AND order must match the declared layout's
    # single-join output exactly.
    expected = (internal_scorer_feature_names(schema) if layout == "internal"
                else scorer_feature_names(schema))
    if feature_names != expected:
        raise ValueError(
            f"model feature_names must exactly equal the {layout} "
            f"build_model_features order "
            f"(got {len(feature_names)} names, expected {len(expected)})")
    forbidden = sorted(set(feature_names) & FEATURE_FORBIDDEN)
    if forbidden:
        raise ValueError(f"association keys cannot be features: {forbidden}")
    embedded = model.get("feature_schema", {})
    if embedded.get("digest") != schema.digest() \
            or embedded.get("version") != schema.version:
        raise ValueError("model feature schema mismatch — refusing to load")
    _identifier(model.get("execution_context_hash"), "execution_context_hash")
    if model.get("source_fingerprint"):
        _identifier(model["source_fingerprint"], "source_fingerprint")
    if model.get("dataset_hash") is not None:
        _identifier(model["dataset_hash"], "dataset_hash")
    for field_name in ("model_id", "parent_model_id"):
        if model.get(field_name) is not None:
            _identifier(model[field_name], field_name)
    if model.get("validation_report") is not None \
            and not isinstance(model["validation_report"], dict):
        raise ValueError("validation_report must be an object when present")
    if model.get("training_configuration") is not None \
            and not isinstance(model["training_configuration"], dict):
        raise ValueError("training_configuration must be an object")
    active = model.get("active_feature_names")
    if active is not None:
        if (not isinstance(active, list) or not active
                or sorted(active) != sorted(set(active))):
            raise ValueError("active_feature_names must be a unique list")
        unknown = sorted(set(active) - set(feature_names))
        if unknown:
            raise ValueError(f"active features outside declared layout: "
                             f"{unknown}")
    inactive = model.get("inactive_features")
    if inactive is not None and not isinstance(inactive, list):
        raise ValueError("inactive_features must be a list when present")

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


# -- pairwise ranking model (Phase 6.3) -------------------------------------

RANKING_MODEL_KIND = "zerg_pairwise_ranking_model_v1"


def ranking_feature_names(schema: ZergFeatureSchema = DEFAULT_SCHEMA,
                          shared_layers: Sequence[str] = ("team", "skill")
                          ) -> List[str]:
    """Feature layout of one ordered comparison: side 'a' then side 'b'."""
    base = scorer_feature_names(schema, shared_layers)
    return [f"a.{name}" for name in base] + [f"b.{name}" for name in base]


def validate_ranking_model(model: Dict[str, Any],
                           schema: ZergFeatureSchema = DEFAULT_SCHEMA
                           ) -> Dict[str, Any]:
    if model.get("schema_version") != 1 \
            or model.get("kind") != RANKING_MODEL_KIND:
        raise ValueError(f"model requires kind={RANKING_MODEL_KIND} version 1")
    feature_names = model.get("feature_names")
    expected = ranking_feature_names(schema)
    if feature_names != expected:
        raise ValueError("ranking model feature_names must exactly equal the "
                         "a./b. pairwise layout in schema order")
    _identifier(model.get("execution_context_hash"),
                "execution_context_hash")
    embedded = model.get("feature_schema", {})
    if embedded.get("digest") != schema.digest():
        raise ValueError("model feature schema mismatch — refusing to load")
    _validate_tree_bundle({"trees": model.get("trees", [])}, feature_names,
                          int(model.get("capacity", {}).get("max_depth", 2)),
                          int(model.get("capacity", {}).get(
                              "number_of_trees", 1)))
    return model


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
