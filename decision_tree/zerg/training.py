"""ZergTrainingBackend — training framework adapter over the existing
decision_tree learning machinery.

Reuse boundary (deliberately narrow and explicit):

- scenario splitting/weighting: ``decision_tree.learning.train.group_split``
  and the equal-scenario-mass contract are imported and used as-is;
- canonical hashing: ``decision_tree.experiment.digest``;
- tree semantics: the same audited contract as ``decision_tree.local.models``
  and ``decision_tree.learning.vector_train`` — weighted-SSE greedy
  threshold splits, midpoint thresholds, per-side minimum row AND minimum
  independent-scene support, masked per-candidate estimates, missing labels
  stay absent. The fitter below is the schema-generalised port of that ONE
  algorithm (the existing entries hard-bind the Protoss feature schema at
  their validation layer), extended with the optional ensemble capacity the
  zerg config exposes. It is not a new training algorithm;
- correction labels follow ``decision_tree.local.training.correction_rows``:
  veto target = negative gain of the frozen model's own winner.

Model capacity (``capacity``) is explicit in every model record: estimator
type, number_of_trees (192 allowed as a capacity option; the default stays
1 — no existing default changes), max_depth, learning_rate,
minimum_leaf_support, minimum_leaf_groups, random_seed, feature schema and
training context.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .dataset import candidate_rows, split_records
from .features import DEFAULT_SCHEMA, ZergFeatureSchema
from .objectives import gain_rows, group_weights_for
from .schema import MODEL_KIND, digest as canonical_digest, \
    scorer_feature_names, validate_scorer_model

DEFAULT_CAPACITY = {
    "estimator_type": "single_threshold_tree",
    "number_of_trees": 1,
    "max_depth": 2,
    "learning_rate": 1.0,
    "minimum_leaf_support": 4,
    "minimum_leaf_groups": 4,
    "random_seed": 0,
}

TOLERANCE = 1e-12


@dataclass
class ZergCandidateAdapter:
    """Convert ZergCandidates / decision records into the flat per-candidate
    training rows the framework understands (Task 7.1 interface)."""

    schema: ZergFeatureSchema = field(default_factory=lambda: DEFAULT_SCHEMA)
    shared_layers: Sequence[str] = ("team", "skill")

    def feature_names(self) -> List[str]:
        return scorer_feature_names(self.schema, self.shared_layers)

    def to_row(self, record: Dict[str, Any], candidate: Dict[str, Any]
               ) -> Optional[Dict[str, Any]]:
        rows = candidate_rows([record], self.schema, self.shared_layers)
        for row in rows:
            if row["candidate_id"] == candidate["candidate_id"]:
                return row
        return None

    def to_rows(self, records: Sequence[Dict[str, Any]]
                ) -> List[Dict[str, Any]]:
        return candidate_rows(records, self.schema, self.shared_layers)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _impute(rows: List[Dict[str, Any]], feature_names: Sequence[str]
            ) -> Dict[str, float]:
    """Deterministic per-feature median over finite training values.

    Recorded inside the model and applied identically at prediction time;
    raw records keep NaN so missing is never confused with true 0."""
    imputation: Dict[str, float] = {}
    for name in feature_names:
        values = sorted(_finite(row["features"][name], name)
                        for row in rows
                        if isinstance(row["features"].get(name), (int, float))
                        and math.isfinite(float(row["features"][name])))
        if not values:
            raise ValueError(f"feature {name} has no finite training value")
        middle = len(values) // 2
        median = values[middle] if len(values) % 2 \
            else (values[middle - 1] + values[middle]) / 2.0
        imputation[name] = median
    return imputation


def _row_vector(row: Dict[str, Any], feature_names: Sequence[str],
                imputation: Dict[str, float]) -> Dict[str, float]:
    vector = {}
    for name in feature_names:
        value = row["features"].get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or not math.isfinite(float(value)):
            value = imputation[name]
        vector[name] = float(value)
    return vector


def _scene_weights(rows: List[Dict[str, Any]]) -> List[float]:
    """Equal scene mass, split across the scene's decision points, then
    across each point's candidate rows."""
    scene_mass = group_weights_for(rows)
    return scene_mass


def _bootstrap_scene_counts(groups: Sequence[str], n_draws: int,
                            seed: int, tree_index: int) -> Dict[str, int]:
    """Deterministic bootstrap over scene groups (hash-counter RNG)."""
    counts = {group: 0 for group in groups}
    ordered = sorted(groups)
    for draw in range(n_draws):
        material = f"{seed}:{tree_index}:{draw}".encode()
        index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") \
            % len(ordered)
        counts[ordered[index]] += 1
    return counts


def fit_threshold_forest(rows: List[Dict[str, Any]], feature_names: Sequence[str],
                         y_field: str, *, capacity: Dict[str, Any],
                         weights: Optional[List[float]] = None
                         ) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Fit one scorer model body (trees / families excluded) from rows.

    Returns ``{"trees": [...], "aggregation": ..., "base_value": ...}`` plus
    the imputation map. Semantics per estimator type:

    - single_threshold_tree: one tree, weight 1.0, mean aggregation;
    - bagged_threshold_tree: deterministic scene bootstrap, mean aggregation;
    - boosted_threshold_tree: SSE gradient boosting on residuals from the
      base mean, learning_rate shrinkage, weighted-sum aggregation.
    """
    spec = {**DEFAULT_CAPACITY, **(capacity or {})}
    estimator = spec["estimator_type"]
    n_trees = int(spec["number_of_trees"])
    max_depth = int(spec["max_depth"])
    learning_rate = float(spec["learning_rate"])
    min_support = int(spec["minimum_leaf_support"])
    min_groups = int(spec["minimum_leaf_groups"])
    seed = int(spec["random_seed"])

    usable = []
    for row in rows:
        target = row.get(y_field)
        if target is None:
            continue
        usable.append(row)
    if not usable:
        raise ValueError(f"no rows with a finite {y_field} label")
    imputation = _impute(usable, feature_names)
    vectors = [_row_vector(row, feature_names, imputation) for row in usable]
    targets = [_finite(row[y_field], y_field) for row in usable]
    row_weights = _scene_weights(usable) if weights is None else weights
    groups = [row["group_id"] for row in usable]

    def stats(indices: Sequence[int]) -> Tuple[float, float]:
        mass = math.fsum(row_weights[i] for i in indices)
        if mass <= 0:
            return 0.0, 0.0
        mean = math.fsum(row_weights[i] * targets[i] for i in indices) / mass
        error = math.fsum(row_weights[i] * (targets[i] - mean) ** 2
                          for i in indices)
        return mean, error

    def build(indices: List[int], depth: int) -> Dict[str, Any]:
        mean, error = stats(indices)
        leaf_groups = sorted({groups[i] for i in indices})
        leaf = {"value": mean, "groups": len(leaf_groups),
                "samples": len(indices), "supported":
                    len(indices) >= min_support and len(leaf_groups) >= min_groups}
        if depth >= max_depth or error <= TOLERANCE:
            return leaf
        best, best_loss = None, error
        for name in feature_names:
            values = sorted({vectors[i][name] for i in indices})
            for low, high in zip(values, values[1:]):
                threshold = low / 2.0 + high / 2.0
                if threshold >= high:
                    threshold = low
                left = [i for i in indices if vectors[i][name] <= threshold]
                right = [i for i in indices if vectors[i][name] > threshold]
                left_groups = {groups[i] for i in left}
                right_groups = {groups[i] for i in right}
                if len(left) < min_support or len(right) < min_support \
                        or len(left_groups) < min_groups \
                        or len(right_groups) < min_groups:
                    continue
                loss = stats(left)[1] + stats(right)[1]
                if loss < best_loss - TOLERANCE:
                    best, best_loss = (name, threshold, left, right), loss
        if best is None:
            return leaf
        name, threshold, left, right = best
        return {"feature": name, "threshold": threshold,
                "left": build(left, depth + 1), "right": build(right, depth + 1)}

    base_value = stats(list(range(len(usable))))[0]

    def tree_value(tree: Dict[str, Any], index: int) -> float:
        node = tree
        while "feature" in node:
            node = node["left"] if vectors[index][node["feature"]] \
                <= node["threshold"] else node["right"]
        return node["value"]

    trees: List[Dict[str, Any]] = []
    if estimator == "single_threshold_tree" or n_trees == 1:
        trees.append({"weight": 1.0, "root": build(list(range(len(usable))), 0)})
        aggregation = "mean"
    elif estimator == "bagged_threshold_tree":
        unique_groups = sorted(set(groups))
        all_indices = list(range(len(usable)))
        for tree_index in range(n_trees):
            counts = _bootstrap_scene_counts(unique_groups, len(unique_groups),
                                             seed, tree_index)
            sample: List[int] = []
            for index in all_indices:
                sample.extend([index] * counts[groups[index]])
            trees.append({"weight": 1.0, "root": build(sample, 0)})
        aggregation = "mean"
    else:                                   # boosted_threshold_tree
        all_indices = list(range(len(usable)))
        current = [base_value] * len(usable)
        for _ in range(n_trees):
            residuals = [targets[i] - current[i] for i in all_indices]
            saved = targets[:]
            targets[:] = residuals          # build()/stats() read `targets`
            tree = build(all_indices, 0)
            targets[:] = saved
            for i in all_indices:
                current[i] += learning_rate * tree_value(tree, i)
            trees.append({"weight": learning_rate, "root": tree})
        aggregation = "weighted_sum"
    body = {"trees": trees, "aggregation": aggregation,
            "base_value": base_value if aggregation == "weighted_sum" else 0.0,
            "missing_imputation": imputation}
    return body, imputation


def _predict_body(body: Dict[str, Any], vector: Dict[str, float]) -> float:
    total = 0.0
    for tree in body["trees"]:
        node = tree["root"]
        while "feature" in node:
            value = vector[node["feature"]]
            node = node["left"] if value <= node["threshold"] else node["right"]
        total += float(tree.get("weight", 1.0)) * float(node["value"])
    if body.get("aggregation") == "mean":
        weights = math.fsum(float(t.get("weight", 1.0)) for t in body["trees"])
        return total / weights if weights else 0.0
    return total + float(body.get("base_value", 0.0))


def _leaf_of(body: Dict[str, Any], vector: Dict[str, float]
             ) -> Optional[Dict[str, Any]]:
    leaves = []
    for tree in body["trees"]:
        node = tree["root"]
        while "feature" in node:
            value = vector[node["feature"]]
            node = node["left"] if value <= node["threshold"] else node["right"]
        leaves.append(node)
    worst = min(leaves, key=lambda leaf: (leaf["supported"], leaf["groups"]))
    return worst


class ZergTrainingBackend:
    """Task facade: imitation / gain / correction / ranking."""

    def __init__(self, schema: ZergFeatureSchema = DEFAULT_SCHEMA,
                 shared_layers: Sequence[str] = ("team", "skill")):
        self.schema = schema
        self.adapter = ZergCandidateAdapter(schema, shared_layers)
        self.feature_names = self.adapter.feature_names()

    # -- core ---------------------------------------------------------------
    def train(self, task: str, records: Sequence[Dict[str, Any]], *,
              scorer: str, execution_context_hash: str,
              source_fingerprint: str = "",
              capacity: Optional[Dict[str, Any]] = None,
              validation_fraction: float = 0.25,
              split_seed: int = 0,
              old_model: Optional[Dict[str, Any]] = None,
              group_by_family: bool = False,
              candidate_skill_filter: Optional[str] = None
              ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if task not in ("imitation", "gain", "correction", "ranking"):
            raise ValueError("unknown training task")
        all_rows = self.adapter.to_rows(records)
        if candidate_skill_filter:
            all_rows = [row for row in all_rows
                        if row["candidate_skill"] == candidate_skill_filter]
        train_records, validation_records = split_records(
            records, validation_fraction=validation_fraction,
            split_seed=split_seed)
        train_rows = [row for row in self.adapter.to_rows(train_records)
                      if not candidate_skill_filter
                      or row["candidate_skill"] == candidate_skill_filter]
        validation_rows = [row for row in self.adapter.to_rows(validation_records)
                           if not candidate_skill_filter
                           or row["candidate_skill"] == candidate_skill_filter]

        if task == "imitation":
            body, report = self._fit_supervised(
                train_rows, validation_rows, y_field="teacher",
                capacity=capacity, group_by_family=group_by_family,
                y_of=self._imitation_targets)
            target = "imitation"
        elif task in ("gain", "ranking"):
            fit_rows, problems = gain_rows(train_rows, strict=True)
            if problems:
                raise ValueError(f"gain audit failed: {problems[:3]}")
            val_rows, _ = gain_rows(validation_rows, strict=False)
            body, report = self._fit_supervised(
                fit_rows, val_rows, y_field="gain", capacity=capacity,
                group_by_family=group_by_family,
                y_of=self._gain_targets)
            target = "gain" if task == "gain" else "ranking"
            report["ranking_limitation"] = (
                "ranking labels equal paired counterfactual gains; anchors "
                "without re-execution evidence carry no ranking label")
        else:
            if old_model is None:
                raise ValueError("correction training requires the frozen "
                                 "old gain model record")
            veto_train = self.correction_rows(train_rows, old_model)
            veto_validation = self.correction_rows(validation_rows, old_model)
            body, report = self._fit_supervised(
                veto_train, veto_validation, y_field="veto_target",
                capacity=capacity, group_by_family=group_by_family,
                y_of=self._veto_targets)
            target = "correction"

        record = {
            "schema_version": 1,
            "kind": MODEL_KIND,
            "scorer": scorer,
            "target": target,
            "feature_names": list(self.feature_names),
            "feature_schema": {"name": self.schema.name,
                               "version": self.schema.version,
                               "digest": self.schema.digest()},
            "execution_context_hash": execution_context_hash,
            "source_fingerprint": source_fingerprint,
            "capacity": {**DEFAULT_CAPACITY, **(capacity or {})},
            "min_gain": 0.0,
            "tie_tolerance": 1e-9,
            "training_context": {
                "task": task,
                "split_seed": split_seed,
                "validation_fraction": validation_fraction,
                "candidate_skill_filter": candidate_skill_filter,
            },
        }
        if group_by_family:
            record["families"] = body["families"]
        else:
            record["trees"] = body["trees"]
            record["aggregation"] = body["aggregation"]
            record["base_value"] = body["base_value"]
            record["missing_imputation"] = body["missing_imputation"]
        record = validate_scorer_model(record, self.schema)
        report["model_sha256"] = canonical_digest(record)
        return record, report

    # -- task wiring --------------------------------------------------------
    @staticmethod
    def _imitation_targets(rows: Sequence[Dict[str, Any]]
                           ) -> List[Optional[float]]:
        return [row.get("teacher") for row in rows]

    @staticmethod
    def _gain_targets(rows: Sequence[Dict[str, Any]]
                      ) -> List[Optional[float]]:
        return [row.get("gain") for row in rows]

    @staticmethod
    def _veto_targets(rows: Sequence[Dict[str, Any]]
                      ) -> List[Optional[float]]:
        return [row.get("veto_target") for row in rows]

    def correction_rows(self, rows: Sequence[Dict[str, Any]],
                        old_model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Veto labels: target = negative gain of the frozen model's own
        winning candidate at each decision point (framework pattern)."""
        by_decision: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            by_decision.setdefault(row["decision_key"], []).append(row)
        out: List[Dict[str, Any]] = []
        old_hash = canonical_digest(old_model)
        for decision_key, group in sorted(by_decision.items()):
            scored = []
            for row in group:
                gain = row.get("gain")
                if gain is None:
                    continue
                scored.append((gain, row))
            if not scored:
                continue
            gain, winner = max(scored, key=lambda pair: pair[0])
            out.append({**winner, "veto_target": -float(gain),
                        "objective": "correction",
                        "old_model_hash": old_hash})
        return out

    # -- fitting with optional family split ---------------------------------
    def _fit_supervised(self, train_rows: List[Dict[str, Any]],
                        validation_rows: List[Dict[str, Any]],
                        *, y_field: str, capacity: Optional[Dict[str, Any]],
                        group_by_family: bool,
                        y_of: Callable[[Sequence[Dict[str, Any]]],
                                       List[Optional[float]]]
                        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        prepared_train = self._attach_targets(train_rows, y_of)
        spec = {**DEFAULT_CAPACITY, **(capacity or {})}
        if group_by_family:
            families: Dict[str, Dict[str, Any]] = {}
            for family in sorted({row["candidate_skill"] for row in prepared_train}):
                part = [row for row in prepared_train
                        if row["candidate_skill"] == family]
                if len(part) < max(1, int(spec["minimum_leaf_support"])):
                    continue
                body, _ = fit_threshold_forest(
                    part, self.feature_names, y_field, capacity=spec)
                families[family] = {"trees": body["trees"],
                                    "aggregation": body["aggregation"],
                                    "base_value": body["base_value"],
                                    "missing_imputation":
                                        body["missing_imputation"]}
            if not families:
                raise ValueError("no candidate family had enough support")
            body = {"families": families}
            report = self._report(body, prepared_train, validation_rows,
                                  y_field, families=families)
            return body, report
        body, _ = fit_threshold_forest(prepared_train, self.feature_names,
                                       y_field, capacity=spec)
        report = self._report(body, prepared_train, validation_rows,
                              y_field)
        return body, report

    @staticmethod
    def _attach_targets(rows: List[Dict[str, Any]],
                        y_of: Callable) -> List[Dict[str, Any]]:
        targets = y_of(rows)
        out = []
        for row, target in zip(rows, targets):
            if target is None:
                continue
            row = dict(row)
            row["_target"] = float(target)
            out.append(row)
        return out

    def _report(self, body: Dict[str, Any], train_rows, validation_rows,
                y_field: str, families: Optional[Dict[str, Any]] = None
                ) -> Dict[str, Any]:
        imputation = (next(iter(families.values()))["missing_imputation"]
                      if families else body["missing_imputation"])

        def evaluate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
            prepared = self._attach_targets(
                rows, lambda rs: [r.get("_target") for r in rs])
            if not prepared:
                return {"n_rows": 0}
            errors, supported = [], 0
            correct = total_teacher = 0
            for row in prepared:
                vector = _row_vector(row, self.feature_names, imputation)
                if families:
                    bundle = families.get(row["candidate_skill"])
                    if bundle is None:
                        continue
                    local = {"trees": bundle["trees"],
                             "aggregation": bundle["aggregation"],
                             "base_value": bundle["base_value"]}
                    prediction = _predict_body(local, vector)
                    leaf = _leaf_of(local, vector)
                else:
                    prediction = _predict_body(body, vector)
                    leaf = _leaf_of(body, vector)
                errors.append((prediction - row["_target"]) ** 2)
                supported += 1 if leaf and leaf.get("supported") else 0
                if row.get("teacher") is not None:
                    total_teacher += 1
                    predicted_choice = prediction >= 0.5
                    correct += predicted_choice == (row["teacher"] >= 0.5)
            n = len(prepared)
            return {"n_rows": n,
                    "mse": math.fsum(errors) / n if n else None,
                    "supported_leaf_fraction": supported / n if n else None,
                    "teacher_agreement": correct / total_teacher
                    if total_teacher else None}

        tree_count, depth = self._tree_stats(body, families)
        return {
            "capacity_used": tree_count,
            "max_depth_used": depth,
            "training": evaluate(train_rows),
            "validation": evaluate(validation_rows),
            "limitations": [
                "offline estimates reuse supplied outcomes; they are not new games",
                "support counts are independent scenes, not rows or candidates",
                "anchor gains cannot be summed into a multi-intervention policy gain",
                "frozen-context models are not deployable after source or config changes",
            ],
        }

    @staticmethod
    def _tree_stats(body: Dict[str, Any],
                    families: Optional[Dict[str, Any]]) -> Tuple[int, int]:
        bundles: List[Dict[str, Any]] = []
        if families:
            bundles.extend(families.values())
        else:
            bundles.append(body)
        count = 0
        max_depth = 0

        def depth_of(node: Dict[str, Any]) -> int:
            if "feature" not in node:
                return 0
            return 1 + max(depth_of(node["left"]), depth_of(node["right"]))

        for bundle in bundles:
            trees = bundle.get("trees", [])
            count += len(trees)
            for tree in trees:
                max_depth = max(max_depth, depth_of(tree["root"]))
        return count, max_depth


def load_scorer_model(path: str) -> "ScorerModel":
    """Load + validate a scorer model file into the runtime model wrapper."""
    from .scorer import ScorerModel
    from .schema import load_json
    return ScorerModel(load_json(path))
