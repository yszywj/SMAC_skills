"""ZergTrainingBackend — training framework adapter over the existing
decision_tree learning machinery (V1.1).

Reuse boundary (deliberately narrow and explicit):

- scenario splitting/weighting: ``decision_tree.learning.train.group_split``
  and the equal-scenario-mass contract are imported and used as-is;
- canonical hashing: ``decision_tree.experiment.digest``;
- tree semantics: the same audited contract as ``decision_tree.local.models``
  and ``decision_tree.learning.vector_train`` — weighted-SSE greedy
  threshold splits, midpoint thresholds, per-side minimum row AND minimum
  independent-scene support, missing labels stay absent. The fitter below is
  the schema-generalised port of that ONE algorithm (the existing entries
  hard-bind the Protoss feature schema at their validation layer), extended
  with the optional ensemble capacity (single / bagged / boosted,
  ``number_of_trees`` up to 192+ as an explicit option; the default stays 1).

Tasks (Phase 6):

- imitation: rows labelled by the actual rule choice;
- gain: audited counterfactual gains only (missing pairs never enter);
- ranking: REAL pairwise ranking — the training unit is the ordered
  candidate comparison (a./b. feature layout, margin target); realised
  paired gains are the evidence, rule-score priors are flagged weak
  supervision and their share is reported. This is NOT gain regression;
- correction: veto target = minus the real gain of the FROZEN old model's
  own winner (old model replayed per decision point; empirically-best
  substitution is forbidden; points without a supported winner or a real
  winner gain are skipped).

Feature vectors on every row come from the single join
``schema.build_model_features`` — identical to online scoring.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .dataset import candidate_rows, split_records
from .features import DEFAULT_SCHEMA, ZergFeatureSchema
from .objectives import correction_rows, gain_rows, group_weights_for, \
    imitation_rows, ranking_pairs
from .schema import MODEL_KIND, RANKING_MODEL_KIND, digest as canonical_digest, \
    internal_scorer_feature_names, ranking_feature_names, \
    scorer_feature_names, validate_ranking_model, validate_scorer_model

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
    """Convert decision records into flat per-candidate training rows
    (features via the single join — Task 1.1)."""

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


def feature_availability(rows: Sequence[Dict[str, Any]],
                         feature_names: Sequence[str]
                         ) -> Dict[str, Dict[str, Any]]:
    """Task 2.1: per-feature finite count, missing ratio and the number of
    independent episodes carrying at least one finite value."""
    stats: Dict[str, Dict[str, Any]] = {}
    total = len(rows)
    for name in feature_names:
        finite_rows = [row for row in rows
                       if isinstance(row["features"].get(name), (int, float))
                       and not isinstance(row["features"].get(name), bool)
                       and math.isfinite(float(row["features"][name]))]
        stats[name] = {
            "finite_count": len(finite_rows),
            "missing_ratio": ((total - len(finite_rows)) / total)
            if total else 1.0,
            "episodes_with_finite": len({row["group_id"]
                                         for row in finite_rows}),
        }
    return stats


def select_active_features(rows: Sequence[Dict[str, Any]],
                           feature_names: Sequence[str],
                           min_finite: int = 1,
                           min_groups: int = 1
                           ) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Task 2.1: drop dataset-wide constant-missing columns from the
    training feature space. Never fills them with 0 — the original records
    keep the full schema; the MODEL just stops declaring dead inputs."""
    availability = feature_availability(rows, feature_names)
    active: List[str] = []
    inactive: List[Dict[str, Any]] = []
    for name in feature_names:
        stat = availability[name]
        if stat["finite_count"] >= min_finite \
                and stat["episodes_with_finite"] >= min_groups:
            active.append(name)
        else:
            inactive.append({"name": name,
                             "reason": "constant_missing_in_training",
                             **stat})
    return active, inactive


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


def _bootstrap_scene_counts(groups: Sequence[str], n_draws: int,
                            seed: int, tree_index: int) -> Dict[str, int]:
    """Deterministic bootstrap over group ids (hash-counter RNG)."""
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
                         weights: Optional[List[float]] = None,
                         active_feature_names: Optional[Sequence[str]] = None
                         ) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Fit one model body (trees) from rows with fields
    ``{group_id, decision_key, features, <y_field>}``.

    Semantics per estimator type:

    - single_threshold_tree: one tree, weight 1.0, mean aggregation;
    - bagged_threshold_tree: deterministic group bootstrap, mean aggregation;
    - boosted_threshold_tree: SSE gradient boosting on residuals from the
      base mean, learning_rate shrinkage, weighted-sum aggregation
      (prediction = base_value + learning_rate * sum(tree leaf values)).
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
    # Task 2.1/2.2: splits and medians cover ACTIVE features only; medians
    # come from THIS (training) row set exclusively.
    active = list(active_feature_names) if active_feature_names \
        else list(feature_names)
    imputation = _impute(usable, active)
    vectors = [_row_vector(row, active, imputation) for row in usable]
    targets = [_finite(row[y_field], y_field) for row in usable]
    row_weights = group_weights_for(usable) if weights is None else weights
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
        for name in active:
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
            "missing_imputation": imputation,
            "active_feature_names": active}
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
    if not leaves:
        return None
    return min(leaves, key=lambda leaf: (leaf.get("supported", False),
                                         leaf.get("groups", 0)))


class ZergTrainingBackend:
    """Task facade: imitation / gain / correction / ranking."""

    def __init__(self, schema: ZergFeatureSchema = DEFAULT_SCHEMA,
                 shared_layers: Sequence[str] = ("team", "skill")):
        self.schema = schema
        self.adapter = ZergCandidateAdapter(schema, shared_layers)
        self.feature_names = self.adapter.feature_names()
        self.shared_layers = tuple(shared_layers)

    # -- core ---------------------------------------------------------------
    def train(self, task: str, records: Sequence[Dict[str, Any]], *,
              scorer: str, execution_context_hash: str,
              source_fingerprint: str = "",
              capacity: Optional[Dict[str, Any]] = None,
              validation_fraction: float = 0.25,
              split_seed: int = 0,
              old_model: Optional[Dict[str, Any]] = None,
              group_by_family: bool = False,
              candidate_skill_filter: Optional[str] = None,
              dataset_hash: Optional[str] = None,
              feature_layout: str = "standard"
              ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if task not in ("imitation", "gain", "correction", "ranking"):
            raise ValueError("unknown training task")
        if feature_layout not in ("standard", "internal"):
            raise ValueError("feature_layout must be 'standard' or 'internal'")
        if feature_layout == "internal":
            raise ValueError("internal-layout models must be trained via "
                             "train_internal()")
        if dataset_hash is None:
            from .schema import _sanitize
            dataset_hash = canonical_digest(_sanitize(list(records)))
        train_records, validation_records = split_records(
            records, validation_fraction=validation_fraction,
            split_seed=split_seed)

        def filtered(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if not candidate_skill_filter:
                return rows
            return [row for row in rows
                    if row["candidate_skill"] == candidate_skill_filter]

        train_rows = filtered(self.adapter.to_rows(train_records))
        validation_rows = filtered(self.adapter.to_rows(validation_records))
        all_rows = filtered(self.adapter.to_rows(records))

        common = {
            "execution_context_hash": execution_context_hash,
            "source_fingerprint": source_fingerprint or None,
            "dataset_hash": dataset_hash,
            "training_configuration": {
                "task": task, "split_seed": split_seed,
                "validation_fraction": validation_fraction,
                "candidate_skill_filter": candidate_skill_filter,
                "group_by_family": group_by_family,
                "scorer": scorer,
            },
        }

        if task == "imitation":
            body, report = self._fit_supervised(
                train_rows, validation_rows, y_field="teacher",
                capacity=capacity, group_by_family=group_by_family,
                task="imitation")
            target = "imitation"
        elif task in ("gain",):
            fit_rows, problems = gain_rows(train_rows, strict=True)
            if problems:
                raise ValueError(f"gain audit failed: {problems[:3]}")
            val_rows, _ = gain_rows(validation_rows, strict=False)
            body, report = self._fit_supervised(
                fit_rows, val_rows, y_field="gain", capacity=capacity,
                group_by_family=group_by_family, task="gain")
            target = "gain"
        elif task == "ranking":
            return self._train_ranking(all_rows, train_rows, validation_rows,
                                       scorer=scorer, common=common,
                                       capacity=capacity)
        else:
            if old_model is None:
                raise ValueError("correction training requires the frozen "
                                 "old gain model record")
            from .scorer import ScorerModel
            old_scorer = ScorerModel(old_model)
            veto_train = correction_rows(train_rows, old_scorer)
            veto_validation = correction_rows(validation_rows, old_scorer)
            body, report = self._fit_supervised(
                veto_train, veto_validation, y_field="veto_target",
                capacity=capacity, group_by_family=group_by_family,
                task="correction")
            target = "correction"
            report["correction_skipped_points"] = {
                "train_points_without_supported_winner_or_gain":
                    report.get("correction_skipped_points", 0)}

        record = {
            "schema_version": 1,
            "kind": MODEL_KIND,
            "scorer": scorer,
            "target": target,
            "feature_layout": feature_layout,
            "feature_names": list(self.feature_names),
            "feature_schema": {"name": self.schema.name,
                               "version": self.schema.version,
                               "digest": self.schema.digest()},
            "capacity": {**DEFAULT_CAPACITY, **(capacity or {})},
            "min_gain": 0.0,
            "tie_tolerance": 1e-9,
            **common,
        }
        if target == "correction":
            record["old_model_hash"] = canonical_digest(old_model)
        if group_by_family:
            record["active_feature_names"] = sorted({name
                                                     for bundle in
                                                     body["families"].values()
                                                     for name in
                                                     bundle.get(
                                                         "active_feature_names",
                                                         [])})
        else:
            record["active_feature_names"] = body.get("active_feature_names",
                                                      list(self.feature_names))
        record["inactive_features"] = report.pop("inactive_features", [])
        if group_by_family:
            record["families"] = body["families"]
        else:
            record["trees"] = body["trees"]
            record["aggregation"] = body["aggregation"]
            record["base_value"] = body["base_value"]
            record["missing_imputation"] = body["missing_imputation"]
        record["validation_report"] = report.get("validation", {})
        record = validate_scorer_model(record, self.schema)
        report["model_sha256"] = canonical_digest(record)
        return record, report

    # -- A-class internal scorers (Phase 3.4/3.5) ---------------------------
    def train_internal(self, task: str, records: Sequence[Dict[str, Any]], *,
                       scorer: str, execution_context_hash: str,
                       source_fingerprint: str = "",
                       capacity: Optional[Dict[str, Any]] = None,
                       validation_fraction: float = 0.25,
                       split_seed: int = 0,
                       old_model: Optional[Dict[str, Any]] = None,
                       group_by_family: bool = False,
                       candidate_skill: Optional[str] = None,
                       unit_type: Optional[str] = None,
                       dataset_hash: Optional[str] = None,
                       min_train_groups: int = 4
                       ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Train a skill-INTERNAL scorer over A-class rows
        (``expand_skill_proposals``) with the unit-level aggregate feature
        layout (``build_internal_model_features``). Routing mirrors online
        scoring: single-unit-type candidates -> type scorer, mixed -> joint.
        """
        if task not in ("imitation", "gain", "correction", "ranking"):
            raise ValueError("unknown training task")
        from .dataset import expand_skill_proposals, export_internal_training_rows
        feature_names = internal_scorer_feature_names(self.schema,
                                                      self.shared_layers)
        rows_all = expand_skill_proposals(records, self.schema,
                                          self.shared_layers)
        rows_all = export_internal_training_rows(rows_all,
                                                 skill=candidate_skill,
                                                 unit_type=unit_type)
        if not rows_all:
            raise ValueError("no A-class proposals in the given records "
                             "(or the records lack the units feature layer)")
        prepared: List[Dict[str, Any]] = []
        for row in rows_all:
            row = dict(row)
            internal = row.pop("features_internal", None)
            if internal is None:
                continue          # record without the units layer
            row["features"] = internal
            # comparisons only WITHIN one internal decision group
            row["decision_key"] = row["decision_group_id"]
            prepared.append(row)
        if dataset_hash is None:
            from .schema import _sanitize
            dataset_hash = canonical_digest(_sanitize(list(records)))
        from decision_tree.learning.train import group_split
        train_rows, validation_rows = group_split(
            prepared, validation_fraction=validation_fraction,
            split_seed=split_seed, min_train_groups=min_train_groups)

        common = {
            "execution_context_hash": execution_context_hash,
            "source_fingerprint": source_fingerprint or None,
            "dataset_hash": dataset_hash,
            "training_configuration": {
                "task": task, "split_seed": split_seed,
                "validation_fraction": validation_fraction,
                "candidate_skill_filter": candidate_skill,
                "unit_type_filter": unit_type,
                "group_by_family": group_by_family,
                "scorer": scorer, "feature_layout": "internal",
                "row_source": "skill_proposals",
            },
        }

        if task == "imitation":
            body, report = self._fit_supervised(
                train_rows, validation_rows, y_field="teacher",
                capacity=capacity, group_by_family=group_by_family,
                task="imitation", feature_names=feature_names)
            target = "imitation"
        elif task == "gain":
            fit_rows, problems = gain_rows(train_rows, strict=True)
            if problems:
                raise ValueError(f"gain audit failed: {problems[:3]}")
            val_rows, _ = gain_rows(validation_rows, strict=False)
            body, report = self._fit_supervised(
                fit_rows, val_rows, y_field="gain", capacity=capacity,
                group_by_family=group_by_family, task="gain",
                feature_names=feature_names)
            target = "gain"
        elif task == "ranking":
            return self._train_ranking(prepared, train_rows, validation_rows,
                                       scorer=scorer, common=common,
                                       capacity=capacity,
                                       feature_names_override=feature_names,
                                       feature_layout="internal")
        else:
            if old_model is None:
                raise ValueError("correction training requires the frozen "
                                 "old gain model record")
            from .scorer import ScorerModel
            veto_train = correction_rows(train_rows, ScorerModel(old_model))
            veto_validation = correction_rows(validation_rows,
                                              ScorerModel(old_model))
            body, report = self._fit_supervised(
                veto_train, veto_validation, y_field="veto_target",
                capacity=capacity, group_by_family=group_by_family,
                task="correction", feature_names=feature_names)
            target = "correction"

        record = {
            "schema_version": 1, "kind": MODEL_KIND, "scorer": scorer,
            "target": target, "feature_layout": "internal",
            "feature_names": feature_names,
            "feature_schema": {"name": self.schema.name,
                               "version": self.schema.version,
                               "digest": self.schema.digest()},
            "capacity": {**DEFAULT_CAPACITY, **(capacity or {})},
            "min_gain": 0.0, "tie_tolerance": 1e-9,
            **common,
        }
        if target == "correction":
            record["old_model_hash"] = canonical_digest(old_model)
        if group_by_family:
            record["families"] = body["families"]
            record["active_feature_names"] = sorted({n for bundle in
                                                     body["families"].values()
                                                     for n in bundle.get(
                                                         "active_feature_names",
                                                         [])})
        else:
            record["trees"] = body["trees"]
            record["aggregation"] = body["aggregation"]
            record["base_value"] = body["base_value"]
            record["missing_imputation"] = body["missing_imputation"]
            record["active_feature_names"] = body.get("active_feature_names",
                                                      feature_names)
        record["inactive_features"] = report.pop("inactive_features", [])
        record["validation_report"] = report.get("validation", {})
        record = validate_scorer_model(record, self.schema)
        report["model_sha256"] = canonical_digest(record)
        return record, report

    # -- pairwise ranking (Phase 6.3) ---------------------------------------
    def _train_ranking(self, all_rows, train_rows, validation_rows, *,
                       scorer: str, common: Dict[str, Any],
                       capacity: Optional[Dict[str, Any]],
                       feature_names_override: Optional[Sequence[str]] = None,
                       feature_layout: str = "standard"
                       ) -> Tuple[Dict, Dict]:
        train_pairs, train_provenance = ranking_pairs(train_rows)
        validation_pairs, validation_provenance = ranking_pairs(
            validation_rows)
        if not train_pairs:
            raise ValueError("ranking training requires pairwise comparisons "
                             "(realised paired gains or rule prior)")
        pair_feature_names = list(feature_names_override) \
            if feature_names_override \
            else ranking_feature_names(self.schema, self.shared_layers)
        # Task 6.1: BOTH directions of every pair are trained, with the
        # reversed margin sign — the model cannot lean on a fixed slot.
        prepared = []
        for pair in train_pairs:
            for higher, lower, sign, direction in (
                    (pair["higher_features"], pair["lower_features"],
                     1.0, "forward"),
                    (pair["lower_features"], pair["higher_features"],
                     -1.0, "reverse")):
                features = {f"a.{k}": v for k, v in higher.items()}
                features.update({f"b.{k}": v for k, v in lower.items()})
                prepared.append({
                    "group_id": pair["group_id"],
                    "decision_key": pair["decision_key"],
                    "features": features,
                    "margin": sign * float(pair["margin"]),
                    "provenance": pair["provenance"],
                    "direction": direction,
                })
        spec = {**DEFAULT_CAPACITY, **(capacity or {})}
        body, _ = fit_threshold_forest(prepared, pair_feature_names,
                                       "margin", capacity=spec)
        record = {
            "schema_version": 1,
            "kind": RANKING_MODEL_KIND,
            "scorer": scorer,
            "target": "ranking",
            "feature_names": pair_feature_names,
            "feature_schema": {"name": self.schema.name,
                               "version": self.schema.version,
                               "digest": self.schema.digest()},
            "trees": body["trees"],
            "aggregation": body["aggregation"],
            "base_value": body["base_value"],
            "missing_imputation": body["missing_imputation"],
            "capacity": spec,
            **common,
        }
        report = {
            "pair_provenance": {"training": train_provenance,
                                "validation": validation_provenance},
            "validation": self._ranking_report(body, pair_feature_names,
                                               validation_pairs),
            "training": self._ranking_report(body, pair_feature_names,
                                             train_pairs),
            "limitations": [
                "pairwise comparisons are local preferences at isolated "
                "decision points; they are not whole-policy returns",
                "rule_score_prior pairs are weak supervision; their share is "
                "reported and must not be read as outcome evidence",
            ],
        }
        record["validation_report"] = report["validation"]
        record = validate_ranking_model(record, self.schema)
        report["model_sha256"] = canonical_digest(record)
        return record, report

    def _ranking_report(self, body, feature_names, pairs) -> Dict[str, Any]:
        """Task 6.3: BOTH input directions must agree before a comparison
        counts as correct; NDCG uses candidate-level realised gains and is
        computed only for decision points where EVERY candidate carries a
        real gain (rule prior is never treated as outcome evidence)."""
        if not pairs:
            return {"n_pairs": 0}
        forward_correct = reverse_correct = both_correct = total = 0
        by_decision: Dict[str, List[Dict[str, Any]]] = {}
        for pair in pairs:
            by_decision.setdefault(pair["decision_key"], []).append(pair)
            ab = _predict_body(body, self._pair_vector(body, feature_names,
                                                       pair))
            ba = _predict_body(body, self._pair_vector(body, feature_names, {
                "higher_features": pair["lower_features"],
                "lower_features": pair["higher_features"]}))
            total += 1
            forward_correct += ab > 0
            reverse_correct += ba < 0
            both_correct += (ab > 0) and (ba < 0)
        ndcgs = []
        for decision_key, group in by_decision.items():
            cands: Dict[str, Dict[str, Any]] = {}
            for pair in group:
                entry = cands.setdefault(pair["higher_id"],
                                         {"features": pair["higher_features"],
                                          "gain": pair.get("higher_gain")})
                entry2 = cands.setdefault(pair["lower_id"],
                                          {"features": pair["lower_features"],
                                           "gain": pair.get("lower_gain")})
                if entry["gain"] is None and pair.get("higher_gain") is not None:
                    entry["gain"] = pair["higher_gain"]
                if entry2["gain"] is None and pair.get("lower_gain") is not None:
                    entry2["gain"] = pair["lower_gain"]
            ids = sorted(cands)
            if len(ids) < 2:
                continue
            gains = {cid: cands[cid]["gain"] for cid in ids}
            # Task 6.3.6/6.3.7: candidate-level, complete-evidence points only
            if any(gain is None for gain in gains.values()):
                continue
            if any(pair["provenance"] != "paired_gain" for pair in group):
                continue

            def pair_score(cid: str) -> float:
                return math.fsum(
                    (_predict_body(body, self._pair_vector(body, feature_names,
                                       {"higher_features": cands[cid]["features"],
                                        "lower_features": cands[other]["features"]}))
                     - _predict_body(body, self._pair_vector(body, feature_names,
                                       {"higher_features": cands[other]["features"],
                                        "lower_features": cands[cid]["features"]})))
                    for other in ids if other != cid)

            ordered = sorted(ids, key=pair_score, reverse=True)
            dcg = math.fsum(gains[cid] / math.log2(rank + 2)
                            for rank, cid in enumerate(ordered))
            ideal = sorted(gains.values(), reverse=True)
            idcg = math.fsum(value / math.log2(rank + 2)
                             for rank, value in enumerate(ideal))
            if idcg > 0:
                ndcgs.append(dcg / idcg)
        return {"n_pairs": total,
                "pairwise_accuracy": both_correct / total if total else None,
                "forward_direction_accuracy": forward_correct / total
                if total else None,
                "reverse_direction_accuracy": reverse_correct / total
                if total else None,
                "ndcg_realised_gain_points": (
                    math.fsum(ndcgs) / len(ndcgs) if ndcgs else None),
                "ndcg_points": len(ndcgs)}

    def _pair_vector(self, body, feature_names, pair) -> Dict[str, float]:
        features = {f"a.{k}": v for k, v in pair["higher_features"].items()}
        features.update({f"b.{k}": v for k, v in pair["lower_features"].items()})
        imputation = body.get("missing_imputation", {})
        vector = {}
        for name in feature_names:
            value = features.get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) \
                    or not math.isfinite(float(value)):
                value = imputation.get(name)
            if value is None:
                value = 0.0      # fully-unknown pair side; recorded reason
            vector[name] = float(value)
        return vector

    # -- supervised fitting with task metrics -------------------------------
    def _fit_supervised(self, train_rows: List[Dict[str, Any]],
                        validation_rows: List[Dict[str, Any]],
                        *, y_field: str, capacity: Optional[Dict[str, Any]],
                        group_by_family: bool, task: str,
                        feature_names: Optional[Sequence[str]] = None
                        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        names = list(feature_names) if feature_names else self.feature_names
        prepared_train = self._attach_targets(train_rows, y_field)
        spec = {**DEFAULT_CAPACITY, **(capacity or {})}
        inactive: List[Dict[str, Any]] = []
        if group_by_family:
            families: Dict[str, Dict[str, Any]] = {}
            for family in sorted({row["candidate_skill"] for row in prepared_train}):
                part = [row for row in prepared_train
                        if row["candidate_skill"] == family]
                if len(part) < max(1, int(spec["minimum_leaf_support"])):
                    continue
                active_f, inactive_f = select_active_features(
                    part, names)
                inactive.extend(inactive_f)
                body, _ = fit_threshold_forest(
                    part, names, y_field, capacity=spec,
                    active_feature_names=active_f)
                families[family] = {"trees": body["trees"],
                                    "aggregation": body["aggregation"],
                                    "base_value": body["base_value"],
                                    "missing_imputation":
                                        body["missing_imputation"],
                                    "active_feature_names":
                                        body["active_feature_names"]}
            if not families:
                raise ValueError("no candidate family had enough support")
            body = {"families": families}
            imputation = next(iter(families.values()))["missing_imputation"]
        else:
            active_f, inactive_f = select_active_features(
                prepared_train, names)
            inactive.extend(inactive_f)
            body, _ = fit_threshold_forest(prepared_train, names,
                                           y_field, capacity=spec,
                                           active_feature_names=active_f)
            imputation = body["missing_imputation"]

        def predict_local(row) -> Tuple[float, Optional[Dict[str, Any]]]:
            if group_by_family:
                bundle = body["families"].get(row["candidate_skill"])
                if bundle is None:
                    return 0.0, None
                active_f = bundle.get("active_feature_names",
                                      self.feature_names)
                vector = _row_vector(row, active_f,
                                     bundle["missing_imputation"])
                local = {"trees": bundle["trees"],
                         "aggregation": bundle["aggregation"],
                         "base_value": bundle["base_value"]}
                return _predict_body(local, vector), _leaf_of(local, vector)
            vector = _row_vector(row, body.get("active_feature_names",
                                               self.feature_names),
                                 imputation)
            return _predict_body(body, vector), _leaf_of(body, vector)

        report = {
            "inactive_features": inactive,
            "training": self._task_metrics(task, prepared_train, y_field,
                                           predict_local),
            "validation": self._task_metrics(task, self._attach_targets(
                validation_rows, y_field) if validation_rows else [],
                y_field, predict_local),
            "limitations": [
                "offline estimates reuse supplied outcomes; they are not new games",
                "support counts are independent episodes, not rows or candidates",
                "anchor gains cannot be summed into a multi-intervention policy gain",
                "frozen-context models are not deployable after source or config changes",
            ],
        }
        tree_count, depth = self._tree_stats(body,
                                             body.get("families")
                                             if group_by_family else None)
        report["capacity_used"] = tree_count
        report["max_depth_used"] = depth
        return body, report

    @staticmethod
    def _attach_targets(rows: List[Dict[str, Any]], y_field: str
                        ) -> List[Dict[str, Any]]:
        out = []
        for row in rows:
            target = row.get(y_field)
            if target is None:
                continue
            row = dict(row)
            row["_target"] = float(target)
            out.append(row)
        return out

    def _task_metrics(self, task, rows, y_field, predict_local) -> Dict[str, Any]:
        """Per-independent-episode metrics (equal episode mass)."""
        if not rows:
            return {"n_rows": 0}
        by_group: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            by_group.setdefault(row["group_id"], []).append(row)

        def group_metrics(group: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
            errors, abs_errors = [], []
            supported = 0
            # point grouping for top-k and ranking metrics
            points: Dict[str, List[Dict[str, Any]]] = {}
            for row in group:
                prediction, leaf = predict_local(row)
                supported += 1 if leaf and leaf.get("supported") else 0
                errors.append((prediction - row["_target"]) ** 2)
                abs_errors.append(abs(prediction - row["_target"]))
                points.setdefault(row["decision_key"], []).append(
                    (prediction, row))
            top1 = topk_total = topk_hit = teacher_rows = 0
            positive = negative = 0
            pos_hit = neg_hit = 0
            for _, members in points.items():
                members_sorted = sorted(members, key=lambda m: -m[0])
                teacher_id = next((m[1]["candidate_id"] for m in members
                                   if m[1].get("teacher") == 1.0), None)
                predicted_ids = [m[1]["candidate_id"] for m in members_sorted]
                if teacher_id is not None:
                    teacher_rows += 1
                    if predicted_ids and predicted_ids[0] == teacher_id:
                        top1 += 1
                    if teacher_id in predicted_ids[:3]:
                        topk_hit += 1
                    topk_total += 1
                if task == "gain":
                    for prediction, row in members:
                        actual = row["_target"]
                        if actual > 0:
                            positive += 1
                            pos_hit += prediction > 0
                        elif actual < 0:
                            negative += 1
                            neg_hit += prediction < 0
            return {
                "n_rows": len(group),
                "mse": math.fsum(errors) / len(errors) if errors else None,
                "supported_leaf_fraction": supported / len(group),
                **({"teacher_top1_agreement": top1 / topk_total
                    if topk_total else None,
                    "teacher_top3_agreement": topk_hit / topk_total
                    if topk_total else None} if task == "imitation" else {}),
                **({"positive_recognition": pos_hit / positive
                    if positive else None,
                    "negative_recognition": neg_hit / negative
                    if negative else None} if task == "gain" else {}),
            }

        groups = [by_group[key] for key in sorted(by_group)]
        metrics = [group_metrics(group) for group in groups]

        def weighted(field: str) -> Optional[float]:
            values = [(m[field], m["n_rows"]) for m in metrics
                      if m.get(field) is not None]
            if not values:
                return None
            return math.fsum(v for v, _ in values) / len(values)

        result = {"n_rows": sum(m["n_rows"] for m in metrics),
                  "n_scenarios": len(metrics)}
        for key in ("mse", "supported_leaf_fraction", "teacher_top1_agreement",
                    "teacher_top3_agreement", "positive_recognition",
                    "negative_recognition"):
            result[key] = weighted(key)
        result["per_scenario"] = [
            {"group_id": group_id, **metric}
            for group_id, metric in zip(sorted(by_group), metrics)]
        if task == "imitation":
            result["rule_score_correlation"] = self._rule_score_correlation(rows)
        if task == "gain":
            result["mae"] = self._per_scenario_mae(rows, predict_local)
        return result

    @staticmethod
    def _per_scenario_mae(rows, predict_local) -> Optional[float]:
        by_group: Dict[str, List[float]] = {}
        for row in rows:
            prediction, _ = predict_local(row)
            by_group.setdefault(row["group_id"], []).append(
                abs(prediction - row["_target"]))
        if not by_group:
            return None
        return math.fsum(math.fsum(v) / len(v)
                         for v in by_group.values()) / len(by_group)

    @staticmethod
    def _rule_score_correlation(rows) -> Optional[float]:
        """Pearson correlation between the declared rule score and the
        teacher label across candidate rows."""
        xs: List[float] = []
        ys: List[float] = []
        for row in rows:
            rule = row.get("rule_score")
            teacher = row.get("teacher")
            if rule is None or teacher is None:
                continue
            if not math.isfinite(float(rule)):
                continue
            xs.append(float(rule))
            ys.append(float(teacher))
        if len(xs) < 2:
            return None
        mean_x = math.fsum(xs) / len(xs)
        mean_y = math.fsum(ys) / len(ys)
        cov = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        var_x = math.fsum((x - mean_x) ** 2 for x in xs)
        var_y = math.fsum((y - mean_y) ** 2 for y in ys)
        if var_x <= 0 or var_y <= 0:
            return None
        return cov / math.sqrt(var_x * var_y)

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


def load_scorer_model(path: str):
    """Load + validate a scorer model file into the runtime model wrapper."""
    from .scorer import ScorerModel
    from .schema import load_json
    return ScorerModel(load_json(path))


def load_ranking_model(path: str):
    """Load + validate a pairwise ranking model file."""
    from .scorer import PairwiseRankModel
    from .schema import load_json
    return PairwiseRankModel(load_json(path))
