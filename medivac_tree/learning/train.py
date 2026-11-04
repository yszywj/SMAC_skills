"""Fit a small, auditable return tree from accepted paired game experiments.

No teacher or training dependency is required. Splits minimize weighted squared
error; each scenario has total weight one so additional seeds do not silently
turn one scenario into many independent maps. Validation uses separate scenario
groups. Its numbers describe the supplied paired experiments, not fresh games.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from medivac_tree.candidates import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from medivac_tree.experiment import context_hash, validate_context
from medivac_tree.selector import predict_gain, validate_model, validate_policy


EVIDENCE_KIND = "matched_observed_prefix"


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def validate_rows(rows, contract_hash):
    """Reject leakage, inconsistent targets and reused scenario/seed pairs."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("accepted paired rows are required; collect real paired games first")
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} must be an object")
        if row.get("evidence_kind") != EVIDENCE_KIND:
            raise ValueError(f"row {index}: only matched observed-prefix evidence is supported")
        if row.get("comparison_contract_hash") != contract_hash:
            raise ValueError(f"row {index}: comparison contract mismatch")
        group = row.get("group_id")
        if not isinstance(group, str) or not group:
            raise ValueError(f"row {index}: scenario group_id is required")
        if type(row.get("sc2_seed")) is not int:
            raise ValueError(f"row {index}: sc2_seed must be an integer")
        pair_key = (group, row["sc2_seed"])
        if pair_key in seen:
            raise ValueError("duplicate scenario/seed pair; repeated files or games are not independent evidence")
        seen.add(pair_key)
        features = row.get("features")
        if not isinstance(features, dict) or set(features) != set(FEATURE_NAMES):
            raise ValueError(f"row {index}: features must match the fixed pre-action feature allowlist")
        for name in FEATURE_NAMES:
            _finite(features[name], f"row {index} feature {name}")
        base = _finite(row.get("base_return"), f"row {index} base_return")
        candidate = _finite(row.get("candidate_return"), f"row {index} candidate_return")
        gain = _finite(row.get("target_gain"), f"row {index} target_gain")
        if not math.isclose(gain, candidate - base, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"row {index}: target_gain must equal candidate_return - base_return")
    return rows


def load_datasets(paths, contract_hash):
    rows, inputs, context = [], [], None
    for value in paths:
        path = Path(value)
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise ValueError(f"{path}: unsupported dataset schema")
        if data.get("evidence_kind") != EVIDENCE_KIND or data.get("comparison_contract_hash") != contract_hash:
            raise ValueError(f"{path}: evidence kind or comparison contract mismatch")
        if data.get("feature_schema") != {"version": FEATURE_SCHEMA_VERSION, "names": list(FEATURE_NAMES)}:
            raise ValueError(f"{path}: feature schema mismatch")
        if not isinstance(data.get("rows"), list) or not isinstance(data.get("rejected", []), list):
            raise ValueError(f"{path}: rows and rejected must be arrays")
        current_context = validate_context(data.get("experiment_context"), data.get("experiment_context_hash"))
        if context is not None and context != current_context:
            raise ValueError("datasets have different experiment contexts; ground controllers and environments cannot be mixed")
        context = current_context
        rows.extend(data["rows"])
        inputs.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                       "accepted_rows": len(data["rows"]), "rejected_pairs": len(data.get("rejected", []))})
    validate_rows(rows, contract_hash)
    return rows, inputs, context


def group_split(rows, *, validation_fraction=.25, split_seed=0, min_train_groups=4):
    """Stable group-level partition; a scenario's seeds stay on the same side."""
    if isinstance(validation_fraction, bool) or not isinstance(validation_fraction, (float, int)) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if type(split_seed) is not int:
        raise ValueError("split_seed must be an integer")
    _positive_int(min_train_groups, "min_train_groups")
    groups = sorted({row["group_id"] for row in rows},
                    key=lambda group: (hashlib.sha256(f"{split_seed}\0{group}".encode()).hexdigest(), group))
    count = max(1, math.ceil(len(groups) * validation_fraction))
    if len(groups) - count < min_train_groups:
        raise ValueError("not enough independent scenarios for training and validation; collect more paired games")
    validation_groups = set(groups[:count])
    train = [row for row in rows if row["group_id"] not in validation_groups]
    validation = [row for row in rows if row["group_id"] in validation_groups]
    return train, validation


def _group_weights(rows):
    counts = {}
    for row in rows:
        counts[row["group_id"]] = counts.get(row["group_id"], 0) + 1
    return [1. / counts[row["group_id"]] for row in rows]


def fit_gain_tree(rows, contract_hash, *, max_depth=2, min_samples_leaf=4, min_leaf_groups=4):
    """Greedy shallow CART with group coverage checked at every proposed leaf."""
    validate_rows(rows, contract_hash)
    if type(max_depth) is not int or not 0 <= max_depth <= 2:
        raise ValueError("max_depth must be 0, 1 or 2 for this initial protocol")
    _positive_int(min_samples_leaf, "min_samples_leaf")
    _positive_int(min_leaf_groups, "min_leaf_groups")
    weights = _group_weights(rows)
    indices = list(range(len(rows)))
    leaves = {}

    def supported(part):
        return len(part) >= min_samples_leaf and len({rows[i]["group_id"] for i in part}) >= min_leaf_groups

    if not supported(indices):
        raise ValueError("training data do not meet minimum leaf row/scenario coverage")

    def mean(part):
        return math.fsum(weights[i] * rows[i]["target_gain"] for i in part) / math.fsum(weights[i] for i in part)

    def error(part):
        average = mean(part)
        return math.fsum(weights[i] * (rows[i]["target_gain"] - average) ** 2 for i in part)

    def build(part, depth, identifier):
        parent_error = error(part)
        best = None
        best_error = parent_error
        tolerance = 1e-12 * max(1., parent_error)
        if depth < max_depth:
            for feature in FEATURE_NAMES:
                values = sorted({float(rows[i]["features"][feature]) for i in part})
                for low, high in zip(values, values[1:]):
                    threshold = low / 2 + high / 2
                    if threshold >= high:  # Adjacent floats may round up.
                        threshold = low
                    yes = [i for i in part if rows[i]["features"][feature] <= threshold]
                    no = [i for i in part if rows[i]["features"][feature] > threshold]
                    if not supported(yes) or not supported(no):
                        continue
                    candidate_error = error(yes) + error(no)
                    if candidate_error < best_error - tolerance:
                        best_error = candidate_error
                        best = feature, threshold, yes, no
        if best is None:
            groups = sorted({rows[i]["group_id"] for i in part})
            leaves[identifier] = {"n_samples": len(part), "n_groups": len(groups), "group_ids": groups,
                                  "weight_sum": math.fsum(weights[i] for i in part)}
            return {"id": identifier, "gain": mean(part), "n_samples": len(part)}
        feature, threshold, yes, no = best
        return {"id": identifier, "feature": feature, "op": "<=", "threshold": threshold,
                "yes": build(yes, depth + 1, identifier + "y"),
                "no": build(no, depth + 1, identifier + "n")}

    root = build(indices, 0, "r")
    model = {"schema_version": 1, "model_kind": "gain_regression_tree",
             "feature_schema_version": FEATURE_SCHEMA_VERSION, "feature_names": list(FEATURE_NAMES),
             "comparison_contract_hash": contract_hash, "root": root,
             "training": {"algorithm": "weighted_squared_error_greedy_cart",
                          "weighting": "each_training_scenario_total_weight_one",
                          "max_depth": max_depth, "min_samples_leaf": min_samples_leaf,
                          "min_leaf_groups": min_leaf_groups, "leaves": leaves,
                          "target": "candidate_return_minus_base_return",
                          "evidence_kind": EVIDENCE_KIND}}
    validate_model(model, contract_hash)
    return model


def evaluate_paired_rows(model, rows):
    """Report empirical paired-return comparisons, retaining zero/negative rows."""
    if not rows:
        raise ValueError("evaluation needs at least one paired row")
    predictions = [predict_gain(model, row["features"])[0] for row in rows]
    weights = _group_weights(rows)
    denominator = math.fsum(weights)
    actual = [row["target_gain"] for row in rows]
    learned = [gain if prediction > 0 else 0. for gain, prediction in zip(actual, predictions)]

    def weighted_mean(values):
        return math.fsum(w * value for w, value in zip(weights, values)) / denominator

    result = {"n_rows": len(rows), "n_groups": len({r["group_id"] for r in rows}),
            "weighting": "each_evaluation_scenario_total_weight_one",
            "base_mean_gain": 0., "always_candidate_mean_gain": weighted_mean(actual),
            "selector_mean_gain": weighted_mean(learned),
            "prediction_mse": weighted_mean([(a - p) ** 2 for a, p in zip(actual, predictions)]),
            "candidate_selections": sum(p > 0 for p in predictions),
            "target_sign_counts": {"negative": sum(g < 0 for g in actual), "zero": sum(g == 0 for g in actual),
                                   "positive": sum(g > 0 for g in actual)},
            "rows": [{"group_id": r["group_id"], "sc2_seed": r["sc2_seed"],
                      "target_gain": r["target_gain"], "predicted_gain": p,
                      "selected": "CANDIDATE" if p > 0 else "BASE",
                      "selected_empirical_gain": value}
                     for r, p, value in zip(rows, predictions, learned)]}
    outcomes = {"win", "loss", "draw", "timeout"}
    if all(row.get("base_outcome") in outcomes and row.get("candidate_outcome") in outcomes for row in rows):
        base_wins = [float(row["base_outcome"] == "win") for row in rows]
        candidate_wins = [float(row["candidate_outcome"] == "win") for row in rows]
        selected_wins = [c if prediction > 0 else b for b, c, prediction in zip(base_wins, candidate_wins, predictions)]
        result["win_metrics"] = {
            "available": True, "scope": "offline_reuse_of_paired_outcomes_not_new_selector_games",
            "base_win_rate": weighted_mean(base_wins), "always_candidate_win_rate": weighted_mean(candidate_wins),
            "selector_win_rate": weighted_mean(selected_wins),
            "selector_win_rate_delta": weighted_mean([s - b for s, b in zip(selected_wins, base_wins)]),
            "weighting": "each_evaluation_scenario_total_weight_one"}
    else:
        result["win_metrics"] = {"available": False, "reason": "complete_paired_outcomes_not_available"}
    return result


def train_policy(dataset_paths, policy_path, *, validation_paths=None, validation_fraction=.25,
                 split_seed=0, max_depth=2, min_samples_leaf=4, min_leaf_groups=4):
    """Build artifacts in memory; callers decide where to write a new version."""
    policy_path = Path(policy_path)
    policy_raw = policy_path.read_bytes()
    policy = json.loads(policy_raw.decode("utf-8-sig"))
    info = validate_policy(policy)
    if policy["selection"]["mode"] != "base":
        raise ValueError("--policy must be the frozen BASE arm envelope")
    contract_hash = info["comparison_contract_hash"]
    rows, input_files, context = load_datasets(dataset_paths, contract_hash)
    if validation_paths:
        validation, validation_files, validation_context = load_datasets(validation_paths, contract_hash)
        if context != validation_context:
            raise ValueError("training and validation experiment contexts differ")
        if {r["group_id"] for r in rows} & {r["group_id"] for r in validation}:
            raise ValueError("training and validation scenario groups overlap")
        train = rows
        split = {"method": "explicit_validation_files"}
    else:
        train, validation = group_split(rows, validation_fraction=validation_fraction,
                                        split_seed=split_seed, min_train_groups=min_leaf_groups)
        validation_files = []
        split = {"method": "sha256_scenario_group_split", "seed": split_seed,
                 "validation_fraction": validation_fraction}
    model = fit_gain_tree(train, contract_hash, max_depth=max_depth,
                          min_samples_leaf=min_samples_leaf, min_leaf_groups=min_leaf_groups)
    train_groups = sorted({r["group_id"] for r in train})
    validation_groups = sorted({r["group_id"] for r in validation})
    split.update(training_groups=train_groups, validation_groups=validation_groups)
    model["training"].update(split=split, inputs=input_files, validation_inputs=validation_files,
                             training_rows_sha256=_json_hash(train), experiment_context=deepcopy(context),
                             experiment_context_hash=context_hash(context))
    output_policy = deepcopy(policy)
    output_policy["version"] += "-gain-" + _json_hash(model)[:12]
    output_policy["selection"] = {"mode": "model", "model": model}
    validate_policy(output_policy)
    report = {"schema_version": 1, "status": "trained_not_game_validated",
              "comparison_contract_hash": contract_hash, "evidence_kind": EVIDENCE_KIND,
              "policy_input": {"path": str(policy_path.resolve()),
                               "sha256": hashlib.sha256(policy_raw).hexdigest()},
              "dataset_inputs": input_files, "validation_inputs": validation_files,
              "experiment_context": deepcopy(context), "experiment_context_hash": context_hash(context),
              "split": split, "training": evaluate_paired_rows(model, train),
              "validation": evaluate_paired_rows(model, validation),
              "limitations": ["Observed-prefix matching is not a restored SC2 internal snapshot.",
                              "Metrics reuse paired BASE/CANDIDATE returns; they are not new selector-policy games.",
                              "The selector acts only at the first eligible opportunity per episode.",
                              "Freeze the policy and evaluate it in new games before claiming an improvement.",
                              "Validation used for redesign becomes development evidence; reserve a separate final test."],
              "runtime_source_hashes": info.get("runtime_source_hashes", {})}
    return output_policy, model, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, type=Path,
                        help="Accepted paired dataset; repeat for separate seed batches")
    parser.add_argument("--policy", type=Path, required=True, help="Frozen BASE envelope JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not exist")
    parser.add_argument("--validation-dataset", action="append", type=Path,
                        help="Optional disjoint scenario groups; otherwise split whole groups")
    parser.add_argument("--validation-fraction", type=float, default=.25)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-depth", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
    parser.add_argument("--min-leaf-groups", type=int, default=4)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new version directory")
    try:
        policy, model, report = train_policy(
            args.dataset, args.policy, validation_paths=args.validation_dataset,
            validation_fraction=args.validation_fraction, split_seed=args.split_seed,
            max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
            min_leaf_groups=args.min_leaf_groups)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (("policy.json", policy), ("model.json", model), ("report.json", report)):
        (args.output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                           encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "status": report["status"],
                      "train_rows": report["training"]["n_rows"],
                      "validation_rows": report["validation"]["n_rows"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
