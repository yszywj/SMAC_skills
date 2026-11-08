"""Fit shallow multi-output gain trees from complete paired skill invocations.

No teacher network or training dependency is used. The label for each candidate
is its whole-episode return minus BASE under the frozen comparison protocol.
Every scenario has total weight one, and all its seeds stay in the same split.
Offline selection estimates reuse the supplied outcomes; they are not new games.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

EVIDENCE_KIND = "matched_observed_prefix"


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def names(value, name):
    if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x for x in value):
        raise ValueError(f"{name} must be a nonempty list of names")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} contains duplicate names")
    return value


def validate_rows(rows, candidate_ids, feature_names):
    """Reject incomplete arms, missing observations, and repeated evidence."""
    names(candidate_ids, "candidate_ids")
    names(feature_names, "feature_names")
    if not isinstance(rows, list) or not rows:
        raise ValueError("No complete accepted comparisons; collect real paired games first")
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("evidence_kind") != EVIDENCE_KIND:
            raise ValueError(f"row {index}: audited matched observed-prefix evidence is required")
        group = row.get("group_id")
        if not isinstance(group, str) or not group or type(row.get("sc2_seed")) is not int:
            raise ValueError(f"row {index}: scenario group_id and integer sc2_seed are required")
        key = (group, row["sc2_seed"])
        if key in seen:
            raise ValueError("duplicate scenario/seed pair; duplicate files are not new evidence")
        seen.add(key)
        features = row.get("features")
        if not isinstance(features, dict) or set(features) != set(feature_names):
            raise ValueError(f"row {index}: features must exactly match the frozen feature schema")
        for feature in feature_names:
            finite(features[feature], f"row {index} feature {feature}")
        base = finite(row.get("base_return"), f"row {index} base_return")
        for field in ("candidate_returns", "gains"):
            if not isinstance(row.get(field), dict) or set(row[field]) != set(candidate_ids):
                raise ValueError(f"row {index}: {field} needs every candidate; unavailable is not zero")
        for candidate in candidate_ids:
            value = finite(row["candidate_returns"][candidate], f"row {index} return {candidate}")
            gain = finite(row["gains"][candidate], f"row {index} gain {candidate}")
            if not math.isclose(gain, value - base, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"row {index}: gain must equal candidate return minus BASE")
        if not isinstance(row.get("gate"), dict) or not row["gate"]:
            raise ValueError(f"row {index}: the matched decision gate must be recorded")
    return rows


def load_datasets(paths, *, purpose="train"):
    """Merge only one frozen protocol, without reusing any scenario/seed pair."""
    from decision_tree.features import FEATURE_NAMES
    from decision_tree.schema import load_policy

    if purpose not in ("train", "validation"):
        raise ValueError("dataset purpose must be train or validation")
    rows, files, reference = [], [], None
    for value in paths:
        path = Path(value)
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8-sig"))
        if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
                or data["schema_version"] != 1 or data.get("kind") != "skill_call_pairs"):
            raise ValueError(f"{path}: unsupported dataset schema")
        allowed_splits = (None, "train", "development") if purpose == "train" else (None, "train", "development", "validation")
        if data.get("split") not in allowed_splits:
            raise ValueError(f"{path}: sealed {data.get('split')} data cannot be used for {purpose}")
        game_version = data.get("game_version")
        if not isinstance(game_version, str) or not game_version.strip():
            raise ValueError("dataset must record the actual nonempty SC2 game_version")
        candidate_ids = names(data.get("candidate_ids"), "candidate_ids")
        feature_names = names(data.get("feature_names"), "feature_names")
        if set(feature_names) != set(FEATURE_NAMES):
            raise ValueError("dataset features differ from the public pre-action feature allowlist")
        context = data.get("context")
        if not isinstance(context, dict) or not context or json_hash(context) != data.get("context_hash"):
            raise ValueError("dataset context fingerprint mismatch")
        contract = data.get("policy_contract")
        if not isinstance(contract, dict):
            raise ValueError("dataset must include its frozen policy_contract")
        base = deepcopy(contract)
        # Dataset contracts omit the arm-specific selection; tolerate an explicit
        # BASE selection, but never erase a recorded candidate/model selection.
        if "selection" in base and base["selection"].get("mode") != "base":
            raise ValueError("policy_contract must omit selection or contain only BASE selection")
        base.setdefault("selection", {"mode": "base", "max_interventions_per_episode": 1})
        policy = load_policy(base)
        if [item["id"] for item in policy["candidates"]] != candidate_ids:
            raise ValueError("dataset candidates differ from the frozen policy")
        if policy["selection"].get("max_interventions_per_episode", 1) != 1:
            raise ValueError("initial gain training requires exactly one opportunity per episode")
        current = {"context": context, "context_hash": data["context_hash"], "game_version": game_version,
                   "policy": policy, "candidate_ids": candidate_ids, "feature_names": feature_names}
        if reference is not None and current != reference:
            raise ValueError("datasets have different contexts, SC2 game versions, candidates, policies or feature schemas")
        reference = current
        if not isinstance(data.get("rows"), list) or not isinstance(data.get("rejected", []), list):
            raise ValueError("dataset rows and rejected entries must be lists")
        rows.extend(data["rows"])
        files.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                      "accepted_rows": len(data["rows"]), "rejected_pairs": len(data.get("rejected", []))})
    if reference is None:
        raise ValueError("at least one paired dataset is required")
    validate_rows(rows, reference["candidate_ids"], reference["feature_names"])
    return rows, files, reference


def group_split(rows, *, validation_fraction=.25, split_seed=0, min_train_groups=4):
    fraction = finite(validation_fraction, "validation_fraction")
    if not 0 < fraction < 1 or type(split_seed) is not int:
        raise ValueError("validation fraction must be between 0 and 1; split seed must be an integer")
    positive_int(min_train_groups, "min_train_groups")
    groups = sorted({row["group_id"] for row in rows},
                    key=lambda group: (hashlib.sha256(f"{split_seed}\0{group}".encode()).hexdigest(), group))
    count = max(1, math.ceil(len(groups) * fraction))
    if len(groups) - count < min_train_groups:
        raise ValueError("not enough independent scenarios for training and validation")
    validation_groups = set(groups[:count])
    return ([r for r in rows if r["group_id"] not in validation_groups],
            [r for r in rows if r["group_id"] in validation_groups])


def group_weights(rows):
    counts = {}
    for row in rows:
        counts[row["group_id"]] = counts.get(row["group_id"], 0) + 1
    return [1. / counts[row["group_id"]] for row in rows]


def fit_gain_tree(rows, candidate_ids, feature_names, *, context_hash,
                  max_depth=2, min_samples_leaf=4, min_leaf_groups=4):
    """Greedy multi-output CART; each leaf needs independent scenario support."""
    validate_rows(rows, candidate_ids, feature_names)
    if type(max_depth) is not int or not 0 <= max_depth <= 2:
        raise ValueError("max_depth must be 0, 1 or 2 for the initial protocol")
    positive_int(min_samples_leaf, "min_samples_leaf")
    positive_int(min_leaf_groups, "min_leaf_groups")
    weights = group_weights(rows)

    def supported(indices):
        return (len(indices) >= min_samples_leaf
                and len({rows[i]["group_id"] for i in indices}) >= min_leaf_groups)

    def means(indices):
        denominator = math.fsum(weights[i] for i in indices)
        return {c: math.fsum(weights[i] * rows[i]["gains"][c] for i in indices) / denominator
                for c in candidate_ids}

    def error(indices):
        averages = means(indices)
        return math.fsum(weights[i] * (rows[i]["gains"][c] - averages[c]) ** 2
                         for c in candidate_ids for i in indices)

    indices = list(range(len(rows)))
    if not supported(indices):
        raise ValueError("training data do not meet minimum leaf sample/scenario coverage")

    def build(indices, depth):
        parent_error = error(indices)
        best_error, best = parent_error, None
        tolerance = 1e-12 * max(1., parent_error)
        if depth < max_depth:
            for feature in feature_names:
                values = sorted({float(rows[i]["features"][feature]) for i in indices})
                for low, high in zip(values, values[1:]):
                    threshold = low / 2 + high / 2
                    if threshold >= high:
                        threshold = low
                    left = [i for i in indices if rows[i]["features"][feature] <= threshold]
                    right = [i for i in indices if rows[i]["features"][feature] > threshold]
                    if not supported(left) or not supported(right):
                        continue
                    candidate_error = error(left) + error(right)
                    if candidate_error < best_error - tolerance:
                        best, best_error = (feature, threshold, left, right), candidate_error
        if best is None:
            groups = sorted({rows[i]["group_id"] for i in indices})
            return {"gains": means(indices), "n_samples": len(indices), "n_groups": len(groups),
                    "group_ids": groups, "weight_sum": math.fsum(weights[i] for i in indices)}
        feature, threshold, left, right = best
        return {"feature": feature, "threshold": threshold,
                "left": build(left, depth + 1), "right": build(right, depth + 1)}

    return {"schema_version": 1, "kind": "skill_gain_tree", "candidate_ids": list(candidate_ids),
            "feature_names": list(feature_names), "context_hash": context_hash, "root": build(indices, 0),
            "training": {"algorithm": "weighted_multioutput_squared_error_greedy_cart",
                         "weighting": "each_scenario_total_weight_one", "max_depth": max_depth,
                         "min_samples_leaf": min_samples_leaf, "min_leaf_groups": min_leaf_groups,
                         "target": "candidate_episode_return_minus_base_episode_return",
                         "evidence_kind": EVIDENCE_KIND}}


def tree_counts(node):
    if "gains" in node:
        return {"nodes": 1, "predicates": 0, "leaves": 1, "depth": 0}
    left, right = tree_counts(node["left"]), tree_counts(node["right"])
    return {"nodes": 1 + left["nodes"] + right["nodes"],
            "predicates": 1 + left["predicates"] + right["predicates"],
            "leaves": left["leaves"] + right["leaves"], "depth": 1 + max(left["depth"], right["depth"])}


def evaluate_paired_rows(model, rows, min_gain=0.):
    from decision_tree.policy import predict_gains
    if not rows:
        raise ValueError("evaluation needs complete paired rows")
    weights = group_weights(rows)
    total = math.fsum(weights)
    details, selected_gains = [], []
    for row in rows:
        predictions = predict_gains(model, row["features"])
        selected, highest = "BASE", max(0., min_gain)
        for candidate in model["candidate_ids"]:
            if predictions[candidate] > highest:
                selected, highest = candidate, predictions[candidate]
        gain = 0. if selected == "BASE" else row["gains"][selected]
        selected_gains.append(gain)
        details.append({"group_id": row["group_id"], "sc2_seed": row["sc2_seed"],
                        "predicted_gains": predictions, "actual_gains": row["gains"],
                        "selected": selected, "selected_empirical_gain": gain})

    def average(values):
        return math.fsum(w * v for w, v in zip(weights, values)) / total

    return {"scope": "offline_reuse_of_paired_outcomes_not_new_selector_games",
            "n_rows": len(rows), "n_groups": len({r["group_id"] for r in rows}),
            "weighting": "each_scenario_total_weight_one", "base_mean_gain": 0.,
            "selector_mean_gain": average(selected_gains),
            "always_candidate_mean_gain": {c: average([r["gains"][c] for r in rows])
                                           for c in model["candidate_ids"]},
            "prediction_mse": {c: average([(r["gains"][c] - d["predicted_gains"][c]) ** 2
                                            for r, d in zip(rows, details)]) for c in model["candidate_ids"]},
            "selection_counts": {c: sum(d["selected"] == c for d in details)
                                 for c in ["BASE", *model["candidate_ids"]]},
            "target_sign_counts": {c: {"positive": sum(r["gains"][c] > 0 for r in rows),
                                       "negative": sum(r["gains"][c] < 0 for r in rows),
                                       "zero": sum(r["gains"][c] == 0 for r in rows)}
                                   for c in model["candidate_ids"]}, "rows": details}


def train_policy(dataset_paths, *, validation_paths=None, validation_fraction=.25,
                 split_seed=0, max_depth=2, min_samples_leaf=4, min_leaf_groups=4):
    """Return policy/model/report without creating any filesystem artifacts."""
    from decision_tree.schema import load_policy, validate_model

    rows, inputs, protocol = load_datasets(dataset_paths)
    if validation_paths:
        validation, validation_inputs, other = load_datasets(validation_paths, purpose="validation")
        if protocol != other:
            raise ValueError("training and validation protocols differ")
        if {r["group_id"] for r in rows} & {r["group_id"] for r in validation}:
            raise ValueError("training and validation scenario groups overlap")
        train, split = rows, {"method": "explicit_validation_files"}
    else:
        train, validation = group_split(rows, validation_fraction=validation_fraction,
                                        split_seed=split_seed, min_train_groups=min_leaf_groups)
        validation_inputs = []
        split = {"method": "sha256_scenario_group_split", "seed": split_seed,
                 "validation_fraction": validation_fraction}
    model = fit_gain_tree(train, protocol["candidate_ids"], protocol["feature_names"],
                          context_hash=protocol["context_hash"], max_depth=max_depth,
                          min_samples_leaf=min_samples_leaf, min_leaf_groups=min_leaf_groups)
    train_groups = sorted({r["group_id"] for r in train})
    validation_groups = sorted({r["group_id"] for r in validation})
    split.update(train_groups=train_groups, validation_groups=validation_groups)
    model["context"] = deepcopy(protocol["context"])
    model["game_version"] = protocol["game_version"]
    model["training"].update(train_groups=train_groups, validation_groups=validation_groups,
                             split=split, inputs=inputs, validation_inputs=validation_inputs,
                             rows_sha256=json_hash(train))
    validate_model(model)
    policy = deepcopy(protocol["policy"])
    min_gain = policy["selection"].get("min_gain", 0.)
    policy["selection"] = {"mode": "model", "max_interventions_per_episode": 1,
                           "min_gain": min_gain, "model": model}
    policy = load_policy(policy)
    report = {"schema_version": 1, "status": "trained_not_game_validated",
              "evidence_kind": EVIDENCE_KIND, "context": deepcopy(protocol["context"]),
              "game_version": protocol["game_version"],
              "context_hash": protocol["context_hash"], "dataset_inputs": inputs,
              "validation_inputs": validation_inputs, "split": split, "tree": tree_counts(model["root"]),
              "model_sha256": json_hash(model), "policy_sha256": json_hash(policy),
              "training": evaluate_paired_rows(model, train, min_gain),
              "validation": evaluate_paired_rows(model, validation, min_gain),
              "limitations": ["Observed-prefix matching is not a restored SC2 internal snapshot.",
                              "Offline metrics reuse the supplied complete candidate outcomes, not new policy games.",
                              "Training targets describe one skill-call intervention per episode.",
                              "No-opportunity/rejected episodes are excluded from labels; report them in full-game evaluation.",
                              "Freeze this model and run disjoint held-out games before claiming improvement.",
                              "Validation used for redesign becomes development evidence; reserve a final test."]}
    return policy, model, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--validation-dataset", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=.25)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-depth", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
    parser.add_argument("--min-leaf-groups", type=int, default=4)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new version")
    try:
        policy, model, report = train_policy(args.dataset, validation_paths=args.validation_dataset,
            validation_fraction=args.validation_fraction, split_seed=args.split_seed,
            max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
            min_leaf_groups=args.min_leaf_groups)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (("policy.json", policy), ("model.json", model), ("report.json", report)):
        with (args.output_dir / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "status": report["status"],
                      "tree": report["tree"], "train_rows": report["training"]["n_rows"],
                      "validation_rows": report["validation"]["n_rows"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
