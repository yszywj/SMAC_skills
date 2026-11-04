"""Supervised distillation of the shared medivac pair scorer (v1).

Default model: one GradientBoostingRegressor shared by all medivacs,
initial hyperparameters 64 trees / depth 3 / lr 0.05 / squared error /
fixed seed, min_samples_leaf starting at 20 pending sample-size audit.
First-round model search is capped at n_estimators {32,64,128} x
max_depth {2,3}; 192 trees is a later capacity ablation, not a default.

Runs in a SEPARATE training environment (scikit-learn); never inside the
SMACv2 adapter venv. Validation splits by episode, never by row.
"""

import argparse
import json
import os
import statistics

DEFAULT_PARAMS = {"n_estimators": 64, "max_depth": 3, "learning_rate": 0.05,
                  "loss": "squared_error", "min_samples_leaf": 20, "random_state": 1708}
SEARCH_GRID = [{"n_estimators": n, "max_depth": d}
               for n in (32, 64, 128) for d in (2, 3)]


def load_groups(path):
    groups = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                groups.append(json.loads(line))
    return groups


def split_by_episode(groups, valid_episodes):
    """Group-level split: every candidate row of an episode stays together."""
    train = [g for g in groups if g["episode_id"] not in valid_episodes]
    valid = [g for g in groups if g["episode_id"] in valid_episodes]
    return train, valid


def rows_and_targets(groups, schema_size=None):
    rows, targets, weights = [], [], []
    for group in groups:
        # Per-group weight normalization: a group with many candidates gets
        # the same total weight as a group with one candidate.
        w = 1.0 / len(group["labels"])
        for target_id, label in group["labels"].items():
            row = group["features"][target_id]
            if schema_size is not None and len(row) != schema_size:
                raise ValueError(f"row width {len(row)} != schema {schema_size}")
            rows.append(row)
            targets.append(label)
            weights.append(w)
    return rows, targets, weights


def fit(train_groups, params=None):
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(**{**DEFAULT_PARAMS, **(params or {})})
    rows, targets, weights = rows_and_targets(train_groups)
    model.fit(rows, targets, sample_weight=weights)
    return model


def joint_groups(groups):
    """Merge per-healer groups into joint decision groups keyed by
    (episode_id, game_loop): all participating healers of one step."""
    merged = {}
    for group in groups:
        key = (group["episode_id"], group["game_loop"])
        merged.setdefault(key, []).append(group)
    return merged


def joint_assignment_agreement(model, groups):
    """Deployment-faithful joint matching agreement.

    For every (episode, game_loop) step: build edges from teacher labels and
    from student predictions, run the SAME forced participation matching the
    deployment path uses (every participating healer picks exactly one
    target), and compare final assignments. Top-1-per-healer equality is NOT
    the same quantity and is reported separately.
    """
    from tree_skills.medivac.matching import forced_matching
    counts = {scope: dict(steps=0, agree=0, healers=0, changed=0, infeasible=0)
              for scope in ("all", "nontrivial")}
    steps = joint_groups(groups)
    nontrivial_steps = 0
    for key, step_groups in sorted(steps.items()):
        # Filter STEPS, never healers: a single-candidate healer can reserve
        # a target and change the joint optimum for a multi-candidate healer.
        nontrivial = any(len(g["labels"]) > 1 for g in step_groups)
        nontrivial_steps += int(nontrivial)
        teacher_edges, student_edges, healers = {}, {}, []
        for group in step_groups:
            healer = group["healer_id"]
            if healer in teacher_edges:
                raise ValueError(f"duplicate healer {healer} in decision step {key}")
            healers.append(healer)
            teacher_edges[healer] = sorted(
                ((label, t) for t, label in group["labels"].items()),
                key=lambda r: (-r[0], r[1]))
            student_edges[healer] = sorted(
                ((float(model.predict([group["features"][t]])[0]), t)
                 for t in group["labels"]),
                key=lambda r: (-r[0], r[1]))
        teacher_match = forced_matching(healers, teacher_edges)
        student_match = forced_matching(healers, student_edges)
        for scope in (("all", "nontrivial") if nontrivial else ("all",)):
            c = counts[scope]
            if teacher_match is None or student_match is None:
                c["infeasible"] += 1
                continue  # Two infeasible assignments do not count as agreement.
            c["steps"] += 1
            c["agree"] += int(teacher_match == student_match)
            c["healers"] += len(healers)
            c["changed"] += sum(teacher_match[h] != student_match[h] for h in healers)
    result = {"n_joint_steps": len(steps),
              "n_nontrivial_joint_steps": nontrivial_steps,
              "single_candidate_step_rate": (1 - nontrivial_steps / len(steps)) if steps else None}
    for scope, c in counts.items():
        result.update({
            f"{scope}_n_feasible_joint_steps": c["steps"],
            f"{scope}_n_infeasible_joint_steps": c["infeasible"],
            f"{scope}_n_participating_healers": c["healers"],
            f"{scope}_joint_assignment_agreement": c["agree"] / c["steps"] if c["steps"] else None,
            f"{scope}_healer_assignment_agreement": 1 - c["changed"] / c["healers"] if c["healers"] else None})
    # Compatibility aliases are explicitly ALL-group diagnostics, not selection criteria.
    result["joint_assignment_agreement"] = result["all_joint_assignment_agreement"]
    result["healer_assignment_agreement"] = result["all_healer_assignment_agreement"]
    return result


def _ranking_metrics(model, groups):
    counts = {scope: dict(groups=0, correct=0, ties=0, errors=[], regrets=[])
              for scope in ("all", "multi_candidate")}
    for group in groups:
        labels, feats = group["labels"], group["features"]
        if not labels:
            raise ValueError("empty candidate group cannot be evaluated")
        pred = {t: float(model.predict([feats[t]])[0]) for t in labels}
        teacher_best = max(labels.values())
        teacher_top = {t for t, v in labels.items() if v == teacher_best}
        # Same ordering as deployment edges: score descending, target id ascending.
        pred_top = min(pred, key=lambda t: (-pred[t], t))
        scopes = ("all", "multi_candidate") if len(labels) > 1 else ("all",)
        for scope in scopes:
            c = counts[scope]
            c["groups"] += 1
            c["ties"] += int(len(teacher_top) > 1)
            c["correct"] += int(len(teacher_top) == 1 and pred_top in teacher_top)
            c["errors"].extend(abs(pred[t] - labels[t]) for t in labels)
            c["regrets"].append(teacher_best - labels[pred_top])
    multi = [g for g in groups if len(g["labels"]) > 1]
    result = {"metrics_version": 2,
              "selection_metric": "multi_candidate_top1_agreement",
              "n_groups": len(groups), "n_multi_candidate_groups": len(multi),
              "n_single_candidate_groups": len(groups) - len(multi),
              "n_episodes": len({g["episode_id"] for g in groups}),
              "n_multi_candidate_episodes": len({g["episode_id"] for g in multi}),
              "n_candidate_rows": sum(len(g["labels"]) for g in groups),
              "single_candidate_group_rate": (len(groups) - len(multi)) / len(groups) if groups else None}
    for scope, c in counts.items():
        decided = c["groups"] - c["ties"]
        result.update({
            f"{scope}_n_tied_groups": c["ties"],
            f"{scope}_n_top1_evaluable_groups": decided,
            f"{scope}_top1_agreement": c["correct"] / decided if decided else None,
            f"{scope}_mae": statistics.mean(c["errors"]) if c["errors"] else None,
            # Regret includes teacher ties: choosing any optimal target costs 0.
            f"{scope}_mean_teacher_regret": statistics.mean(c["regrets"]) if c["regrets"] else None})
    result.update(joint_assignment_agreement(model, groups))
    result.update({"top1_agreement": result["all_top1_agreement"],
                   "n_tied_groups": result["all_n_tied_groups"],
                   "mae": result["all_mae"]})
    return result


class ConstantZeroScorer:
    """Untrained control that exposes agreement inflated by forced choices."""

    def predict(self, rows):
        return [0.0] * len(rows)


def evaluate_ranking(model, groups):
    """Report informative decisions separately from trivial forced choices.

    Selection uses unique-teacher-best MULTI-candidate groups only. Joint
    metrics retain all healers within a selected nontrivial decision step.
    Legacy unprefixed metrics remain all-group diagnostics for old callers.
    """
    result = _ranking_metrics(model, groups)
    result["constant_zero_baseline"] = _ranking_metrics(ConstantZeroScorer(), groups)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shard", required=True)
    parser.add_argument("--valid-episodes", required=True,
                        help="JSON list of episode ids reserved for validation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--search", action="store_true",
                        help="run the capped first-round grid; else fit defaults")
    args = parser.parse_args(argv)
    if os.path.isdir(args.output_dir) and os.listdir(args.output_dir):
        raise SystemExit(f"refusing to overwrite non-empty {args.output_dir}")

    groups = load_groups(args.train_shard)
    with open(args.valid_episodes, encoding="utf-8") as fh:
        valid_episodes = set(json.load(fh))
    train, valid = split_by_episode(groups, valid_episodes)
    if not valid:
        raise SystemExit("validation split is empty; refusing to fit")
    if not train:
        raise SystemExit("training split is empty; refusing to fit")
    if not any(len(g["labels"]) > 1 and
               list(g["labels"].values()).count(max(g["labels"].values())) == 1
               for g in valid):
        raise SystemExit("validation has no multi-candidate groups with a unique teacher best; refusing model selection")

    candidates = SEARCH_GRID if args.search else [None]
    best = None
    report = []
    for params in candidates:
        model = fit(train, params)
        metrics = evaluate_ranking(model, valid)
        report.append({"params": params or DEFAULT_PARAMS, **metrics})
        score = metrics["multi_candidate_top1_agreement"]
        if score is not None and (best is None or score > best[0]):
            best = (score, params, model)
    score, params, model = best
    os.makedirs(args.output_dir, exist_ok=True)
    from ..export.exporter import export_gbr
    from ..medivac.features import FEATURE_NAMES, SCHEMA_VERSION
    export_gbr(model, FEATURE_NAMES, SCHEMA_VERSION,
               os.path.join(args.output_dir, "model.json"),
               meta_extra={"source_shard": os.path.abspath(args.train_shard),
                           "metrics_version": 2,
                           "selection_metric": "multi_candidate_top1_agreement",
                           "search_report": report, "selected_params": params or DEFAULT_PARAMS})
    with open(os.path.join(args.output_dir, "train_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps({"selected": params or DEFAULT_PARAMS,
                      "selection_metric": "multi_candidate_top1_agreement",
                      "valid_multi_candidate_top1": score,
                      "report": report}, indent=1))


if __name__ == "__main__":
    main()
