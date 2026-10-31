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
import random

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
    agree = total = 0
    changed_healers = 0
    all_healers = 0
    for key, step_groups in sorted(joint_groups(groups).items()):
        teacher_edges, student_edges, healers = {}, {}, []
        for group in step_groups:
            healer = group["healer_id"]
            healers.append(healer)
            teacher_edges[healer] = sorted(
                ((label, t) for t, label in group["labels"].items()),
                key=lambda r: (-r[0], r[1]))
            student_edges[healer] = sorted(
                ((float(model.predict([group["features"][t]])[0]), t)
                 for t in group["labels"]),
                key=lambda r: (-r[0], r[1]))
        teacher_match = forced_matching(healers, teacher_edges) or {}
        student_match = forced_matching(healers, student_edges) or {}
        total += 1
        if teacher_match == student_match:
            agree += 1
        all_healers += len(healers)
        changed_healers += sum(1 for h in healers
                               if teacher_match.get(h) != student_match.get(h))
    return {"n_joint_steps": total,
            "joint_assignment_agreement": agree / total if total else None,
            "healer_assignment_agreement": 1 - changed_healers / all_healers if all_healers else None,
            "single_candidate_step_rate": (sum(1 for gs in joint_groups(groups).values()
                                               if len(gs) == 1 and len(gs[0]["labels"]) == 1)
                                           / len(joint_groups(groups))) if groups else None}


def evaluate_ranking(model, groups):
    """Group-level behaviour metrics: what actually matters for matching.

    - top1_agreement: argmax of predicted vs teacher label per healer group,
      non-tie groups only.
    - joint_assignment_agreement: deployment-faithful joint matching over
      all healers of the same decision step (see joint_assignment_agreement).
    - mae on rows for reference.
    """
    import statistics
    top1 = tie = 0
    errors = []
    for group in groups:
        labels, feats = group["labels"], group["features"]
        pred = {t: float(model.predict([feats[t]])[0]) for t in labels}
        teacher_best = max(labels.values())
        teacher_top = {t for t, v in labels.items() if v == teacher_best}
        pred_top = max(pred, key=lambda t: (pred[t], t))
        if len(teacher_top) > 1:
            tie += 1
        elif pred_top in teacher_top:
            top1 += 1
        errors.extend(abs(pred[t] - labels[t]) for t in labels)
    decided = len(groups) - tie
    result = {"n_groups": len(groups), "n_tied_groups": tie,
              "top1_agreement": top1 / decided if decided else None,
              "mae": statistics.mean(errors) if errors else None}
    result.update(joint_assignment_agreement(model, groups))
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
    valid_episodes = set(json.loads(open(args.valid_episodes, encoding="utf-8").read()))
    train, valid = split_by_episode(groups, valid_episodes)
    if not valid:
        raise SystemExit("validation split is empty; refusing to fit")

    candidates = SEARCH_GRID if args.search else [None]
    best = None
    report = []
    for params in candidates:
        model = fit(train, params)
        metrics = evaluate_ranking(model, valid)
        report.append({"params": params or DEFAULT_PARAMS, **metrics})
        score = metrics["top1_agreement"]
        if score is not None and (best is None or score > best[0]):
            best = (score, params, model)
    score, params, model = best
    os.makedirs(args.output_dir, exist_ok=True)
    from ..export.exporter import export_gbr
    from ..medivac.features import FEATURE_NAMES, SCHEMA_VERSION
    export_gbr(model, FEATURE_NAMES, SCHEMA_VERSION,
               os.path.join(args.output_dir, "model.json"),
               meta_extra={"source_shard": os.path.abspath(args.train_shard),
                           "search_report": report, "selected_params": params or DEFAULT_PARAMS})
    with open(os.path.join(args.output_dir, "train_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps({"selected": params or DEFAULT_PARAMS, "valid_top1": score,
                      "report": report}, indent=1))


if __name__ == "__main__":
    main()
