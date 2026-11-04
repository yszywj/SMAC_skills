"""Compare completed BASE and CANDIDATE/MODEL game runs without creating labels.

All valid completed pairs count, including episodes without a proposal and
episodes with an engine-rejected command. Command failures remain policy
outcomes; the stricter counterfactual dataset audit is reported separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from medivac_tree.candidates import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from medivac_tree.experiment import context_hash, experiment_context, validate_context
from .dataset import (EVIDENCE_KIND, _episode, _feedback, _finite, _load, _matched_row,
                      _require, _run_contract)

ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError)


def _reasons(rejected):
    return dict(Counter(reason for row in rejected for reason in row["reasons"]))


def _signs(values):
    return {"positive": sum(v > 0 for v in values), "zero": sum(v == 0 for v in values),
            "negative": sum(v < 0 for v in values)}


def _diagnostics(episode, mode, contract_hash):
    opportunities = episode["opportunities"]
    _require(len(opportunities) <= 1, "multiple_opportunities_in_episode")
    reasons, missing, proposals, eligible = Counter(), 0, 0, 0
    for tree in episode["trees"].values():
        for decision in tree["decisions"]:
            audit = decision.get("candidate_audit")
            if not isinstance(audit, dict):
                missing += 1
                continue
            reason = audit.get("reason")
            _require(isinstance(reason, str) and reason, "malformed_candidate_diagnostic")
            reasons[reason] += 1
            proposals += int(reason == "proposed")
            eligible += int(audit.get("eligible") is True)
    selected = Counter()
    feedback_checks = []
    for index, loop, decision, selection in opportunities:
        _require(selection.get("comparison_contract_hash") == contract_hash, "opportunity_contract_mismatch")
        choice = selection.get("selected")
        _require(choice in ("BASE", "CANDIDATE"), "unknown_selected_arm")
        _require(mode == "model" or choice == mode.upper(), "arm_selection_mismatch")
        if mode == "model":
            _require(_finite(selection.get("predicted_gain")) and isinstance(selection.get("path"), list),
                     "missing_model_prediction_audit")
            _require(choice == ("CANDIDATE" if selection["predicted_gain"] > 0 else "BASE"),
                     "model_choice_prediction_mismatch")
        selected[choice] += 1
        try:
            raw, audit = _feedback(episode, index, decision, selection[choice.lower()],
                                   allow_existing_order=choice == "BASE")
            feedback_checks.append({"game_loop": loop, "actor": decision["actor"], "accepted_for_label": True,
                                    "submission": raw["submission"], **audit})
        except ERRORS as exc:
            # Do not selectively remove a policy's failed intervention from
            # full-episode performance. It is unusable as a clean return label.
            feedback_checks.append({"game_loop": loop, "actor": decision.get("actor"),
                                    "accepted_for_label": False, "reason": str(exc)})
    return {"opportunities": len(opportunities), "selected": dict(selected),
            "candidate_audit_reasons": dict(reasons), "diagnostic_records_missing": missing,
            "eligible_diagnostic_records": eligible, "proposal_diagnostic_records": proposals,
            "intervention_feedback": feedback_checks,
            "no_opportunity": len(opportunities) == 0}


def _aggregate_diagnostics(rows, arm):
    reasons, selected, failed = Counter(), Counter(), Counter()
    for row in rows:
        diagnostic = row[arm + "_diagnostics"]
        reasons.update(diagnostic["candidate_audit_reasons"])
        selected.update(diagnostic["selected"])
        failed.update(f["reason"] for f in diagnostic["intervention_feedback"] if not f["accepted_for_label"])
    count = sum(row[arm + "_diagnostics"]["opportunities"] for row in rows)
    return {"opportunity_count": count, "episodes_with_opportunity": count,
            "episodes_without_opportunity": len(rows) - count,
            "candidate_coverage_fraction": count / len(rows) if rows else None,
            "selected_counts": dict(selected), "candidate_audit_reason_counts": dict(reasons),
            "diagnostic_records_missing": sum(row[arm + "_diagnostics"]["diagnostic_records_missing"] for row in rows),
            "feedback_failure_counts": dict(failed)}


def _dataset_support(path, contract_hash, context):
    raw = Path(path).read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    _require(data.get("schema_version") == 1 and data.get("evidence_kind") == EVIDENCE_KIND,
             "unsupported_dataset_evidence")
    _require(data.get("comparison_contract_hash") == contract_hash, "dataset_contract_mismatch")
    _require(data.get("feature_schema") == {"version": FEATURE_SCHEMA_VERSION, "names": list(FEATURE_NAMES)},
             "dataset_feature_schema_mismatch")
    _require(validate_context(data.get("experiment_context"), data.get("experiment_context_hash")) == context,
             "dataset_experiment_context_mismatch")
    rows, rejected = data.get("rows"), data.get("rejected")
    _require(isinstance(rows, list) and isinstance(rejected, list), "malformed_dataset_rows")
    if rows:
        from .train import validate_rows
        validate_rows(rows, contract_hash)
    groups = len({row["group_id"] for row in rows})
    validation_groups = max(1, math.ceil(groups * .25)) if groups else 0
    train_groups = groups - validation_groups
    warnings = []
    if not rows:
        warnings.append("no_accepted_training_labels")
    if train_groups < 4:
        warnings.append("insufficient_independent_groups_for_default_train_validation_split")
    if rows and len({row["target_gain"] for row in rows}) == 1:
        warnings.append("all_accepted_returns_have_the_same_gain")
    varying = [name for name in FEATURE_NAMES if len({row["features"][name] for row in rows}) > 1]
    if rows and not varying:
        warnings.append("no_feature_variation_to_learn_a_split")
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
            "label_count": len(rows), "independent_groups": groups, "gain_sign_counts": _signs([r["target_gain"] for r in rows]),
            "rejected_count": len(rejected), "rejection_reason_counts": _reasons(rejected),
            "default_minimum_train_groups": 4, "default_validation_fraction": .25,
            "default_split_train_groups": train_groups, "default_split_validation_groups": validation_groups,
            "meets_default_group_minimum": train_groups >= 4, "varying_features": varying, "warnings": warnings,
            "interpretation": "Data coverage only; no claim that a useful split or generalizing policy has been learned."}


def build_report(base_run, other_run, *, dataset=None):
    result = {"schema_version": 1, "report_kind": "paired_whole_episode_comparison",
              "status": "rejected", "training_labels_generated": False, "pairs": [], "rejected": [],
              "limitations": ["Independent runs matched by scenario, seed, configuration and initial observation; no hidden-state equivalence claim.",
                              "All valid completed episodes count, including no-opportunity episodes and command failures.",
                              "Small-sample observed performance is not evidence of generalization."]}
    try:
        base, other = _load(base_run), _load(other_run)
        mode = other["run.json"].get("medivac_tree_metadata", {}).get("selection_mode")
        _require(mode in ("candidate", "model"), "other_mode_must_be_candidate_or_model")
        meta = _run_contract(base, other, False, other_mode=mode)
        context = experiment_context(base["run.json"])
        _require(context == experiment_context(other["run.json"]), "experiment_context_mismatch")
        result.update(other_mode=mode, comparison_contract_hash=meta["comparison_contract_hash"],
                      experiment_context=context, experiment_context_hash=context_hash(context),
                      feature_schema=meta["feature_schema"], expected_pairs=len(base["manifest.json"]["episodes"]),
                      sources={"base": {k: base[k] for k in ("path", "file_hashes")},
                               "other": {k: other[k] for k in ("path", "file_hashes")}})
        if dataset is not None:
            result["learning_support"] = _dataset_support(dataset, meta["comparison_contract_hash"], context)
    except ERRORS as exc:
        result["rejected"].append({"episode": None, "reasons": [str(exc)]})
        result["rejection_reason_counts"] = _reasons(result["rejected"])
        return result
    label_rejections = []
    for episode, scenario in enumerate(base["manifest.json"]["episodes"], 1):
        try:
            b, o = _episode(base, episode, scenario), _episode(other, episode, scenario)
            _require(b["start"] == o["start"], "initial_state_or_game_version_mismatch")
            bd = _diagnostics(b, "base", meta["comparison_contract_hash"])
            od = _diagnostics(o, mode, meta["comparison_contract_hash"])
            eligibility = {"eligible": False, "reason": "model_run_is_not_a_forced_candidate_training_arm"}
            if mode == "candidate":
                try:
                    _matched_row(base, other, episode, scenario, meta)
                    eligibility = {"eligible": True}
                except ERRORS as exc:
                    eligibility = {"eligible": False, "reason": str(exc)}
            if not eligibility["eligible"]:
                label_rejections.append({"episode": episode, "reasons": [eligibility["reason"]]})
            result["pairs"].append({"episode": episode, "group_id": scenario["sha256"], "sc2_seed": scenario["sc2_seed"],
                "base_outcome": b["record"]["outcome"], "other_outcome": o["record"]["outcome"],
                "base_return": b["record"]["reward"], "other_return": o["record"]["reward"],
                "return_difference": o["record"]["reward"] - b["record"]["reward"],
                "base_diagnostics": bd, "other_diagnostics": od, "training_eligibility": eligibility})
        except ERRORS as exc:
            result["rejected"].append({"episode": episode, "group_id": scenario["sha256"], "reasons": [str(exc)]})
    pairs, n = result["pairs"], len(result["pairs"])
    result["status"] = "complete" if n == result["expected_pairs"] else "partial" if n else "rejected"
    result["rejection_reason_counts"] = _reasons(result["rejected"])
    result["training_eligibility"] = {"eligible_pairs": sum(r["training_eligibility"]["eligible"] for r in pairs),
                                      "ineligible_pairs": len(label_rejections), "reason_counts": _reasons(label_rejections)}
    differences = [row["return_difference"] for row in pairs]
    result["summary"] = {"compared_pairs": n, "independent_groups": len({r["group_id"] for r in pairs}),
        "outcome_pairs": dict(Counter(row["base_outcome"] + "->" + row["other_outcome"] for row in pairs)),
        "both_win": sum(r["base_outcome"] == r["other_outcome"] == "win" for r in pairs),
        "base_only_win": sum(r["base_outcome"] == "win" and r["other_outcome"] != "win" for r in pairs),
        "other_only_win": sum(r["other_outcome"] == "win" and r["base_outcome"] != "win" for r in pairs),
        "neither_win": sum(r["base_outcome"] != "win" and r["other_outcome"] != "win" for r in pairs),
        "mean_return_difference": sum(differences) / n if n else None, "return_difference_sign_counts": _signs(differences)}
    for arm in ("base", "other"):
        outcomes = Counter(r[arm + "_outcome"] for r in pairs)
        result["summary"][arm] = {"outcomes": dict(outcomes), "wins": outcomes["win"],
            "win_rate": outcomes["win"] / n if n else None,
            "mean_return": sum(r[arm + "_return"] for r in pairs) / n if n else None,
            **_aggregate_diagnostics(pairs, arm)}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", required=True, type=Path)
    parser.add_argument("--other-run", required=True, type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite")
    result = build_report(args.base_run, args.other_run, dataset=args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "compared_pairs": len(result["pairs"]),
                      "rejected": len(result["rejected"])}))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
