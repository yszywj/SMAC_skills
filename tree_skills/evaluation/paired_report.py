"""Paired win-rate report for two runs (A vs B) on the same manifest.

Per-episode pairing by manifest episode order; win/loss per episode_summary
in events ('episode_completed'). Reports overall + medivac-subset win
rates, paired difference with bootstrap 95% CI over episodes, and invalid
episode counts. No-medivac episodes stay in the overall denominator.

Usage: python tree_skills/evaluation/paired_report.py <runA> <runB> [--bootstrap 10000]
"""

import argparse
import json
import math
import os
import random


def load_episodes(run_dir):
    episodes = {}
    with open(os.path.join(run_dir, "events.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            event = json.loads(line)
            if event.get("event") == "episode_completed":
                if event["episode"] in episodes:
                    raise ValueError(f"duplicate completed episode {event['episode']} in {run_dir}")
                episodes[event["episode"]] = {
                    "win": bool(event.get("win") or event.get("outcome") == "win"),
                    "steps": event.get("steps"),
                    "scenario": event.get("scenario_sha256"),
                    "initial_state": event.get("initial_state_sha256"),
                    "enemy_elimination_fraction": event.get("enemy_elimination_fraction"),
                    "allied_survival_fraction": event.get("allied_survival_fraction"),
                }
    return episodes


def load_manifest(run_dir):
    path = os.path.join(run_dir, "manifest.json")
    if not os.path.isfile(path):
        path = os.path.join(run_dir, "..", "manifest.json")
    if not os.path.isfile(path):
        raise ValueError(f"manifest.json missing in {run_dir}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_manifest_medivac(run_dir):
    """episode -> n_medivac from the manifest's episode configs."""
    manifest = load_manifest(run_dir)
    out = {}
    for i, episode in enumerate(manifest["episodes"], 1):
        team = episode["episode_config"]["team_gen"]["ally_team"]
        out[i] = sum(1 for t in team if "medivac" in str(t).lower())
    return out


def paired_bootstrap(diffs, rounds=10000, seed=0):
    """Paired percentile bootstrap plus a conservative uncertainty bound.

    The bootstrap is explicitly flagged when its input is constant. A
    Hoeffding bound for independent pairs in [-1, 1] remains non-degenerate
    even if every observed difference is zero; it is intentionally broad.
    """
    if rounds < 1:
        raise ValueError("bootstrap rounds must be positive")
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return None
    means = []
    for _ in range(rounds):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    point = sum(diffs) / n
    radius = math.sqrt(2 * math.log(2 / 0.05) / n)
    degenerate = len(set(diffs)) == 1
    result = {"point": point,
            "ci95_low": means[int(0.025 * rounds)],
            "ci95_high": means[int(0.975 * rounds)],
            "method": "paired_percentile_bootstrap",
            "bootstrap_degenerate": degenerate,
            "conservative_ci95": {"method": "hoeffding_independent_pairs",
                                   "low": max(-1.0, point - radius),
                                   "high": min(1.0, point + radius)},
            "warning": ("All observed pair differences are identical. The bootstrap interval "
                        "is degenerate and does not establish the population effect exactly."
                        if degenerate else None)}
    if all(d == 0 for d in diffs):
        # Exact one-sided 95% bound from P(no discordance)=(1-p)^n.
        upper = 1 - 0.05 ** (1 / n)
        result["zero_discordance_bound"] = {
            "method": "exact_binomial_one_sided_95",
            "discordance_probability_upper": upper,
            "implied_absolute_mean_effect_upper": upper,
            "assumption": "independent identically distributed scenario pairs"}
    return result


def validate_pairs(a, b, expected, allow_incomplete=False):
    """Never pair by episode number without matching scenario AND initial state."""
    missing_a, missing_b = sorted(expected - set(a)), sorted(expected - set(b))
    extra_a, extra_b = sorted(set(a) - expected), sorted(set(b) - expected)
    if extra_a or extra_b:
        raise ValueError(f"episodes outside manifest: A={extra_a}, B={extra_b}")
    if (missing_a or missing_b) and not allow_incomplete:
        raise ValueError(f"incomplete runs: missing A={missing_a}, missing B={missing_b}")
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("no shared completed episodes between runs")
    for k in common:
        for field in ("scenario", "initial_state"):
            if not a[k].get(field) or not b[k].get(field):
                raise ValueError(f"episode {k}: missing {field} hash; pairing unverified")
            if a[k][field] != b[k][field]:
                raise ValueError(f"episode {k}: {field} hash mismatch")
    return common, {"expected_episodes": len(expected), "paired_episodes": len(common),
                    "missing_completed_in_a": missing_a, "missing_completed_in_b": missing_b,
                    "complete": not (missing_a or missing_b),
                    "scenario_hashes_match": True, "initial_state_hashes_match": True}


def paired_stats(a, b, keys, rounds=10000):
    wins_a = sum(a[k]["win"] for k in keys)
    wins_b = sum(b[k]["win"] for k in keys)
    table = {"both_win": 0, "a_win_b_loss": 0, "a_loss_b_win": 0, "both_loss": 0}
    for k in keys:
        label = ("both_win" if a[k]["win"] and b[k]["win"] else
                 "a_win_b_loss" if a[k]["win"] else
                 "a_loss_b_win" if b[k]["win"] else "both_loss")
        table[label] += 1
    discordant = table["a_win_b_loss"] + table["a_loss_b_win"]
    smaller = min(table["a_win_b_loss"], table["a_loss_b_win"])
    mcnemar_p = (min(1.0, 2 * sum(math.comb(discordant, i) for i in range(smaller + 1))
                     / 2 ** discordant) if discordant else 1.0)
    diffs = [int(b[k]["win"]) - int(a[k]["win"]) for k in keys]
    secondary = {}
    for field in ("enemy_elimination_fraction", "allied_survival_fraction", "steps"):
        bounded = field != "steps"
        pairs = []
        for k in keys:
            values = (a[k].get(field), b[k].get(field))
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                       and math.isfinite(v) and v >= 0 and (not bounded or v <= 1)
                       for v in values):
                continue
            pairs.append(values)
        n = len(pairs)
        deltas = [vb - va for va, vb in pairs]
        secondary[field] = {
            "n_pairs": n, "n_missing_or_invalid_pairs": len(keys) - n,
            "mean_a": sum(va for va, _ in pairs) / n if n else None,
            "mean_b": sum(vb for _, vb in pairs) / n if n else None,
            "paired_mean_delta": sum(deltas) / n if n else None,
            "scope": "means use the same complete, finite pairs; delta is B minus A"}
        if bounded:
            secondary[field]["paired_diff"] = paired_bootstrap(deltas, rounds)
        # Step differences are not bounded by [-1,1]. Deliberately do not
        # apply the fraction/win-rate Hoeffding bound to episode duration.
    return {"n": len(keys), "wins_a": wins_a, "wins_b": wins_b,
            "win_rate_a": wins_a / len(keys), "win_rate_b": wins_b / len(keys),
            "paired_outcomes": table, "n_discordant": discordant,
            "exact_mcnemar_two_sided_p": mcnemar_p,
            "paired_diff": paired_bootstrap(diffs, rounds),
            "secondary_outcomes": secondary}


def build_report(run_a, run_b, rounds=10000, allow_incomplete=False):
    manifest_a, manifest_b = load_manifest(run_a), load_manifest(run_b)
    if manifest_a != manifest_b:
        raise ValueError("run manifests differ; cannot claim paired comparison")
    a, b = load_episodes(run_a), load_episodes(run_b)
    expected = set(range(1, len(manifest_a["episodes"]) + 1))
    common, audit = validate_pairs(a, b, expected, allow_incomplete)
    for k in common:
        expected_hash = manifest_a["episodes"][k - 1].get("sha256")
        if not expected_hash or a[k]["scenario"] != expected_hash:
            raise ValueError(f"episode {k}: completed scenario hash does not match manifest")
    audit["scenario_hashes_match_manifest"] = True
    run_status = {}
    for name, path in (("a", run_a), ("b", run_b)):
        summary_path = os.path.join(path, "summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path, encoding="utf-8") as fh:
                summary = json.load(fh)
            run_status[name] = {"status": summary.get("status"), "error": summary.get("error")}
            if not allow_incomplete and summary.get("status") != "complete":
                raise ValueError(f"run {name} status is {summary.get('status')!r}; not complete")
    medivac = load_manifest_medivac(run_a)
    groups = {"overall": common,
              "with_medivac": [k for k in common if medivac[k] > 0],
              "no_medivac": [k for k in common if medivac[k] == 0],
              "one_medivac": [k for k in common if medivac[k] == 1],
              "two_plus_medivac": [k for k in common if medivac[k] >= 2]}
    return {"report_version": 2, "run_a": str(run_a), "run_b": str(run_b),
            "pairing_audit": audit, "run_status": run_status,
            "episodes_only_in_a": sorted(set(a) - set(b)),
            "episodes_only_in_b": sorted(set(b) - set(a)),
            "interpretation": ("Scenario pairs are the statistical units. Bootstrap intervals are "
                               "descriptive screening estimates; identical outcomes do not prove "
                               "equivalence. Generalization requires independent target scenarios."),
            "groups": {name: paired_stats(a, b, keys, rounds)
                       for name, keys in groups.items() if keys}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="explicitly report completed pairs only, with missing episodes listed")
    args = parser.parse_args()

    try:
        report = build_report(args.run_a, args.run_b, args.bootstrap, args.allow_incomplete)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
