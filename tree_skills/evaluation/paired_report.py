"""Paired win-rate report for two runs (A vs B) on the same manifest.

Per-episode pairing by manifest episode order; win/loss per episode_summary
in events ('episode_completed'). Reports overall + medivac-subset win
rates, paired difference with bootstrap 95% CI over episodes, and invalid
episode counts. No-medivac episodes stay in the overall denominator.

Usage: python tree_skills/evaluation/paired_report.py <runA> <runB> [--bootstrap 10000]
"""

import argparse
import json
import os
import random


def load_episodes(run_dir):
    episodes = {}
    with open(os.path.join(run_dir, "events.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            event = json.loads(line)
            if event.get("event") == "episode_completed":
                episodes[event["episode"]] = {
                    "win": bool(event.get("win") or event.get("outcome") == "win"),
                    "steps": event.get("steps"),
                    "scenario": event.get("scenario_sha256"),
                    "enemy_elimination_fraction": event.get("enemy_elimination_fraction"),
                    "allied_survival_fraction": event.get("allied_survival_fraction"),
                }
    return episodes


def load_manifest_medivac(run_dir):
    """episode -> n_medivac from the manifest's episode configs."""
    path = os.path.join(run_dir, "manifest.json")
    if not os.path.isfile(path):
        path = os.path.join(run_dir, "..", "manifest.json")
    if not os.path.isfile(path):
        return {}
    manifest = json.load(open(path, encoding="utf-8"))
    out = {}
    for i, episode in enumerate(manifest["episodes"], 1):
        team = episode["episode_config"]["team_gen"]["ally_team"]
        out[i] = sum(1 for t in team if "medivac" in str(t).lower())
    return out


def paired_bootstrap(diffs, rounds=10000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return None
    means = []
    for _ in range(rounds):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return {"point": sum(diffs) / n,
            "ci95_low": means[int(0.025 * rounds)],
            "ci95_high": means[int(0.975 * rounds)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    a, b = load_episodes(args.run_a), load_episodes(args.run_b)
    medivac = load_manifest_medivac(args.run_a)
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("no shared episodes between runs")

    def stats(keys):
        wins_a = sum(1 for k in keys if a[k]["win"])
        wins_b = sum(1 for k in keys if b[k]["win"])
        diffs = [int(b[k]["win"]) - int(a[k]["win"]) for k in keys]
        return {"n": len(keys), "wins_a": wins_a, "wins_b": wins_b,
                "win_rate_a": wins_a / len(keys), "win_rate_b": wins_b / len(keys),
                "paired_diff": paired_bootstrap(diffs, args.bootstrap)}

    groups = {"overall": common,
              "no_medivac": [k for k in common if medivac.get(k, 0) == 0],
              "one_medivac": [k for k in common if medivac.get(k) == 1],
              "two_plus_medivac": [k for k in common if medivac.get(k, 0) >= 2]}
    report = {"run_a": args.run_a, "run_b": args.run_b,
              "episodes_only_in_a": sorted(set(a) - set(b)),
              "episodes_only_in_b": sorted(set(b) - set(a)),
              "groups": {name: stats(keys) for name, keys in groups.items() if keys}}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
