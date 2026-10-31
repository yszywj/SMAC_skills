"""G3 equivalence check: compare per-step FINAL ACTIONS of two runs
(A vs A-plug) on the same manifest. Reads events.jsonl 'skill_step' rows.

Usage: python tree_skills/scripts/compare_g3_runs.py <runA> <runB>
"""

import json
import os
import sys


def load_actions(run_dir):
    path = os.path.join(run_dir, "events.jsonl")
    steps = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            event = json.loads(line)
            if event.get("event") != "step":
                continue  # final submitted actions live on "step" rows
            steps[(event["episode"], event["step"])] = list(event["actions"])
    return steps


def main(run_a, run_b):
    a, b = load_actions(run_a), load_actions(run_b)
    if not a:
        raise SystemExit("no action rows found in run A (check telemetry keys)")
    keys = sorted(set(a) | set(b))
    mismatch = [k for k in keys if a.get(k) != b.get(k)]
    report = {"run_a": run_a, "run_b": run_b, "n_steps": len(keys),
              "steps_in_a_only": len([k for k in a if k not in b]),
              "steps_in_b_only": len([k for k in b if k not in a]),
              "n_mismatched_steps": len(mismatch), "mismatched": mismatch[:20]}
    print(json.dumps(report, indent=1))
    verdict = "G3 ACTION EQUIVALENCE: " + ("PASS" if not mismatch and
                                           keys and len(a) == len(b) else "FAIL")
    print(verdict)
    return 0 if verdict.endswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
