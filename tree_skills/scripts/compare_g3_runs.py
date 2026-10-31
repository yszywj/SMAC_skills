"""G3 equivalence check between two runs on the same manifest.

Compares, per (episode, step):
- final SUBMITTED actions (events.jsonl 'step' rows);
- joint heal assignments, retreat actions, attack assignments, keep
  decisions and roles from 'skill_step' telemetry (the state-equivalence
  surface the runner exposes; race_memory internals are not emitted and
  are covered by offline unit tests instead);
plus manifest identity and per-episode initial-state hashes.

Usage: python tree_skills/scripts/compare_g3_runs.py <runA> <runB>
Exit 0 = equivalent, 1 = mismatch (details printed).
"""

import json
import os
import sys


def load_rows(run_dir):
    rows = {"step": {}, "skill": {}, "initial_state": {}}
    with open(os.path.join(run_dir, "events.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            event = json.loads(line)
            kind = event.get("event")
            key = (event.get("episode"), event.get("step"))
            if kind == "step":
                rows["step"][key] = list(event["actions"])
            elif kind == "skill_step":
                healing = (event.get("race_telemetry") or {}).get("healing") or {}
                keep = healing.get("keep_decisions") or {}
                rows["skill"][key] = {
                    "assignments": healing.get("assignments"),
                    "retreat_actions": event.get("retreat_actions"),
                    "attack_assignments": event.get("attack_assignments"),
                    "keep_selected": {a: d.get("selected") for a, d in keep.items()},
                    "roles": event.get("roles"),
                }
            elif kind == "episode_started":
                rows["initial_state"][event["episode"]] = event.get("initial_state_sha256")
    return rows


def diff_maps(name, a, b):
    return [{"what": name, "step": key, "a": a.get(key), "b": b.get(key)}
            for key in sorted(set(a) | set(b), key=lambda k: (k is None, k))
            if a.get(key) != b.get(key)]


def main(run_a, run_b):
    for run in (run_a, run_b):
        if not os.path.isfile(os.path.join(run, "events.jsonl")):
            raise SystemExit(f"missing events.jsonl in {run}")
    manifest_a = manifest_b = None
    if os.path.isfile(os.path.join(run_a, "manifest.json")):
        manifest_a = json.load(open(os.path.join(run_a, "manifest.json"), encoding="utf-8"))
    if os.path.isfile(os.path.join(run_b, "manifest.json")):
        manifest_b = json.load(open(os.path.join(run_b, "manifest.json"), encoding="utf-8"))
    manifest_equal = (manifest_a == manifest_b) if (manifest_a and manifest_b) else None

    a, b = load_rows(run_a), load_rows(run_b)
    problems = diff_maps("final_actions", a["step"], b["step"])
    for field in ("assignments", "retreat_actions", "attack_assignments",
                  "keep_selected", "roles"):
        problems += diff_maps(f"telemetry:{field}",
                              {k: v[field] for k, v in a["skill"].items()},
                              {k: v[field] for k, v in b["skill"].items()})
    problems += diff_maps("initial_state_sha256",
                          a["initial_state"], b["initial_state"])

    report = {"run_a": run_a, "run_b": run_b,
              "manifest_identical": manifest_equal,
              "n_steps_compared": len(a["step"]),
              "n_problems": len(problems), "problems": problems[:15]}
    print(json.dumps(report, indent=1, ensure_ascii=False))
    verdict = "G3 EQUIVALENCE: " + ("PASS" if not problems and manifest_equal else "FAIL")
    print(verdict)
    return 0 if verdict.endswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
