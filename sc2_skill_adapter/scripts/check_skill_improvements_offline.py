"""Fixed-input regression/ablation; these are NOT counterfactual matches."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge.skill_runtime import load_manager


def main(source, output):
    observations, actual = [], {}
    for event in map(json.loads, (source / "events.jsonl").open(encoding="utf-8")):
        if event["event"] == "skill_observation":
            observations.append(event)
        elif event["event"] == "step":
            actual[event["episode"], event["step"] - 1] = event["actions"]
    cases = {
        "legacy_off": ({}, {"mode": "off"}),
        "zero_consumers": ({}, {"mode": "contextual", "spatial_context": True,
                                "focus_value_weight": 0, "focus_switch_margin": 0,
                                "retreat_value_margin": 0, "heal_value_weight": 0}),
        "value_only": ({}, {"mode": "contextual", "spatial_context": True}),
        "approach_only": ({"coordinated_approach": True}, {"mode": "off"}),
        "safe_reentry_only": ({"safe_reentry": True}, {"mode": "off"}),
        "full": ({"coordinated_approach": True, "safe_reentry": True},
                 {"mode": "contextual", "spatial_context": True}),
    }
    results = {}
    for name, (combat, values) in cases.items():
        manager = load_manager()(max_steps=4, combat_params={
            "window_fire": True, "range_preserving_move": True, "target_scoring": True,
            **combat}, unit_value_params=values)
        last_episode, counts = None, Counter()
        for event in observations:
            if event["episode"] != last_episode:
                manager.reset()
                last_episode = event["episode"]
            if not manager.active:
                manager.select("adaptive_focus_fire", posture="balanced")
            payload = event["payload"]
            before = copy.deepcopy(payload)
            reference = actual[event["episode"], event["step"]]
            actions, telemetry = manager.step(payload, len(reference))
            assert payload == before, (name, event["episode"], event["step"], "input mutated")
            for i, action in enumerate(actions):
                question = payload["questions"].get(f"u{i}")
                assert str(action) in question["criteria"] if question else action == 0
            changed = sum(a != b for a, b in zip(actions, reference))
            counts["steps"] += 1
            counts["changed_team_steps"] += bool(changed)
            counts["changed_unit_actions"] += changed
            counts["value_evaluations"] += telemetry["unit_value_evaluation_calls"]
            counts["approach_unit_steps"] += len(telemetry["approach_diagnostics"])
            counts["safe_reentry_evaluations"] += len(telemetry["safe_reentry_diagnostics"])
            json.dumps(telemetry, allow_nan=False)
        results[name] = dict(counts)
    assert results["legacy_off"]["changed_unit_actions"] == 0, results["legacy_off"]
    assert results["zero_consumers"]["changed_unit_actions"] == 0, results["zero_consumers"]
    report = {"source": str(source), "results": results,
              "scope": "Recorded states stay fixed; candidate actions do not produce next states. "
                       "Checks compatibility, legal actions, immutable inputs and finite telemetry; "
                       "does not measure win rates or improvement."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/native-10v10-v2-skill-n20-seed110")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/skill_improvements_20261023/offline_checks.json")
    args = parser.parse_args()
    main(args.source.resolve(), args.output.resolve())
