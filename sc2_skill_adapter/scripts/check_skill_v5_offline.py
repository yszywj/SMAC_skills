"""Recorded-state compatibility and safety checks, never simulated win rates."""
import copy
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge.skill_runtime import load_manager


def main():
    reports = []
    base = dict(window_fire=True, range_preserving_move=True, target_scoring=True,
                coordinated_approach=True, safe_reentry=True)
    for size in (5, 10):
        run = ROOT / f"runs/native-{size}v{size}-v4-skill-n20-seed150"
        actual, observations = {}, []
        for event in map(json.loads, (run / "events.jsonl").open(encoding="utf-8")):
            if event["event"] == "skill_observation":
                observations.append(event)
            elif event["event"] == "step":
                actual[event["episode"], event["step"] - 1] = event["actions"]
        for label, added in (("new_flags_off", {}), ("v5_on_old_schema", dict(
                accurate_damage=True, critical_escape=True, planned_reentry=True))):
            manager = load_manager()(max_steps=4, combat_params={**base, **added},
                                     unit_value_params=dict(mode="contextual", spatial_context=True))
            counts, previous = Counter(), None
            for event in observations:
                if previous != event["episode"]:
                    manager.reset()
                    previous = event["episode"]
                if not manager.active:
                    manager.select("adaptive_focus_fire", posture="balanced")
                payload = event["payload"]
                before = copy.deepcopy(payload)
                reference = actual[event["episode"], event["step"]]
                actions, telemetry = manager.step(payload, len(reference))
                assert before == payload
                json.dumps(telemetry, allow_nan=False)
                for i, action in enumerate(actions):
                    question = payload["questions"].get(f"u{i}")
                    assert str(action) in question["criteria"] if question else action == 0
                changed = sum(a != b for a, b in zip(actions, reference))
                counts["steps"] += 1
                counts["changed_team_steps"] += bool(changed)
                counts["changed_unit_actions"] += changed
            if not added:
                assert counts["changed_unit_actions"] == 0, (size, label, counts)
            reports.append(dict(size=size, case=label, **counts))
    result = dict(results=reports, scope="Fixed recorded states; not counterfactual games. "
                  "v4 observations lack the new damage fields, so v5 uses its explicit unknown-data fallback. "
                  "Native damage correctness is checked separately using the saved native audit and trace.")
    output = ROOT / "analysis/skill_repairs_v5_20261023/offline_checks_v4.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
