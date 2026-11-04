import json
from pathlib import Path
import tempfile
import unittest

from tree_skills.evaluation.analyze_teacher_diagnostic import audit_events


def rows(arm="C"):
    allies = {"u0": {"unit_type_id": 54, "alive": True, "health": 100, "max_health": 150, "orders": []},
              "u1": {"unit_type_id": 48, "alive": True, "health": 30, "max_health": 45, "orders": []}}
    payload = {"state": {"game_loop": 8, "allies": allies},
               "questions": {"u0": {"criteria": {"3": "move", "7": "heal"}}, "u1": {"criteria": {"3": "move"}}}}
    decisions = [{"agent": i, "controlled": arm == "D" or i == 0, "unit_type": allies[f"u{i}"]["unit_type_id"],
                  "logits": [0, 0, 0, 5, 0, 0, 0, 0], "native_avail": [0, 0, 0, 1, 0, 0, 0, 1],
                  "teacher_action": 3, "rule_action": None if arm == "D" else 3, "final_action": 3}
                 for i in range(2)]
    feedback = {"response_mapping": "exact", "unmatched_observation_errors": [],
                "actors": {f"u{i}": {"requested_action_id": "3", "response": {"name": "Success"},
                                     "submission": "accepted", "observation_errors": []} for i in range(2)}}
    result = [{"event": "policy_observation" if arm == "D" else "skill_observation", "episode": 1, "payload": payload},
              {"event": "native_teacher_step", "episode": 1, "game_loop": 8, "decisions": decisions},
              {"event": "step", "episode": 1, "game_loop": 8, "actions": [3, 3], "feedback": feedback}]
    if arm == "C": result.insert(1, {"event": "skill_step"})
    return result


class DiagnosticAnalysisTests(unittest.TestCase):
    def audit(self, events, arm="C"):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "events.jsonl").write_text("\n".join(json.dumps(r) for r in events), encoding="utf-8")
            return audit_events(p, arm)

    def test_valid_native_trace(self):
        for arm in ("C", "D"):
            self.assertTrue(self.audit(rows(arm), arm)["submitted_actions_and_teacher_permissions_verified"])

    def test_reject_ground_override(self):
        events = rows()
        events[-2]["decisions"][1]["rule_action"] = 7
        with self.assertRaisesRegex(ValueError, "Ground rule action"):
            self.audit(events)

    def test_reject_teacher_override_and_wrong_feedback(self):
        events = rows()
        events[-2]["decisions"][0]["final_action"] = 7
        with self.assertRaises(ValueError): self.audit(events)
        events = rows()
        events[-1]["feedback"]["actors"]["u0"]["requested_action_id"] = "7"
        with self.assertRaisesRegex(ValueError, "feedback request"):
            self.audit(events)

    def test_full_teacher_cannot_execute_rules(self):
        events = rows("D") + [{"event": "skill_step"}]
        with self.assertRaisesRegex(ValueError, "Rule policy executed"):
            self.audit(events, "D")


if __name__ == "__main__": unittest.main()
