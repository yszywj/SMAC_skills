import copy
import json
import unittest

from skills.local_v1.tests.test_library import fixture
from skills.local_v2.retreat import analyze


def injured_actor():
    payload, baseline = fixture()
    state = payload["state"]
    state["allies"]["u0"].update(health=5, shield=0, radius=0.5)
    for enemy in state["visible_enemies"].values():
        enemy["radius"] = 0.5
    state["unit_type_profiles"]["74"]["movement_speed_native"] = 1.5
    state["unit_type_profiles"]["48"]["weapons"] = [{"range": 1.5, "target_type": "ground"}]
    for relation in state["relations"]["u0"].values():
        relation.update(native_weapon_range=6, smac_attack_distance_limit=6)
    return payload, baseline


class RetreatTests(unittest.TestCase):
    def test_escaping_visible_weapon_coverage_keeps_a_return_shot(self):
        payload, baseline = injured_actor()
        before = copy.deepcopy((payload, baseline))
        report = analyze(payload, baseline, "u0")
        self.assertEqual(report["recommended_action"], 5)
        options = {o["action"]: o for o in report["options"]}
        self.assertEqual(set(options), {2, 3, 4, 5, 6})
        self.assertEqual(options[6]["known_covering_enemies"], ["e0"])
        self.assertEqual(options[5]["known_covering_enemies"], [])
        self.assertTrue(options[5]["next_counterfire_possible"])
        self.assertEqual(options[5]["estimated_next_position"], [-0.75, 0.0])
        self.assertEqual((payload, baseline), before)
        json.dumps(report, allow_nan=False)

    def test_keep_ready_attack_when_all_moves_remain_exposed(self):
        payload, baseline = injured_actor()
        payload["state"]["unit_type_profiles"]["48"]["weapons"][0]["range"] = 12
        report = analyze(payload, baseline, "u0")
        self.assertEqual(report["recommended_action"], report["base_action"])
        self.assertEqual(len(report["options"]), 5)

    def test_unknown_weapon_envelope_keeps_base_and_exploration_menu(self):
        payload, baseline = injured_actor()
        state = payload["state"]
        del state["unit_type_profiles"]["48"]["weapons"][0]["range"]
        del state["unit_type_profiles"]["74"]["movement_speed_native"]
        report = analyze(payload, baseline, "u0")
        self.assertEqual(report["recommended_action"], 6)
        west = next(x for x in report["options"] if x["action"] == 5)
        self.assertEqual(west["unknown_enemy_coverage"], ["e0", "e1"])
        self.assertIsNone(west["estimated_next_position"])
        self.assertIsNone(west["next_counterfire_possible"])

    def test_masked_direction_excluded_base_direction_listed_once(self):
        payload, baseline = injured_actor()
        del payload["questions"]["u0"]["criteria"]["3"]
        baseline["actions"][0] = 5
        report = analyze(payload, baseline, "u0")
        self.assertEqual(report["recommended_action"], 5)
        self.assertEqual([o["action"] for o in report["options"]], [5, 2, 4])

    def test_scope_and_legal_base(self):
        payload, baseline = injured_actor()
        self.assertIsNone(analyze(payload, baseline, "u1"))  # healthy
        baseline["roles"]["u0"] = "retreat"
        self.assertIsNone(analyze(payload, baseline, "u0"))
        baseline["roles"]["u0"] = "attack"
        del payload["questions"]["u0"]["criteria"]["6"]
        self.assertIsNone(analyze(payload, baseline, "u0"))

    def test_legal_direction_with_missing_destination_remains_in_menu(self):
        payload, baseline = injured_actor()
        payload["state"]["movement_candidates"]["u0"]["5"]["destination"] = [None, 0.0]
        report = analyze(payload, baseline, "u0")
        west = next(o for o in report["options"] if o["action"] == 5)
        self.assertIsNone(west["destination"])
        self.assertIsNone(west["nearest_threat_distance"])
        self.assertNotEqual(report["recommended_action"], 5)


if __name__ == "__main__":
    unittest.main()
