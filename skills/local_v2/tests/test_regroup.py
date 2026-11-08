from copy import deepcopy
import json
import unittest

from skills.local_v2.regroup import analyze


def fixture():
    # Decision 0 in the saved Protoss scene 4915023 (u6). Use only positions and
    # public action menus available before either arm submitted an action.
    points = [(2.992, 20.360), (4.213, 28.292), (5.499, 23.176), (6.269, 18.135),
              (7.841, 26.265), (10.967, 2.312), (13.863, 13.951), (2.993, 10.901),
              (4.402, 2.438), (7.206, 28.111)]
    allies = {f"u{i}": {"alive": True, "type": "stalker", "position": list(p)}
              for i, p in enumerate(points)}
    actor = "u6"
    x, y = points[6]
    moves = {str(action): {"destination": [x + dx, y + dy]}
             for action, dx, dy in [(2, 0, 2), (3, 0, -2), (4, 2, 0), (5, -2, 0)]}
    payload = {"state": {"allies": allies, "visible_enemies": {"e1": {"position": [18.277, y]}},
                         "movement_candidates": {actor: moves}, "relations": {}},
               "questions": {key: {"criteria": {str(a): "legal" for a in range(1, 6)}}
                             for key in allies}}
    baseline = {"actions": [4] * len(allies), "roles": {k: "reposition" for k in allies}}
    return payload, baseline


class RegroupTests(unittest.TestCase):
    def test_real_known_geometry_preserves_base_and_all_four_directions(self):
        payload, baseline = fixture()
        original = deepcopy((payload, baseline))
        result = analyze(payload, baseline, "u6")
        self.assertEqual((payload, baseline), original)
        self.assertEqual((result["base_action"], result["legacy_action"],
                          result["recommended_action"]), (4, 5, 2))
        options = {row["action"]: row for row in result["options"]}
        self.assertEqual(list(options), [2, 3, 4, 5])
        self.assertAlmostEqual(options[2]["support_distance"], 7.9018, places=3)
        self.assertAlmostEqual(options[2]["contact_distance"], 4.8460, places=3)
        self.assertAlmostEqual(options[5]["support_distance"], 6.9856, places=3)
        self.assertAlmostEqual(options[4]["support_gain_vs_base"], 0.0)
        json.dumps(result, allow_nan=False)

    def test_unknown_contact_preserves_base_without_discarding_alternatives(self):
        payload, baseline = fixture()
        payload["state"]["visible_enemies"] = {}
        baseline["actions"][6] = 1
        result = analyze(payload, baseline, "u6")
        self.assertIsNone(result["contact_id"])
        self.assertEqual(result["recommended_action"], 1)
        self.assertEqual([o["action"] for o in result["options"]], [1, 2, 3, 4, 5])
        self.assertTrue(all(o["contact_cost_vs_base"] is None for o in result["options"]))

    def test_supporting_ally_is_selected_by_legal_attack_and_not_in_retreat(self):
        payload, baseline = fixture()
        for friend in ("u0", "u3"):
            payload["state"]["relations"][friend] = {"e1": {"smac_attack_available": True}}
            payload["questions"][friend]["criteria"]["7"] = "legal attack"
        baseline["roles"]["u3"] = "retreat"
        result = analyze(payload, baseline, "u6")
        self.assertEqual(result["support_basis"], "can_attack_contact")
        self.assertEqual(result["support_ids"], ["u0"])
        self.assertTrue(all(o["nearest_support_id"] == "u0" for o in result["options"]))

    def test_no_supporters_keeps_base_even_with_known_contact(self):
        payload, baseline = fixture()
        for friend in baseline["roles"]:
            if friend != "u6":
                baseline["roles"][friend] = "recover"
        result = analyze(payload, baseline, "u6")
        self.assertEqual(result["support_ids"], [])
        self.assertEqual(result["recommended_action"], result["base_action"])
        self.assertTrue(all(o["margin_vs_base"] is None for o in result["options"]))

    def test_only_legal_directions_and_incomplete_legal_data(self):
        payload, baseline = fixture()
        del payload["questions"]["u6"]["criteria"]["3"]
        result = analyze(payload, baseline, "u6")
        self.assertEqual([o["action"] for o in result["options"]], [2, 4, 5])
        del payload["state"]["movement_candidates"]["u6"]["2"]
        with self.assertRaisesRegex(ValueError, "missing its observed destination"):
            analyze(payload, baseline, "u6")

    def test_unchanged_or_equal_improvement_preserves_base(self):
        payload, baseline = fixture()
        baseline["actions"][6] = 2
        result = analyze(payload, baseline, "u6")
        self.assertEqual(result["recommended_action"], 2)
        self.assertEqual(len(result["options"]), 4)

    def test_stable_ties_and_failure_of_actor_admission(self):
        payload, baseline = fixture()
        moves = payload["state"]["movement_candidates"]["u6"]
        moves["3"] = deepcopy(moves["2"])
        payload["state"]["movement_candidates"]["u6"] = dict(reversed(list(moves.items())))
        self.assertEqual(analyze(payload, baseline, "u6")["recommended_action"], 2)
        payload["questions"]["u6"]["criteria"]["7"] = "legal attack"
        payload["state"]["relations"]["u6"] = {"e1": {"smac_attack_available": True}}
        self.assertIsNone(analyze(payload, baseline, "u6"))
        payload["state"]["relations"]["u6"] = {}
        payload["state"]["allies"]["u3"]["position"] = [13.863, 19.951]
        self.assertIsNone(analyze(payload, baseline, "u6"))  # exactly 6
