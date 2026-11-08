from copy import deepcopy
import unittest

from skills.local_v2.retarget import analyze as analyze_v2
from skills.local_v2.tests.test_retarget import combat
from skills.local_v3.retarget import analyze


class RetargetV3Tests(unittest.TestCase):
    def test_already_covered_source_releases_a_ready_shot_without_mutation(self):
        payload, baseline = combat()
        payload["state"]["visible_enemies"]["e0"]["health"] = 8.0
        before = deepcopy((payload, baseline))
        result = analyze(payload, baseline, "u0")
        self.assertEqual((payload, baseline), before)
        self.assertEqual(result["recommended_action"], 7)
        self.assertEqual(result["source_useful_damage"], [0.0, 0.0])
        self.assertEqual(result["options"][1]["own_marginal_gain_lower"], 10.0)

    def test_do_not_pay_more_source_damage_for_a_small_target_finish(self):
        payload, baseline = combat()
        payload["state"]["visible_enemies"]["e1"]["health"] = 2.0
        self.assertEqual(analyze_v2(payload, baseline, "u0")["recommended_action"], 7)
        result = analyze(payload, baseline, "u0")
        self.assertEqual(result["recommended_action"], 6)
        destination = result["options"][1]
        self.assertTrue(destination["adds_kill_certain"])
        self.assertEqual(destination["own_marginal_gain_lower"], -8.0)
        self.assertEqual(destination["guard_reason"], "source_damage_cost_exceeds_target_gain")

    def test_other_assignment_does_not_block_a_needed_cooperative_finish(self):
        payload, baseline = combat()
        baseline["actions"][1] = 7
        payload["state"]["visible_enemies"]["e1"]["health"] = 20.0
        result = analyze(payload, baseline, "u0")
        destination = result["options"][1]
        self.assertTrue(destination["other_assignment_present"])
        self.assertEqual(destination["destination_remaining_health"], [10.0, 10.0])
        self.assertTrue(destination["eligible"])
        self.assertEqual(result["recommended_action"], 7)

    def test_do_not_join_a_target_already_covered_by_ready_allies(self):
        payload, baseline = combat()
        baseline["actions"][1] = 7
        payload["state"]["visible_enemies"]["e1"]["health"] = 8.0
        result = analyze(payload, baseline, "u0")
        self.assertEqual(result["recommended_action"], 6)
        self.assertEqual(result["options"][1]["guard_reason"], "target_already_covered")

    def test_source_finish_and_cooldown_preserve_original(self):
        payload, baseline = combat()
        payload["state"]["visible_enemies"]["e0"]["health"] = 15.0
        result = analyze(payload, baseline, "u0")
        self.assertEqual((result["recommended_action"], result["reason"]), (6, "preserve_source_finish"))
        payload["state"]["allies"]["u0"]["weapon_ready"] = False
        result = analyze(payload, baseline, "u0")
        self.assertEqual((result["recommended_action"], result["reason"]), (6, "wait_weapon_ready"))
        self.assertEqual([option["action"] for option in result["options"]], [6, 7])

    def test_uncertain_source_or_destination_firepower_is_preserved_as_unknown(self):
        for target in ("e0", "e1"):
            with self.subTest(target=target):
                payload, baseline = combat()
                baseline["actions"][1] = 6 if target == "e0" else 7
                payload["state"]["relations"]["u1"][target]["in_weapon_range"] = None
                result = analyze(payload, baseline, "u0")
                self.assertEqual(result["recommended_action"], 6)
                option = next(option for option in result["options"] if option["target"] == target)
                self.assertIsNone(option["destination_remaining_health"])
                self.assertIsNone(option["own_marginal_gain_lower"])
                self.assertTrue(option["other_assignment_present"])

    def test_movement_order_and_domain_restrictions(self):
        payload, baseline = combat()
        baseline["actions"][0] = 4
        result = analyze(payload, baseline, "u0")
        self.assertEqual((result["recommended_action"], result["reason"]), (4, "preserve_nonattack_order"))
        self.assertEqual([option["action"] for option in result["options"]], [4, 6, 7])
        baseline["roles"]["u0"] = "retreat"
        self.assertIsNone(analyze(payload, baseline, "u0"))


if __name__ == "__main__":
    unittest.main()
