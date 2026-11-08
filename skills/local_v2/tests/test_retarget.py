from copy import deepcopy
import unittest

from skills.local_v1.tests.test_library import fixture
from skills.local_v2.retarget import analyze


def combat():
    payload, baseline = fixture()
    state = payload["state"]
    for ally in state["allies"].values():
        ally["attack_upgrade_level"] = 0
    for enemy in state["visible_enemies"].values():
        enemy["armor_upgrade_level"] = 0
    state["unit_type_profiles"]["74"]["weapons"][0]["damage_bonus"] = []
    state["unit_type_profiles"]["48"]["attributes"] = []
    return payload, baseline


class RetargetV2Tests(unittest.TestCase):
    def test_leaving_old_kill_keeps_base_even_if_new_target_could_also_die(self):
        payload, baseline = combat()
        enemies = payload["state"]["visible_enemies"]
        enemies["e0"]["health"] = 15.0
        enemies["e1"]["health"] = 8.0
        result = analyze(payload, baseline, "u0")
        old, alternate = result["options"]
        self.assertTrue(old["adds_kill_certain"])
        self.assertTrue(alternate["adds_kill_certain"])
        self.assertEqual(old["other_ready_damage"], [10.0, 10.0])
        self.assertEqual(old["useful_damage"], [5.0, 5.0])
        self.assertEqual(result["recommended_action"], 6)

    def test_distributable_shot_and_tie_prefer_existing_order(self):
        payload, baseline = combat()
        enemies = payload["state"]["visible_enemies"]
        enemies["e0"]["health"] = 8.0
        before = deepcopy((payload, baseline))
        result = analyze(payload, baseline, "u0")
        self.assertEqual((payload, baseline), before)
        self.assertEqual(result["recommended_action"], 7)
        self.assertEqual(result["options"][0]["overkill_damage"], [10.0, 10.0])
        enemies["e0"]["health"] = enemies["e1"]["health"] = 40.0
        baseline["actions"][0] = 7
        self.assertEqual(analyze(payload, baseline, "u0")["recommended_action"], 7)

    def test_creates_a_new_finish_without_giving_up_old_finish(self):
        payload, baseline = combat()
        payload["state"]["visible_enemies"]["e1"]["health"] = 8.0
        self.assertEqual(analyze(payload, baseline, "u0")["recommended_action"], 7)

    def test_unknown_allied_range_is_not_assumed_zero_ready_budget(self):
        payload, baseline = combat()
        payload["state"]["relations"]["u1"]["e0"]["in_weapon_range"] = None
        payload["state"]["visible_enemies"]["e1"]["health"] = 8.0
        result = analyze(payload, baseline, "u0")
        self.assertEqual([x["action"] for x in result["options"]], [6, 7])
        self.assertIsNone(result["options"][0]["other_ready_damage"])
        self.assertEqual(result["options"][0]["unknown_volley_actors"], ["u1"])
        self.assertEqual(result["recommended_action"], 6)

    def test_missing_catalogue_or_enemy_health_abstains_without_suppressing_exploration(self):
        payload, baseline = combat()
        del payload["state"]["unit_type_profiles"]["48"]["armor"]
        result = analyze(payload, baseline, "u0")
        self.assertEqual([x["target"] for x in result["options"]], ["e0", "e1"])
        self.assertIsNone(result["options"][0]["actor_ready_damage"])
        self.assertEqual(result["recommended_action"], 6)
        payload, baseline = combat()
        del payload["state"]["visible_enemies"]["e0"]["shield"]
        result = analyze(payload, baseline, "u0")
        self.assertIsNone(result["options"][0]["target_effective_health"])
        self.assertEqual(result["recommended_action"], 6)

    def test_all_legal_targets_even_on_cooldown_or_when_baseline_moves(self):
        payload, baseline = combat()
        state = payload["state"]
        for i in range(2, 6):
            state["visible_enemies"][f"e{i}"] = deepcopy(state["visible_enemies"]["e0"])
            state["relations"]["u0"][f"e{i}"] = deepcopy(state["relations"]["u0"]["e0"])
            payload["questions"]["u0"]["criteria"][str(6+i)] = "attack"
        state["allies"]["u0"]["weapon_ready"] = False
        # e2 is masked; e3 is not visible; e4 is visible but unavailable.
        del payload["questions"]["u0"]["criteria"]["8"]
        del state["visible_enemies"]["e3"]
        state["relations"]["u0"]["e4"]["smac_attack_available"] = False
        result = analyze(payload, baseline, "u0")
        self.assertEqual([x["action"] for x in result["options"]], [6, 7, 11])
        self.assertEqual(result["recommended_action"], 6)
        baseline["actions"][0] = 4
        state["allies"]["u0"]["weapon_ready"] = True
        result = analyze(payload, baseline, "u0")
        self.assertEqual([x["action"] for x in result["options"]], [4, 6, 7, 11])
        self.assertEqual(result["recommended_action"], 4)

    def test_outside_original_family_domain(self):
        payload, baseline = combat()
        baseline["roles"]["u0"] = "recover"
        self.assertIsNone(analyze(payload, baseline, "u0"))
        baseline["roles"]["u0"] = "all_in_attack"
        del payload["questions"]["u0"]["criteria"]["7"]
        self.assertIsNone(analyze(payload, baseline, "u0"))


if __name__ == "__main__":
    unittest.main()
