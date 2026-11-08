import unittest

from decision_tree.scripts.local_v2_offline_fit import applies, summarize
from decision_tree.scripts.local_v2_offline_selector import _compare


class OfflineComparisons(unittest.TestCase):
    def test_scene_and_actor_step_have_equal_mass_despite_extra_target_bindings(self):
        def obs(seed, step, gain):
            return ({"seed": seed, "group_id": str(seed), "actor": "u0", "step": step}, gain, 0)
        rows = [obs(1, 0, 6), obs(1, 0, 2), obs(1, 1, 0), obs(2, 0, -2)]
        result = summarize(rows)
        self.assertEqual((result["scenes"], result["actor_steps"], result["bindings"]), (2, 3, 4))
        self.assertEqual(result["mean_return_gain"], 0)
        self.assertEqual(result["return_gain_without_one_scene_min"], -2)

    def test_missing_feature_and_late_step_fall_back_to_old_action(self):
        row = {"step": 14, "features": {"old_other_assigned": None}}
        rule = "ally_also_on_old_alternative"
        self.assertFalse(applies(rule, row, 14))
        row["features"]["old_other_assigned"] = 1
        self.assertTrue(applies(rule, row, 14))
        row["step"] = 15
        self.assertFalse(applies(rule, row, 14))

    def test_identical_missing_effect_proves_relative_zero_only_at_same_step(self):
        def endpoint(step):
            return {"effect": "actor_action_key", "step": step, "label": None}
        scenes = [{"seed": seed, "arms": {"a": endpoint(3), "b": endpoint(step)}}
                  for seed, step in [(1, 3), (2, 4)]]
        stats = _compare(scenes, "a", "b")
        self.assertEqual(stats["paired_terminal_scenes"], 0)
        self.assertEqual(stats["same_effect_without_terminal_seeds"], [1])
        self.assertEqual(stats["relative_unknown_seeds"], [2])
        self.assertEqual(stats["relative_known_mean_gain"], 0)
