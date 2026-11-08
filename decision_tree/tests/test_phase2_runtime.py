"""Second campaign decisions exercised with the actual adaptive manager hook."""
from copy import deepcopy
import unittest

from decision_tree.phase2.policy import REGROUP_POLICY, Selector, accepted_attack, normalize_policy
from decision_tree.phase2.run import Phase2Controller, carrier_policy
from sc2_skill_adapter.bridge.skill_runtime import load_manager
from skills.precontact_regroup.test_skill import regroup_observation


def controller(policy):
    return Phase2Controller(load_manager()(), carrier_policy(policy), policy, posture="balanced",
                            context_hash=REGROUP_POLICY["model"]["context_hash"])


def obs(distance, game_loop=0, feedback=None):
    payload = regroup_observation(distance)
    payload["state"].update(game_loop=game_loop, last_action_result=feedback)
    return payload


class Phase2RuntimeTests(unittest.TestCase):
    def test_wait_then_native_attack_then_near_without_second_change(self):
        c = controller({"mode": "regroup", "rules": {"far": "wait", "near_pre": "decline"}})
        first = obs(15)
        actions, _, events = c.step(first, 6)
        self.assertEqual(events[0]["phase2"]["action"], "wait")
        self.assertEqual(events[0]["phase2"]["considered_id"], "regroup_step:u0:4")
        self.assertEqual(events[0]["proposed_id"], "BASE")
        self.assertEqual(events[0]["proposals_used"], 0)
        self.assertEqual(c.step(deepcopy(first), 6)[2], [])
        c.manager.commit_final_actions(actions)
        fb = {"actors": {"u1": {"submission": "accepted", "requested_action_id": "7"}}}
        actions, _, events = c.step(obs(12, 8, fb), 6)
        second = events[0]
        self.assertEqual(second["phase2"]["bucket"], "near_engaged")
        self.assertEqual(second["selected_id"], "regroup_step:u0:4")
        self.assertEqual([i for i in range(6) if actions[i] != second["baseline_actions"][i]], [0])
        c.manager.commit_final_actions(actions)
        event = c.step(obs(9, 16), 6)[2][0]
        self.assertEqual((event["phase2"]["action"], event["phase2"]["team_fired_before"]), ("spent", True))
        c.manager.reset()
        c.reset()
        event = c.step(obs(9), 6)[2][0]
        self.assertEqual((event["phase2"]["bucket"], event["phase2"]["action"]), ("near_pre", "decline"))

    def test_decline_closes_episode_and_unknown_is_separate(self):
        c = controller({"mode": "regroup", "rules": {"unknown": "decline"}})
        first = c.step(obs(None), 6)[2][0]
        self.assertEqual(first["phase2"]["bucket"], "unknown")
        self.assertTrue(first["vetoed"])
        self.assertEqual(first["final_actions"], first["baseline_actions"])
        self.assertEqual(c.step(obs(10, 8), 6)[2][0]["phase2"]["action"], "spent")

    def test_accepted_attack_uses_completed_receipt(self):
        for status, action, expected in [("accepted", 7, True), ("rejected", 7, False), ("accepted", 4, False)]:
            self.assertEqual(accepted_attack({"actors": {"u0": {"submission": status,
                                                               "requested_action_id": str(action)}}}), expected)
        self.assertFalse(accepted_attack(None))

    def test_forced_and_base_use_native_contracts(self):
        c = controller({"mode": "base"})
        first = c.step(obs(9), 6)[2][0]
        self.assertEqual(first["selected_id"], "BASE")
        self.assertTrue(any(item["family"] == "regroup_step" for item in first["candidates"]))
        forced = controller({"mode": "forced", "candidate_id": "regroup_step:u0:4", "anchor_step": 1})
        self.assertEqual(forced.step(obs(9), 6)[2][0]["selected_id"], "BASE")
        self.assertEqual(forced.step(obs(9, 8), 6)[2][0]["selected_id"], "regroup_step:u0:4")
        with self.assertRaisesRegex(ValueError, "Forced candidate absent"):
            Selector({"mode": "forced", "candidate_id": "regroup_step:u0:4", "anchor_step": 0}).decide([], 0, False)

    def test_fallback_filters_before_binding_and_requires_prior_fire(self):
        policy = {"mode": "regroup", "fallback": {"family": "retarget"}}
        options = [{"id": "retarget:u2:8", "family": "retarget", "features": {"actor_health_ratio": 0.9,
                                                                               "actor_weapon_ready": 0.0}},
                   {"id": "retarget:u3:8", "family": "retarget", "features": {"actor_health_ratio": 0.9,
                                                                               "actor_weapon_ready": 1.0}}]
        selector = Selector(policy)
        self.assertEqual(selector.decide(options, 0, False)["action"], "none")
        self.assertEqual(selector.decide(options, 1, True)["selected_id"], "retarget:u3:8")
        self.assertEqual(selector.decide(options, 2, True)["action"], "spent")

    def test_precontact_branch_tree_and_invalid_inputs(self):
        policy = {"mode": "regroup", "rules": {"near_pre": {
            "feature": "nearest_enemy_before", "threshold": 8, "left": "wait", "right": "execute"}}}
        self.assertEqual(controller(policy).step(obs(8), 6)[2][0]["phase2"]["action"], "wait")
        self.assertEqual(controller(policy).step(obs(10), 6)[2][0]["phase2"]["action"], "execute")
        with self.assertRaises(ValueError):
            normalize_policy({"mode": "regroup", "rules": {"near_pre": {"feature": "final_return"}}})
        with self.assertRaises(ValueError):
            normalize_policy({"mode": "regroup", "fallback": {"family": "retreat_veto"}})


if __name__ == "__main__":
    unittest.main()
