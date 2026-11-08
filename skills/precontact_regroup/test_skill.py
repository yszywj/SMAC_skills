"""Conformance of the packaged skill at the Adaptive manager's commit boundary."""
from copy import deepcopy
import unittest

from decision_tree.local.models import model_hash
from decision_tree.local.runtime import load_policy
from decision_tree.tests.test_integration import observed
from decision_tree.tests.test_invocation import combat_scene
from sc2_skill_adapter.bridge.skill_runtime import load_manager
from skills.current.tests.test_group_skills import ally, scene
from skills.precontact_regroup import POLICY_PATH, attach, propose


def regroup_observation(distance=12.0, ally_count=6):
    # u0 can move east to the group. Adaptive's ordinary approach moves north.
    positions = [(0, 0)] + [(20, i) for i in range(ally_count - 1)]
    enemies = {} if distance is None else {"e0": ally((0, distance))}
    return observed(scene(positions, enemies))


class PrecontactRegroupTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy(POLICY_PATH)
        self.manager = load_manager()()
        self.controller = attach(self.manager, context_hash=self.policy["model"]["context_hash"])

    def step(self, payload):
        actions, _, events = self.controller.step(payload, len(payload["state"]["allies"]))
        self.assertEqual(len(events), 1)
        self.assertEqual(actions, events[0]["final_actions"])
        return events[0]

    def test_packaged_models_and_manager_hook(self):
        gain, veto = self.policy["model"], self.policy["veto_model"]
        self.assertEqual(set(gain["families"]), {"regroup_step"})
        self.assertEqual(set(veto["families"]), {"regroup_step"})
        self.assertEqual(veto["old_model_hash"], model_hash(gain))
        self.assertEqual(self.controller.policy, self.policy)
        self.assertIs(self.manager.local_decision_hook.__self__, self.controller)
        with self.assertRaisesRegex(ValueError, "context"):
            attach(load_manager()(), context_hash="wrong context")

    def test_distance_twelve_changes_one_unit_once_then_returns_to_base(self):
        payload = regroup_observation()
        first = self.step(payload)
        changed = [i for i, (old, new) in enumerate(zip(first["baseline_actions"], first["final_actions"]))
                   if old != new]
        self.assertEqual(changed, [0])
        self.assertEqual(first["selected_id"], "regroup_step:u0:4")
        self.assertEqual(first["final_actions"][0], 4)
        payload["state"]["game_loop"] += 8
        following = self.step(payload)
        self.assertEqual(following["selected_id"], "BASE")
        self.assertEqual(following["final_actions"], following["baseline_actions"])
        self.assertEqual(self.controller.interventions, 1)

    def test_far_veto_consumes_proposal_until_next_episode(self):
        first = self.step(regroup_observation(12.01))
        self.assertTrue(first["vetoed"])
        self.assertEqual(first["proposed_id"], "regroup_step:u0:4")
        self.assertEqual(first["final_actions"], first["baseline_actions"])
        near = regroup_observation(11.99)
        near["state"]["game_loop"] += 8
        following = self.step(near)
        self.assertEqual(following["reason"], "intervention_budget_exhausted")
        self.assertEqual((self.controller.proposals_used, self.controller.interventions), (1, 0))
        self.manager.reset()
        self.controller.reset()
        near["state"]["episode"] = 2
        self.assertEqual(self.step(near)["selected_id"], "regroup_step:u0:4")

    def test_missing_visible_enemy_keeps_the_original_unknown_distance_behavior(self):
        event = self.step(regroup_observation(None))
        proposed = next(c for c in event["candidates"] if c["id"] == event["selected_id"])
        self.assertEqual(proposed["features"]["nearest_enemy_before"], -1)
        self.assertEqual(event["selected_id"], "regroup_step:u0:4")

    def test_at_most_five_allies_abstains_even_when_a_legal_patch_exists(self):
        event = self.step(regroup_observation(12, ally_count=5))
        self.assertTrue(any(c["family"] == "regroup_step" for c in event["candidates"]))
        self.assertEqual(event["proposed_id"], "BASE")
        self.assertEqual(event["proposals_used"], 0)

    def test_compact_formation_has_no_regroup_proposal(self):
        payload = observed(scene([(i, 0) for i in range(6)], {"e0": ally((0, 12))}))
        event = self.step(payload)
        self.assertFalse(any(c["family"] == "regroup_step" for c in event["candidates"]))
        self.assertEqual(event["final_actions"], event["baseline_actions"])
        self.assertEqual(event["proposals_used"], 0)

    def test_isolated_unit_with_no_legal_move_cannot_regroup(self):
        payload = regroup_observation()
        payload["questions"]["u0"]["criteria"] = {"1": "Stop"}
        payload["state"]["movement_candidates"]["u0"] = {}
        event = self.step(payload)
        self.assertFalse(any(c["family"] == "regroup_step" for c in event["candidates"]))
        self.assertEqual(event["final_actions"], event["baseline_actions"])
        self.assertEqual(event["proposals_used"], 0)

    def test_other_legal_families_are_never_selected(self):
        event = self.step(observed(combat_scene()))
        self.assertTrue(any(c["family"] == "retarget" for c in event["candidates"]))
        self.assertEqual(event["proposed_id"], "BASE")
        self.assertEqual(event["final_actions"], event["baseline_actions"])

    def test_binding_chosen_before_veto_without_trying_second_unit(self):
        positions = [(0, 0), (40, 0), (20, 0), (20, 1), (20, 2), (20, 3)]
        event = self.step(observed(scene(positions, {"e0": ally((40, 10))})))
        options = {c["id"]: c["features"]["nearest_enemy_before"] for c in event["candidates"]
                   if c["family"] == "regroup_step"}
        self.assertGreater(options["regroup_step:u0:4"], 12)
        self.assertLessEqual(options["regroup_step:u1:5"], 12)
        self.assertEqual(event["proposed_id"], "regroup_step:u0:4")
        self.assertTrue(event["vetoed"])
        self.assertEqual(event["selected_id"], "BASE")

    def test_pure_proposal_matches_first_manager_decision_without_mutation(self):
        original_hook = self.manager.local_decision_hook
        queries = []

        def compare_at_commit(payload, baseline):
            saved = deepcopy((payload, baseline))
            query = propose(payload, baseline)
            self.assertEqual(propose(payload, baseline), query)
            self.assertEqual((payload, baseline), saved)
            queries.append(query)
            return original_hook(payload, baseline)

        self.manager.local_decision_hook = compare_at_commit
        cases = [
            ("near", regroup_observation(12), "regroup_step:u0:4", False),
            ("far", regroup_observation(13), "regroup_step:u0:4", True),
            ("other skill", observed(combat_scene()), "BASE", False),
            ("no candidate", observed(scene([(i, 0) for i in range(6)])), "BASE", False),
        ]
        for label, payload, expected_id, expected_veto in cases:
            with self.subTest(label=label):
                self.manager.reset()
                self.controller.reset()
                queries.clear()
                event = self.step(payload)
                self.assertEqual(len(queries), 1)
                candidate, vetoed = queries[0]
                self.assertEqual(candidate["id"] if candidate else "BASE", expected_id)
                self.assertEqual(vetoed, expected_veto)
                self.assertEqual((event["proposed_id"], event["vetoed"]), (expected_id, expected_veto))
                if candidate is not None:
                    self.assertIn(candidate, event["candidates"])


if __name__ == "__main__":
    unittest.main()
