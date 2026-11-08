from copy import deepcopy
import unittest

from decision_tree.conditional.policy import normalize_policy
from decision_tree.conditional.runtime import ConditionalController
from decision_tree.conditional.runtime import CARRIER
from decision_tree.local import runtime
from decision_tree.phase2.policy import REGROUP_POLICY
from decision_tree.tests.test_phase2_runtime import obs
from sc2_skill_adapter.bridge.skill_runtime import load_manager
from skills.local_v2.library import catalog as old_catalog
from skills.local_v2.tests.test_regroup import fixture
from skills.local_v3.library import catalog
from skills.local_v3.run import installed


def controller(policy=None):
    return ConditionalController(load_manager()(), policy,
                                 context_hash=REGROUP_POLICY["model"]["context_hash"])


def step(owner, payload):
    actions, telemetry, events = owner.step(payload, len(payload["state"]["allies"]))
    owner.manager.commit_final_actions(actions)
    return actions, telemetry, events[0]


class RepeatedRuntimeTests(unittest.TestCase):
    def test_native_factory_installs_and_restores_repeated_controller(self):
        original = runtime.LocalController
        policy = normalize_policy()
        with installed(policy, {"test": True}):
            owner = runtime.LocalController(load_manager()(), runtime.load_policy(CARRIER),
                                            posture="balanced",
                                            context_hash=REGROUP_POLICY["model"]["context_hash"])
            self.assertIsInstance(owner, ConditionalController)
            for clock in (0, 8, 16):
                event = step(owner, obs(10, clock))[2]
            self.assertEqual(event["interventions"], 2)
        self.assertIs(runtime.LocalController, original)

    def test_wait_then_repeat_after_actor_cooldown_and_cache(self):
        owner = controller()
        self.assertEqual(step(owner, obs(18))[2]["interventions"], 0)
        first = step(owner, obs(10, 8))[2]
        self.assertEqual(first["selected_id"], "regroup_step:u0:4")
        self.assertEqual(first["interventions"], 1)
        cache = owner.step(deepcopy(obs(10, 8)), 6)
        self.assertEqual(cache[2], [])
        waiting = step(owner, obs(10, 16))[2]
        self.assertEqual(waiting["selected_id"], "BASE")
        self.assertEqual(waiting["interventions"], 1)
        again = step(owner, obs(10, 24))[2]
        self.assertEqual(again["selected_id"], "regroup_step:u0:4")
        self.assertEqual(again["interventions"], 2)
        self.assertEqual([i for i, (a, b) in enumerate(zip(again["baseline_actions"], again["final_actions"])) if a != b], [0])
        self.assertEqual(owner.manager._last_move["u0"], 4)

    def test_unknown_wait_and_budget_only_changes_when_action_changes(self):
        owner = controller({"max_interventions": 1})
        waiting = step(owner, obs(None))[2]
        self.assertEqual((waiting["selected_id"], waiting["proposals_used"]), ("BASE", 0))
        self.assertIn("wait_unknown_contact", [c["reason"] for c in waiting["considered"]])
        self.assertEqual(step(owner, obs(10, 8))[2]["interventions"], 1)
        done = step(owner, obs(10, 24))[2]
        self.assertEqual((done["selected_id"], done["reason"]), ("BASE", "budget_exhausted"))
        owner.manager.reset()
        owner.reset()
        self.assertEqual(step(owner, obs(10))[2]["interventions"], 1)

    def test_full_exploration_effects_survive_direction_change(self):
        payload, base = fixture()
        for actor, unit in payload["state"]["allies"].items():
            x, y = unit["position"]
            payload["state"]["movement_candidates"][actor] = {
                str(a): {"destination": [x + dx, y + dy]}
                for a, dx, dy in ((2, 0, 2), (3, 0, -2), (4, 2, 0), (5, -2, 0))}
        original = deepcopy((payload, base))
        old = old_catalog(payload, base, [])
        new = catalog(payload, base)
        self.assertEqual(new["effects"], old["effects"])
        row = next(b for b in new["bindings"] if b["family"] == "regroup_step" and b["actor"] == "u6")
        self.assertEqual(row["recommended_action"], row["legacy_action"])
        self.assertEqual((payload, base), original)

    def test_invalid_config_and_clock(self):
        for policy in ({"actor_cooldown_steps": 0}, {"priority": ["retarget", "retarget"]},
                       {"max_interventions": 0}, {"regroup_unknown": "guess"}):
            with self.assertRaises(ValueError):
                normalize_policy(policy)
        owner = controller()
        step(owner, obs(10))
        changed = obs(11)
        with self.assertRaisesRegex(ValueError, "same clock"):
            owner.step(changed, 6)
