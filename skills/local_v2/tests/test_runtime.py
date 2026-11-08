"""Exercise v2 through the real adaptive manager's precommit and history update."""
from copy import deepcopy
import unittest

from decision_tree.local.runtime import LocalController
from decision_tree.phase2.policy import REGROUP_POLICY
from decision_tree.tests.test_integration import observed
from decision_tree.tests.test_phase2_runtime import controller as old_controller, obs
from sc2_skill_adapter.bridge.skill_runtime import load_manager
from skills.current.tests.test_group_skills import ally, scene
from skills.local_v1.tests.replay_contract import CORE, normalized
from skills.local_v2.run import attach
from skills.local_v2.tests.test_regroup import fixture as regroup_geometry


CONTEXT = REGROUP_POLICY["model"]["context_hash"]


def owner(policy, version="v2", forced=None):
    return attach(load_manager()(), policy, implementation=version, context_hash=CONTEXT, forced=forced)


def step(controller, observation):
    actions, telemetry, events = controller.step(deepcopy(observation), len(observation["state"]["allies"]))
    controller.manager.commit_final_actions(actions)
    return actions, normalized(telemetry), events[0]


class ActionRuntimeTests(unittest.TestCase):
    def assert_manager_history(self, actual, expected):
        for key in CORE:
            self.assertEqual(normalized(getattr(actual.manager, key)), normalized(getattr(expected.manager, key)), key)

    def test_keep_frozen_actor_and_wait_timing_even_when_v2_chooses_base(self):
        policy = {"mode": "regroup", "rules": {"far": "wait"}}
        v1, v2, adaptive = old_controller(policy), owner(policy), old_controller({"mode": "base"})
        for clock, distance in ((0, 16), (8, 10), (16, 10)):
            payload = obs(distance, clock)
            _, _, original = step(v1, payload)
            actions, telemetry, event = step(v2, payload)
            base_actions, base_telemetry, _ = step(adaptive, payload)
            meta = event["local_v2"]
            self.assertEqual(meta["frozen_selected_id"], original["selected_id"])
            self.assertEqual(meta["frozen_proposed_id"], original["proposed_id"])
            self.assertEqual(event["phase2"]["action"], {0:"wait", 8:"execute", 16:"spent"}[clock])
            self.assertEqual((actions, telemetry), (base_actions, base_telemetry))
            self.assert_manager_history(v2, adaptive)
            if clock == 8:
                self.assertEqual((meta["actor"], meta["frozen_selected_id"], event["selected_id"]),
                                 ("u0","regroup_step:u0:4","BASE"))
                self.assertEqual((event["proposals_used"], event["interventions"]), (1,0))

    def test_v1_parity_with_frozen_numeric_and_phase2_selectors(self):
        for policy in ({"mode":"regroup"}, REGROUP_POLICY):
            with self.subTest(policy_kind=policy.get("kind", "phase2")):
                native = (old_controller(policy) if "mode" in policy else
                          LocalController(load_manager()(), policy, context_hash=CONTEXT))
                wrapped = owner(policy,"v1")
                for clock in (0, 8, 16):
                    payload=obs(10,clock)
                    a,t,e=step(native,payload)
                    b,u,f=step(wrapped,payload)
                    self.assertEqual((b,u,f["selected_id"],f["proposed_id"],f["vetoed"]),
                                     (a,t,e["selected_id"],e["proposed_id"],e["vetoed"]))
                    self.assert_manager_history(wrapped,native)

    def test_changed_v2_direction_commits_only_frozen_actor(self):
        # Rotate the published regroup geometry so the synthetic manager's
        # ordinary northward BASE step is aligned with the visible contact.
        geometry, _ = regroup_geometry()
        units = geometry["state"]["allies"]
        enemy = geometry["state"]["visible_enemies"]["e1"]["position"]
        points = [(32 - u["position"][1], u["position"][0]) for u in units.values()]
        state = scene(points, {"e1": ally((32 - enemy[1], enemy[0]))}, distance=2)
        state["last_action_result"] = None
        payload = observed(state)
        v1 = owner({"mode": "regroup"}, "v1")
        v2 = owner({"mode": "regroup"}, "v2")
        before, _, old = step(v1, payload)
        after, _, new = step(v2, payload)
        self.assertEqual(old["selected_id"], "regroup_step:u6:3")
        self.assertEqual(new["local_v2"]["frozen_selected_id"], old["selected_id"])
        self.assertEqual(new["selected_id"], "regroup_step:u6:5")
        self.assertEqual(old["baseline_actions"], new["baseline_actions"])
        self.assertEqual([slot for slot in range(10) if before[slot]!=after[slot]], [6])
        self.assertEqual(v2.manager._last_move["u6"], 5)
        payload["state"]["game_loop"] += 8
        following = step(v2, payload)[2]
        self.assertEqual((following["selected_id"], following["interventions"]), ("BASE", 1))

    def test_same_clock_cache_and_reset(self):
        c = owner({"mode":"regroup"})
        first = obs(10)
        actions, telemetry, event = step(c, first)
        count = c.manager.stats["steps"]
        cached, cached_telemetry, again = c.step(deepcopy(first),6)
        self.assertEqual((cached,normalized(cached_telemetry),again), (actions,telemetry,[]))
        self.assertEqual(c.manager.stats["steps"],count)
        changed = deepcopy(first)
        changed["state"]["allies"]["u0"]["health"] -= 1
        with self.assertRaisesRegex(ValueError,"same clock"):
            c.step(changed,6)
        c.manager.reset()
        c.reset()
        self.assertEqual((c.interventions,c.proposals_used,c.step_number), (0,0,0))
        first["state"]["episode"] = 2
        self.assertEqual(step(c,first)[2]["local_v2"],event["local_v2"])

    def test_force_reaches_an_action_the_old_tree_would_not_select(self):
        c = owner({"mode":"base"},forced=(1,"regroup_step:u0:3"))
        first = step(c,obs(10))[2]
        self.assertEqual((first["selected_id"],first["proposals_used"]), ("BASE",0))
        changed = step(c,obs(10,8))[2]
        self.assertTrue(changed["local_v2"]["forced"])
        self.assertEqual((changed["local_v2"]["frozen_selected_id"], changed["selected_id"]),
                         ("BASE","regroup_step:u0:3"))
        self.assertNotIn("regroup_step:u0:3", [x["id"] for x in changed["candidates"]])
        self.assertEqual([i for i,(a,b) in enumerate(zip(changed["baseline_actions"],changed["final_actions"]))
                          if a!=b], [0])
        self.assertEqual(changed["final_actions"][0],3)
        after = step(c,obs(10,16))[2]
        self.assertFalse(after["local_v2"]["forced"])
        self.assertEqual(after["final_actions"],after["baseline_actions"])
        self.assertEqual((after["proposals_used"], after["interventions"]), (1,1))

    def test_forced_base_alias_preserves_all_roles_and_history(self):
        c = owner({"mode":"base"},forced=(0,"regroup_step:u0:2"))
        adaptive = old_controller({"mode":"base"})
        for clock in (0,8):
            payload=obs(10,clock)
            a,t,event=step(c,payload)
            b,u,_=step(adaptive,payload)
            self.assertEqual((a,t),(b,u))
            self.assertEqual(event["selected_id"],"BASE")
            self.assert_manager_history(c,adaptive)
            if clock == 0:
                self.assertTrue(event["local_v2"]["forced"])
                self.assertEqual((event["proposals_used"],event["interventions"]), (1,0))


if __name__ == "__main__":
    unittest.main()
