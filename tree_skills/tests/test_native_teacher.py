import unittest
from types import SimpleNamespace as NS

from tree_skills.integration.native_teacher import NativeTeacherRuntime, masked_native_argmax, validate_control


class FakeClient:
    model_args = {"action_size": 20}
    def __init__(self):
        self.resets, self.requests, self.inits = [], [], []
    def init(self, dims): self.inits.append(dims)
    def reset_episode(self, ep): self.resets.append(ep)
    def logits(self, ep, loop, agents):
        self.requests.append((ep, loop, agents))
        return {a["agent"]: [0, 1, 2, 9, 3, 4, 8, 7, 6] + [1000] * 11 for a in agents}


class FakeEnv:
    n_agents = n_enemies = 3
    medivac_id = 54
    agents = {0: NS(unit_type=48, health=45), 1: NS(unit_type=54, health=100),
              2: NS(unit_type=51, health=0)}
    def get_obs_move_feats_size(self): return 4
    def get_obs_own_feats_size(self): return 5
    def get_obs_enemy_feats_size(self): return (3, 5)
    def get_obs_ally_feats_size(self): return (2, 5)
    def get_obs_agent(self, a): return [float(a)] * 34
    def get_avail_agent_actions(self, a):
        return [1] + [0] * 9 if a == 2 else [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]


class NativeTeacherTests(unittest.TestCase):
    def test_mask_padding_keep_and_tie(self):
        self.assertEqual(masked_native_argmax([99, 1, 4, 4, 1000], [0, 1, 1, 1, 1], 4), 2)
        self.assertEqual(masked_native_argmax([0, 100], [1, 0], 2), 0)

    def test_invalid_logits_mask_fail_closed(self):
        for logits, avail, n in (([0, float("nan")], [1, 1], 2),
                                  ([0], [1, 1], 2), ([0, 0], [0, 0, 1], 2)):
            with self.assertRaises(ValueError): masked_native_argmax(logits, avail, n)

    def test_medivac_only_and_all_agent_history(self):
        client = FakeClient()
        runtime = NativeTeacherRuntime("medivac", "test", client=client)
        rules = [6, 9, 0]
        actions, trace = runtime.choose("ep1", 8, FakeEnv(), rules)
        self.assertEqual(actions, [6, 3, 0])
        self.assertEqual(rules, [6, 9, 0])
        self.assertEqual([d["controlled"] for d in trace["decisions"]], [False, True, False])
        self.assertEqual(len(client.requests[0][2]), 3)
        runtime.choose("ep1", 16, FakeEnv(), rules)
        runtime.choose("ep2", 8, FakeEnv(), rules)
        self.assertEqual(client.resets, ["ep1", "ep2"])
        self.assertEqual(len(client.inits), 1)
        with self.assertRaises(ValueError): runtime.choose("ep2", 8, FakeEnv(), rules)

    def test_full_teacher_dead_noop_and_no_rules(self):
        runtime = NativeTeacherRuntime("all", "test", client=FakeClient())
        actions, trace = runtime.choose("ep1", 8, FakeEnv())
        self.assertEqual(actions, [3, 3, 0])
        self.assertTrue(all(d["rule_action"] is None for d in trace["decisions"]))
        with self.assertRaises(ValueError): runtime.choose("ep1", 16, FakeEnv(), [6, 9, 0])

    def test_exclusive_and_explicit_configuration(self):
        args = NS(native_teacher_control="all", race_profile="terran", skill_version="current",
                  pair_mode="off", native_teacher_checkpoint="checkpoint.pt")
        config = {"map_name": "10gen_terran", "unit_balance": "native"}
        validate_control(args, "adaptive-skill", config)
        args.pair_mode = "teacher-score"
        with self.assertRaises(ValueError): validate_control(args, "adaptive-skill", config)
        args.pair_mode = "off"
        with self.assertRaises(ValueError): validate_control(args, "nearest", config)


if __name__ == "__main__": unittest.main()
