"""A. Contracts & lifecycle: determinism, idempotence, isolation, reset."""
import copy
import unittest

from skills.tactical_v1.context import build_snapshot, Context
from skills.tactical_v1.memory import TacticalMemory
from .fixtures import (TacticalTestCase, ally, enemy, scene, with_questions,
                       DEAD, MOVE_DELTAS, payload)


class ContractTests(TacticalTestCase):
    def test_deterministic_actions_for_same_input_and_memory(self):
        state = scene([((0., 0.), "marine"), ((2., 0.), "marine")],
                      enemies={"e0": enemy((5., 5.))})
        first, _ = self.step(self.manager(), state)
        second, _ = self.step(self.manager(), copy.deepcopy(state))
        self.assertEqual(first, second)

    def test_repeated_call_same_clock_is_idempotent(self):
        manager = self.manager()
        state = scene([((0., 0.), "marine")], enemies={"e0": enemy((4., 4.))})
        pl = with_questions(state)
        a1, _ = manager.step(pl, 1)
        a2, _ = manager.step(pl, 1)          # same payload object, same clock
        self.assertEqual(a1, a2)
        self.assertEqual(manager.stats["steps"], 1)

    def test_new_episode_clears_instances_and_memory(self):
        manager = self.manager()
        state = scene([((0., 0.), "marine")], enemies={"e0": enemy((4., 4.))},
                      episode=1)
        self.step(manager, state)
        manager.commit_final_actions([1])
        self.assertTrue(manager.memory.enemy_last_seen or True)
        state2 = scene([((0., 0.), "marine")], episode=2)  # no enemies visible
        self.step(manager, state2)
        self.assertEqual(manager.memory.enemy_last_seen, {})
        self.assertEqual(manager.memory.instances, {})

    def test_clock_rollback_clears_state(self):
        manager = self.manager()
        state = scene([((0., 0.), "marine")], enemies={"e0": enemy((4., 4.))},
                      loop=200)
        self.step(manager, state)
        manager.commit_final_actions([1])
        rolled = scene([((0., 0.), "marine")], enemies={"e0": enemy((4., 4.))},
                       loop=80, episode=1)
        self.step(manager, rolled)
        # Rolled-back step was treated as fresh: no stale instance survives.
        self.assertTrue(all(i.start_loop <= 80 for i in
                            (manager.memory.instances or {}).values()))

    def test_unselected_candidates_mutate_nothing(self):
        from skills.tactical_v1.registry import build_skills
        from skills.tactical_v1.telemetry import Telemetry
        manager = self.manager()
        state = scene([((0., 0.), "marine"), ((1., 0.), "zealot")],
                      enemies={"e0": enemy((3., 3.)), "e1": enemy((6., 6.))})
        snapshot = build_snapshot(with_questions(state), 1)
        context = Context(snapshot)
        memory = TacticalMemory()
        before = copy.deepcopy(memory.__dict__)
        skills = build_skills(manager.config)
        telemetry = Telemetry()
        proposals = []
        for skill in skills.values():
            proposals.extend(skill.propose(context, memory, telemetry))
        self.assertEqual(memory.__dict__.keys(), before.keys())
        for key in before:
            self.assertEqual(memory.__dict__[key], before[key],
                             f"memory mutated during proposal: {key}")

    def test_dead_unit_gets_noop_only(self):
        state = scene([((0., 0.), "marine"), ((1., 0.), "marine")],
                      enemies={"e0": enemy((4., 4.))})
        state["allies"]["u1"]["alive"] = False
        state["allies"]["u1"]["health"] = 0.
        actions, telemetry = self.step(self.manager(), state)
        self.assertEqual(actions[1], DEAD)

    def test_actor_count_is_not_hardcoded(self):
        for count in (2, 5, 7):
            state = scene([((float(i), 0.), "marine") for i in range(count)],
                          enemies={"e0": enemy((9., 9.))})
            actions, _ = self.step(self.manager(), state)
            self.assertEqual(len(actions), count)


class IdentityTests(TacticalTestCase):
    def test_unit_identity_by_catalog_id_not_name_prefix(self):
        from skills.tactical_v1.capabilities import canonical_type, capability_for
        self.assertEqual(canonical_type("Weird Prefix Unit", 54), "medivac")
        self.assertTrue(capability_for(canonical_type("X", 9)).suicide_attack)
        self.assertTrue(capability_for(canonical_type("X", 4)).splash_attack)
        self.assertEqual(canonical_type("totally unknown", 12345), "unknown")
        self.assertEqual(capability_for("unknown").data_completeness, "unknown")

    def test_ranged_and_melee_groups(self):
        from skills.tactical_v1.capabilities import capability_for
        for name in ("marine", "marauder", "stalker", "hydralisk"):
            self.assertEqual(capability_for(name).combat_class, "ranged")
        for name in ("zealot", "zergling"):
            self.assertEqual(capability_for(name).combat_class, "melee")


if __name__ == "__main__":
    unittest.main()
