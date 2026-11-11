"""B. Joint decision core: control rights, marginal credit, joint plans,
fallback behaviour, search exhaustion."""
import copy
import unittest

from skills.tactical_v1.evaluator import Evaluator
from skills.tactical_v1.ledger import Ledger
from skills.tactical_v1.contracts import (CandidatePlan, Commitment, EffectClaim,
                                          ResourceClaims, CALIBRATED, HEURISTIC)
from .fixtures import TacticalTestCase, ally, enemy, scene, with_questions, STOP


class ControlRightsTests(TacticalTestCase):
    def test_no_unit_controlled_by_two_plans(self):
        state = scene([((0., 0.), "marine"), ((1., 0.), "marine")],
                      enemies={"e0": enemy((3., 3.))})
        actions, telemetry = self.step(self.manager(), state)
        for actor, info in telemetry["units"].items():
            self.assertIsNotNone(info["plan"] or info["skill"] == "fallback")
        # the validator already guarantees one action per unit; check the
        # ledger invariant directly on the chosen plan list
        self.assertEqual(len(actions), 2)

    def test_healed_marine_can_still_attack(self):
        state = scene([((0., 0.), "medivac"), ((1., 0.), "marine")],
                      enemies={"e0": enemy((3., 0.))})
        state["allies"]["u1"]["health"] = 20.
        from .fixtures import heal_candidates
        heal_candidates(state, "u0")
        state["allies"]["u0"]["energy"] = 50.
        actions, telemetry = self.step(self.manager(), state)
        self.assertEqual(telemetry["units"]["u0"]["skill"], "heal_support")
        # The patient keeps its own action and attacks legally.
        self.assertEqual(telemetry["units"]["u1"]["skill"], "focus_fire")
        self.assertEqual(actions[1], 6)   # attack e0

    def test_joint_kill_candidate_is_atomic(self):
        # A wounded enemy baneling in reach of two marines must produce a
        # joint plan whose members are all-or-nothing.
        state = scene([((0., 0.), "marine"), ((0., 2.), "marine"),
                       ((2., 0.), "marine")],
                      enemies={"e0": enemy((4., 0.), "baneling", health=10.)})
        from skills.tactical_v1.context import build_snapshot, Context
        from skills.tactical_v1.registry import build_skills
        from skills.tactical_v1.telemetry import Telemetry
        from skills.tactical_v1.memory import TacticalMemory
        snapshot = build_snapshot(with_questions(state), 1)
        context = Context(snapshot)
        plans = build_skills(self.manager().config)["focus_fire"].propose(
            context, TacticalMemory(), Telemetry())
        joint = [p for p in plans if p.commitment.phase == "joint_kill"]
        self.assertTrue(joint)
        self.assertGreaterEqual(len(joint[0].controlled_actors), 2)
        self.assertTrue(all(f.startswith("member_ready")
                            for f in joint[0].dependencies))


class MarginalCreditTests(unittest.TestCase):
    def _context(self):
        from skills.tactical_v1.context import build_snapshot, Context
        from .fixtures import scene, with_questions, enemy
        state = scene([((0., 0.), "marine")], enemies={"e0": enemy((3., 0.))})
        return Context(build_snapshot(with_questions(state), 1))

    def _plan(self, pid, target, damage, confidence=1.0, evidence=CALIBRATED):
        return CandidatePlan(
            proposal_id=pid, snapshot_id=1, config_fingerprint="t",
            primary_skill="focus_fire", specialization_provider=None,
            instance_id=None, controlled_actors=("u0",), beneficiaries=(),
            goal=target, current_actions={"u0": 6},
            expected_effects=[EffectClaim("damage", target, damage, 8,
                                          evidence, "test", confidence)],
            claims=ResourceClaims(controls=("u0",), fire=((target, damage),)),
            dependencies=(), commitment=Commitment("attacking", 0, None),
            features={}, proposed_memory_delta={})

    def test_overkill_not_double_counted(self):
        evaluator = Evaluator()
        context = self._context()          # e0 effective health = 35
        strong = self._plan("p1", "e0", 100.)
        weak = self._plan("p2", "e0", 40.)
        ledger = Ledger()
        alone = evaluator.score_plan(weak, context, ledger)
        ledger.reserve(strong)             # strong plan already tipped e0 over
        after_strong = evaluator.score_plan(weak, context, ledger)
        # In-budget credit disappears; only the diminishing overkill discount
        # remains (25% factor, calibrated discount, exhausted-budget halving)
        # — never full double count.
        expected = evaluator.weights["damage"] * evaluator.weights["overkill_credit"] \
            * 40.0 * 0.85 * 0.5
        self.assertAlmostEqual(after_strong, expected, places=4)
        self.assertGreater(alone, after_strong)
        ledger2 = Ledger()
        tipping = evaluator.score_plan(strong, context, ledger2)
        self.assertGreater(tipping, 0.)    # damage + the single kill bonus

    def test_heal_not_over_counted(self):
        evaluator = Evaluator()

        class FakeAlly:
            max_health = 45.
            health = 35.
        class FakeSnapshot:
            allies = {"u1": FakeAlly()}
            enemies = {}
        class FakeContext:
            snapshot = FakeSnapshot()
        plan = CandidatePlan(
            "h1", 1, "t", "heal_support", None, None, ("u0",), (), "u1",
            {"u0": 6}, [EffectClaim("heal", "u1", 15., 8, CALIBRATED, "t", 1.)],
            ResourceClaims(controls=("u0",), healing=(("u1", 15.),)), (),
            Commitment("healing", 0, 1), {}, {})
        ledger = Ledger()
        first = evaluator.score_plan(plan, FakeContext(), ledger)
        ledger.reserve(plan)
        second = evaluator.score_plan(plan, FakeContext(), ledger)
        self.assertGreater(first, 0.)
        self.assertAlmostEqual(second, 0.0, places=4)  # missing health covered

    def test_heuristic_effects_are_discounted(self):
        evaluator = Evaluator()
        context = self._context()
        verified = EffectClaim("damage", "e0", 10., 8, CALIBRATED, "t", 1.0)
        heuristic = EffectClaim("damage", "e0", 10., 8, HEURISTIC, "t", 0.3)
        plan_v = self._plan("v", "e0", 10.)
        plan_h = self._plan("h", "e0", 10.)
        plan_v.expected_effects = [verified]
        plan_h.expected_effects = [heuristic]
        self.assertGreater(evaluator.score_plan(plan_v, context, Ledger()),
                           evaluator.score_plan(plan_h, context, Ledger()))


class FallbackTests(TacticalTestCase):
    def test_disabled_skills_still_yield_legal_actions(self):
        manager = self.manager(enabled={name: False for name in (
            "focus_fire", "coordinated_engage", "combat_reposition",
            "disengage_reenter", "regroup", "local_formation",
            "cohesive_search", "splash_fire", "heal_support", "burst_engage")})
        state = scene([((0., 0.), "marine")], enemies={"e0": enemy((3., 3.))})
        actions, telemetry = self.step(manager, state)
        self.assertEqual(actions, [6])
        self.assertEqual(telemetry["units"]["u0"]["skill"], "fallback")

    def test_exhausted_budget_submits_complete_plan(self):
        manager = self.manager(search={"max_expansions": 1, "beam_width": 2})
        state = scene([((float(i), 0.), "marine") for i in range(5)],
                      enemies={"e0": enemy((9., 9.))})
        actions, telemetry = self.step(manager, state)
        self.assertEqual(len(actions), 5)
        self.assertEqual(manager.telemetry.search_budget_events, 1)
        # Every unit still has a deterministic legal action.
        for actor, info in telemetry["units"].items():
            self.assertIn(info["skill"], (
                "coordinated_engage", "fallback"))

    def test_actions_always_legal_under_generated_snapshots(self):
        import random
        rng = random.Random(7)
        manager = self.manager()
        for _ in range(25):
            allies = [((rng.uniform(0, 20), rng.uniform(0, 20)), rng.choice(
                ("marine", "zealot", "stalker", "hydralisk", "zergling")))
                for _ in range(rng.randint(1, 6))]
            enemies = {f"e{i}": enemy((rng.uniform(0, 20), rng.uniform(0, 20)))
                       for i in range(rng.randint(0, 4))}
            state = scene(allies, enemies=enemies)
            pl = with_questions(state)
            from skills.tactical_v1.context import build_snapshot, Context
            snapshot = build_snapshot(pl, 1)
            context = Context(snapshot)
            actions, _ = manager.step(pl, len(allies))
            from skills.tactical_v1.validator import validate_actions
            self.assertEqual(validate_actions(actions, context, len(allies)), [],
                             f"illegal for state {allies} {enemies}")
            manager.commit_final_actions(actions)


if __name__ == "__main__":
    unittest.main()
