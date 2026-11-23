"""Risk-envelope and future-opportunity regressions, with a recorded failure.

These are offline observations and legal action comparisons, never SC2 runs.
"""
import copy
import json
import pathlib
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.contracts import Commitment, ResourceClaims
from skills.tactical_v1.evaluator import Evaluator
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.common import make_plan
from .fixtures import enemy, scene, with_questions


def context(state):
    return Context(build_snapshot(with_questions(state), state["game_loop"]))


def plan(c, actions, objective, data=None):
    return make_plan("combat_rejoin", tuple(actions), "test", actions, [],
        ResourceClaims(controls=tuple(actions)), Commitment("joining"), {},
        c.snapshot.snapshot_id, "test", atomic=len(actions) > 1,
        goal_kind="area", goal_data=dict(data or {}, objective=objective))


class RiskForecastTests(unittest.TestCase):
    def test_current_contact_and_visible_pursuit_remain_distinct(self):
        c = context(scene([((0., 0.), "marine")],
                          {"e0": enemy((3.5, 0.))}, attack_range=2.))
        self.assertEqual(c.risk_at("u0", (0., 0.)).threats, ())
        future = c.forecast_risk("u0", (0., 0.))
        self.assertEqual(future.current_threats, ())
        self.assertEqual(future.reachable_threats, ("e0",))
        self.assertEqual(c.risk_at("u0", (0., 0.)).threats, ())

    def test_risk_continues_growing_after_three_enemies(self):
        evaluator = Evaluator()
        penalties = []
        for n in (1, 3, 6):
            c = context(scene([((0., 0.), "marine")],
                {f"e{i}": enemy((2., .01 * i)) for i in range(n)}))
            penalties.append(evaluator.score_joint(c, [], {"u0": 1})[1]["risk"])
        self.assertGreater(penalties[0], penalties[1])
        self.assertGreater(penalties[1], penalties[2])

    def test_known_weapon_strength_and_effective_health_change_pressure(self):
        evaluator = Evaluator()
        weak = scene([((0., 0.), "marine")], {"e0": enemy((2., 0.), "zealot")})
        strong = scene([((0., 0.), "marine")], {"e0": enemy((2., 0.), "colossus")})
        weak_cost = evaluator.score_joint(context(weak), [], {"u0": 1})[1]["risk"]
        strong_cost = evaluator.score_joint(context(strong), [], {"u0": 1})[1]["risk"]
        self.assertLess(strong_cost, weak_cost)
        damaged = copy.deepcopy(strong)
        damaged["allies"]["u0"]["health"] = 10.
        self.assertLess(evaluator.score_joint(context(damaged), [], {"u0": 1})[1]["risk"],
                        strong_cost)

    def test_unknown_enemy_data_stays_unknown_but_not_safe(self):
        state = scene([((0., 0.), "marine")], {"e0": enemy((2., 0.))})
        state["unit_type_profiles"].pop("105")
        c = context(state)
        forecast = c.forecast_risk("u0", (0., 0.))
        self.assertIsNone(forecast.damage_estimates["e0"])
        self.assertIn("e0", forecast.heuristic_enemies)
        self.assertIn("e0", c.forecast_risk("u0", (100., 0.)).reachable_threats)
        self.assertLess(Evaluator().score_joint(c, [], {"u0": 1})[1]["risk"], 0.)

    def test_domain_incompatible_enemy_cannot_create_forecast_threat(self):
        state = scene([((0., 0.), "medivac")], {"e0": enemy((1., 0.))},
                      enemy_can_reach=False)
        c = context(state)
        self.assertEqual(c.forecast_risk("u0", (0., 0.)).threats, ())

    def test_uncertain_escape_retains_some_origin_exposure(self):
        state = scene([((0., 0.), "marine")], {"e0": enemy((2., 0.))}, distance=10.)
        state["unit_type_profiles"]["48"]["movement_speed_native"] = 20.
        c = context(state)
        predicted = c.predict_position("u0", 5)
        self.assertFalse(predicted.certain)
        self.assertEqual(c.forecast_risk("u0", predicted.position).threats, ())
        conservative = Evaluator().score_joint(c, [], {"u0": 5})[1]["risk"]
        optimistic = Evaluator(forecast_config={"uncertain_position_floor": 0.}).score_joint(
            c, [], {"u0": 5})[1]["risk"]
        self.assertLess(conservative, optimistic)
        self.assertEqual(optimistic, 0.)

    def test_future_mask_opportunity_does_not_submit_illegal_current_attack(self):
        state = scene([((0., 0.), "stalker")], {"e0": enemy((6.5, 0.))},
                      attack_range=6.)
        c = context(state)
        self.assertFalse(c.can_issue_attack("u0", "e0"))
        opportunity = c.projected_attack_opportunity("u0", "e0", c.predict_position("u0", 4).position)
        self.assertTrue(opportunity.available)
        self.assertTrue(opportunity.ready)
        self.assertNotIn(6, c.legal_actions("u0"))
        after = Evaluator().score_joint(c, [], {"u0": 4})[1]
        self.assertGreater(after["future_attack"], 0.)
        self.assertEqual(after["useful_damage"], 0.)
        self.assertEqual(after["expected_kills"], 0.)
        self.assertEqual(after["coverage"], 0.)

    def test_current_shot_cannot_be_repackaged_as_future_reward(self):
        c = context(scene([((0., 0.), "marine")], {"e0": enemy((4., 0.))}))
        evaluator = Evaluator()
        shooting = evaluator.score_joint(c, [], {"u0": 6})[1]
        postponing = evaluator.score_joint(c, [], {"u0": 4})[1]
        self.assertGreater(shooting["useful_damage"], 0.)
        self.assertEqual(shooting["future_attack"], 0.)
        self.assertEqual(postponing["future_attack"], 0.)

    def test_cooling_splash_position_has_future_value_without_present_coverage(self):
        state = scene([((0., 0.), "colossus")],
                      {"e0": enemy((6.8, 0.), health=200., max_health=200.),
                       "e1": enemy((6.8, 1.), health=200., max_health=200.)}, attack_range=7.)
        state["allies"]["u0"].update(weapon_cooldown_loops=12., weapon_ready=False)
        c = context(state)
        enabled = Evaluator().score_joint(c, [], {"u0": 4})[1]
        disabled = Evaluator(mechanisms={"splash": False, "burst": True}).score_joint(c, [], {"u0": 4})[1]
        zero_weight = Evaluator(splash_weight=0.).score_joint(c, [], {"u0": 4})[1]
        leaving_range = Evaluator().score_joint(c, [], {"u0": 5})[1]
        self.assertGreater(enabled["future_attack"], disabled["future_attack"])
        self.assertEqual(disabled["future_attack"], zero_weight["future_attack"])
        self.assertGreater(enabled["future_attack"], leaving_range["future_attack"])
        self.assertEqual(leaving_range["future_attack"], 0.)
        self.assertEqual(enabled["coverage"], 0.)
        self.assertEqual(enabled["useful_damage"], 0.)

    def test_urgency_config_changes_objective_threat_value_not_plan_packaging(self):
        # Threat removal requires a known lethal volley; chip damage alone
        # does not suppress an enemy weapon or protect threatened allies.
        c = context(scene([((0., 0.), "marine"), ((0., 1.), "marine")],
                          {"e0": enemy((2., 0.), health=1.)}))
        actions = {"u0": 6, "u1": 6}
        disabled = Evaluator(urgency_value=0.).score_joint(c, [], actions)[1]
        evaluator = Evaluator(urgency_value=1.)
        bare = evaluator.score_joint(c, [], actions)[1]
        wrapped = evaluator.score_joint(c, [plan(c, actions, {"kind": "none"})], actions)[1]
        self.assertEqual(disabled["target_threat"], 0.)
        self.assertGreater(bare["target_threat"], 0.)
        self.assertEqual(bare, wrapped)
        self.assertEqual(bare["useful_damage"], disabled["useful_damage"])
        self.assertLessEqual(bare["target_threat"], evaluator.weights["target_threat"])

    def test_joint_rejoin_credits_final_pair_progress_once_and_cover_still_fires(self):
        c = context(scene([((0., 0.), "stalker"), ((10., 0.), "stalker")],
                          {"e0": enemy((14., 0.))}, attack_range=6.))
        data = {"joiners": ["u0"], "supporters": ["u1"], "support_radius": 6.,
                "readiness_fraction": 1.}
        join = {"u0": 4, "u1": 6}
        wait = {"u0": 1, "u1": 6}
        evaluator = Evaluator()
        approach = evaluator.score_joint(c, [plan(c, join, {"kind": "combat_rejoin"}, data)], join)[1]
        holding = evaluator.score_joint(c, [plan(c, wait, {"kind": "combat_rejoin"}, data)], wait)[1]
        self.assertGreater(approach["link"], 0.)
        self.assertEqual(holding["link"], 0.)
        self.assertLess(holding["rally_wait"], 0.)
        self.assertEqual(approach["useful_damage"], holding["useful_damage"])
        self.assertGreater(approach["useful_damage"], 0.)
        # If the cover unit also retreats away, the declaration alone earns nothing.
        divergent = {"u0": 4, "u1": 4}
        misleading = evaluator.score_joint(c,
            [plan(c, join, {"kind": "combat_rejoin"}, data)], divergent)[1]
        self.assertEqual(misleading["link"], 0.)

    def test_recorded_safe_prediction_includes_the_visible_pursuing_zealot(self):
        root = pathlib.Path(__file__).resolve().parents[3]
        log = root / "runs/tactical_protoss_10v10_3x20_postfix_20261113/6z4s/events.jsonl"
        if not log.exists():
            self.skipTest("optional real-run regression log is not present")
        payload = None
        with log.open(encoding="utf-8") as stream:
            for line in stream:
                if '"event": "skill_observation"' not in line:
                    continue
                row = json.loads(line)
                if row["episode"] == 1 and row["step"] == 34:
                    payload = row["payload"]
                    break
        self.assertIsNotNone(payload)
        c = Context(build_snapshot(payload, 35))
        point = c.predict_position("u5", 5).position
        self.assertEqual(c.risk_at("u5", point).threats, ())
        self.assertIn("e9", c.forecast_risk("u5", point).reachable_threats)
        self.assertLess(Evaluator().score_joint(c, [], {"u5": 5})[1]["risk"], 0.)

    def test_combat_rejoin_commitment_survives_chip_damage_but_not_cover_emergency(self):
        state = scene([((0., 0.), "stalker"), ((10., 0.), "stalker")],
                      {"e0": enemy((14., 0.), "stalker")}, loop=88)
        memory = TacticalMemory()
        memory.create_instance("combat_rejoin", ("u0", "u1"), "group", "rejoining",
            80, 64, goal_kind="area", extra={"min_loops": 16,
                "interrupt_on": ("danger_change", "illegal"),
                "supporters": ["u1"], "joiners": ["u0"],
                "initial_threat_counts": {"u0": 0, "u1": 1}})
        state["allies"]["u1"].update(health=44., health_lost_since_previous_decision=1.)
        actions = {"u0": 4, "u1": 6}
        evaluator = Evaluator()
        chip = evaluator.score_joint(context(state), [], actions, memory)[1]
        self.assertLess(chip["commitment_switch"], 0.)
        state["allies"]["u1"].update(health=5., health_lost_since_previous_decision=20.)
        emergency = evaluator.score_joint(context(state), [], actions, memory)[1]
        self.assertEqual(emergency["commitment_switch"], 0.)


if __name__ == "__main__":
    unittest.main()
