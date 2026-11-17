"""Participation across roles without turning safety into fictitious damage."""
import copy
import unittest

from skills.tactical_v1.battle_pressure import joint_pressure
from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.evaluator import Evaluator
from skills.tactical_v1.memory import TacticalMemory
from .fixtures import enemy, scene, with_questions


def context(state):
    return Context(build_snapshot(with_questions(state), 1))


class SustainedContactScoringTests(unittest.TestCase):
    def rear_scene(self, name="stalker", reach=6.):
        state = scene([((0., 0.), name), ((10., 0.), name)],
                      {"e0": enemy((12., 0.), "zergling")}, attack_range=reach)
        for unit in state["allies"].values():
            unit.update(health=160., max_health=160., effective_health=160.)
        return state

    def test_multistep_advance_values_participation_across_combat_roles(self):
        for name, reach in (("marine", 5.), ("zealot", 2.),
                            ("hydralisk", 5.), ("colossus", 7.)):
            with self.subTest(role=name):
                c = context(self.rear_scene(name, reach))
                evaluator = Evaluator()
                hold = evaluator.score_joint(c, [], {"u0": 1, "u1": 6})[1]
                move = evaluator.score_joint(c, [], {"u0": 4, "u1": 6})[1]
                self.assertEqual(move["future_attack"], 0.)
                self.assertGreater(move["sustained_contact"], 0.)
                self.assertLess(hold["combat_idle"], 0.)
                self.assertEqual(hold["useful_damage"], move["useful_damage"])
                self.assertEqual(hold["expected_kills"], move["expected_kills"])

    def test_no_persistent_advance_credit_before_observed_combat(self):
        state = self.rear_scene()
        state["allies"]["u1"]["alive"] = False
        c = context(state)
        self.assertEqual(Evaluator().score_joint(c, [], {"u0": 4, "u1": 0})[1]
                         ["sustained_contact"], 0.)

    def test_known_safe_low_reserve_can_advance_but_unknown_geometry_cannot(self):
        for kind in ("critical", "unknown_range", "unknown_speed", "incompatible", "unknown_mask"):
            with self.subTest(kind=kind):
                state = self.rear_scene()
                if kind == "critical":
                    state["allies"]["u0"].update(health=5., effective_health=5.)
                elif kind == "unknown_range":
                    state["relations"]["u0"]["e0"]["native_weapon_range"] = None
                elif kind == "incompatible":
                    state["relations"]["u0"]["e0"]["weapon_range_status"] = "no_compatible_weapon"
                elif kind == "unknown_mask":
                    state["relations"]["u0"]["e0"].pop("smac_attack_distance_limit")
                    state["action_contract"]["candidate_center_distance_by_type"] = {}
                else:
                    state["unit_type_profiles"]["74"].pop("movement_speed_native")
                c = context(state)
                value = Evaluator().score_joint(c, [], {"u0": 4, "u1": 6})[1]['sustained_contact']
                if kind == 'critical':
                    # Low HP alone does not make a currently safe route lethal.
                    self.assertGreater(value, 0.)
                else:
                    self.assertEqual(value, 0.)

    def test_ready_current_shot_cannot_gain_credit_by_postponement(self):
        state = scene([((0., 0.), "stalker")], {"e0": enemy((4., 0.))}, attack_range=6.)
        c = context(state)
        for action in (4, 5):
            self.assertEqual(Evaluator().score_joint(c, [], {"u0": action})[1]
                             ["sustained_contact"], 0.)

    def test_dangerous_advance_is_not_rewarded_and_escape_is_not_penalized(self):
        state = scene([((0., 0.), "zealot"), ((6., 0.), "zealot")],
                      {"e0": enemy((8., 0.), "colossus")}, attack_range=2.)
        state["allies"]["u0"].update(health=25., max_health=100., effective_health=25.)
        c = context(state)
        for action in (4, 5):
            self.assertEqual(Evaluator().score_joint(c, [], {"u0": action, "u1": 6})[1]
                             ["sustained_contact"], 0.)

    def test_risk_and_exposure_combine_per_actor_without_adding_same_volley(self):
        state = scene([((0., 0.), "stalker"), ((30., 0.), "stalker")],
                      {"e0": enemy((4., 0.), "colossus")}, attack_range=6.)
        c = context(state)
        actions = {"u0": 1, "u1": 1}
        for weights in ({}, {"risk": 0.}, {"risk": .02}, {"unsupported_frontline": 0.}):
            with self.subTest(weights=weights):
                evaluator = Evaluator(weights)
                report = evaluator.effects(c, actions)
                before = copy.deepcopy(report)
                costs, _ = evaluator._forecast_costs(c, report)
                pressure = joint_pressure(c, actions, report)
                breakdown = evaluator.score_joint(c, [], actions)[1]
                expected = sum(max(costs.get(actor, 0.), evaluator.weights["unsupported_frontline"] * exposure)
                               for actor, exposure in pressure["unsupported_actors"].items())
                self.assertAlmostEqual(-breakdown["risk"] - breakdown["unsupported_frontline"], expected)
                self.assertEqual(report, before)

    def test_nonlethal_shooting_does_not_reduce_exposure(self):
        state = scene([((0., 0.), "stalker")],
                      {"e0": enemy((4., 0.), "colossus", health=300., max_health=300.)}, attack_range=6.)
        c, evaluator = context(state), Evaluator()
        shooting = evaluator.score_joint(c, [], {"u0": 6})[1]
        holding = evaluator.score_joint(c, [], {"u0": 1})[1]
        self.assertEqual(shooting["risk"], holding["risk"])
        self.assertEqual(shooting["unsupported_frontline"], holding["unsupported_frontline"])
        self.assertEqual(shooting["target_threat"], 0.)

    def test_stalled_press_increases_feasible_advance_urgency_without_changing_risk(self):
        c, evaluator, memory = context(self.rear_scene()), Evaluator(), TacticalMemory()
        state = {"engaged": True, "phase": "engage", "targets": ("e0",)}
        memory.battle_state = lambda _: dict(state)
        memory.battle_target = lambda _c, _a: "e0"
        results = {}
        for phase in ("engage", "press", "engage"):
            state["phase"] = phase
            hold = evaluator.score_joint(c, [], {"u0": 1, "u1": 6}, memory)[1]
            move = evaluator.score_joint(c, [], {"u0": 4, "u1": 6}, memory)[1]
            if phase in results:
                self.assertEqual(results[phase], (hold, move))
            results[phase] = (hold, move)
        ordinary_hold, ordinary_move = results["engage"]
        pressed_hold, pressed_move = results["press"]
        self.assertAlmostEqual(pressed_move["sustained_contact"],
                               1.25 * ordinary_move["sustained_contact"])
        self.assertAlmostEqual(pressed_hold["combat_idle"],
                               1.25 * ordinary_hold["combat_idle"])
        self.assertGreater(pressed_move["total"] - pressed_hold["total"],
                           ordinary_move["total"] - ordinary_hold["total"])
        for key in ("risk", "unsupported_frontline", "useful_damage", "expected_kills"):
            self.assertEqual(pressed_move[key], ordinary_move[key])

    def test_press_does_not_boost_low_reserve_or_postpone_ready_shot(self):
        for kind in ("critical", "ready"):
            with self.subTest(kind=kind):
                state = self.rear_scene()
                if kind == "critical":
                    state["allies"]["u0"].update(health=5., effective_health=5.)
                else:
                    state = scene([((0., 0.), "stalker"), ((1., 0.), "stalker")],
                                  {"e0": enemy((4., 0.))}, attack_range=6.)
                c, memory = context(state), TacticalMemory()
                memory.battle_state = lambda _: {"engaged": True, "phase": "press", "targets": ("e0",)}
                memory.battle_target = lambda _c, _a: "e0"
                result = Evaluator().score_joint(c, [], {"u0": 4, "u1": 6}, memory)[1]
                if kind == 'critical':
                    ordinary = Evaluator().score_joint(c, [], {'u0': 4, 'u1': 6})[1]
                    self.assertGreater(result['sustained_contact'], 0.)
                    self.assertEqual(result['sustained_contact'], ordinary['sustained_contact'])
                else:
                    self.assertEqual(result["sustained_contact"], 0.)

    def test_locally_reachable_battle_target_joins_distant_global_objective(self):
        state = scene([((0., 0.), "stalker"), ((-8., 0.), "stalker")],
                      {"e0": enemy((30., 0.), "zergling"),
                       "e1": enemy((-10., 0.), "zergling")}, attack_range=6.)
        c, evaluator, memory = context(state), Evaluator(), TacticalMemory()
        memory.battle_state = lambda _: {"engaged": True, "phase": "engage", "targets": ("e0",)}
        local = {"target": None}
        memory.battle_target = lambda _c, _a: local["target"]
        unsupported = evaluator.score_joint(c, [], {"u0": 5, "u1": 7}, memory)[1]
        local["target"] = "e1"
        supporting = evaluator.score_joint(c, [], {"u0": 5, "u1": 7}, memory)[1]
        self.assertLess(unsupported["sustained_contact"], 0.)
        self.assertGreater(supporting["sustained_contact"], 0.)


if __name__ == "__main__":
    unittest.main()
