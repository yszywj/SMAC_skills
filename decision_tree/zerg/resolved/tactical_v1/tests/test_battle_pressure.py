"""Known exposure and lethal pressure relief, without false fire suppression."""
import copy
import unittest

from skills.tactical_v1.battle_pressure import enemy_pressure, joint_pressure
from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.effects import physical_effects
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.focus_fire import FocusFireSkill
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import enemy, scene, with_questions


def context(state):
    return Context(build_snapshot(with_questions(state), 1))


def measured(state, actions):
    c = context(state)
    report = physical_effects(c, actions)
    before = copy.deepcopy(report)
    result = joint_pressure(c, actions, report)
    assert before == report, "pressure forecast must not change the damage ledger"
    return result, report


class BattlePressureTests(unittest.TestCase):
    def test_exposing_a_fragile_unit_increases_enemy_pressure(self):
        state = scene([((0., 0.), "stalker")], {"e0": enemy((4., 0.), "stalker")}, attack_range=6.)
        ordinary = enemy_pressure(context(state), "e0")
        state["allies"]["u0"]["health"] = 10.
        fragile = enemy_pressure(context(state), "e0")
        self.assertGreater(fragile, ordinary)
        self.assertLessEqual(fragile, 1.)

    def test_late_id_high_pressure_attacker_survives_candidate_quota(self):
        enemies = {f"e{i}": enemy((3., i * .01), "zealot", health=200., max_health=200.)
                   for i in range(6)}
        enemies["e9"] = enemy((5., 0.), "colossus", health=200., max_health=200.)
        c = context(scene([((0., 0.), "stalker")], enemies, attack_range=6.))
        skill = FocusFireSkill({"search": {"max_candidates_per_unit": 2}})
        ordered = skill._ordered_targets(c, "u0", sorted(enemies), "e0")
        self.assertEqual(ordered[:2], ["e0", "e9"])
        self.assertGreater(enemy_pressure(c, "e9"), enemy_pressure(c, "e0"))

    def test_pressure_ranking_does_not_hardcode_ranged_unit_types(self):
        enemies = {"e0": enemy((9., 0.), "stalker", health=200., max_health=200.),
                   "e9": enemy((1., 0.), "zealot", health=200., max_health=200.)}
        c = context(scene([((0., 0.), "colossus")], enemies, attack_range=10.))
        self.assertGreater(enemy_pressure(c, "e9"), enemy_pressure(c, "e0"))
        order = FocusFireSkill({})._ordered_targets(c, "u0", sorted(enemies), None)
        self.assertEqual(order[0], "e9")

    def test_immediate_kill_class_survives_pressure_ranking(self):
        enemies = {"e0": enemy((3., 0.), "zealot", health=1.),
                   "e9": enemy((5., 0.), "colossus", health=200., max_health=200.)}
        c = context(scene([((0., 0.), "stalker")], enemies, attack_range=6.))
        plans = FocusFireSkill({"search": {"max_candidates_per_unit": 1}}).propose(
            c, TacticalMemory(), Telemetry())
        self.assertEqual([p.goal for p in plans], ["e0"])

    def support_scene(self, rear_type):
        state = scene([((0., 0.), "zealot"), ((0., 1.), rear_type)],
                      {"e0": enemy((4., 0.), "stalker", health=200., max_health=200.)},
                      attack_range=6.)
        state["allies"]["u0"]["health"] = 15.
        for actor in ["u0"] + (["u1"] if rear_type == "zealot" else []):
            state["relations"][actor]["e0"].update(native_weapon_range=.1,
                in_weapon_range=False, weapon_range_status="out_of_range",
                smac_attack_available=False, smac_attack_distance_limit=2.)
        return state

    def test_ordinary_ranged_fire_does_not_cancel_frontline_exposure(self):
        melee, _ = measured(self.support_scene("zealot"), {"u0": 1, "u1": 1})
        ranged, _ = measured(self.support_scene("stalker"), {"u0": 1, "u1": 6})
        self.assertEqual(melee["unsupported_actors"]["u0"],
                         ranged["unsupported_actors"]["u0"])
        self.assertEqual(ranged["pressure_relief"], 0.)

    def test_movement_into_future_support_has_no_current_damage_credit(self):
        state = scene([((0., 0.), "zealot"), ((12., 0.), "stalker")],
                      {"e0": enemy((5., 0.), "stalker", health=200., max_health=200.)},
                      attack_range=6.)
        state["allies"]["u0"]["health"] = 15.
        state["relations"]["u0"]["e0"].update(native_weapon_range=.1,
            in_weapon_range=False, weapon_range_status="out_of_range",
            smac_attack_available=False, smac_attack_distance_limit=2.)
        holding, hold_report = measured(state, {"u0": 1, "u1": 1})
        joining, move_report = measured(state, {"u0": 1, "u1": 5})
        self.assertEqual(joining["unsupported_actors"]["u0"], holding["unsupported_actors"]["u0"])
        self.assertEqual(move_report.damage_known, {})
        self.assertEqual(joining["pressure_relief"], 0.)

    def test_nonlethal_target_allocation_does_not_reduce_exposure(self):
        state = self.support_scene("stalker")
        state["visible_enemies"]["e1"] = enemy((-4., 0.), "stalker", health=200., max_health=200.)
        for actor in state["relations"]:
            state["relations"][actor]["e1"] = dict(state["relations"][actor]["e0"])
        left, _ = measured(state, {"u0": 1, "u1": 6})
        both_idle, _ = measured(state, {"u0": 1, "u1": 1})
        self.assertGreater(left["unsupported_actors"]["u0"], 0.)
        self.assertEqual(left["unsupported_actors"], both_idle["unsupported_actors"])
        self.assertEqual(left["pressure_relief"], 0.)

    def test_unknown_enemy_weapon_cannot_create_quantified_relief(self):
        state = scene([((0., 0.), "stalker")], {"e0": enemy((3., 0.))}, attack_range=6.)
        del state["unit_type_profiles"]["105"]
        result, report = measured(state, {"u0": 6})
        self.assertGreater(result["unknown_enemy_pairs"], 0)
        self.assertEqual(result["pressure_relief"], 0.)

    def test_incompatible_enemy_does_not_threaten_flying_healer(self):
        state = scene([((0., 0.), "medivac")], {"e0": enemy((1., 0.), "zealot")},
                      enemy_can_reach=False)
        result, _ = measured(state, {"u0": 1})
        self.assertEqual(result["unsupported_exposure"], 0.)
        self.assertEqual(enemy_pressure(context(state), "e0"), 0.)

    def test_known_lethal_shot_earns_relief_once_without_assuming_first_strike(self):
        state = scene([((0., 0.), "stalker")], {"e0": enemy((3., 0.), "zealot", health=1.)}, attack_range=6.)
        result, report = measured(state, {"u0": 6})
        waiting, _ = measured(state, {"u0": 1})
        self.assertGreater(result["pressure_relief"], 0.)
        self.assertGreater(result["unsupported_exposure"], 0.)
        self.assertEqual(result["unsupported_exposure"], waiting["unsupported_exposure"])
        self.assertEqual(len(report.damage_known), 1)

    def test_nominal_lethal_but_uncertain_lower_bound_has_no_relief(self):
        c = context(scene([((0., 0.), "stalker")],
                          {"e0": enemy((3., 0.), "zealot", health=10.)}, attack_range=6.))
        report = physical_effects(c, {"u0": 6})
        report.damage_bounds["e0"] = (5., 20.)
        self.assertGreaterEqual(report.damage_known["e0"], 10.)
        self.assertEqual(joint_pressure(c, {"u0": 6}, report)["pressure_relief"], 0.)

    def test_cooling_attack_has_no_relief_even_when_nominal_volley_is_lethal(self):
        state = scene([((0., 0.), "stalker")],
                      {"e0": enemy((3., 0.), "zealot", health=1.)}, attack_range=6.)
        state["allies"]["u0"]["weapon_cooldown_loops"] = 100.
        result, report = measured(state, {"u0": 6})
        self.assertEqual(report.damage_known, {})
        self.assertEqual(result["pressure_relief"], 0.)

    def test_more_overkill_sources_do_not_duplicate_threat_relief(self):
        state = scene([((0., 0.), "stalker"), ((0., 1.), "stalker")],
                      {"e0": enemy((3., 0.), "zealot", health=1.)}, attack_range=6.)
        one, _ = measured(state, {"u0": 6, "u1": 1})
        two, _ = measured(state, {"u0": 6, "u1": 6})
        self.assertEqual(one["pressure_relief"], two["pressure_relief"])
        self.assertLessEqual(two["pressure_relief"], 1.)
        self.assertEqual(one["unsupported_actors"], two["unsupported_actors"])

    def test_helper_results_are_stable_and_do_not_depend_on_prior_calls(self):
        c = context(self.support_scene("stalker"))
        actions = {"u0": 1, "u1": 6}
        report = physical_effects(c, actions)
        expected = joint_pressure(c, actions, report)
        other = {"u0": 5, "u1": 5}
        joint_pressure(c, other, physical_effects(c, other))
        self.assertEqual(expected, joint_pressure(c, actions, report))


if __name__ == "__main__":
    unittest.main()
