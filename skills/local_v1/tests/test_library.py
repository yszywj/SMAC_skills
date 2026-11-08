import copy
import math
import unittest

from skills.local_v1 import (FEATURE_NAMES, generate_candidates, apply_candidate,
                            deduplicate_candidates, effect_key)


def fixture():
    allies = {}
    for i, position in enumerate(((0., 0.), (0., 1.))):
        allies[f"u{i}"] = dict(alive=True, type="Stalker", unit_type_id=74,
                                position=list(position), health=80., shield=80.,
                                max_health=80., max_shield=80., weapon_ready=True,
                                is_flying=False)
    enemies = {"e0": dict(alive=True, type="Marine", unit_type_id=48, position=[2., 0.],
                          health=30., shield=0., max_health=45., max_shield=0., is_flying=False),
               "e1": dict(alive=True, type="Marine", unit_type_id=48, position=[3., 0.],
                          health=20., shield=0., max_health=45., max_shield=0., is_flying=False)}
    state = dict(allies=allies, visible_enemies=enemies, game_loop=8, step_mul=8,
                 relations={}, movement_candidates={}, heal_candidates={},
                 unit_type_profiles={"74": {"known": True, "weapons": [
                     {"target_type": "any", "base_damage": 10., "hits_per_attack": 1}]},
                                     "48": {"known": True, "armor": 0.}})
    questions = {}
    for actor, unit in allies.items():
        state["relations"][actor] = {e: dict(smac_attack_available=True, in_weapon_range=True,
                                              in_enemy_weapon_range=True,
                                              distance=math.dist(unit["position"], enemy["position"]))
                                     for e, enemy in enemies.items()}
        moves = {}
        for action, delta in ((2, (0, 1)), (3, (0, -1)), (4, (1, 0)), (5, (-1, 0))):
            destination = [unit["position"][j] + delta[j] for j in range(2)]
            moves[str(action)] = dict(destination=destination,
                                     distances_to_visible_enemies={e: math.dist(destination, enemy["position"])
                                                                  for e, enemy in enemies.items()})
        state["movement_candidates"][actor] = moves
        questions[actor] = {"criteria": {str(i): "legal" for i in range(1, 8)}}
    baseline = dict(actions=[6, 6], roles={"u0": "attack", "u1": "attack"},
                    attack_assignments={"u0": "e0", "u1": "e0"},
                    predicted_damage={"e0": 20.}, estimated_overkill={"e0": 0.},
                    immediate_damage={"e0": 20.}, window_only_damage={},
                    all_in=False, all_in_reason=None, phase="engage", reengaged=[],
                    attack_units=["u0", "u1"], retreat_units=[], support_units=[],
                    retreat_actions={}, retreat_reasons={},
                    approach_state_updates={"u0": {"old": True}, "u1": {"keep": True}},
                    engagement_state_updates={"u0": {"old": True}},
                    movement_diagnostics={"u0": {"old": True}},
                    approach_diagnostics={}, reentry_decisions={}, unit_value_diagnostics={})
    return {"state": state, "questions": questions}, baseline


class LocalSkillTests(unittest.TestCase):
    def test_retarget_is_scoped_and_final_budget_moves_only_its_contribution(self):
        payload, baseline = fixture()
        before = copy.deepcopy((payload, baseline))
        candidate = next(c for c in generate_candidates(payload, baseline)
                         if c["family"] == "retarget" and c["actors"] == ["u0"])
        result = apply_candidate(payload, baseline, candidate)
        self.assertEqual((payload, baseline), before)
        self.assertEqual(result["actions"], [7, 6])
        self.assertEqual(result["attack_assignments"], {"u0": "e1", "u1": "e0"})
        self.assertEqual(result["predicted_damage"], {"e1": 10., "e0": 10.})
        self.assertNotIn("u0", result["approach_state_updates"])
        self.assertEqual(result["approach_state_updates"]["u1"], {"keep": True})
        self.assertNotIn("u0", result["engagement_state_updates"])
        self.assertEqual(result["local_changed_actors"], ["u0"])
        self.assertEqual(set(candidate["features"]), set(FEATURE_NAMES))
        self.assertTrue(all(math.isfinite(x) for x in candidate["features"].values()))

    def test_mask_and_visibility_both_required_for_attack(self):
        payload, baseline = fixture()
        del payload["questions"]["u0"]["criteria"]["7"]
        self.assertFalse(any(c["actors"] == ["u0"] for c in generate_candidates(payload, baseline)))
        payload["questions"]["u0"]["criteria"]["7"] = "attack"
        del payload["state"]["visible_enemies"]["e1"]
        self.assertEqual(generate_candidates(payload, baseline), [])

    def test_missing_menu_abstains_and_medical_actor_never_owned(self):
        payload, baseline = fixture()
        payload["state"]["allies"]["u0"]["type"] = "Medivac"
        del payload["questions"]["u1"]
        self.assertEqual(generate_candidates(payload, baseline), [])

    def test_duplicate_effects_alias_instead_of_extra_candidates(self):
        payload, baseline = fixture()
        candidate = generate_candidates(payload, baseline)[0]
        alias = copy.deepcopy(candidate)
        alias["id"] = "retarget-other-rule:u0:7"
        result = deduplicate_candidates([candidate, alias])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["aliases"], [alias["id"]])
        self.assertEqual(result[0]["effect_key"], candidate["effect_key"])

    def test_deterministic_cap_does_not_modify_inputs(self):
        payload, baseline = fixture()
        actual = generate_candidates(payload, baseline, {"max_candidates": 1})
        payload["state"]["allies"] = dict(reversed(list(payload["state"]["allies"].items())))
        self.assertEqual(generate_candidates(payload, baseline, {"max_candidates": 1}), actual)
        self.assertEqual(len(actual), 1)

    def test_retreat_veto_removes_retreat_and_reentry_proposal(self):
        payload, baseline = fixture()
        baseline["actions"][0] = 5
        baseline["roles"]["u0"] = "retreat"
        baseline["retreat_actions"] = {"u0": 5}
        baseline["reentry_decisions"] = {"u0": {"old": True}}
        candidate = next(c for c in generate_candidates(payload, baseline) if c["family"] == "retreat_veto")
        result = apply_candidate(payload, baseline, candidate)
        self.assertEqual(result["roles"]["u0"], "attack")
        self.assertNotIn("u0", result["retreat_actions"])
        self.assertNotIn("u0", result["reentry_decisions"])
        self.assertIn("u0", result["reengaged"])

    def test_retreat_step_removes_firepower_from_final_budget(self):
        payload, baseline = fixture()
        payload["state"]["allies"]["u0"].update(health=5., shield=0.)
        candidate = next(c for c in generate_candidates(payload, baseline) if c["family"] == "retreat_step")
        result = apply_candidate(payload, baseline, candidate)
        self.assertEqual(candidate["actors"], ["u0"])
        self.assertEqual(result["attack_assignments"], {"u1": "e0"})
        self.assertEqual(result["predicted_damage"], {"e0": 10.})
        self.assertEqual(result["retreat_units"], ["u0"])
        self.assertEqual(result["actions"][1], baseline["actions"][1])

    def test_regroup_requires_isolation_and_no_legal_attack(self):
        payload, baseline = fixture()
        payload["state"]["allies"]["u0"]["position"] = [-20., 0.]
        payload["state"]["relations"]["u0"] = {}
        payload["state"]["movement_candidates"]["u0"] = {
            "4": {"destination": [-19., 0.]}, "5": {"destination": [-21., 0.]}}
        baseline["actions"][0], baseline["roles"]["u0"] = 1, "reposition"
        candidate = next(c for c in generate_candidates(payload, baseline) if c["family"] == "regroup_step")
        self.assertEqual(candidate["actions"], {"u0": 4})
        result = apply_candidate(payload, baseline, candidate)
        self.assertEqual(result["actions"], [4, 6])

    def test_rejects_widened_scope_stale_mask_noop_and_corrupt_fingerprint(self):
        payload, baseline = fixture()
        candidate = generate_candidates(payload, baseline)[0]
        bad = copy.deepcopy(candidate)
        bad["actions"]["u1"] = 7
        with self.assertRaises(ValueError):
            apply_candidate(payload, baseline, bad)
        bad = copy.deepcopy(candidate)
        bad["effect_key"] = "corrupt"
        with self.assertRaises(ValueError):
            apply_candidate(payload, baseline, bad)
        bad = copy.deepcopy(candidate)
        bad["actions"]["u0"] = 6
        bad["effect_key"] = effect_key(bad)
        with self.assertRaises(ValueError):
            apply_candidate(payload, baseline, bad)
        del payload["questions"]["u0"]["criteria"]["7"]
        with self.assertRaises(ValueError):
            apply_candidate(payload, baseline, candidate)

    def test_unknown_weapon_data_not_credited_as_damage(self):
        payload, baseline = fixture()
        payload["state"]["unit_type_profiles"] = {}
        candidate = generate_candidates(payload, baseline)[0]
        result = apply_candidate(payload, baseline, candidate)
        self.assertEqual(sum(result["predicted_damage"].values()), 0.)
        self.assertEqual(result["damage_model"]["unknown_actors"], ["u0", "u1"])

    def test_actual_adaptive_allocator_contract(self):
        from skills.current.skills.adaptive_focus_fire import allocate_adaptive_focus_fire
        payload, _ = fixture()
        baseline = allocate_adaptive_focus_fire(payload["state"], 2)
        candidates = generate_candidates(payload, baseline)
        self.assertTrue(candidates)
        for candidate in candidates:
            final = apply_candidate(payload, baseline, candidate)
            changed = [i for i, (a, b) in enumerate(zip(baseline["actions"], final["actions"])) if a != b]
            self.assertEqual(changed, [int(candidate["actors"][0][1:])])


if __name__ == "__main__":
    unittest.main()
