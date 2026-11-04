"""Mode isolation and actual portable SkillManager wiring contracts."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from medivac_tree import MedivacTree
from medivac_tree.tests.fixtures import observe_heal, payload, scene, unit
from sc2_skill_adapter.bridge.skill_runtime import (
    _load_module, load_manager, manager_configuration)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.terran = _load_module("skills.terran_combat")

    def state(self):
        state = scene([unit("Marine", (0, 0), health=20),
                       unit("Marauder", (1, 0), health=80),
                       unit("Medivac", (0, 1)), unit("Medivac", (1, 1))])
        observe_heal(state, "u2", "u0")
        return state

    def test_default_and_explicit_legacy_are_identical(self):
        state = self.state()
        implicit = self.terran.allocate_terran_combat(state, 4)
        explicit = self.terran.allocate_terran_combat(
            state, 4, medivac_policy_mode="legacy", medical_patient_coordination=True)
        self.assertEqual(implicit, explicit)
        self.assertNotIn("medivac_tree", implicit["race_telemetry"])

    def test_shadow_preserves_actions_and_all_legacy_state_over_multiple_windows(self):
        state = self.state()
        legacy_memory, shadow_memory = {}, {}
        runtime = MedivacTree()
        for _ in range(5):
            legacy = self.terran.allocate_terran_combat(state, 4, race_memory=legacy_memory)
            shadow = self.terran.allocate_terran_combat(
                state, 4, race_memory=shadow_memory, medivac_policy_mode="tree-shadow",
                medivac_tree=runtime)
            legacy_memory, shadow_memory = legacy["race_memory"], shadow["race_memory"]
            comparison = copy.deepcopy(shadow)
            self.assertIn("medivac_tree_shadow", comparison["race_memory"])
            comparison["race_memory"].pop("medivac_tree_shadow")
            self.assertEqual(comparison["race_telemetry"].pop("medivac_tree")["mode"], "tree-shadow")
            self.assertEqual(legacy, comparison)
            state["game_loop"] += 8

    def test_active_tree_bypasses_legacy_healing_and_preserves_base_ground_actions(self):
        state = self.state()
        base_results = []
        original = self.terran.allocate_adaptive_focus_fire

        def capture_base(*args, **kwargs):
            result = original(*args, **kwargs)
            base_results.append(copy.deepcopy(result))
            return result

        with patch.object(self.terran, "allocate_adaptive_focus_fire", side_effect=capture_base), \
                patch.object(self.terran, "assign_healing", side_effect=AssertionError("legacy healer used")), \
                patch.object(self.terran, "_support_retreat", side_effect=AssertionError("patient override used")):
            result = self.terran.allocate_terran_combat(
                state, 4, medivac_policy_mode="tree", medivac_tree=MedivacTree(),
                medical_patient_coordination=False)
        self.assertEqual(len(base_results), 1)
        self.assertEqual(result["actions"][:2], base_results[0]["actions"][:2])
        tree_trace = result["race_telemetry"]["medivac_tree"]
        for actor, decision in tree_trace["decisions"].items():
            self.assertEqual(decision["final_action"], result["actions"][int(actor[1:])])
            self.assertEqual(decision["action"], decision["final_action"])
            self.assertIsNone(decision["override_reason"])
        self.assertEqual(result["race_memory"].keys(), {"medivac_tree"})

    def test_no_healer_active_mode_leaves_every_base_action_unchanged(self):
        state = scene([unit("Marine", (0, 0)), unit("Marauder", (1, 0))])
        baseline = self.terran.allocate_adaptive_focus_fire(
            state, 2, parameters={"known_compatible_threats": True})
        result = self.terran.allocate_terran_combat(
            state, 2, medivac_policy_mode="tree", medivac_tree=MedivacTree(),
            medical_patient_coordination=False)
        self.assertEqual(result["actions"], baseline["actions"])
        self.assertEqual(result["race_telemetry"]["medivac_tree"]["decisions"], {})

    def test_portable_manager_builds_medical_mask_from_questions_without_mutation(self):
        state = self.state()
        request = payload(state)
        # The bridge normally carries its authoritative mask in criteria.
        del request["state"]["legal_actions"]
        before = copy.deepcopy(request)
        manager = load_manager()(race_profile="terran", medivac_policy_mode="tree",
                                 medivac_tree=MedivacTree(), medical_patient_coordination=False)
        manager.select("adaptive_focus_fire", posture="balanced")
        actions, telemetry = manager.step(request, 4)
        self.assertEqual(request, before)
        self.assertEqual(len(actions), 4)
        self.assertEqual(telemetry["race_telemetry"]["medivac_tree"]["mode"], "tree")
        for actor in ("u2", "u3"):
            self.assertIn(str(actions[int(actor[1:])]), request["questions"][actor]["criteria"])

    def test_manager_rejects_unsafe_mode_combinations(self):
        factory = load_manager()
        cases = [
            {"race_profile": "off", "medivac_policy_mode": "tree", "medivac_tree": MedivacTree()},
            {"race_profile": "terran", "medivac_policy_mode": "unknown"},
            {"race_profile": "terran", "medivac_policy_mode": "tree"},
            {"race_profile": "terran", "medivac_policy_mode": "tree", "medivac_tree": MedivacTree()},
            {"race_profile": "terran", "medivac_policy_mode": "tree-shadow", "medivac_tree": MedivacTree(),
             "pair_scoring": "teacher"},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                factory(**kwargs)

    def config(self, **changes):
        values = dict(skill_version="current", race_profile="terran", medivac_policy="tree",
                      medivac_tree=None, medical_patient_coordination=None,
                      terran_keep_heal_order=True, pair_mode="off", native_teacher_control="off")
        values.update(changes)
        return manager_configuration(SimpleNamespace(**values))

    def test_configuration_automatically_freezes_ground_actions_for_active_tree(self):
        configured = self.config()
        self.assertFalse(configured["manager_kwargs"]["medical_patient_coordination"])
        self.assertEqual(configured["manager_kwargs"]["medivac_policy_mode"], "tree")
        self.assertIsInstance(configured["manager_kwargs"]["medivac_tree"], MedivacTree)

    def test_configuration_rejects_teacher_pair_overlay_and_missing_keep(self):
        for changes in ({"pair_mode": "teacher"}, {"native_teacher_control": "teacher"},
                        {"terran_keep_heal_order": False}, {"enable_regroup": True},
                        {"race_profile": "zerg"}, {"medical_patient_coordination": "on"},
                        {"medivac_policy": "legacy", "medivac_tree": "any.json"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.config(**changes)


if __name__ == "__main__":
    unittest.main()
