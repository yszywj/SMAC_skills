"""Behavior tests at the public tree boundary, independent of node layout."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from medivac_tree import MedivacTree
from medivac_tree.tests.fixtures import observe_heal, scene, unit


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tree = MedivacTree()

    def decide(self, state, memory=None):
        result = self.tree.decide(state, len(state["allies"]), memory=memory)
        # Every emitted action must remain in the bridge-provided legal set.
        for actor, action in result["actions"].items():
            self.assertIn(action, state["legal_actions"][actor])
            self.assertEqual(result["decisions"][actor]["action"], action)
        return result

    def test_dead_medivac_is_noop_and_not_assigned(self):
        dead = unit("Medivac", health=0)
        dead["alive"] = False
        state = scene([dead, unit("Marine", (1, 0), health=20)])
        result = self.decide(state)
        self.assertEqual(result["actions"], {"u0": 0})
        self.assertEqual(result["assignments"], {})
        self.assertEqual(result["decisions"]["u0"]["intent"], "NOOP")

    def test_tree_controls_only_medivacs(self):
        state = scene([unit("Marine", health=20), unit("Marauder"), unit("Medivac")])
        result = self.decide(state)
        self.assertEqual(set(result["actions"]), {"u2"})

    def test_legal_healing_binds_current_actor_and_patient(self):
        for medivac_index, target_index in [(0, 2), (2, 1)]:
            units = [unit("Marine", (1, 0)) for _ in range(3)]
            units[medivac_index] = unit("Medivac")
            units[target_index]["health"] = 20
            with self.subTest(medivac_index=medivac_index, target_index=target_index):
                result = self.decide(scene(units))
                actor, target = f"u{medivac_index}", f"u{target_index}"
                self.assertEqual(result["assignments"], {actor: target})
                self.assertEqual(result["actions"][actor], 6 + target_index)
                self.assertEqual(result["decisions"][actor]["target"], target)

    def test_full_health_never_becomes_heal_even_with_stale_candidate(self):
        state = scene([unit("Medivac"), unit("Marine", (1, 0))])
        state["heal_candidates"]["u0"]["7"] = {
            "target": "u1", "distance": 1, "missing_health": 20}
        state["legal_actions"]["u0"].append(7)
        result = self.decide(state)
        self.assertNotEqual(result["actions"]["u0"], 7)
        self.assertEqual(result["assignments"], {})

    def test_zero_energy_never_heals_even_with_stale_candidate(self):
        state = scene([unit("Medivac", energy=0), unit("Marine", (1, 0), health=20)])
        state["heal_candidates"]["u0"]["7"] = {
            "target": "u1", "distance": 1, "missing_health": 25}
        state["legal_actions"]["u0"].append(7)
        result = self.decide(state)
        self.assertNotEqual(result["actions"]["u0"], 7)
        self.assertEqual(result["assignments"], {})
        self.assertNotEqual(result["decisions"]["u0"]["intent"], "APPROACH")

    def test_masked_heal_is_never_submitted(self):
        state = scene([unit("Medivac"), unit("Marine", (1, 0), health=20)])
        state["legal_actions"]["u0"].remove(7)
        result = self.decide(state)
        self.assertNotEqual(result["actions"]["u0"], 7)
        self.assertEqual(result["assignments"], {})

    def test_missing_authoritative_medivac_mask_is_an_error(self):
        state = scene([unit("Medivac")])
        state["legal_actions"].pop("u0")
        with self.assertRaises(ValueError):
            self.tree.decide(state, 1)

    def test_ground_only_threat_does_not_interrupt_low_health_air_healer(self):
        state = scene([unit("Medivac", health=20), unit("Marine", (0, 1), health=20)],
                      {"e0": unit("Marauder", (1, 0))})
        result = self.decide(state)
        self.assertEqual(result["actions"]["u0"], 7)
        self.assertNotEqual(result["decisions"]["u0"]["intent"], "RETREAT")

    def test_air_capable_threat_triggers_legal_retreat_away_from_enemy(self):
        state = scene([unit("Medivac", health=20), unit("Marine", (0, 1), health=20)],
                      {"e0": unit("Marine", (5, 0))})
        result = self.decide(state)
        self.assertEqual(result["actions"]["u0"], 5)
        self.assertEqual(result["decisions"]["u0"]["intent"], "RETREAT")
        self.assertNotIn("u0", result["assignments"])

    def test_true_keep_is_adapter_action_not_stop(self):
        # An observed native heal order can follow a patient outside the
        # radius in which the bridge would allow a newly issued heal.
        state = scene([unit("Medivac"), unit("Marine", (8, 0), health=30)])
        keep_action = observe_heal(state, "u0", "u1")
        result = self.decide(state)
        self.assertEqual(result["actions"]["u0"], keep_action)
        self.assertNotEqual(keep_action, 1)
        self.assertEqual(result["decisions"]["u0"]["intent"], "KEEP_HEAL")
        self.assertEqual(result["assignments"], {"u0": "u1"})

    def test_without_keep_extension_current_order_uses_legal_heal_command(self):
        state = scene([unit("Medivac"), unit("Marine", (1, 0), health=30)])
        observe_heal(state, "u0", "u1", keep=False)
        result = self.decide(state)
        self.assertEqual(result["actions"]["u0"], 7)
        self.assertNotEqual(result["decisions"]["u0"]["intent"], "KEEP_HEAL")

    def test_keep_candidate_cannot_override_raw_mask(self):
        state = scene([unit("Medivac"), unit("Marine", (1, 0), health=30)])
        keep_action = observe_heal(state, "u0", "u1")
        state["legal_actions"]["u0"].remove(keep_action)
        result = self.decide(state)
        self.assertEqual(result["actions"]["u0"], 7)
        self.assertNotEqual(result["decisions"]["u0"]["intent"], "KEEP_HEAL")

    def test_no_patient_and_no_movement_produces_explicit_stop(self):
        state = scene([unit("Medivac")])
        state["movement_candidates"]["u0"] = {}
        state["legal_actions"]["u0"] = [1]
        result = self.decide(state)
        self.assertEqual(result["actions"], {"u0": 1})
        self.assertEqual(result["decisions"]["u0"]["intent"], "STOP")
        self.assertEqual(result["assignments"], {})

    def test_new_reservations_prevent_two_healers_choosing_same_patient(self):
        state = scene([unit("Medivac"), unit("Medivac", (0, 1)),
                       unit("Marine", (1, 0), health=20),
                       unit("Marine", (1, 1), health=25)])
        result = self.decide(state)
        self.assertEqual(set(result["assignments"]), {"u0", "u1"})
        self.assertEqual(set(result["assignments"].values()), {"u2", "u3"})

    def test_observed_occupancy_is_reserved_before_new_choices(self):
        state = scene([unit("Medivac"), unit("Medivac", (0, 1)),
                       unit("Marine", (1, 0), health=20),
                       unit("Marine", (1, 1), health=25)])
        observe_heal(state, "u1", "u2")
        result = self.decide(state)
        self.assertEqual(result["assignments"], {"u0": "u3", "u1": "u2"})

    def test_retreat_does_not_claim_native_order_already_released(self):
        state = scene([unit("Medivac", health=20), unit("Medivac", (0, 1)),
                       unit("Marine", (1, 0), health=20)],
                      {"e0": unit("Marine", (5, 0))})
        observe_heal(state, "u0", "u2")
        result = self.decide(state)
        self.assertEqual(result["decisions"]["u0"]["intent"], "RETREAT")
        self.assertNotIn("u2", result["assignments"].values())

    def test_trace_matches_leaf_and_action_and_inputs_are_not_mutated(self):
        state = scene([unit("Medivac"), unit("Marine", (1, 0), health=20)])
        before = copy.deepcopy(state)
        first = self.decide(state)
        memory_before = copy.deepcopy(first["memory"])
        second = self.decide(state, first["memory"])
        repeated = self.decide(state, first["memory"])
        self.assertEqual(second, repeated)
        self.assertEqual(state, before)
        self.assertEqual(first["memory"], memory_before)
        decision = second["decisions"]["u0"]
        self.assertTrue(decision["leaf"])
        self.assertTrue(decision["path"])
        self.assertTrue(decision["reason"])
        self.assertEqual(decision["target"], second["assignments"]["u0"])
        node = self.tree.document["root"]
        for split in decision["path"]:
            self.assertEqual(split["node"], node["id"])
            node = node["yes"] if split["result"] else node["no"]
        self.assertEqual(node["id"], decision["leaf"])
        self.assertEqual(node["intent"], decision["intent"])
        self.assertEqual(decision["memory_after"], second["memory"]["actors"]["u0"])
        # Traces must be persistable, not hold runtime objects or state aliases.
        json.dumps(second, allow_nan=False)

    def test_episode_change_discards_old_decision_history(self):
        state = scene([unit("Medivac"), unit("Marine", (1, 0), health=30)])
        observe_heal(state, "u0", "u1")
        old = self.decide(state)["memory"]
        changed = copy.deepcopy(state)
        changed["episode"] = 2
        changed["game_loop"] = 0
        with_old = self.decide(changed, old)
        fresh = self.decide(changed)
        self.assertEqual(with_old["actions"], fresh["actions"])
        self.assertEqual(with_old["assignments"], fresh["assignments"])
        self.assertEqual(with_old["memory"], fresh["memory"])
        self.assertEqual(with_old["memory"]["episode"], 2)
        self.assertEqual(with_old["memory_reset_reason"], "episode_changed")

    def test_stalled_keep_is_bounded_and_does_not_invent_healing_progress(self):
        state = scene([unit("Medivac"), unit("Marine", (8, 0), health=30)])
        observe_heal(state, "u0", "u1")
        intents, memory = [], None
        # No energy consumption or patient health change is observed.
        for i in range(8):
            state["game_loop"] = (i + 1) * 8
            result = self.decide(state, memory)
            memory = result["memory"]
            intents.append(result["decisions"]["u0"]["intent"])
        self.assertEqual(intents[0], "KEEP_HEAL")
        self.assertTrue(any(intent != "KEEP_HEAL" for intent in intents[1:]), intents)
        self.assertEqual(state["allies"]["u1"]["health"], 30)

    def test_rescue_switch_has_cooldown_when_native_order_has_not_changed(self):
        state = scene([unit("Medivac"), unit("Marine", (1, 0), health=30),
                       unit("Marine", (0, 1), health=5)])
        observe_heal(state, "u0", "u1")
        first = self.decide(state)
        self.assertEqual(first["decisions"]["u0"]["intent"], "HEAL_RESCUE")
        self.assertEqual(first["actions"]["u0"], 8)
        state["game_loop"] += 8
        second = self.decide(state, first["memory"])
        self.assertNotEqual(second["decisions"]["u0"]["intent"], "HEAL_RESCUE")
        self.assertEqual(state["allies"]["u0"]["orders"][0]["target_unit"], "u1")
        state["game_loop"] += 32
        recovered = self.decide(state, second["memory"])
        self.assertEqual(recovered["decisions"]["u0"]["intent"], "HEAL_RESCUE")

    def test_approach_without_observed_progress_is_not_persistent_forever(self):
        state = scene([unit("Medivac"), unit("Marine", (10, 0), health=20)])
        memory, intents = None, []
        for index in range(8):
            state["game_loop"] = (index + 1) * 8
            result = self.decide(state, memory)
            memory = result["memory"]
            intents.append(result["decisions"]["u0"]["intent"])
        self.assertEqual(intents[0], "APPROACH")
        self.assertTrue(any(intent != "APPROACH" for intent in intents[1:]), intents)
        self.assertEqual(result["assignments"], {})

    def test_identity_change_and_loop_rewind_reset_history(self):
        state = scene([unit("Medivac"), unit("Marine", (8, 0), health=30)])
        observe_heal(state, "u0", "u1")
        memory = None
        for index in range(3):
            state["game_loop"] = (index + 1) * 8
            memory = self.decide(state, memory)["memory"]
        for change in ("actor_tag", "loop_rewind"):
            with self.subTest(change=change):
                changed = copy.deepcopy(state)
                if change == "actor_tag":
                    changed["allies"]["u0"]["tag"] = "new-episode-unit"
                else:
                    changed["game_loop"] = 0
                reset = self.decide(changed, memory)
                fresh = self.decide(changed)
                self.assertEqual(reset["actions"], fresh["actions"])
                self.assertEqual(reset["memory"], fresh["memory"])

    def test_same_loop_cannot_be_reused_for_a_different_observation(self):
        state = scene([unit("Medivac"), unit("Marine", (1, 0), health=20)])
        first = self.decide(state)
        changed = copy.deepcopy(state)
        changed["allies"]["u1"]["health"] = 10
        with self.assertRaises(ValueError):
            self.tree.decide(changed, 2, first["memory"])

    def test_import_and_inference_do_not_need_training_libraries(self):
        workspace = Path(__file__).resolve().parents[2]
        code = """
import importlib.abc, sys
class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'sklearn', 'tree_skills'}:
            raise AssertionError('Forbidden training dependency: ' + fullname)
sys.meta_path.insert(0, BlockTrainingImports())
from medivac_tree import MedivacTree
from medivac_tree.tests.fixtures import scene, unit
state = scene([unit('Medivac'), unit('Marine', (1, 0), health=20)])
assert MedivacTree().decide(state, 2)['actions']['u0'] == 7
"""
        completed = subprocess.run([sys.executable, "-c", code], cwd=workspace,
                                   text=True, capture_output=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
