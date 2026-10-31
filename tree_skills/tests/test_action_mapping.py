"""Action-mapping tests (P0-2). Pure python; runs in the adapter venv."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tree_skills.teacher.action_mapping import (ActionMappingError,
                                                SmacActionMap)


class TestActionMapping(unittest.TestCase):
    def setUp(self):
        self.m = SmacActionMap(n_agents=10, n_enemies=10,
                               teacher_action_size=20)

    def test_native_layout(self):
        self.assertEqual(self.m.native_n_actions, 16)
        self.assertEqual(self.m.heal_slot(0), 6)
        self.assertEqual(self.m.heal_slot(9), 15)
        self.assertEqual(self.m.attack_slot_of_enemy[0], 6)
        self.assertEqual(self.m.attack_slot_of_enemy[9], 15)
        self.assertEqual(self.m.ally_of_heal_slot(6), 0)
        self.assertEqual(self.m.ally_of_heal_slot(15), 9)

    def test_bidirectional_roundtrip(self):
        for ally in range(10):
            self.assertEqual(self.m.ally_of_heal_slot(self.m.heal_slot(ally)), ally)

    def test_keep_extension_never_a_teacher_slot(self):
        keep = self.m.keep_action_id()
        self.assertEqual(keep, 16)
        with self.assertRaises(ActionMappingError):
            self.m.validate_teacher_slots([keep])
        with self.assertRaises(ActionMappingError):
            self.m.ally_of_heal_slot(keep)

    def test_padding_slots_rejected(self):
        # checkpoint head is 20 wide; 16..19 are cross-env padding
        with self.assertRaises(ActionMappingError):
            self.m.validate_teacher_slots([16, 17, 18, 19])
        for slot in range(16):
            self.assertTrue(self.m.is_native_slot(slot))
        self.assertTrue(self.m.is_padding_slot(19))
        self.assertFalse(self.m.is_padding_slot(15))

    def test_env_wider_than_teacher_head_rejected(self):
        with self.assertRaises(ActionMappingError):
            SmacActionMap(n_agents=15, n_enemies=15, teacher_action_size=20)

    def test_heal_and_attack_share_integer_range(self):
        """6+j is heal-ally-j for a medivac and attack-enemy-j for a marine:
        one integer, two semantics — callers must pick by unit type."""
        for j in range(10):
            self.assertEqual(self.m.heal_slot(j), self.m.attack_slot_of_enemy[j])


if __name__ == "__main__":
    unittest.main()
