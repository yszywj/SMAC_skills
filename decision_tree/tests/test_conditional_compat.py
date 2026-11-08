import unittest
from unittest.mock import patch

from decision_tree.scripts.conditional_compat import canonical, native_command, audit_compatibility
from skills.local_v3 import run as native


class ConditionalAuditCompatibilityTests(unittest.TestCase):
    def test_json_sequences_normalize_without_float_tolerance(self):
        raw = {"point": (11.864, 14.998), "actions": [4]}
        self.assertEqual(canonical(raw), {"point": [11.864, 14.998], "actions": [4]})
        self.assertIsInstance(raw["point"], tuple)
        self.assertNotEqual(canonical(raw), {"point": [11.865, 14.998], "actions": [4]})
        with self.assertRaises(ValueError):
            canonical({"x": float("nan")})

    def test_only_v3_native_command_is_redirected(self):
        v3 = ["python", "-B", "-m", "skills.local_v3.run", "--policy", "p.json"]
        self.assertEqual(native_command(v3),
                         ["python", "-B", "-m", "decision_tree.scripts.conditional_compat",
                          "native", "--policy", "p.json"])
        adaptive = ["python", "-B", "-m", "skills.local_v2.run"]
        self.assertEqual(native_command(adaptive), adaptive)
        self.assertEqual(v3[3], "skills.local_v3.run")

    def test_controller_replacement_is_limited_to_audit(self):
        original = native.ConditionalController
        seen = []

        def audit():
            seen.append(native.ConditionalController)
            raise ValueError("test")

        with patch.object(native, "_audit_episode", audit):
            with audit_compatibility():
                self.assertIs(native.ConditionalController, original)
                with self.assertRaisesRegex(ValueError, "test"):
                    native._audit_episode()
                self.assertIs(native.ConditionalController, original)
            self.assertIs(native._audit_episode, audit)
        self.assertTrue(issubclass(seen[0], original))
