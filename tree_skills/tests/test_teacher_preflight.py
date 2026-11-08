"""Teacher registration works in a fresh checkout without published local audits."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tree_skills.teacher import preflight


class TeacherPreflightTests(unittest.TestCase):
    def test_missing_manifest_is_absent_and_can_register_local_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "TEACHER_MANIFEST.json"
            checkpoint = root / "fixture.pt"
            checkpoint.write_bytes(b"synthetic checkpoint bytes")
            expected = [{"file": checkpoint.name, "size": checkpoint.stat().st_size,
                         "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                         "url": "https://example.invalid/fixture.pt", "license": "MIT"}]
            with patch.object(preflight, "MANIFEST_PATH", str(manifest)), \
                    patch.object(preflight, "EXPECTED", expected):
                initial = preflight.teacher_status()
                self.assertEqual(initial["state"], "absent")
                self.assertEqual(initial["checkpoints"], {})
                self.assertFalse(manifest.exists())
                registered = preflight.register_checkpoints(str(root))
                self.assertEqual(registered["state"], "downloaded")
                self.assertEqual(registered["checkpoints"][checkpoint.name]["status"], "registered")
                self.assertEqual(json.loads(manifest.read_text()), registered)
                preflight.record_audit(checkpoint.name, {"ok": True, "details": {"fixture": True}})
                self.assertEqual(preflight.register_checkpoints(str(root))["checkpoints"][checkpoint.name]["status"], "audited")

    def test_existing_manifest_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "TEACHER_MANIFEST.json"
            original = {"state": "ready", "checkpoints": {"existing.pt": {"status": "audited"}},
                        "custom_state": "keep existing local audit"}
            manifest.write_text(json.dumps(original), encoding="utf-8")
            before = manifest.read_bytes()
            with patch.object(preflight, "MANIFEST_PATH", str(manifest)):
                self.assertEqual(preflight.teacher_status(), original)
            self.assertEqual(manifest.read_bytes(), before)

    def test_invalid_existing_manifest_does_not_silently_reset_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "TEACHER_MANIFEST.json"
            manifest.write_text("{incomplete", encoding="utf-8")
            with patch.object(preflight, "MANIFEST_PATH", str(manifest)):
                with self.assertRaises(json.JSONDecodeError):
                    preflight.teacher_status()


if __name__ == "__main__":
    unittest.main()
