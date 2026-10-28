"""Verify maintenance regeneration against migrated sources in a temporary folder."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import vendor_controller


class VendorControllerTests(unittest.TestCase):
    def test_mixed_source_snapshot_records_actual_paths_without_overwriting_adapter(self):
        names = ("client.py", "feedback.py", "policy.py", "planner.py")
        protected = [*(vendor_controller.DEST / name for name in names),
                     vendor_controller.ROOT / "SOURCE_MANIFEST.json"]
        before = {path: path.read_bytes() for path in protected}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "bridge"
            destination.mkdir()
            manifest_path = root / "SOURCE_MANIFEST.json"
            vendor_controller.main(destination, manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(set(manifest), set(names))
            for name in names:
                source = (vendor_controller.SKILL_SOURCE if name == "client.py"
                          else vendor_controller.SOURCE) / name
                source_name = source.relative_to(vendor_controller.ROOT.parent).as_posix()
                self.assertEqual(manifest[name]["source"], source_name)
                self.assertEqual(manifest[name]["source_sha256"],
                                 hashlib.sha256(source.read_bytes()).hexdigest())
                generated = (destination / name).read_text(encoding="utf-8")
                self.assertIn(source_name, generated.splitlines()[0])
                ast.parse(generated, filename=str(destination / name))
            self.assertEqual(manifest["client.py"]["source"], "skills/current/client.py")
        self.assertEqual(before, {path: path.read_bytes() for path in protected})


if __name__ == "__main__":
    unittest.main()
