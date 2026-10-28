"""Verify subprocess isolation and relocated sources without starting SC2."""
import json
import os
from pathlib import Path
import subprocess
import unittest

from adapter import ROOT, WORKSPACE, GAME_ROOT, launch_spec


class UnifiedLauncherTests(unittest.TestCase):
    def probe(self, backend, code):
        command, cwd, environment = launch_spec(backend, [])
        result = subprocess.run([*command[:4], "-c", code], cwd=cwd,
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_v1_sources_maps_and_vendored_dependency(self):
        paths = self.probe("smacv1", "import json, pysc2; from pathlib import Path; "
                           "from jev_micro import environment, skill_manager; "
                           "from jev_micro.catalog import map_source; "
                           "print(json.dumps([str(Path(p).resolve()) for p in "
                           "(environment.__file__, skill_manager.__file__, pysc2.__file__, map_source('3m'))]))")
        self.assertEqual(Path(paths[0]).parent, ROOT / "smacv1/jev_micro")
        self.assertEqual(Path(paths[1]).parent, WORKSPACE / "skills/current")
        self.assertEqual(Path(paths[2]).parent, GAME_ROOT / "micro/pysc2")
        self.assertTrue(Path(paths[3]).is_file())

    def test_v2_sources_and_dependency_isolation(self):
        paths = self.probe("smacv2", "import json, pysc2; from bridge import environment; "
                           "from bridge.skill_runtime import load_manager; "
                           "print(json.dumps([pysc2.__file__, environment.__file__, load_manager().__module__]))")
        self.assertNotIn("game_skill", paths[0])
        self.assertEqual(Path(paths[1]).resolve().parent, ROOT / "bridge")
        self.assertIn("_smacv2_portable_jev_skills", paths[2])

    def test_launcher_resolves_dependencies_without_legacy_junctions(self):
        command, cwd, environment = launch_spec("smacv1", ["--help"])
        self.assertEqual(cwd, GAME_ROOT / "micro")
        self.assertTrue(Path(command[0]).is_file())
        self.assertEqual(environment["PYTHONPATH"].split(os.pathsep)[0],
                         str(ROOT / "smacv1"))
        paths = self.probe("smacv1", "import json; from jev_micro.run import REPO, SKILL_SOURCE; "
                           "print(json.dumps([str(REPO), str(SKILL_SOURCE)]))")
        self.assertEqual(Path(paths[0]), GAME_ROOT / "micro")
        self.assertEqual(Path(paths[1]), WORKSPACE / "skills/current")

    def test_help_through_both_backends(self):
        for backend, arguments in (("smacv1", ["--help"]),
                                   ("smacv1", ["suite", "--help"]),
                                   ("smacv2", ["--help"])):
            command, cwd, environment = launch_spec(backend, arguments)
            result = subprocess.run(command, cwd=cwd, env=environment,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)
