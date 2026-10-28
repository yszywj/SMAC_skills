"""Execute the real PowerShell wrapper with a local fake Python command, never SC2."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/run_improved_skill.ps1"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")
V5_FLAGS = {"--adaptive-accurate-damage", "--adaptive-critical-escape", "--adaptive-planned-reentry"}
V6_FLAG = "--adaptive-engagement-commitment"
V7_FLAG = "--adaptive-approach-progress-guard"


def ps_string(value):
    return "'" + str(value).replace("'", "''") + "'"


@unittest.skipUnless(POWERSHELL, "PowerShell is required for the Windows run wrapper")
class ImprovedSkillWrapperTests(unittest.TestCase):
    def execute_wrapper(self, root, version=None, existing=False, extra="", skill_version=None):
        scripts, configs = root / "scripts", root / "configs"
        scripts.mkdir()
        configs.mkdir()
        source = SOURCE if skill_version is None else SOURCE.with_name("run_skill_version.ps1")
        script = scripts / source.name
        shutil.copyfile(source, script)
        manifest = configs / "protoss_10_vs_10_native_n20_seed150.json"
        manifest.write_text(json.dumps({"episodes": [{} for _ in range(20)], "seed": 150,
            "env_args": {"capability_config": {"n_units": 10, "n_enemies": 10}, "unit_balance": "native"}}))
        expected_version = version or "v6"
        output = root / "runs" / f"native-10v10-{expected_version}-skill-n20-seed150"
        if existing:
            output.mkdir(parents=True)
            (output / "preserve.txt").write_text("historical data")
        capture = root / "captured_arguments.json"
        harness = root / "harness.ps1"
        wrapper_python = root / ".venv/Scripts/python.exe"
        version_arg = " -PolicyVersion " + version if version else ""
        if skill_version is not None:
            version_arg = f" -SkillVersion {skill_version} -Seed 150 -OutputName {output.name}"
        harness.write_text("""
$ErrorActionPreference = 'Stop'
$wrapperPython = PYTHON_PATH
$capturePath = CAPTURE_PATH
Set-Item -LiteralPath "Function:$wrapperPython" -Value {
    param([Parameter(ValueFromRemainingArguments=$true)][object[]]$CapturedArguments)
    if ($CapturedArguments[0] -ne 'run.py' -or $CapturedArguments[1] -ne 'run') {
        throw 'Existing manifest should be reused; scenario generation is forbidden in this fixture'
    }
    ConvertTo-Json -InputObject @($CapturedArguments) | Set-Content -LiteralPath $capturePath
    $destinationIndex = [array]::IndexOf($CapturedArguments, '--output-dir') + 1
    $destination = $CapturedArguments[$destinationIndex]
    New-Item -ItemType Directory -Path $destination | Out-Null
    @{status='complete'; completed_episodes=20; skill_version='current'} | ConvertTo-Json |
        Set-Content -LiteralPath (Join-Path $destination 'summary.json')
    $global:LASTEXITCODE = 0
}
& WRAPPER_PATH -Size 10 VERSION_ARGUMENT
""".replace("PYTHON_PATH", ps_string(wrapper_python)).replace("CAPTURE_PATH", ps_string(capture))
            .replace("WRAPPER_PATH", ps_string(script)).replace("VERSION_ARGUMENT", version_arg + " " + extra), encoding="utf-8")
        result = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                                 "-File", str(harness)], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=30)
        arguments = json.loads(capture.read_text(encoding="utf-8-sig")) if capture.exists() else None
        return result, arguments, output, manifest

    def test_v4_through_v7_preserve_flags_and_default_to_paired_v6_output(self):
        for version in ("v4", "v5", "v6", "v7", None):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temporary:
                result, args, output, manifest = self.execute_wrapper(Path(temporary), version)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(args[args.index("--manifest") + 1], str(manifest))
                self.assertEqual(args[args.index("--output-dir") + 1], str(output))
                self.assertEqual(args[args.index("--method") + 1], "adaptive-skill")
                self.assertEqual(args[args.index("--unit-value-mode") + 1], "contextual")
                for flag in ("--adaptive-window-fire", "--adaptive-range-preserving-move", "--adaptive-target-scoring",
                             "--adaptive-coordinated-approach", "--adaptive-safe-reentry", "--unit-value-spatial-context"):
                    self.assertIn(flag, args)
                self.assertEqual(V5_FLAGS.intersection(args), set() if version == "v4" else V5_FLAGS)
                self.assertEqual(V6_FLAG in args, version in ("v6", "v7", None))
                self.assertEqual(V7_FLAG in args, version == "v7")
                self.assertEqual(len(args), len(set(args)))

    def test_existing_directory_is_rejected_before_python_and_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, args, output, _ = self.execute_wrapper(Path(temporary), existing=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Run directory already exists", result.stderr)
            self.assertIsNone(args)
            self.assertEqual((output / "preserve.txt").read_text(), "historical data")

    def test_both_wrappers_forward_each_spatial_ablation_and_layout(self):
        modes = {"": set(), "-EnableRegroup": {"--enable-regroup", "--auto-spatial-dispatch"},
                 "-EnableCohesiveSearch": {"--enable-cohesive-search", "--auto-spatial-dispatch"},
                 "-EnableCombatPositioning": {"--enable-combat-positioning"},
                 "-EnableRegroup -EnableCohesiveSearch -EnableCombatPositioning": {
                     "--enable-regroup", "--enable-cohesive-search", "--enable-combat-positioning",
                     "--auto-spatial-dispatch"}}
        all_flags = set.union(*modes.values())
        for skill_version in (None, "current"):
            for extra, expected in modes.items():
                with self.subTest(wrapper=skill_version, extra=extra), tempfile.TemporaryDirectory() as temporary:
                    result, args, _, _ = self.execute_wrapper(Path(temporary), extra=extra,
                                                             skill_version=skill_version)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(set(args) & all_flags, expected)
            with tempfile.TemporaryDirectory() as temporary:
                result, args, _, _ = self.execute_wrapper(Path(temporary), skill_version=skill_version,
                    extra="-EnableCombatPositioning -PositioningSpacing spread -PositioningLayout front_back")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(args[args.index("--positioning-spacing") + 1], "spread")
                self.assertEqual(args[args.index("--positioning-layout") + 1], "front_back")

    def test_version_wrapper_rejects_spatial_flags_on_both_baselines(self):
        for version in ("baseline_v2_70pct", "baseline_v6_20261024"):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temporary:
                result, args, _, _ = self.execute_wrapper(Path(temporary), skill_version=version,
                                                          extra="-EnableRegroup")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Historical skill versions", result.stderr)
                self.assertIsNone(args)


if __name__ == "__main__":
    unittest.main()
