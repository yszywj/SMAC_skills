"""Run the predeclared 200-game experiment with frozen inputs and resumable groups."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUTPUT = ROOT / "analysis/spatial_ablation_20261025"
GROUPS = {"all": ["-EnableRegroup", "-EnableCohesiveSearch", "-EnableCombatPositioning"],
          "off": [], "regroup": ["-EnableRegroup"], "search": ["-EnableCohesiveSearch"],
          "positioning": ["-EnableCombatPositioning"]}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def hashes(paths):
    return {p.relative_to(WORKSPACE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Start or resume the frozen experiment")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    python = ROOT / ".venv/Scripts/python.exe"
    manifests = []
    for size in (5, 10):
        path = ROOT / f"configs/protoss_{size}_vs_{size}_native_n20_seed210.json"
        if not path.exists():
            subprocess.run([str(python), "-B", "run.py", "scenarios", "--size", str(size),
                "--episodes", "20", "--seed", "210", "--unit-balance", "native", "--output", str(path)],
                cwd=ROOT, check=True)
        manifests.append(path)
    paths = [*sorted((WORKSPACE/"skills/current").glob("*.py")),
             *sorted((WORKSPACE/"skills/current/skills").glob("*.py")),
             *sorted((ROOT/"bridge").glob("*.py")), *sorted((ROOT/"scripts").glob("*.ps1")),
             WORKSPACE/"game_skill/micro/jev_micro/run.py", *manifests, Path(__file__),
             OUTPUT/"PROTOCOL.md"]
    paths += [WORKSPACE/"smacv2/smacv2/env/starcraft2"/name for name in
              ("starcraft2.py", "wrapper.py", "distributions.py", "maps/SMAC_Maps/10gen_protoss.SC2Map")]
    current = hashes(paths)
    frozen_path = OUTPUT/"frozen_hashes.json"
    if frozen_path.exists():
        if json.loads(frozen_path.read_text(encoding="utf-8")) != current:
            raise RuntimeError("Frozen inputs changed; do not mix experimental versions")
    else:
        for path in paths:
            target = OUTPUT/"frozen"/path.relative_to(WORKSPACE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        save(frozen_path, current)
    if not args.run:
        print("Prepared manifests and frozen inputs", flush=True)
        return
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if not shell:
        raise RuntimeError("PowerShell not found")
    ledger = OUTPUT/"execution.json"
    records = json.loads(ledger.read_text()) if ledger.exists() else []
    for group, flags in GROUPS.items():
        for size in (5, 10):
            if hashes(paths) != current:
                raise RuntimeError("Experimental inputs changed during execution")
            name = f"spatial-20261025-{group}-{size}v{size}-n20-seed210"
            destination = ROOT/"runs"/name
            summary_path = destination/"summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                if summary["status"] == "complete" and summary["completed_episodes"] == 20:
                    print(f"Already complete: {group} {size}v{size}", flush=True)
                    continue
            if destination.exists():
                raise RuntimeError(f"Incomplete output preserved; inspect before retrying: {destination}")
            command = [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
                       str(ROOT/"scripts/run_improved_skill.ps1"), "-Size", str(size), "-Episodes", "20",
                       "-Seed", "210", "-PolicyVersion", "v7", "-OutputName", name, *flags]
            record = {"group": group, "size": size, "output": str(destination), "command": command,
                      "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
            records.append(record)
            save(ledger, records)
            print(f"Starting {group} {size}v{size}, 20 episodes", flush=True)
            with (OUTPUT/f"{group}-{size}v{size}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            record.update(returncode=result.returncode, finished_utc=datetime.now(timezone.utc).isoformat())
            record["status"] = "complete" if result.returncode == 0 else "failed"
            save(ledger, records)
            if result.returncode:
                raise RuntimeError(f"Failed {group} {size}v{size}; retained logs and output")
            if hashes(paths) != current:
                raise RuntimeError("Experimental inputs changed during a group")
            summary = json.loads(summary_path.read_text())
            if summary["status"] != "complete" or summary["completed_episodes"] != 20:
                raise RuntimeError("Incomplete group")
            print(f"Completed {group} {size}v{size}: {summary['outcomes']}", flush=True)


if __name__ == "__main__":
    main()
