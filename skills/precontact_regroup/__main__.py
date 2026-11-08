"""Run adaptive with the pre-contact regroup skill on a Protoss 10v10 manifest."""

import argparse
from pathlib import Path
import subprocess
import sys

from decision_tree.scripts.study import DEFAULT_FLAGS
from . import POLICY_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sc2-path", type=Path)
    args = parser.parse_args()

    runner = Path(__file__).resolve().parents[2] / "sc2_skill_adapter" / "run.py"
    command = [sys.executable, "-B", str(runner), "run", *DEFAULT_FLAGS,
               "--local-policy", str(POLICY_PATH), "--manifest", str(args.manifest),
               "--output-dir", str(args.output_dir)]
    if args.sc2_path:
        command.extend(["--sc2-path", str(args.sc2_path)])
    return subprocess.run(command).returncode


if __name__ == "__main__":
    raise SystemExit(main())
