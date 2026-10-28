"""Unified launcher for isolated SMACv1 (SMAC-Hard) and SMACv2 backends."""

import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
GAME_ROOT = next((parent / "game_skill" for parent in (WORKSPACE, WORKSPACE.parent)
                  if (parent / "game_skill/micro/pysc2/__init__.py").is_file()),
                 WORKSPACE / "game_skill")
V1_COMMANDS = {
    "run": "jev_micro.run",
    "suite": "jev_micro.suite",
    "video": "jev_micro.replay_video",
    "realtime": "jev_micro.realtime",
    "unit-value-ablation": "jev_micro.unit_value_ablation",
}


def launch_spec(backend, arguments):
    """Return a command, cwd and environment without importing either backend."""
    binary_suffix = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if backend == "smacv1":
        binary = GAME_ROOT / ".venvs/micro" / binary_suffix
        cwd = GAME_ROOT / "micro"
        command = arguments[0] if arguments and arguments[0] in V1_COMMANDS else "run"
        forwarded = arguments[1:] if arguments and arguments[0] in V1_COMMANDS else arguments
        environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "smacv1"), str(cwd)))
        invocation = ["-m", V1_COMMANDS[command], *forwarded]
    elif backend == "smacv2":
        binary = ROOT / ".venv" / binary_suffix
        cwd = ROOT
        environment["PYTHONPATH"] = str(ROOT)
        invocation = [str(ROOT / "run.py"), *arguments]
    else:
        raise ValueError(f"Unknown backend: {backend}")
    if not binary.is_file():
        raise FileNotFoundError(f"Missing {backend} Python environment: {binary}")
    return [str(binary), "-X", "utf8", "-B", *invocation], cwd, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "smacv1 [run|suite|video|realtime|unit-value-ablation] ARGS; "
        "smacv2 ARGS. Use BACKEND --help for backend options. "
        "Relative data paths retain each backend's original working directory."))
    parser.add_argument("backend", choices=("smacv1", "smacv2"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    try:
        command, cwd, environment = launch_spec(args.backend, forwarded)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    return subprocess.call(command, cwd=cwd, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
