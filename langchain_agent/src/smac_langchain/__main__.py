"""smac-langchain CLI.

Commands: doctor | demo | codex-smoke | agent-smoke | run | smoke-test
Real-model and real-game paths are opt-in (--live-llm / backend smacv2).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config, resolve_path
from . import __version__


def make_backend(name: str, config: dict) -> object:
    if name == "mock":
        from .backends.mock import MockBackend
        return MockBackend(resolve_path("skills/current"))
    if name == "smacv2":
        from .backends.rpc_client import RPCClientBackend
        python = config.get("paths", {}).get("backend_python") or sys.executable
        worker = Path(__file__).resolve().parents[2] / "workers" / "smacv2_worker.py"
        if not worker.is_file():
            raise SystemExit(f"missing worker: {worker}")
        return RPCClientBackend([python, str(worker)], cwd=str(resolve_path(".")))
    raise SystemExit(f"unsupported backend: {name} (smacv1 is not implemented in "
                     "this version and is never faked with smacv2)")


def fresh_run_dir(base: Path, prefix: str) -> Path:
    import time
    candidate = base / f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"
    if candidate.exists():
        raise SystemExit(f"will not overwrite existing run dir: {candidate}")
    return candidate


def cmd_doctor(args) -> int:
    config = load_config(args.config)
    report = {"version": __version__, "config": config["_config_path"]}
    from .llm.codex_client import find_codex, CodexError, CodexCapabilityProbe
    try:
        exe = find_codex(config.get("codex", {}).get("codex_path") or None)
        report["codex_executable"] = str(exe)
        report["codex_version"] = CodexCapabilityProbe.version(exe)
        help_text = CodexCapabilityProbe.exec_help(exe)
        for flag in ("--output-schema", "--output-last-message", "--json",
                     "--sandbox", "--ephemeral", "--disable"):
            report[f"codex_supports_{flag.strip('-').replace('-', '_')}"] = flag in help_text
    except (CodexError, OSError) as exc:
        report["codex_executable"] = None
        report["codex_error"] = str(exc)
    import importlib.metadata as im
    report["deps"] = {}
    for name in ("langchain", "langchain-core", "pydantic"):
        try:
            report["deps"][name] = im.version(name)
        except im.PackageNotFoundError:
            report["deps"][name] = None
    paths = {}
    for key in ("skills_current", "smac_root", "sc2_root"):
        value = config.get("paths", {}).get(key)
        if value:
            resolved = resolve_path(value)
            paths[key] = {"path": str(resolved), "exists": resolved.exists()}
    report["paths"] = paths
    worker = Path(config["_pkg_root"]) / "workers" / "smacv2_worker.py"
    report["smacv2_worker"] = {"path": str(worker), "exists": worker.is_file()}
    # offline mock backend self-check (no SC2, no model)
    try:
        from .backends.mock import MockBackend
        backend = MockBackend(resolve_path("skills/current"))
        backend.start(0, 1, {"max_steps": 4})
        snap = backend.snapshot()
        backend.execute(next(iter(snap["candidates"])), 2)
        backend.close()
        report["mock_backend"] = "ok"
    except Exception as exc:  # noqa: BLE001
        report["mock_backend"] = f"error: {exc}"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_demo(args) -> int:
    if args.backend != "mock":
        raise SystemExit("demo requires --backend mock (offline)")
    if args.model != "mock":
        raise SystemExit("demo requires --model mock (offline; no Codex launch)")
    config = load_config(args.config)
    from .agent import run_agent_episode
    backend = make_backend("mock", config)
    run_dir = fresh_run_dir(resolve_path(config["paths"].get("run_output", "langchain_agent/runs")),
                            "demo-mock")
    summary = run_agent_episode(backend, config, run_dir, episode_id=1, seed=args.seed,
                                model_name="mock")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"run dir: {run_dir}")
    return 0


def cmd_codex_smoke(args) -> int:
    if not args.live_llm:
        print("codex-smoke does nothing without --live-llm (real model calls are opt-in)")
        return 1
    config = load_config(args.config)
    from .llm.codex_client import CodexCLI, CodexError
    client = CodexCLI(
        model=args.model or config["codex"].get("model"),
        reasoning_effort=config["codex"].get("reasoning_effort", "medium"),
        timeout_seconds=config["codex"].get("timeout_seconds", 180),
        executable=config["codex"].get("codex_path") or None,
        request_root=".codex_smoke")
    prompt = ('Reply with {"kind":"final","tool_name":null,"arguments_json":null,'
              '"final_text":"pong"} exactly.')
    try:
        envelope = client.request(prompt)
    except CodexError as exc:
        print(f"FAIL: {exc}")
        return 2
    print(json.dumps({"ok": True, "kind": envelope["kind"], "stats": client.stats},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_agent_smoke(args) -> int:
    config = load_config(args.config)
    if args.backend != "mock":
        raise SystemExit("agent-smoke requires --backend mock (no SC2 launch)")
    if not args.live_llm:
        raise SystemExit("agent-smoke is a REAL Codex test; pass --live-llm")
    from .agent import run_agent_episode
    backend = make_backend("mock", config)
    run_dir = fresh_run_dir(resolve_path(config["paths"].get("run_output", "langchain_agent/runs")),
                            "agent-smoke")
    summary = run_agent_episode(backend, config, run_dir, episode_id=1, seed=args.seed,
                                model_name=args.model or config["codex"].get("model"),
                                live=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_run(args) -> int:
    config = load_config(args.config)
    backend = make_backend(args.backend, config)
    run_dir = fresh_run_dir(resolve_path(config["paths"].get("run_output", "langchain_agent/runs")),
                            f"run-{args.policy}-{args.backend}")
    from .agent import run_agent_episode, run_pure_skill_episode
    if args.policy == "pure-skill":
        summary = run_pure_skill_episode(backend, config, run_dir,
                                         episode_id=args.episode, seed=args.seed)
    elif args.policy == "agent":
        if not args.live_llm:
            raise SystemExit("policy agent performs real model calls; pass --live-llm "
                             "(or use `demo` for the offline mock loop)")
        summary = run_agent_episode(backend, config, run_dir, episode_id=args.episode,
                                    seed=args.seed,
                                    model_name=args.model or config["codex"].get("model"),
                                    live=True)
    else:
        raise SystemExit(f"unknown policy: {args.policy}")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"run dir: {run_dir}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="smac-langchain")
    parser.add_argument("--config", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="local checks; no model, no game")
    demo = sub.add_parser("demo", help="offline full agent demo (mock model + mock backend)")
    demo.add_argument("--backend", default="mock")
    demo.add_argument("--model", default="mock")
    demo.add_argument("--seed", type=int, default=1)
    codex_smoke = sub.add_parser("codex-smoke", help="real Codex structured-output test")
    codex_smoke.add_argument("--live-llm", action="store_true")
    codex_smoke.add_argument("--model", default=None)
    smoke = sub.add_parser("agent-smoke", help="real Codex + mock backend loop")
    smoke.add_argument("--live-llm", action="store_true")
    smoke.add_argument("--model", default=None)
    smoke.add_argument("--backend", default="mock")
    smoke.add_argument("--seed", type=int, default=1)
    run = sub.add_parser("run", help="run an episode on a backend")
    run.add_argument("--backend", default="mock", choices=("mock", "smacv2"))
    run.add_argument("--policy", default="pure-skill", choices=("pure-skill", "agent"))
    run.add_argument("--live-llm", action="store_true")
    run.add_argument("--model", default=None)
    run.add_argument("--episode", type=int, default=1)
    run.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    handlers = {"doctor": cmd_doctor, "demo": cmd_demo, "codex-smoke": cmd_codex_smoke,
                "agent-smoke": cmd_agent_smoke, "run": cmd_run}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
