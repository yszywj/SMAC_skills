"""JSONL telemetry: decisions, commands, model stats, provenance.

Model-visible data and hidden/evaluation data are written to separate files.
No credentials, raw auth errors or reasoning traces are ever logged.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


class Telemetry:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.events_path = self.run_dir / "events.jsonl"
        self._events = self.events_path.open("w", encoding="utf-8")
        self._hidden_path = self.run_dir / "hidden_events.jsonl"
        self._hidden = None  # opened lazily, only if hidden data is ever recorded

    def __call__(self, event: str, **fields: Any) -> None:
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **fields}
        self._events.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._events.flush()

    def hidden(self, event: str, **fields: Any) -> None:
        """Evaluation-only data, kept out of events.jsonl (never model-visible)."""
        if self._hidden is None:
            self._hidden = self._hidden_path.open("w", encoding="utf-8")
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **fields}
        self._hidden.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._hidden.flush()

    def provenance(self, config: dict[str, Any], extra_files: list[Path] | None = None) -> dict:
        """Record dependency versions and source hashes (SHA-256, no Git metadata)."""
        import importlib.metadata as im
        versions = {}
        for name in ("langchain", "langchain-core", "pydantic"):
            try:
                versions[name] = im.version(name)
            except im.PackageNotFoundError:
                versions[name] = None
        versions["python"] = sys.version.split()[0]
        hashes = {}
        roots = [Path(config.get("_pkg_root", "."))]
        for source in (extra_files or []):
            roots.append(Path(source))
        seen: set[Path] = set()
        for root in roots:
            root = Path(root)
            candidates = [root] if root.is_file() else sorted(root.rglob("*.py"))
            for path in candidates:
                if path in seen or not path.is_file():
                    continue
                seen.add(path)
                try:
                    hashes[str(path)] = hashlib.sha256(
                        path.read_bytes()).hexdigest()[:16]
                except OSError:
                    continue
        record = {"versions": versions, "source_sha256": hashes,
                  "config": {k: v for k, v in config.items() if not k.startswith("_")},
                  "codex": {"model": config.get("codex", {}).get("model"),
                            "reasoning_effort": config.get("codex", {}).get("reasoning_effort")}}
        self("provenance", **record)
        return record

    def close(self) -> None:
        self._events.close()
        if self._hidden is not None:
            self._hidden.close()
