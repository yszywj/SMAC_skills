#!/usr/bin/env python3
"""Audit the latest zerg tactical_v1 source (base + overlay merge).

Linux-first companion to decision_tree/zerg/source.py. Reports the merged
tree composition, the overlay/base differences and the whole-source
fingerprint that every model context record carries.

Usage (from the repository root):
    python decision_tree/scripts/linux/inspect_zerg.py
    python decision_tree/scripts/linux/inspect_zerg.py --json out.json
    python decision_tree/scripts/linux/inspect_zerg.py --no-resolve
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision_tree.zerg.source import (DEFAULT_BASE_DIR, DEFAULT_OVERLAY_DIR,
                                       ZergSkillSourceResolver, _scan,
                                       merge_file_map, sha256_file)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--overlay-dir", default=DEFAULT_OVERLAY_DIR)
    parser.add_argument("--output-dir",
                        default="decision_tree/zerg/resolved/tactical_v1")
    parser.add_argument("--workspace", default=str(ROOT))
    parser.add_argument("--no-resolve", action="store_true",
                        help="report the merge plan without materialising it")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="write the report as JSON")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    base = Path(args.base_dir)
    overlay = Path(args.overlay_dir)
    if not base.is_absolute():
        base = workspace / base
    if not overlay.is_absolute():
        overlay = workspace / overlay

    merged = merge_file_map(base, overlay)
    base_files = set(_scan(base))
    overlay_files = set(_scan(overlay))
    rows = []
    for rel in sorted(merged):
        origin, path = merged[rel]
        rows.append({"path": rel, "source": origin,
                     "sha256": sha256_file(path)})

    resolver = ZergSkillSourceResolver(
        base_dir=str(base), overlay_dir=str(overlay),
        output_dir=args.output_dir, workspace=str(workspace))
    changed = None
    if not args.no_resolve:
        result = resolver.resolve()
        changed = resolver.has_changed()
        fingerprint = result.fingerprint
        counts = result.manifest["counts"]
        syntax_errors = result.manifest["syntax_errors"]
    else:
        from decision_tree.zerg.source import source_fingerprint
        fingerprint = source_fingerprint(base, overlay)
        counts = {"base": sum(1 for r in rows if r["source"] == "base"),
                  "overlay": sum(1 for r in rows if r["source"] == "overlay")}
        syntax_errors = []

    report = {
        "workspace": str(workspace),
        "base_dir": str(base), "overlay_dir": str(overlay),
        "fingerprint": fingerprint,
        "file_count": len(rows),
        "counts": counts,
        "syntax_errors": syntax_errors,
        "resolved": not args.no_resolve,
        "changed_since_last_manifest": changed,
        "overlay_only": sorted(overlay_files - base_files),
        "base_only": sorted(base_files - overlay_files),
    }
    print(json.dumps({k: v for k, v in report.items()
                      if k != "overlay_only"}, indent=2))
    if report["overlay_only"]:
        print("overlay-only files:")
        for path in report["overlay_only"]:
            print(f"  + {path}")
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({**report, "files": rows}, indent=2) + "\n",
                       encoding="utf-8")
        print(f"report written: {out}")
    return 1 if syntax_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
