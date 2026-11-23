#!/usr/bin/env python3
"""Resolve the complete tactical_v1 source tree (base + overlay -> staging).

Linux entry for ZergSkillSourceResolver. Deterministic and repeatable; the
output carries source_manifest.json with per-file provenance and the
whole-source fingerprint used in model context records.

Usage (from the repository root):
    python decision_tree/scripts/linux/resolve_source.py
    python decision_tree/scripts/linux/resolve_source.py --clean
    python decision_tree/scripts/linux/resolve_source.py --check-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision_tree.zerg.source import DEFAULT_BASE_DIR, DEFAULT_OUTPUT_DIR, \
    DEFAULT_OVERLAY_DIR, ZergSkillSourceResolver, load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--overlay-dir", default=DEFAULT_OVERLAY_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workspace", default=str(ROOT))
    parser.add_argument("--clean", action="store_true",
                        help="rebuild the output tree from scratch")
    parser.add_argument("--check-only", action="store_true",
                        help="only report whether the inputs changed since "
                             "the last resolve")
    args = parser.parse_args()

    resolver = ZergSkillSourceResolver(
        base_dir=args.base_dir, overlay_dir=args.overlay_dir,
        output_dir=args.output_dir, workspace=args.workspace)

    if args.check_only:
        manifest_path = Path(resolver.output_dir) / "source_manifest.json"
        if not manifest_path.is_file():
            print(json.dumps({"resolved": False,
                              "changed": None,
                              "manifest": str(manifest_path)}))
            return 2
        changed = resolver.has_changed()
        manifest = load_manifest(str(resolver.output_dir))
        print(json.dumps({"resolved": True, "changed": changed,
                          "fingerprint": manifest["fingerprint"],
                          "counts": manifest["counts"]}))
        return 1 if changed else 0

    result = resolver.resolve(clean_output=args.clean)
    print(json.dumps({
        "output_dir": str(result.output_dir),
        "fingerprint": result.fingerprint,
        "files": result.file_count,
        "counts": {
            "base": result.base_count,
            "overlay": result.overlay_count,
            "overridden": result.overridden_count,
            "overlay_only": result.overlay_only_count,
        },
        "syntax_errors": result.manifest["syntax_errors"],
    }, indent=2))
    return 1 if result.manifest["syntax_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
