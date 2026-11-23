"""Deterministic source resolution for the latest Zerg tactical_v1.

The newest skill update lives as an incremental snapshot (overlay) inside a
run directory; the base package lives at ``skills/tactical_v1``. The runnable
package is: base for every file, overlay overwriting same-named files,
overlay-only files added. This module materialises that merge into an
explicit staging directory — it never mutates either input tree and never
relies on sys.path tricks or dynamic multi-package imports to express the
overlay semantics.

Outputs are deterministic and repeatable: the same inputs always produce the
same file set, the same per-file SHA256 rows and the same whole-source
fingerprint, so the fingerprint can be embedded in model context records.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

MANIFEST_SCHEMA = "zerg_source_manifest_v1"
FINGERPRINT_SCHEMA = "zerg_source_fingerprint_v1"

# Non-source artifacts that never take part in resolution or fingerprinting.
IGNORED_DIRS = frozenset({"__pycache__"})
IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})

DEFAULT_BASE_DIR = "skills/tactical_v1"
DEFAULT_OVERLAY_DIR = ("runs/contact_all_attack_20261122_230107_662982/"
                       "marginal_near/skills/tactical_v1")
DEFAULT_OUTPUT_DIR = "decision_tree/zerg/resolved/tactical_v1"


def _scan(root: Path) -> Dict[str, Path]:
    """Recursively collect candidate files under ``root`` keyed by relative
    posix path, in sorted order (deterministic across platforms)."""
    found: Dict[str, Path] = {}
    if not root.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {root}")
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        parts = Path(rel).parts
        if any(part in IGNORED_DIRS for part in parts):
            continue
        if path.suffix.lower() in IGNORED_SUFFIXES:
            continue
        found[rel] = path
    return found


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_file_map(base_dir: Path,
                   overlay_dir: Path) -> Dict[str, Tuple[str, Path]]:
    """Compute the merged file map without touching disk outputs.

    Returns ``relative_path -> ("base" | "overlay", resolved path)``.
    """
    merged = {rel: ("base", path) for rel, path in _scan(base_dir).items()}
    for rel, path in _scan(overlay_dir).items():
        merged[rel] = ("overlay", path)
    return merged


def source_fingerprint(base_dir: Path, overlay_dir: Path) -> str:
    """Whole-source fingerprint of the would-be merged tree.

    Hashes the schema tag, then for every file in sorted path order: the
    relative path, the winning source tag and the file's SHA256. Byte-level
    content only — timestamps, permissions and directory layout are excluded,
    so equal inputs always give an equal fingerprint.
    """
    merged = merge_file_map(Path(base_dir), Path(overlay_dir))
    digest = hashlib.sha256(FINGERPRINT_SCHEMA.encode())
    for rel in sorted(merged):
        origin, path = merged[rel]
        digest.update(rel.encode())
        digest.update(origin.encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


@dataclass
class ResolveResult:
    output_dir: Path
    fingerprint: str
    manifest: Dict = field(repr=False)
    base_count: int = 0
    overlay_count: int = 0
    overridden_count: int = 0
    overlay_only_count: int = 0

    @property
    def file_count(self) -> int:
        return len(self.manifest["files"])


class ZergSkillSourceResolver:
    """Materialise base+overlay into one complete, traceable source tree."""

    def __init__(self, base_dir: Optional[str] = None,
                 overlay_dir: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 workspace: Optional[str] = None):
        self.workspace = Path(workspace) if workspace else self._default_workspace()
        self.base_dir = self._resolve_input(base_dir or DEFAULT_BASE_DIR)
        self.overlay_dir = self._resolve_input(overlay_dir or DEFAULT_OVERLAY_DIR)
        out = output_dir or DEFAULT_OUTPUT_DIR
        self.output_dir = Path(out) if Path(out).is_absolute() \
            else self.workspace / out

    @staticmethod
    def _default_workspace() -> Path:
        # decision_tree/zerg/source.py -> decision_tree -> repo root
        return Path(__file__).resolve().parents[2]

    def _resolve_input(self, spec: str) -> Path:
        path = Path(spec)
        if path.is_absolute():
            return path
        candidate = self.workspace / path
        if candidate.exists():
            return candidate
        # Allow callers to pass paths relative to the decision_tree package
        # (e.g. "zerg/resolved/tactical_v1" for a previous output).
        alt = Path(__file__).resolve().parent / path
        if alt.exists():
            return alt
        return candidate

    # -- resolution ---------------------------------------------------------

    def resolve(self, verify_syntax: bool = True,
                clean_output: bool = False) -> ResolveResult:
        """Materialise the merged tree into ``output_dir``.

        Staging: files are first copied into a private staging directory and
        moved into place at the end, so a failure never leaves a half-written
        tree. ``clean_output=True`` replaces an existing output wholesale;
        by default an existing identical tree is verified rather than
        rewritten (idempotent, cheap re-runs).
        """
        merged = merge_file_map(self.base_dir, self.overlay_dir)
        rows = self._build_rows(merged)

        if clean_output and self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        if self.output_dir.exists():
            self._verify_existing(rows)
        else:
            self._materialise(merged)

        syntax_errors: List[str] = []
        if verify_syntax:
            syntax_errors = self._check_syntax()

        fingerprint = self._fingerprint_from_rows(rows)
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "fingerprint_schema": FINGERPRINT_SCHEMA,
            "fingerprint": fingerprint,
            "base_dir": str(self.base_dir),
            "overlay_dir": str(self.overlay_dir),
            "output_dir": str(self.output_dir),
            "counts": {
                "base": sum(1 for r in rows if r["source"] == "base"
                            and not r["overridden"]),
                "overlay": sum(1 for r in rows if r["source"] == "overlay"),
                "overridden": sum(1 for r in rows if r["overridden"]),
                "total": len(rows),
            },
            "syntax_errors": syntax_errors,
            "files": rows,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / "source_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                 encoding="utf-8")

        counts = manifest["counts"]
        return ResolveResult(
            output_dir=self.output_dir,
            fingerprint=fingerprint,
            manifest=manifest,
            base_count=counts["base"],
            overlay_count=counts["overlay"],
            overridden_count=counts["overridden"],
            overlay_only_count=sum(1 for r in rows
                                   if r["source"] == "overlay"
                                   and not r["overridden"]))

    def _build_rows(self, merged: Dict[str, Tuple[str, Path]]) -> List[Dict]:
        base_files = set(_scan(self.base_dir))
        rows = []
        for rel in sorted(merged):
            origin, path = merged[rel]
            rows.append({
                "path": rel,
                "source": origin,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "overridden": origin == "overlay" and rel in base_files,
            })
        return rows

    def _fingerprint_from_rows(self, rows: Iterable[Dict]) -> str:
        digest = hashlib.sha256(FINGERPRINT_SCHEMA.encode())
        for row in sorted(rows, key=lambda r: r["path"]):
            digest.update(row["path"].encode())
            digest.update(row["source"].encode())
            digest.update(row["sha256"].encode())
        return digest.hexdigest()

    def _materialise(self, merged: Dict[str, Tuple[str, Path]]) -> None:
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="zerg_source_staging_",
                                        dir=str(self.output_dir.parent)))
        try:
            for rel in sorted(merged):
                _, src = merged[rel]
                target = staging / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, target)
            self.output_dir.parent.mkdir(parents=True, exist_ok=True)
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
            shutil.move(str(staging), str(self.output_dir))
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def _verify_existing(self, rows: List[Dict]) -> None:
        """Check an already-materialised output still matches the merge map."""
        existing = set(_scan(self.output_dir))
        expected = {row["path"] for row in rows}
        existing.discard("source_manifest.json")
        missing = expected - existing
        extra = existing - expected
        if missing or extra:
            raise RuntimeError(
                f"resolved source tree drifted from merge map; "
                f"missing={sorted(missing)} extra={sorted(extra)}; "
                f"re-run resolve(clean_output=True)")
        for row in rows:
            current = sha256_file(self.output_dir / row["path"])
            if current != row["sha256"]:
                raise RuntimeError(
                    f"resolved file content drift: {row['path']}; "
                    f"re-run resolve(clean_output=True)")

    def _check_syntax(self, tree: Optional[Path] = None) -> List[str]:
        root = tree or self.output_dir
        errors = []
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            try:
                compile(path.read_text(encoding="utf-8"), rel, "exec")
            except SyntaxError as exc:
                errors.append(f"{rel}: {exc}")
        return errors

    # -- change detection ---------------------------------------------------

    def has_changed(self, previous_manifest_path: Optional[str] = None,
                    base_dir: Optional[str] = None,
                    overlay_dir: Optional[str] = None) -> bool:
        """True when the (possibly new) base/overlay pair would resolve to a
        different source tree than the recorded manifest's fingerprint."""
        if previous_manifest_path is None:
            previous_manifest_path = str(
                self.output_dir / "source_manifest.json")
        previous = json.loads(Path(previous_manifest_path).read_text(
            encoding="utf-8"))
        base = self._resolve_input(base_dir) if base_dir else self.base_dir
        overlay = (self._resolve_input(overlay_dir)
                   if overlay_dir else self.overlay_dir)
        return source_fingerprint(base, overlay) != previous.get("fingerprint")


def load_manifest(output_dir: str) -> Dict:
    manifest_path = Path(output_dir) / "source_manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))
