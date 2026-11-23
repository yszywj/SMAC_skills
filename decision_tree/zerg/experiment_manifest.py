"""Experiment manifests: frozen declarations of bounded collection runs.

A manifest freezes one experiment (identity, arms, budget configuration,
environment/context binding) BEFORE collection; progress records append
without mutating the frozen part. Group association between experiment arms
and BASE uses (run_id, episode_id, snapshot_id, game_loop). Nothing here
runs games.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .schema import digest as canonical_digest

MANIFEST_KIND = "zerg_experiment_manifest_v1"


def make_manifest(*, experiment_id: str, mode: str,
                  base_policy: str,
                  candidate_arms: Sequence[Dict[str, Any]],
                  exploration_config: Dict[str, Any],
                  source_fingerprint: str,
                  execution_context_hash: str,
                  scene_groups: Optional[Sequence[str]] = None,
                  description: str = "") -> Dict[str, Any]:
    if mode not in ("shadow", "counterfactual_pairs", "validation"):
        raise ValueError("unknown experiment mode")
    if not candidate_arms:
        raise ValueError("at least one candidate arm is required")
    manifest = {
        "kind": MANIFEST_KIND,
        "experiment_id": experiment_id,
        "mode": mode,
        "base_policy": base_policy,          # always the existing rule skill
        "candidate_arms": [dict(arm) for arm in candidate_arms],
        "exploration_config": exploration_config,
        "source_fingerprint": source_fingerprint,
        "execution_context_hash": execution_context_hash,
        "scene_groups": list(scene_groups or []),
        "description": description,
        "status": "declared",
    }
    manifest["manifest_sha256"] = canonical_digest(manifest)
    return manifest


def freeze(path: str, manifest: Dict[str, Any]) -> Path:
    """Write-once freeze: an existing manifest is never overwritten."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"experiment manifest already frozen: {target}")
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return target


def record_progress(path: str, rows: Sequence[Dict[str, Any]]) -> int:
    """Append progress rows (JSONL) beside the frozen manifest. Each row is
    an association record: {experiment_id, run_id, episode_id, snapshot_id,
    game_loop, arm_id, completion_status}. Unfinished rows stay flagged."""
    progress_path = Path(path).with_suffix(".progress.jsonl")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with progress_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            if not row.get("experiment_id"):
                raise ValueError("progress rows require experiment_id")
            row = dict(row)
            row.setdefault("completion_status", "unknown")
            row.setdefault("unfinished", True)
            handle.write(json.dumps(row, sort_keys=True,
                                    separators=(",", ":")) + "\n")
            written += 1
    return written


def summarize(path: str) -> Dict[str, Any]:
    """Frozen manifest + progress aggregation (unfinished counts kept)."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    progress_path = Path(path).with_suffix(".progress.jsonl")
    rows: List[Dict[str, Any]] = []
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    per_arm: Dict[str, Dict[str, int]] = {}
    for row in rows:
        arm = str(row.get("arm_id", "unknown"))
        bucket = per_arm.setdefault(arm, {"rows": 0, "unfinished": 0})
        bucket["rows"] += 1
        bucket["unfinished"] += 1 if row.get("unfinished") else 0
    return {"manifest": manifest, "progress_rows": len(rows),
            "per_arm": per_arm,
            "frozen_sha256": manifest.get("manifest_sha256")}


ANCHOR_MANIFEST_KIND = "zerg_anchor_manifest_v1"


def make_anchor_manifest(*, experiment_id: str, source_fingerprint: str,
                         execution_context_hash: str,
                         anchors: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """V1.3 Task 2.9: the independent per-anchor evidence registry.

    ``anchors`` maps anchor_id -> {initial_state_hash, prefix_hash,
    baseline_id}. The manifest FREEzes the collection-side truth for start
    state, decision prefix, execution context and source, so pair JSONLs
    can never validate themselves by their own declarations. Write-once via
    :func:`freeze`.
    """
    if not anchors:
        raise ValueError("anchor manifest requires at least one anchor")
    normalized = {}
    for anchor_id, evidence in anchors.items():
        for field in ("initial_state_hash", "prefix_hash", "baseline_id"):
            if not evidence.get(field):
                raise ValueError(f"anchor {anchor_id!r} missing {field}")
        normalized[str(anchor_id)] = {
            "initial_state_hash": str(evidence["initial_state_hash"]),
            "prefix_hash": str(evidence["prefix_hash"]),
            "baseline_id": str(evidence["baseline_id"]),
        }
    manifest = {
        "kind": ANCHOR_MANIFEST_KIND,
        "experiment_id": experiment_id,
        "source_fingerprint": source_fingerprint,
        "execution_context_hash": execution_context_hash,
        "anchors": normalized,
        "status": "declared",
    }
    manifest["manifest_sha256"] = canonical_digest(manifest)
    return manifest


def validate_anchor_evidence(manifest: Dict[str, Any], pair: Dict[str, Any]
                             ) -> List[str]:
    """Cross-check ONE pair against the frozen anchor manifest. Returns the
    mismatch list (empty = consistent)."""
    problems: List[str] = []
    anchor_id = str(pair.get("anchor_id"))
    anchors = manifest.get("anchors", {})
    if anchor_id not in anchors:
        return [f"anchor {anchor_id!r} not in anchor manifest"]
    evidence = anchors[anchor_id]
    if str(pair.get("initial_state_hash") or "") != \
            evidence["initial_state_hash"]:
        problems.append("initial_state_hash differs from anchor manifest")
    if str(pair.get("prefix_hash") or "") != evidence["prefix_hash"]:
        problems.append("prefix_hash differs from anchor manifest")
    if str(pair.get("baseline_id") or "") != evidence["baseline_id"]:
        problems.append(f"baseline_id {pair.get('baseline_id')!r} is not the "
                        f"declared BASE ({evidence['baseline_id']!r})")
    if pair.get("execution_context_hash") is not None and \
            str(pair["execution_context_hash"]) != \
            str(manifest.get("execution_context_hash")):
        problems.append("execution_context_hash differs from anchor manifest")
    if pair.get("source_fingerprint") is not None and \
            str(pair["source_fingerprint"]) != \
            str(manifest.get("source_fingerprint")):
        problems.append("source_fingerprint differs from anchor manifest")
    return problems
