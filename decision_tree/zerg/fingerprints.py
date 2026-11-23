"""Execution-context and model fingerprints for the Zerg tree framework.

Deployment gating follows the existing framework contract: a model may only
be loaded into a runtime whose frozen execution context hash equals the hash
recorded inside the model. The hash covers:

- the resolved tactical_v1 source fingerprint (base+overlay merge);
- the effective tactical configuration actually passed to the SkillManager;
- the runtime policy contract (mode, scorer enablement, thresholds);
- the feature schema digest and version.

``episode_id`` / ``scene_id`` never enter the hash — they are association
keys, not identity inputs.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from .features import DEFAULT_SCHEMA, ZergFeatureSchema
from .source import source_fingerprint
from .schema import digest as canonical_digest

RUNTIME_POLICY_CONTRACT = "zerg_tree_runtime_v1"


def source_hash(workspace: str, base_dir: Optional[str] = None,
                overlay_dir: Optional[str] = None) -> str:
    """Whole-source fingerprint of the resolved tactical_v1 tree."""
    from pathlib import Path
    root = Path(workspace)
    base = root / (base_dir or "skills/tactical_v1")
    overlay = root / (overlay_dir or (
        "runs/contact_all_attack_20261122_230107_662982/"
        "marginal_near/skills/tactical_v1"))
    return source_fingerprint(base, overlay)


def runtime_policy_contract(mode: str, scorers: Iterable[str],
                            thresholds: Optional[Dict[str, Any]] = None,
                            schema: ZergFeatureSchema = DEFAULT_SCHEMA
                            ) -> Dict[str, Any]:
    return {
        "contract": RUNTIME_POLICY_CONTRACT,
        "mode": mode,
        "scorers": sorted(set(scorers)),
        "thresholds": dict(thresholds or {}),
        "feature_schema": {"name": schema.name, "version": schema.version,
                           "digest": schema.digest()},
    }


def execution_context(source_fingerprint_value: str,
                      tactical_config: Dict[str, Any],
                      policy: Dict[str, Any],
                      schema: ZergFeatureSchema = DEFAULT_SCHEMA,
                      extra: Optional[Dict[str, Any]] = None
                      ) -> Dict[str, Any]:
    context = {
        "policy_contract": policy,
        "source_fingerprint": source_fingerprint_value,
        "tactical_config": tactical_config,
        "feature_schema": {"name": schema.name, "version": schema.version,
                           "digest": schema.digest()},
    }
    if extra:
        context["extra"] = extra
    return context


def execution_context_hash(context: Dict[str, Any]) -> str:
    return canonical_digest(context)


def model_fingerprint(model_record: Dict[str, Any]) -> str:
    return canonical_digest(model_record)


def decision_context_hash(source_fingerprint_value: str,
                          execution_context_hash_value: str,
                          episode_id: str, scene_id: str,
                          snapshot_id: int, game_loop: int,
                          schema: ZergFeatureSchema = DEFAULT_SCHEMA
                          ) -> str:
    """Stable key binding one decision point to its provenance (association
    use only — these fields are rejected as model features)."""
    return canonical_digest({
        "source": source_fingerprint_value,
        "context": execution_context_hash_value,
        "episode_id": episode_id, "scene_id": scene_id,
        "snapshot_id": snapshot_id, "game_loop": game_loop,
        "feature_schema": schema.digest()})


def validate_deployment(model_record: Dict[str, Any],
                        context: Dict[str, Any]) -> None:
    """Refuse to load a model whose recorded context does not match the live
    one (same contract as decision_tree.experiment.validate_deployment)."""
    expected = model_record.get("execution_context_hash")
    actual = execution_context_hash(context)
    if expected != actual:
        raise ValueError(
            "model execution context mismatch: refusing to load "
            f"(model={str(expected)[:16]}… live={actual[:16]}…)")
    embedded = model_record.get("context")
    if embedded is not None and embedded != context:
        raise ValueError("model embedded context differs from live context")


def assert_model_source(model_record: Dict[str, Any],
                        live_source_fingerprint: str,
                        allow_missing: bool = False) -> None:
    """Task 3.2: a model recorded under one source fingerprint cannot run on
    another. Shadow mode is bound by the same rule — data collected against
    the wrong source must never be marked as a valid version."""
    recorded = model_record.get("source_fingerprint")
    if recorded is None:
        if allow_missing:
            return
        raise ValueError("model lacks source_fingerprint — refusing to load")
    if recorded != live_source_fingerprint:
        raise ValueError(
            "model source fingerprint mismatch: refusing to load "
            f"(model={str(recorded)[:16]}… live={live_source_fingerprint[:16]}…)")
