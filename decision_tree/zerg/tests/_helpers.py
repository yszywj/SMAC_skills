"""Shared synthetic-audit helpers for gain/ranking/correction training tests.

Builds a FULLY audited gain dataset (anchor manifest + execution receipts)
from the synthetic ``_records`` fixtures, so training entry points receive
their required audit envelope without any fabricated shortcuts.
"""
from __future__ import annotations

from decision_tree.zerg.experiment_manifest import make_anchor_manifest
from decision_tree.zerg.pair_dataset import build_gain_dataset, \
    make_execution_receipt

EXPERIMENT_ID = "exp_test_audit"


def synthetic_audit(records, *, scope: str = "joint", skill: str = None,
                    execution_context_hash: str = None,
                    source_fingerprint: str = None):
    """Build (bundle, envelope) for the given records.

    Every record contributes one BASE+candidate pair with matching
    execution receipts; the manifest freezes start/prefix/baseline evidence.
    ``execution_context_hash``/``source_fingerprint`` override the record
    values so the envelope can be bound to the TRAINING context.
    """
    pairs = []
    receipts = []
    anchors = {}
    if not records:
        raise ValueError("no records")
    source_fp = source_fingerprint or records[0]["source_hash"]
    ctx_hash = execution_context_hash or records[0]["execution_context_hash"]
    # The whole chain shares ONE execution context: when the caller binds
    # the audit to the live runtime context, the record COPIES are re-stamped
    # so records/pairs/receipts/manifest all agree (originals untouched).
    if source_fingerprint or execution_context_hash:
        records = [dict(record, execution_context_hash=ctx_hash,
                        source_hash=source_fp)
                   for record in records]
    for index, record in enumerate(records):
        anchor_id = f"{record['episode_id']}:{record['snapshot_id']}"
        if scope == "joint":
            candidate = record["candidates"][1]
            candidate_id = candidate["candidate_id"]
        else:
            proposal = (record.get("skill_proposals") or {}).get(
                skill or "", [])
            if not proposal:
                continue
            candidate = proposal[0]
            candidate_id = candidate["candidate_id"]
        candidate_return = candidate.get("candidate_return")
        if candidate_return is None:
            candidate_return = 1.2 + 0.01 * index
        baseline_return = record.get("baseline_return")
        if baseline_return is None:
            baseline_return = 1.0
        pair_id = f"pair_{index:03d}"
        pairs.append({
            "schema": "zerg_counterfactual_pair_v1",
            "experiment_id": EXPERIMENT_ID,
            "run_id": record["run_id"],
            "episode_id": record["episode_id"],
            "scene_config_id": record.get("scene_config_id"),
            "anchor_id": anchor_id,
            "pair_id": pair_id,
            "baseline_id": "BASE",
            "candidate_id": candidate_id,
            "decision_key": f"{record['run_id']}:{record['episode_id']}:"
                            f"{record['snapshot_id']}:"
                            f"{record['game_loop']}",
            "initial_state_hash": "a" * 64,
            "prefix_hash": "b" * 64,
            "execution_context_hash": ctx_hash,
            "source_fingerprint": source_fp,
            "intervention_type": "single_step",
            "intervention_horizon": 1,
            "candidate_return": candidate_return,
            "baseline_return": baseline_return,
            "outcome": "win",
            "completion_status": "complete",
            "scope": scope,
            "skill": skill,
        })
        anchors[anchor_id] = {
            "initial_state_hash": "a" * 64,
            "prefix_hash": "b" * 64,
            "baseline_id": "BASE",
        }
        for branch_id, branch_return in (("BASE", baseline_return),
                                         ("candidate", candidate_return)):
            receipts.append(make_execution_receipt(
                experiment_id=EXPERIMENT_ID,
                anchor_id=anchor_id,
                baseline_id="BASE",
                candidate_id=candidate_id,
                branch_id=branch_id,
                initial_state_hash="a" * 64,
                prefix_hash="b" * 64,
                source_fingerprint=source_fp,
                execution_context_hash=ctx_hash,
                intervention_type="single_step",
                intervention_horizon=1,
                execution_started=True,
                execution_completed=True,
                episode_finished=True,
                execution_error=None,
                outcome="win",
                return_value=branch_return,
                receipt_id=f"{pair_id}:{branch_id}",
                receipt_source="test_driver",
            ))
    manifest = make_anchor_manifest(
        experiment_id=EXPERIMENT_ID,
        source_fingerprint=source_fp,
        execution_context_hash=ctx_hash,
        anchors=anchors)
    bundle = build_gain_dataset(records, pairs,
                                anchor_manifest=manifest,
                                execution_receipts=receipts)
    envelope = bundle.export_envelope(scope)
    return bundle, envelope
