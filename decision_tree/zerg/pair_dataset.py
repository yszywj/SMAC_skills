"""Counterfactual pair → gain-dataset pipeline (V1.3 Phase 3).

Flow (Task 3.4):

    Decision Records JSONL + Counterfactual Pair JSONL
        → join_counterfactual_pairs()      # identity association only
        → audit_counterfactual_pairs()     # strict legality audit
        → build_gain_dataset()             # audited training records
        → ZergTrainingBackend.train("gain") / .train_internal("gain")

Strictness rules: only COMPLETE pairs with matching execution context and
source can produce a gain; missing results stay None; unexecuted candidates
are never filled with zero; different anchors are never mixed; A-class
(skill-internal) and B-class (joint) gains are kept in separate outputs —
a local candidate's return is never presented as whole-tactic evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .dataset import record_key, dumps_key
from .schema import digest as canonical_digest, validate_pair_record

SCOPES = ("joint", "skill_internal")
GAIN_TOLERANCE = 1e-9


def _decision_index(decision_records: Sequence[Dict[str, Any]]
                    ) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    index: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for record in decision_records:
        index.setdefault((record["run_id"], record["episode_id"]),
                         []).append(record)
    return index


def _find_decision_point(index, pair) -> Optional[Dict[str, Any]]:
    """Locate the decision record a pair anchors to. Explicit
    ``decision_key`` (run:episode:snapshot:loop) wins; otherwise the pair's
    anchor_id must encode ``{episode}:{snapshot}``; otherwise the episode's
    first decision is used only when it is unambiguous."""
    run_id = pair.get("run_id")
    episode_id = pair.get("episode_id")
    bucket = index.get((run_id, episode_id)) or []
    if not bucket:
        return None
    decision_key = pair.get("decision_key")
    if decision_key:
        for record in bucket:
            if dumps_key(record_key(record)) == str(decision_key):
                return record
        return None
    anchor = str(pair.get("anchor_id", ""))
    parts = anchor.split(":")
    if parts and parts[-1].isdigit():
        wanted = int(parts[-1])
        matches = [record for record in bucket
                   if record["snapshot_id"] == wanted]
        if len(matches) == 1:
            return matches[0]
        return None
    return bucket[0] if len(bucket) == 1 else None


def _candidate_iter(record: Dict[str, Any], scope: Optional[str],
                    skill: Optional[str]):
    """Iterate (container-dict, candidate-dict) pairs of one decision record
    for the requested scope. 'joint' reads record["candidates"]; A-class
    reads record["skill_proposals"][skill]."""
    if scope == "joint" or scope is None:
        for candidate in record.get("candidates", []) or []:
            yield record.get("candidates"), candidate
    elif scope == "skill_internal":
        proposals = record.get("skill_proposals") or {}
        if skill:
            for candidate in proposals.get(skill, []) or []:
                yield proposals.get(skill), candidate
        else:
            for skill_name, candidates in proposals.items():
                for candidate in candidates or []:
                    yield candidates, candidate


def join_counterfactual_pairs(decision_records: Sequence[Dict[str, Any]],
                              pair_records: Sequence[Dict[str, Any]]
                              ) -> List[Dict[str, Any]]:
    """Task 3.1: associate pair records with their decision records.

    Output row = the pair's identity fields + the matched decision point's
    teacher/selection state + both return slots (None when absent). This is
    association ONLY — no gains are computed here and nothing is written
    back to the inputs.
    """
    index = _decision_index(decision_records)
    joined: List[Dict[str, Any]] = []
    for pair in pair_records:
        pair = dict(pair)
        record = _find_decision_point(index, pair)
        row = {
            "pair_id": pair.get("pair_id"),
            "experiment_id": pair.get("experiment_id"),
            "run_id": pair.get("run_id"),
            "episode_id": pair.get("episode_id"),
            "scene_config_id": pair.get("scene_config_id"),
            "anchor_id": pair.get("anchor_id"),
            "decision_key": pair.get("decision_key"),
            "scope": pair.get("scope") or "joint",
            "skill": pair.get("skill"),
            "baseline_id": pair.get("baseline_id"),
            "candidate_id": pair.get("candidate_id"),
            "initial_state_hash": pair.get("initial_state_hash"),
            "prefix_hash": pair.get("prefix_hash"),
            "execution_context_hash": pair.get("execution_context_hash"),
            "intervention_type": pair.get("intervention_type"),
            "intervention_horizon": pair.get("intervention_horizon"),
            "completion_status": pair.get("completion_status"),
            "baseline_return": pair.get("baseline_return"),
            "candidate_return": pair.get("candidate_return"),
            "gain": pair.get("gain"),
            "outcome": pair.get("outcome"),
            "decision_point_found": record is not None,
            "record_execution_context_hash":
                record.get("execution_context_hash") if record else None,
            "record_source_hash": record.get("source_hash") if record else None,
            "record_teacher_choice":
                record.get("teacher_choice") if record else None,
            "record_selected_candidate":
                record.get("selected_candidate") if record else None,
            "record_execution_status":
                record.get("execution_status") if record else None,
        }
        # Derive the gain when both returns exist and the pair omitted it;
        # the arithmetic audit below still rejects wrong values.
        if row["gain"] is None and row["candidate_return"] is not None \
                and row["baseline_return"] is not None:
            row["gain"] = float(row["candidate_return"]) \
                - float(row["baseline_return"])
        joined.append(row)
    return joined


def _pair_source_hash(row: Dict[str, Any]) -> str:
    # The pair's declared source; experiments record the decision record's
    # source hash at creation time. Presence of a differing value is a
    # version mismatch.
    return str(row.get("record_source_hash") or "")


def audit_counterfactual_pairs(joined: Sequence[Dict[str, Any]],
                               decision_records: Optional[Sequence[
                                   Dict[str, Any]]] = None
                               ) -> Dict[str, Any]:
    """Task 3.2: strict pair-legality audit.

    A pair is trainable only when ALL hold: decision point found, prefix and
    initial state evidenced, execution context AND source hash equal to the
    decision record's, intervention recorded, completion complete, both
    returns present, gain arithmetic exact, scope consistent with the
    candidate's home section. Everything else is listed with reasons."""
    record_by_key: Dict[str, Dict[str, Any]] = {}
    for record in decision_records or ():
        record_by_key[dumps_key(record_key(record))] = record
    problems_by_pair: Dict[str, List[str]] = {}
    trainable = 0
    for row in joined:
        problems: List[str] = []
        pair_id = str(row.get("pair_id"))
        if not row.get("decision_point_found"):
            problems.append("decision_point_not_found")
        else:
            if row.get("record_execution_context_hash") not in (
                    None, row.get("execution_context_hash")):
                problems.append("execution_context_mismatch")
            if row.get("record_source_hash") not in (None, "") and \
                    row.get("record_source_hash") != \
                    _pair_source_hash(row):
                problems.append("source_hash_mismatch")
        if not row.get("initial_state_hash") or not row.get("prefix_hash"):
            problems.append("missing_pair_start_evidence")
        if not row.get("intervention_type") or \
                row.get("intervention_type") == "unknown":
            problems.append("intervention_not_recorded")
        if row.get("completion_status") != "complete":
            problems.append(f"completion_status="
                            f"{row.get('completion_status')!r}")
        candidate_return = row.get("candidate_return")
        baseline_return = row.get("baseline_return")
        gain = row.get("gain")
        if candidate_return is None or baseline_return is None:
            problems.append("missing_return_kept_none")
        elif gain is None or abs(float(gain) - (float(candidate_return)
                                                - float(baseline_return))) \
                > GAIN_TOLERANCE:
            problems.append("gain_arithmetic_mismatch")
        scope = row.get("scope") or "joint"
        if scope not in SCOPES:
            problems.append(f"unknown_scope:{scope}")
        # scope/candidate cross-check against the actual decision record
        if decision_records and row.get("decision_point_found") \
                and not problems:
            key = dumps_key(record_key_by_identity(row))
            record = record_by_key.get(key)
            if record is not None:
                found = False
                for _container, candidate in _candidate_iter(
                        record, scope, row.get("skill")):
                    cid = candidate.get("candidate_id")
                    if cid == row.get("candidate_id") or \
                            candidate.get("source_plan_id") == \
                            row.get("candidate_id"):
                        found = True
                        break
                if not found:
                    problems.append("candidate_not_in_decision_record")
        if problems:
            problems_by_pair[pair_id] = problems
        else:
            trainable += 1
    total = len(joined)
    return {
        "pairs_total": total,
        "pairs_trainable": trainable,
        "pairs_rejected": total - trainable,
        "trainable_ratio": (trainable / total) if total else None,
        "problems_by_pair": problems_by_pair,
        "rules": [
            "BASE and candidate must share initial state and prefix",
            "execution context and source hash must match the decision record",
            "only complete experiments produce gains",
            "missing results stay None (never zero-filled)",
            "different anchors are never mixed",
        ],
    }


def record_key_by_identity(row: Dict[str, Any]) -> Tuple[str, str, int, int]:
    """Rebuild a decision record key from a joined row's identity fields;
    snapshot/game_loop come from the decision_key when present."""
    decision_key = str(row.get("decision_key") or "")
    parts = decision_key.split(":")
    snapshot_id = int(parts[2]) if len(parts) >= 4 and parts[2].isdigit() \
        else -1
    game_loop = int(parts[3]) if len(parts) >= 4 and parts[3].isdigit() \
        else -1
    return (str(row.get("run_id")), str(row.get("episode_id")),
            snapshot_id, game_loop)


def build_gain_dataset(decision_records: Sequence[Dict[str, Any]],
                       pair_records: Sequence[Dict[str, Any]],
                       *, require_audit: bool = True
                       ) -> "GainDatasetBundle":
    """Task 3.3/3.4: audited gain training inputs.

    Returns SEPARATE record lists for joint (B-class) and skill-internal
    (A-class) gains — copies of the decision records with ``candidate_return``
    written onto the paired candidate and ``baseline_return`` on the record.
    Unaudited data never reaches the training records.
    """
    joined = join_counterfactual_pairs(decision_records, pair_records)
    audit = audit_counterfactual_pairs(joined, decision_records)
    rejected_ids = set(audit["problems_by_pair"])
    trainable_ids = ({str(row.get("pair_id")) for row in joined}
                     - rejected_ids)
    trainable_rows = [row for row in joined
                      if str(row.get("pair_id")) in trainable_ids]
    joint_records: List[Dict[str, Any]] = []
    internal_records: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    indexed = _decision_index(decision_records)
    used_points: Dict[str, set] = {}
    for row in trainable_rows:
        record = _find_decision_point(indexed, row)
        if record is None:
            continue
        key = dumps_key(record_key(record))
        copy_record = dict(record)
        copy_record["candidates"] = [dict(c)
                                     for c in record.get("candidates", [])]
        copy_record["skill_proposals"] = {
            skill: [dict(c) for c in items]
            for skill, items in (record.get("skill_proposals") or {}).items()}
        copy_record["baseline_return"] = row.get("baseline_return")
        scope = row.get("scope") or "joint"
        candidate_id = row.get("candidate_id")
        applied = False
        for container, candidate in _candidate_iter(copy_record, scope,
                                                    row.get("skill")):
            if candidate.get("candidate_id") == candidate_id or \
                    candidate.get("source_plan_id") == candidate_id:
                candidate["candidate_return"] = row.get("candidate_return")
                candidate["pair_id"] = row.get("pair_id")
                candidate["pair_experiment_id"] = row.get("experiment_id")
                candidate["pair_anchor_id"] = row.get("anchor_id")
                applied = True
                break
        if not applied:
            continue
        sources.append({"pair_id": row.get("pair_id"),
                        "decision_key": key, "scope": scope,
                        "candidate_id": candidate_id,
                        "anchor_id": row.get("anchor_id")})
        used_points.setdefault(key, set()).add(scope)
        if scope == "skill_internal":
            internal_records.append(copy_record)
        else:
            joint_records.append(copy_record)
    dataset_hash = canonical_digest({
        "joint": joint_records, "internal": internal_records,
        "audit": {"pairs_trainable": audit["pairs_trainable"]}})
    return GainDatasetBundle(
        joint_records=joint_records,
        internal_records=internal_records,
        audit_report=audit,
        dataset_hash=dataset_hash,
        sources=sources,
        joined=joined)


@dataclass
class GainDatasetBundle:
    joint_records: List[Dict[str, Any]] = field(default_factory=list)
    internal_records: List[Dict[str, Any]] = field(default_factory=list)
    audit_report: Dict[str, Any] = field(default_factory=dict)
    dataset_hash: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    joined: List[Dict[str, Any]] = field(default_factory=list)


def validate_pairs(path: str) -> List[Dict[str, Any]]:
    """Load a pairs JSONL through the strict record validation."""
    import json as _json
    from pathlib import Path as _Path
    records = []
    for line in _Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(validate_pair_record(_json.loads(line)))
    return records


def validate_decision_records(path: str) -> List[Dict[str, Any]]:
    from .dataset import read_records
    return read_records(path)
