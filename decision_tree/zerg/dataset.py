"""Decision-point records, JSONL persistence, splits and row conversion.

One record = one decision moment: shared global features, every candidate
considered, the rule teacher choice, the model's shadow opinion, what finally
executed, and (later, filled by the collection runs) the episode outcome.

Splitting discipline (inherited from the paired-experiment framework):

- the split unit is an independent SCENE (``scene_id``, falling back to
  ``episode_id``); every candidate row of a scene stays on one side;
- weights are group-normalised: each decision point carries total mass one,
  split evenly across its candidate rows, so scenes with many candidates
  cannot dominate training.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .features import DEFAULT_SCHEMA, ZergFeatureSchema
from .schema import DECISION_RECORD_SCHEMA, dumps, flatten_row_features, \
    scorer_feature_names, validate_decision_record


def build_decision_record(*, episode_id: str, scene_id: str,
                          snapshot, summary: Dict[str, Any],
                          candidates: Sequence[Any],
                          global_features: Dict[str, Any],
                          teacher_choice: Optional[str],
                          model_scores: Optional[List[Dict[str, Any]]] = None,
                          selected_candidate: Optional[str] = None,
                          final_actions: Optional[Dict[str, int]] = None,
                          execution_status: str = "pending",
                          source_hash: str = "",
                          execution_context_hash: str = "",
                          reward_provenance: Optional[Dict[str, Any]] = None,
                          schema: ZergFeatureSchema = DEFAULT_SCHEMA
                          ) -> Dict[str, Any]:
    """Assemble one decision record (validated before returning)."""
    record = {
        "schema": DECISION_RECORD_SCHEMA,
        "episode_id": episode_id,
        "scene_id": scene_id,
        "game_loop": int(getattr(snapshot, "game_loop", 0) or 0),
        "snapshot_id": int(getattr(snapshot, "snapshot_id", 0) or 0),
        "source_hash": source_hash,
        "execution_context_hash": execution_context_hash,
        "feature_schema_version": schema.version,
        "global_features": {
            "team": dict(global_features.get("team", {})),
            "skill": dict(global_features.get("skill", {})),
            "units": {actor: dict(values)
                      for actor, values in global_features.get("units", {}).items()},
            "unit_types": dict(global_features.get("unit_types", {})),
        },
        "candidates": [candidate.to_record() for candidate in candidates],
        "teacher_choice": teacher_choice,
        "model_scores": list(model_scores or []),
        "selected_candidate": selected_candidate,
        "final_actions": dict(final_actions or {}),
        "execution_status": execution_status,
        "outcome": None,
        "return": None,
        "reward_provenance": reward_provenance,
    }
    return validate_decision_record(record, schema)


def finalise_outcome(record: Dict[str, Any], *, outcome: str,
                     episode_return: Optional[float],
                     ally_losses: int, enemy_losses: int,
                     damage_dealt: Optional[float] = None,
                     damage_received: Optional[float] = None,
                     reward_provenance: Optional[Dict[str, Any]] = None
                     ) -> Dict[str, Any]:
    """Attach the episode-outcome teacher signal. Terminal returns are kept
    strictly separate from any local candidate gain."""
    if outcome not in ("win", "loss", "draw", "timeout"):
        raise ValueError("unknown outcome")
    record["outcome"] = outcome
    record["return"] = episode_return
    provenance = record.setdefault("reward_provenance", {}) or {}
    provenance.update({
        "ally_losses": int(ally_losses),
        "enemy_losses": int(enemy_losses),
        "damage_dealt": damage_dealt,
        "damage_received": damage_received,
    })
    if reward_provenance:
        provenance.update(reward_provenance)
    return record


class DecisionRecordWriter:
    """Append-only JSONL writer; duplicates are refused (repeated
    observations must never be recorded twice)."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen_keys: set = set()
        if self.path.exists():
            for record in read_records(str(self.path)):
                self._seen_keys.add(record_key(record))

    def write(self, record: Dict[str, Any]) -> bool:
        key = record_key(record)
        if key in self._seen_keys:
            return False
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(dumps(record))
        self._seen_keys.add(key)
        return True


def record_key(record: Dict[str, Any]) -> Tuple:
    return (record["episode_id"], record["snapshot_id"], record["game_loop"])


def read_records(path: str,
                 schema: ZergFeatureSchema = DEFAULT_SCHEMA
                 ) -> List[Dict[str, Any]]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(validate_decision_record(json.loads(line), schema))
    return records


def scene_group(record: Dict[str, Any]) -> str:
    return record.get("scene_id") or record.get("episode_id")


def split_records(records: Sequence[Dict[str, Any]],
                  validation_fraction: float = 0.25,
                  split_seed: int = 0,
                  min_train_groups: int = 4
                  ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Scene-level split using the framework's deterministic sha256 group
    split (same scene never straddles train/validation)."""
    from decision_tree.learning.train import group_split
    rows = [{"group_id": scene_group(record), **{"_record": record}}
            for record in records]
    train_rows, validation_rows = group_split(
        rows, validation_fraction=validation_fraction,
        split_seed=split_seed, min_train_groups=min_train_groups)
    return ([row["_record"] for row in train_rows],
            [row["_record"] for row in validation_rows])


def decision_point_weights(records: Sequence[Dict[str, Any]]
                           ) -> List[float]:
    """Group-level weight normalisation: each decision point carries mass 1,
    shared equally by its candidate rows (built downstream); this helper
    gives each RECORD equal mass — candidate-level weighting happens in
    :func:`candidate_rows`."""
    counts: Dict[Tuple, int] = {}
    for record in records:
        counts[record_key(record)] = counts.get(record_key(record), 0) + 1
    return [1.0 / counts[record_key(record)] for record in records]


def candidate_rows(records: Sequence[Dict[str, Any]],
                   schema: ZergFeatureSchema = DEFAULT_SCHEMA,
                   shared_layers: Sequence[str] = ("team", "skill")
                   ) -> List[Dict[str, Any]]:
    """Flatten records into per-candidate training rows.

    Row fields (association keys kept for joining, never usable as features):
    ``group_id`` (scene), ``episode_id``, ``decision_key``, ``candidate_id``,
    ``candidate_family``, ``features`` (flat, fixed schema), ``teacher``,
    ``rule_rank``, plus optional realised outcome fields when present.
    """
    rows: List[Dict[str, Any]] = []
    counts: Dict[Tuple, int] = {}
    for record in records:
        counts[record_key(record)] = counts.get(record_key(record), 0) + 1
    for record in records:
        key = record_key(record)
        weight = 1.0 / counts[key]
        legal = [candidate for candidate in record["candidates"]
                 if candidate.get("legal")]
        rule_order = [candidate["candidate_id"] for candidate in legal]
        teacher = record.get("teacher_choice")
        for candidate in record["candidates"]:
            if not candidate.get("legal"):
                continue
            rows.append({
                "group_id": scene_group(record),
                "episode_id": record["episode_id"],
                "decision_key": dumps_key(key),
                "candidate_id": candidate["candidate_id"],
                "candidate_family": candidate["candidate_family"],
                "candidate_skill": candidate["candidate_skill"],
                "unit_type": (record.get("global_features", {})
                              .get("unit_types", {}) or {}).get(
                    (candidate.get("controlled_actors") or [""])[0]),
                "features": flatten_row_features(
                    record["global_features"], candidate.get("features", {}),
                    schema, shared_layers),
                "weight": weight,
                "teacher": 1.0 if teacher == candidate["candidate_id"] else 0.0,
                "rule_rank": float(rule_order.index(candidate["candidate_id"]))
                if candidate["candidate_id"] in rule_order else -1.0,
                "candidate_return": candidate.get("candidate_return"),
                "base_return": record.get("base_return"),
                "gain": (candidate["candidate_return"] - record["base_return"])
                if candidate.get("candidate_return") is not None
                and record.get("base_return") is not None else None,
            })
    return rows


def dumps_key(key: Tuple) -> str:
    return ":".join(str(part) for part in key)


def feature_names_for(schema: ZergFeatureSchema = DEFAULT_SCHEMA,
                      shared_layers: Sequence[str] = ("team", "skill")
                      ) -> List[str]:
    return scorer_feature_names(schema, shared_layers)
