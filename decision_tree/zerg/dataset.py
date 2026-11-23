"""Decision-point records, JSONL persistence, splits and row conversion (V1.1).

One record = one decision moment: shared global features, every candidate
considered, the rule teacher choice, the model's shadow opinion, what finally
executed, and (later, filled by the collection runs) the episode outcome.

Identity discipline (Task 2.1/2.2):

- every record carries ``run_id`` (collection batch), ``episode_id``
  (unique game), ``scene_config_id`` (roster+environment configuration),
  ``seed`` and ``map_id``. ``map_id`` is descriptive metadata — NEVER the
  grouping key for training splits;
- ``record_key`` = (run_id, episode_id, snapshot_id, game_loop): different
  collection batches can never collide, repeated observations of the same
  decision are still deduplicated;
- the split unit for local candidate models is the independent EPISODE; a
  scene-config holdout split is available for cross-configuration
  generalisation; group lists are written to a manifest file.

Weights: each decision point carries total mass one, split evenly across
its candidate rows, so episodes with many candidates cannot dominate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .features import DEFAULT_SCHEMA, ZergFeatureSchema
from .schema import DECISION_RECORD_SCHEMA, PAIR_RECORD_SCHEMA, dumps, \
    build_internal_model_features, build_model_features, \
    scorer_feature_names, validate_decision_record, validate_pair_record, \
    audit_pair, pair_is_trainable

RecordKey = Tuple[str, str, int, int]


def build_decision_record(*, run_id: str, episode_id: str,
                          scene_config_id: Optional[str] = None,
                          seed: Optional[int] = None,
                          map_id: Optional[str] = None,
                          snapshot=None,
                          snapshot_id: int = 0,
                          game_loop: int = 0,
                          summary: Optional[Dict[str, Any]] = None,
                          candidates: Sequence[Any] = (),
                          global_features: Optional[Dict[str, Any]] = None,
                          teacher_choice: Optional[str] = None,
                          model_scores: Optional[List[Dict[str, Any]]] = None,
                          selected_candidate: Optional[str] = None,
                          final_actions: Optional[Dict[str, int]] = None,
                          execution_status: str = "pending",
                          source_hash: str = "",
                          execution_context_hash: str = "",
                          reward_provenance: Optional[Dict[str, Any]] = None,
                          schema: ZergFeatureSchema = DEFAULT_SCHEMA
                          ) -> Dict[str, Any]:
    """Assemble one decision record (validated before returning).

    ``snapshot`` may be passed directly (preferred); otherwise explicit
    ``snapshot_id``/``game_loop`` must be supplied.
    """
    if snapshot is not None:
        snapshot_id = int(getattr(snapshot, "snapshot_id", snapshot_id) or 0)
        game_loop = int(getattr(snapshot, "game_loop", game_loop) or 0)
    record = {
        "schema": DECISION_RECORD_SCHEMA,
        "run_id": run_id,
        "episode_id": episode_id,
        "scene_config_id": scene_config_id,
        "seed": seed,
        "map_id": map_id,
        "game_loop": game_loop,
        "snapshot_id": snapshot_id,
        "source_hash": source_hash,
        "execution_context_hash": execution_context_hash,
        "feature_schema_version": schema.version,
        "global_features": {
            "team": dict((global_features or {}).get("team", {})),
            "skill": dict((global_features or {}).get("skill", {})),
            "units": {actor: dict(values) for actor, values
                      in (global_features or {}).get("units", {}).items()},
            "unit_types": dict((global_features or {})
                               .get("unit_types", {})),
        },
        "candidates": [candidate.to_record() if hasattr(candidate,
                                                        "to_record")
                       else dict(candidate) for candidate in candidates],
        "teacher_choice": teacher_choice,
        "model_scores": list(model_scores or []),
        "selected_candidate": selected_candidate,
        "final_actions": dict(final_actions or {}),
        "execution_status": execution_status,
        "outcome": None,
        "return": None,
        "baseline_return": None,
        "reward_provenance": reward_provenance,
    }
    return validate_decision_record(record, schema)


def finalise_outcome(record: Dict[str, Any], *, outcome: str,
                     episode_return: Optional[float],
                     ally_losses: int, enemy_losses: int,
                     damage_dealt: Optional[float] = None,
                     damage_received: Optional[float] = None,
                     baseline_return: Optional[float] = None,
                     reward_provenance: Optional[Dict[str, Any]] = None
                     ) -> Dict[str, Any]:
    """Attach the episode-outcome teacher signal. Terminal returns are kept
    strictly separate from any local candidate gain; ``baseline_return``
    (paired BASE episode return) is stored alongside when evidenced."""
    if outcome not in ("win", "loss", "draw", "timeout"):
        raise ValueError("unknown outcome")
    record["outcome"] = outcome
    record["return"] = episode_return
    if baseline_return is not None:
        record["baseline_return"] = baseline_return
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


# -- counterfactual pair helpers (Task 2.3) ---------------------------------

def build_pair_record(*, experiment_id: str, run_id: str, episode_id: str,
                      scene_config_id: Optional[str], anchor_id: str,
                      pair_id: str, baseline_id: str, candidate_id: str,
                      initial_state_hash: Optional[str],
                      prefix_hash: Optional[str],
                      execution_context_hash: str,
                      intervention_type: str,
                      intervention_horizon: int,
                      candidate_return: Optional[float] = None,
                      baseline_return: Optional[float] = None,
                      outcome: Optional[str] = None,
                      completion_status: str = "unknown",
                      candidate_id_at_point: Optional[str] = None,
                      ) -> Dict[str, Any]:
    """One BASE-vs-candidate comparison. ``candidate_return`` and
    ``baseline_return`` default to None (absent, never zero); ``gain`` is
    derived and only written when BOTH returns exist. Single-step and
    multi-step continuous interventions are different ``intervention_type``
    values and must never be mixed in one interpretation."""
    record = {
        "schema": PAIR_RECORD_SCHEMA,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "episode_id": episode_id,
        "scene_config_id": scene_config_id,
        "anchor_id": anchor_id,
        "pair_id": pair_id,
        "baseline_id": baseline_id,
        "candidate_id": candidate_id,
        "candidate_id_at_point": candidate_id_at_point,
        "initial_state_hash": initial_state_hash,
        "prefix_hash": prefix_hash,
        "execution_context_hash": execution_context_hash,
        "intervention_type": intervention_type,
        "intervention_horizon": int(intervention_horizon),
        "candidate_return": candidate_return,
        "baseline_return": baseline_return,
        "gain": (candidate_return - baseline_return)
        if candidate_return is not None and baseline_return is not None
        else None,
        "outcome": outcome,
        "completion_status": completion_status,
    }
    return validate_pair_record(record)


class PairWriter:
    """Append-only JSONL writer for counterfactual pair records,
    deduplicated by (experiment_id, pair_id)."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set = set()
        if self.path.exists():
            for record in read_pairs(str(self.path)):
                self._seen.add((record["experiment_id"], record["pair_id"]))

    def write(self, record: Dict[str, Any]) -> bool:
        record = validate_pair_record(record)
        key = (record["experiment_id"], record["pair_id"])
        if key in self._seen:
            return False
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(dumps(record))
        self._seen.add(key)
        return True


def read_pairs(path: str) -> List[Dict[str, Any]]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(validate_pair_record(json.loads(line)))
    return records


def trainable_gain_pairs(path_or_records) -> Tuple[List[Dict[str, Any]],
                                                   List[Dict[str, Any]]]:
    """Split pairs into (trainable-with-gain, incomplete). Incomplete pairs
    are retained and reported, never silently dropped or fabricated."""
    records = path_or_records if isinstance(path_or_records, list) \
        else read_pairs(path_or_records)
    usable, incomplete = [], []
    for record in records:
        if record.get("gain") is not None and pair_is_trainable(record):
            usable.append(record)
        else:
            incomplete.append(record)
    return usable, incomplete


# -- persistence ------------------------------------------------------------

class DecisionRecordWriter:
    """Append-only JSONL writer; duplicates are refused (repeated
    observations must never be recorded twice). Keys include run_id, so
    different collection batches never collide."""

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


def record_key(record: Dict[str, Any]) -> RecordKey:
    """Task 2.2: (run_id, episode_id, snapshot_id, game_loop)."""
    return (record["run_id"], record["episode_id"],
            record["snapshot_id"], record["game_loop"])


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


# -- grouping and splits (Task 2.2) -----------------------------------------

def episode_group(record: Dict[str, Any]) -> str:
    """Local candidate models group by independent EPISODE
    (run_id:episode_id — an episode identity is unique across runs)."""
    return f"{record['run_id']}:{record['episode_id']}"


def scene_config_group(record: Dict[str, Any]) -> str:
    """Cross-configuration holdout groups by scene configuration
    (roster+environment), falling back to the episode group."""
    return record.get("scene_config_id") or episode_group(record)


def split_records(records: Sequence[Dict[str, Any]],
                  validation_fraction: float = 0.25,
                  split_seed: int = 0,
                  min_train_groups: int = 4,
                  group_by: str = "episode"
                  ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic group split (framework sha256 mechanism). ``group_by``
    is ``episode`` (default, local models) or ``scene_config`` (cross-config
    holdout). One group is never split across sides."""
    from decision_tree.learning.train import group_split
    group_of = episode_group if group_by == "episode" else scene_config_group
    rows = [{"group_id": group_of(record), "_record": record}
            for record in records]
    train_rows, validation_rows = group_split(
        rows, validation_fraction=validation_fraction,
        split_seed=split_seed, min_train_groups=min_train_groups)
    return ([row["_record"] for row in train_rows],
            [row["_record"] for row in validation_rows])


def write_split_manifest(path: str, train_records, validation_records,
                         test_records: Optional[Sequence] = None,
                         *, split_seed: int = 0,
                         group_by: str = "episode") -> Dict[str, Any]:
    """Persist the train/validation(/test) group lists — auditable split
    bookkeeping; disjointness is asserted here."""
    group_of = episode_group if group_by == "episode" else scene_config_group
    train_groups = sorted({group_of(r) for r in train_records})
    validation_groups = sorted({group_of(r) for r in validation_records})
    test_groups = sorted({group_of(r) for r in (test_records or [])})
    overlap = (set(train_groups) & set(validation_groups)) \
        | (set(train_groups) & set(test_groups)) \
        | (set(validation_groups) & set(test_groups))
    if overlap:
        raise ValueError(f"split groups overlap: {sorted(overlap)[:5]}")
    manifest = {
        "kind": "zerg_split_manifest_v1",
        "group_by": group_by,
        "split_seed": split_seed,
        "train_groups": train_groups,
        "validation_groups": validation_groups,
        "test_groups": test_groups,
        "counts": {"train_records": len(train_records),
                   "validation_records": len(validation_records),
                   "test_records": len(test_records or [])},
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    return manifest


def decision_point_weights(records: Sequence[Dict[str, Any]]) -> List[float]:
    """Equal mass per decision point (candidate-level weighting happens in
    :func:`candidate_rows`)."""
    counts: Dict[RecordKey, int] = {}
    for record in records:
        counts[record_key(record)] = counts.get(record_key(record), 0) + 1
    return [1.0 / counts[record_key(record)] for record in records]


# -- training rows ----------------------------------------------------------

def candidate_rows(records: Sequence[Dict[str, Any]],
                   schema: ZergFeatureSchema = DEFAULT_SCHEMA,
                   shared_layers: Sequence[str] = ("team", "skill")
                   ) -> List[Dict[str, Any]]:
    """Flatten records into per-candidate training rows.

    ``features`` comes from THE single join (``build_model_features``) so
    offline rows are identical to online scoring vectors. Identity keys
    (group_id / episode_id / decision_key / candidate_id) are kept for
    joining and are forbidden as model features by the schema layer.
    """
    rows: List[Dict[str, Any]] = []
    counts: Dict[RecordKey, int] = {}
    for record in records:
        counts[record_key(record)] = counts.get(record_key(record), 0) + 1
    for record in records:
        key = record_key(record)
        weight = 1.0 / counts[key]
        legal = [candidate for candidate in record["candidates"]
                 if candidate.get("legal")]
        rule_order = [candidate["candidate_id"] for candidate in legal]
        teacher = record.get("teacher_choice")
        rule_scores = {candidate["candidate_id"]: candidate.get("rule_score")
                       for candidate in record["candidates"]}
        for candidate in record["candidates"]:
            if not candidate.get("legal"):
                continue
            rows.append({
                "group_id": episode_group(record),
                "scene_config_id": record.get("scene_config_id"),
                "run_id": record["run_id"],
                "episode_id": record["episode_id"],
                "decision_key": dumps_key(key),
                "candidate_id": candidate["candidate_id"],
                "candidate_family": candidate["candidate_family"],
                "candidate_skill": candidate["candidate_skill"],
                "unit_type": (record.get("global_features", {})
                              .get("unit_types", {}) or {}).get(
                    (candidate.get("controlled_actors") or [""])[0]),
                "features": build_model_features(
                    record["global_features"], candidate.get("features", {}),
                    schema, shared_layers),
                "rule_score": rule_scores.get(candidate["candidate_id"]),
                "weight": weight,
                "teacher": 1.0 if teacher == candidate["candidate_id"] else 0.0,
                "rule_rank": float(rule_order.index(candidate["candidate_id"]))
                if candidate["candidate_id"] in rule_order else -1.0,
                "candidate_return": candidate.get("candidate_return"),
                "baseline_return": record.get("baseline_return"),
                "base_return": record.get("baseline_return"),
                "gain": (candidate["candidate_return"]
                         - record["baseline_return"])
                if candidate.get("candidate_return") is not None
                and record.get("baseline_return") is not None else None,
            })
    return rows


def dumps_key(key: Tuple) -> str:
    return ":".join(str(part) for part in key)


def feature_names_for(schema: ZergFeatureSchema = DEFAULT_SCHEMA,
                      shared_layers: Sequence[str] = ("team", "skill")
                      ) -> List[str]:
    return scorer_feature_names(schema, shared_layers)


# -- A-class (skill-internal) training rows (Phase 3) -----------------------

def _matches_final_actions(candidate: Dict[str, Any],
                           final_actions: Dict[str, int]) -> bool:
    """A-class teacher evidence: every controlled actor of the candidate has
    its proposed action present AND unchanged in the final action list."""
    controlled = candidate.get("controlled_actors") or ()
    proposed = candidate.get("proposed_actions") or {}
    if not controlled:
        return False
    for actor in controlled:
        if actor not in final_actions:
            return False
        if actor in proposed and final_actions[actor] != proposed[actor]:
            return False
    return True


def proposal_group_id(candidate: Dict[str, Any]) -> str:
    """Task 3.2: distinguish independent control tasks of the SAME skill at
    one decision point. Uses the plan's task identity (task_key, falling
    back to instance_id, falling back to the sorted controlled-actor set) —
    never a model feature."""
    task_key = candidate.get("task_key")
    if task_key:
        return f"task:{task_key}"
    instance_id = candidate.get("instance_id")
    if instance_id:
        return f"inst:{instance_id}"
    actors = sorted(candidate.get("controlled_actors") or [])
    return "actors:" + "|".join(actors) if actors else "actors:unknown"


def decision_group_id(record: Dict[str, Any], skill: str,
                      candidate: Dict[str, Any]) -> str:
    """run_id + episode_id + snapshot_id + skill + proposal_group_id.
    Comparisons (teacher labels, ranking) are only valid WITHIN one group."""
    key = record_key(record)
    return f"{key[0]}:{key[1]}:{key[2]}:{skill}:{proposal_group_id(candidate)}"


def expand_skill_proposals(records: Sequence[Dict[str, Any]],
                           schema: ZergFeatureSchema = DEFAULT_SCHEMA,
                           shared_layers: Sequence[str] = ("team", "skill")
                           ) -> List[Dict[str, Any]]:
    """Task 3.1: turn ``record["skill_proposals"]`` into training rows.

    - one candidate group per (decision point, skill, proposal task);
      A-class rows are NEVER merged into the B-class joint candidate set;
    - same state feature schema and the SAME single join as B-class rows;
    - Task 3.3 teacher labels derived from the Coordinator's FINAL choice:
      ``matched_final_actions`` = every controlled actor's proposed action
      survived into ``final_actions``; the (unique) matching candidate of a
      group carries ``teacher=1``. An unselected candidate is just
      unselected — never labelled as a rule error;
    - ``executed`` additionally requires ``execution_status == "committed"``.
    """
    rows: List[Dict[str, Any]] = []
    for record in records:
        proposals = record.get("skill_proposals") or {}
        if not proposals:
            continue
        final_actions = record.get("final_actions", {}) or {}
        committed = record.get("execution_status") == "committed"
        for skill, candidates in proposals.items():
            # group rows first so teacher labels resolve within a group
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for candidate in candidates:
                if not candidate.get("legal"):
                    continue
                groups.setdefault(proposal_group_id(candidate),
                                  []).append(candidate)
            for group_key, members in groups.items():
                matched_ids = [
                    candidate["candidate_id"] for candidate in members
                    if _matches_final_actions(candidate, final_actions)
                ]
                teacher_id = matched_ids[0] if len(matched_ids) == 1 else None
                for candidate in members:
                    candidate_id = candidate["candidate_id"]
                    matched = candidate_id in matched_ids
                    controlled = tuple(candidate.get("controlled_actors")
                                       or ())
                    try:
                        features_internal = build_internal_model_features(
                            record["global_features"],
                            candidate.get("features", {}), controlled,
                            schema, shared_layers)
                    except Exception:
                        features_internal = None   # units layer absent
                    rows.append({
                        "group_id": episode_group(record),
                        "decision_group_id": decision_group_id(
                            record, skill, candidate),
                        "proposal_group_id": group_key,
                        "decision_key": dumps_key(record_key(record)),
                        "run_id": record["run_id"],
                        "episode_id": record["episode_id"],
                        "candidate_id": candidate_id,
                        "candidate_family": candidate.get(
                            "candidate_family", "skill_internal"),
                        "candidate_skill": skill,
                        "controlled_actors": list(controlled),
                        "source_plan_id": candidate.get("source_plan_id"),
                        "resource_claims": candidate.get("resource_claims",
                                                         {}),
                        "dependencies": candidate.get("dependencies", []),
                        "unit_type": (record.get("global_features", {})
                                      .get("unit_types", {}) or {}).get(
                            controlled[0]) if controlled else None,
                        "features": build_model_features(
                            record["global_features"],
                            candidate.get("features", {}), schema,
                            shared_layers),
                        "features_internal": features_internal,
                        "weight": 1.0 / max(1, len(members)),
                        "teacher": 1.0 if teacher_id == candidate_id else 0.0,
                        "matched_final_actions": matched,
                        "entered_final_plan": matched,
                        "executed": bool(matched and committed),
                        "final_action_kept": matched,
                        "rule_score": candidate.get("rule_score"),
                        "candidate_return": candidate.get("candidate_return"),
                        "baseline_return": record.get("baseline_return"),
                        "gain": (candidate.get("candidate_return")
                                 - record["baseline_return"])
                        if candidate.get("candidate_return") is not None
                        and record.get("baseline_return") is not None
                        else None,
                        "objective": "skill_internal",
                    })
    return rows


def export_internal_training_rows(rows: Sequence[Dict[str, Any]], *,
                                  skill: Optional[str] = None,
                                  unit_type: Optional[str] = None,
                                  family: Optional[str] = None
                                  ) -> List[Dict[str, Any]]:
    """Task 3.6: filter A-class rows by skill name, unit type or family for
    per-scorer training data exports."""
    picked = []
    for row in rows:
        if skill is not None and row["candidate_skill"] != skill:
            continue
        if unit_type is not None and row.get("unit_type") != unit_type:
            continue
        if family is not None and row["candidate_family"] != family:
            continue
        picked.append(row)
    return picked


# -- counterfactual pair splits (Task 1.3.2/1.3.3) ---------------------------

def anchor_group(pair: Dict[str, Any]) -> str:
    """One anchor = one partition unit: every branch (BASE + candidates) of
    the same anchor stays on the same side."""
    return f"{pair['run_id']}:{pair['episode_id']}:{pair['anchor_id']}"


def split_pairs(pairs: Sequence[Dict[str, Any]],
                validation_fraction: float = 0.25,
                split_seed: int = 0,
                min_train_groups: int = 4
                ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Anchor-level deterministic split: BASE and candidate branches of one
    anchor can never straddle train/validation."""
    from decision_tree.learning.train import group_split
    rows = [{"group_id": anchor_group(pair), "_pair": pair}
            for pair in pairs]
    train_rows, validation_rows = group_split(
        rows, validation_fraction=validation_fraction,
        split_seed=split_seed, min_train_groups=min_train_groups)
    return ([row["_pair"] for row in train_rows],
            [row["_pair"] for row in validation_rows])
