"""Runtime integration: off / shadow / tree_score (V1.2).

Safety contract:

- ``off``: inert; ``wrap`` returns the manager untouched.
- ``shadow``: candidates from the manager's own decision are scored and the
  disagreement recorded to a separate JSONL log; actions, memory and the
  tactical Telemetry are never touched.
- ``tree_score``: guarded by ``_tree_score_ready`` (model present, target
  correct, source fingerprint and feature schema verified, COMPLETE branch
  plan objects available, validator importable, manager plan state
  updatable). Replacement goes through ``selector.apply_complete_plan``
  (whole Coordinator branch, full Validator re-run, bounded repair, full
  BASE fallback). If any gate fails — or the selected candidate has no
  complete plan objects — Tree Score stays effectively Shadow-only; an
  action-list swap is never used as a bypass.

Identity (V1.2 Phase 1):

- ``run_id`` is generated ONCE per runtime (or taken from config) and never
  changes for the lifetime of the collection process; ``reset`` clears
  episode state only. Every decision record, event and pair record carries
  it.
- ``episode_id`` is unique inside the run: repeated episode numbers get a
  deterministic ``_r<n>`` suffix when a new game actually starts.
- ``scene_config_id`` comes from the experiment manifest when provided,
  otherwise from a canonical hash over map + roster + birth positions +
  environment parameters (never map name or unit count alone); its source
  is stored on every record.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .candidates import CompletePlanCandidate, branch_scoring_features, \
    build_branch_candidates, joint_candidates_from_summary
from .dataset import DecisionRecordWriter, build_decision_record, \
    dumps_key, finalise_outcome, record_key
from .features import DEFAULT_SCHEMA, ZergFeatureSchema, StateFeatureExtractor
from .fingerprints import assert_model_source, execution_context, \
    execution_context_hash, runtime_policy_contract, source_hash
from .proposal_collector import PRIORITY_SKILLS, SkillProposalCollector
from .schema import digest as canonical_digest, load_json
from .dataset import proposal_group_id, _matches_final_actions
from .scorer import ScorerModel, score_candidates, score_routed
from .selector import apply_complete_plan, select_complete_plan, \
    select_skill_plan

MODES = ("off", "shadow", "tree_score")
RUNTIME_CONFIG_KIND = "zerg_tree_runtime_config"
DEFAULT_CONFIG_PATH = "decision_tree/zerg/configs/runtime.json"
DEFAULT_RECORDS_PATH = "decision_tree/zerg/artifacts/records/shadow.jsonl"
DEFAULT_EVENTS_PATH = "decision_tree/zerg/artifacts/events/events.jsonl"
DEFAULT_SCORER_SLOTS = ("zergling_scorer", "hydralisk_scorer",
                        "baneling_scorer", "joint_scorer")
EVENT_KINDS = ("proposed", "selected", "submitted", "committed",
               "rejected", "episode_finished")
SORTABLE_MODEL_TARGETS = ("gain", "ranking", "imitation")


_RUN_SEQ = {"n": 0}


def generate_run_id() -> str:
    """Task 1.1: one identity per collection process. Clock + pid + an
    in-process sequence: a new process necessarily gets a new run_id, and
    repeated construction inside one process still yields unique ids. Once
    created, a runtime's run_id never changes."""
    _RUN_SEQ["n"] += 1
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return f"run_{stamp}_{os.getpid()}_{_RUN_SEQ['n']:03d}"


def default_config(workspace: str = ".") -> Dict[str, Any]:
    return {
        "kind": RUNTIME_CONFIG_KIND,
        "mode": "off",
        "workspace": workspace,
        "source_root": workspace or ".",
        "run_id": None,
        "scene_config": None,          # {"scene_config_id": ..., "source":
                                       #  "experiment_manifest"} or None
        "action_space_version": "smac_v2_standard_v1",
        "observation_protocol": "shared_team_vision_snapshot_v1",
        "model_algorithm": "weighted_sse_threshold_tree_v1",
        "base_dir": "skills/tactical_v1",
        "overlay_dir": ("runs/contact_all_attack_20261122_230107_662982/"
                        "marginal_near/skills/tactical_v1"),
        "records_path": DEFAULT_RECORDS_PATH,
        "events_path": DEFAULT_EVENTS_PATH,
        "collect_skill_proposals": True,
        "skill_proposal_skills": list(PRIORITY_SKILLS),
        "scorers": {name: {"enabled": False, "model_path": None,
                           "min_gain": 0.0} for name in DEFAULT_SCORER_SLOTS},
        "selector": {"enabled": True, "min_gain": 0.0,
                     "tie_tolerance": 1e-9},
        "recording": {"shadow": True, "tree_score": True},
    }


def validate_runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if config.get("kind") != RUNTIME_CONFIG_KIND:
        raise ValueError("runtime config requires kind="
                         f"{RUNTIME_CONFIG_KIND}")
    if config.get("mode") not in MODES:
        raise ValueError("mode must be one of " + ", ".join(MODES))
    for name in config.get("scorers", {}):
        if name not in DEFAULT_SCORER_SLOTS:
            raise ValueError(f"unknown scorer slot {name}")
    scene = config.get("scene_config")
    if scene is not None:
        if not isinstance(scene, dict) or not scene.get("scene_config_id"):
            raise ValueError("scene_config requires scene_config_id")
        scene.setdefault("source", "experiment_manifest")
    return config


def runtime_policy_for_scorers(mode: str, scorers) -> Dict[str, Any]:
    return runtime_policy_contract(mode, scorers, {}, DEFAULT_SCHEMA)


def actions_hash(actions: Any) -> str:
    material = json.dumps(list(actions) if isinstance(actions, (list, tuple))
                          else dict(actions), sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def scene_config_identity(snapshot, map_field: str,
                          include_positions: bool = False
                          ) -> Tuple[str, str]:
    """Task 1.2: canonical scene-config hash over everything that actually
    shapes the game: map, step_mul, roster (actor -> type id + birth
    position), declared enemy count and search location. Map name or unit
    count alone is never the identity."""
    allies = getattr(snapshot, "allies", {}) or {}
    # Composition identity: unit types of the declared roster. Health and
    # remaining steps are PER-STEP STATE and never enter. Birth positions
    # are included ONLY for the episode's opening snapshot (snapshot_id==1)
    # and are then frozen per episode — current mid-battle positions are
    # never used (Task 5.3).
    if include_positions:
        roster = sorted(
            (actor, int(getattr(unit, "unit_type_id", 0) or 0),
             [round(float(v), 3) for v in (getattr(unit, "position", ())
                                           or ())])
            for actor, unit in allies.items()
            if getattr(unit, "alive", True))
    else:
        roster = sorted((actor, int(getattr(unit, "unit_type_id", 0) or 0))
                        for actor, unit in allies.items()
                        if getattr(unit, "alive", True))
    identity = {
        "map": str(map_field),
        "step_mul": int(getattr(snapshot, "step_mul", 0) or 0),
        "n_allies": int(getattr(snapshot, "n_allies", 0) or 0),
        "n_enemies": int(getattr(snapshot, "n_enemies", 0) or 0),
        "search_location": [round(float(v), 3) for v in
                            (getattr(snapshot, "search_location", ()) or ())],
        "roster": roster,
    }
    return f"sconf_{canonical_digest(identity)[:16]}", "canonical_hash"


class EventLog:
    """Append-only JSONL event log, separate from tactical Telemetry."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, kind: str, **fields: Any) -> None:
        if kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {kind}")
        row = {"event": kind, **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True,
                                    separators=(",", ":"),
                                    ensure_ascii=False,
                                    allow_nan=False) + "\n")


class ZergTreeRuntime:
    """Loads models, computes fingerprints, records shadow decisions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 config_path: Optional[str] = None,
                 workspace: Optional[str] = None,
                 tactical_config: Optional[Dict[str, Any]] = None,
                 schema: ZergFeatureSchema = DEFAULT_SCHEMA):
        if config is None:
            config = (load_json(config_path) if config_path
                      else default_config())
        self.config = validate_runtime_config(copy.deepcopy(config))
        if workspace:
            self.config["workspace"] = workspace
        self.workspace = Path(self.config.get("workspace", "."))
        self.schema = schema
        self.tactical_config = tactical_config or {}
        self.mode = self.config["mode"]
        self._started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         time.gmtime())
        self.extractor = StateFeatureExtractor(schema)
        # Task 1.1: run identity lives and dies with this runtime object.
        self._run_id = str(self.config.get("run_id") or generate_run_id())
        self._episode_seen: Dict[str, int] = {}
        self._episode_closed = False
        self._scene_cache: Dict[str, Tuple[str, str]] = {}
        self._last_loop: Optional[int] = None
        self._last_snapshot_id: Optional[int] = None
        self._models: Dict[str, Optional[ScorerModel]] = {}
        self._context_hash: Optional[str] = None
        self._source_fingerprint: Optional[str] = None
        self._writer: Optional[DecisionRecordWriter] = None
        self._events: Optional[EventLog] = None
        self._buffer: Dict[Tuple, Dict[str, Any]] = {}
        self._active_collector: Optional[SkillProposalCollector] = None
        self.bundle_provenance: Optional[Dict[str, Any]] = None
        self.last_error: Optional[str] = None
        self.stats: Dict[str, int] = {
            "decisions_seen": 0, "decisions_recorded": 0,
            "shadow_disagreements": 0, "tree_score_changes": 0,
            "tree_score_fallbacks": 0, "tree_score_blocked": 0,
            "model_errors": 0, "routed_candidates": {}}
        if self.mode != "off":
            self._build_execution_context()
            self._load_models()

    # -- identity -----------------------------------------------------------
    @property
    def run_id(self) -> str:
        return self._run_id

    def new_run(self, run_id: Optional[str] = None) -> str:
        """Explicitly start a new collection run inside this process (same
        runtime object, fresh identity, episode bookkeeping cleared)."""
        self._run_id = str(run_id or generate_run_id())
        self._episode_seen.clear()
        self._episode_closed = False
        self._scene_cache.clear()
        self._last_loop = None
        self._last_snapshot_id = None
        self._buffer.clear()
        if self._active_collector is not None:
            self._active_collector.start_run(self._run_id)
        return self._run_id

    def reset(self) -> None:
        """Episode-state reset only; the run_id survives (Task 1.1.6)."""
        self._episode_seen.clear()
        self._episode_closed = False
        self._scene_cache.clear()
        self._last_loop = None
        self._last_snapshot_id = None
        self._buffer.clear()

    def resolve_episode_id(self, raw_episode: str, snapshot_id: int,
                           game_loop: int) -> str:
        """Unique episode id inside the run. A repeat of an episode number
        becomes a NEW game when a snapshot/loop rollover is observed OR when
        the caller finalised the previous episode (explicit boundary)."""
        key = str(raw_episode)
        is_new_game = self._episode_closed \
            or (self._last_snapshot_id is not None
                and snapshot_id < self._last_snapshot_id) \
            or (self._last_loop is not None and game_loop < self._last_loop)
        self._episode_closed = False
        if is_new_game:
            count = self._episode_seen.get(key, 0)
            self._episode_seen[key] = count + 1
            if count:
                key = f"{key}_r{count + 1}"
        elif key not in self._episode_seen:
            self._episode_seen[key] = 1
        self._last_loop = game_loop
        self._last_snapshot_id = snapshot_id
        return key

    def _scene_config(self, snapshot, map_field: str) -> Tuple[str, str]:
        configured = self.config.get("scene_config")
        if configured and configured.get("scene_config_id"):
            return str(configured["scene_config_id"]), \
                str(configured.get("source", "experiment_manifest"))
        episode = getattr(self, "_pending_identity", None)
        episode_key = episode[1] if episode else None
        if episode_key:
            cached = self._scene_cache.get(episode_key)
            if cached is not None:
                return cached
        # Birth positions enter ONLY at the episode's opening snapshot; the
        # identity is then frozen for the whole episode.
        include_positions = int(getattr(snapshot, "snapshot_id", 0) or 0) == 1
        identity = scene_config_identity(snapshot, map_field,
                                         include_positions=include_positions)
        identity = (identity[0], identity[1] +
                    ("+birth_positions" if include_positions else ""))
        if episode_key:
            self._scene_cache[episode_key] = identity
        return identity

    # -- execution context --------------------------------------------------
    def _build_execution_context(self,
                                 tactical_config: Optional[Dict] = None
                                 ) -> None:
        effective_config = tactical_config if tactical_config is not None \
            else self.tactical_config
        self._policy = runtime_policy_contract(
            self.mode,
            [name for name, slot in self.config["scorers"].items()
             if slot.get("enabled")],
            {"selector": self.config.get("selector", {})},
            self.schema)
        context = execution_context(
            source_hash(str(self.config.get("source_root") or self.workspace),
                        self.config.get("base_dir"),
                        self.config.get("overlay_dir")),
            effective_config, self._policy, self.schema,
            extra={
                "action_space_version":
                    self.config.get("action_space_version"),
                "observation_protocol":
                    self.config.get("observation_protocol"),
                "model_algorithm": self.config.get("model_algorithm"),
            })
        self._context = context
        self._context_hash = execution_context_hash(context)
        self._source_fingerprint = context["source_fingerprint"]

    def rebind_execution_context(self, tactical_config: Dict[str, Any]) -> None:
        """Task 5.3: rebuild the frozen context around the manager's ACTUAL
        effective config (never silently an empty dict), then re-validate
        every loaded model against it."""
        self.tactical_config = dict(tactical_config or {})
        self._build_execution_context(self.tactical_config)
        from .fingerprints import validate_deployment
        for name, model in self._models.items():
            if model is None:
                continue
            validate_deployment(model.record, self._context)
            assert_model_source(model.record, self._source_fingerprint)

    # -- model loading ------------------------------------------------------
    def _load_models(self) -> None:
        for name, slot in self.config["scorers"].items():
            path = slot.get("model_path")
            if not slot.get("enabled") or not path:
                self._models[name] = None
                continue
            resolved = Path(path)
            if not resolved.is_absolute():
                resolved = self.workspace / path
            if not resolved.is_file():
                self._models[name] = None
                continue
            record = load_json(str(resolved))
            if record.get("scorer") != name:
                raise ValueError(f"model file {path} declares scorer "
                                 f"{record.get('scorer')!r}, expected {name!r}")
            if record.get("target") not in SORTABLE_MODEL_TARGETS:
                raise ValueError(f"model file {path} has unsupported target "
                                 f"{record.get('target')!r}")
            self._models[name] = ScorerModel(record)
            from .fingerprints import validate_deployment
            validate_deployment(self._models[name].record, self._context)
            # Task 3.2/5.1: source fingerprint must match the live tree in
            # EVERY mode — shadow data is never marked as a valid version
            # against the wrong source.
            assert_model_source(self._models[name].record,
                                self._source_fingerprint,
                                allow_missing=False)

    # -- integration --------------------------------------------------------
    def wrap(self, manager: Any) -> Any:
        """Accept a raw SkillManager OR a
        :class:`factory.ZergManagerBundle` (preferred — Task 5.1). In ``off``
        mode the manager itself is returned untouched."""
        if self.mode == "off":
            return getattr(manager, "manager", manager)
        controller = _RuntimeController(manager, self)
        actual = controller.manager
        if hasattr(manager, "source_fingerprint"):
            self._verify_bundle(manager, actual)
            if not self.tactical_config and manager.effective_config:
                self.rebind_execution_context(manager.effective_config)
        if self.config.get("collect_skill_proposals", True):
            collector = SkillProposalCollector(
                enabled=self.config.get("skill_proposal_skills",
                                        PRIORITY_SKILLS),
                schema=self.schema).install(actual)
            collector.start_run(self._run_id)
            controller.collector = collector
            self._active_collector = collector
        return controller

    def _verify_bundle(self, bundle, manager: Any) -> None:
        """Task 5.1: verify the manager ACTUALLY comes from the resolved
        tree the runtime fingerprinted — config declarations are never
        accepted as proof."""
        module = type(manager).__module__
        module_file = str(getattr(import_module_of(manager), "__file__", ""))
        source_root = Path(self.config.get("source_root")
                           or self.workspace)
        expected_root = (source_root / "decision_tree/zerg/resolved") \
            .resolve()
        try:
            inside_resolved = Path(module_file).resolve().is_relative_to(
                expected_root)
        except AttributeError:
            inside_resolved = str(Path(module_file).resolve()).startswith(
                str(expected_root))
        if not inside_resolved:
            raise ValueError(
                f"manager module {module} loaded from {module_file!r} is "
                "NOT the resolved tree — refusing to wrap")
        if bundle.source_fingerprint != self._source_fingerprint:
            raise ValueError(
                "bundle source fingerprint "
                f"{str(bundle.source_fingerprint)[:16]}… != runtime "
                f"fingerprint {str(self._source_fingerprint)[:16]}… — "
                "refusing to wrap")
        self.bundle_provenance = {
            "module_path": str(bundle.module_path),
            "module_file": module_file,
            "source_fingerprint": bundle.source_fingerprint,
            "manifest_path": str(bundle.manifest_path),
            "effective_config_digest": canonical_digest(
                bundle.effective_config),
        }

    def scorer_model(self, name: str) -> Optional[ScorerModel]:
        return self._models.get(name)

    @property
    def execution_context_hash(self) -> Optional[str]:
        return self._context_hash

    @property
    def source_fingerprint(self) -> Optional[str]:
        return self._source_fingerprint

    def write_run_manifest(self) -> Optional[Path]:
        """Task 5.8: a reproducibility manifest for this run — identity,
        scene configs seen so far, fingerprints, mode and config digest."""
        if self.mode == "off":
            return None
        records_path = self.config.get("records_path") or DEFAULT_RECORDS_PATH
        resolved = Path(records_path)
        if not resolved.is_absolute():
            resolved = self.workspace / resolved
        manifest = {
            "kind": "zerg_run_manifest_v1",
            "run_id": self._run_id,
            "mode": self.mode,
            "source_fingerprint": self._source_fingerprint,
            "execution_context_hash": self._context_hash,
            "episode_ids_seen": sorted(self._episode_seen),
            "scene_configs": dict(self._scene_cache),
            "config_digest": canonical_digest(self.config),
            "started_at": getattr(self, "_started_at", None),
        }
        resolved.parent.mkdir(parents=True, exist_ok=True)
        out = resolved.parent / f"run_manifest_{self._run_id}.json"
        out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        return out

    # -- sinks --------------------------------------------------------------
    def _writer_for(self) -> DecisionRecordWriter:
        if self._writer is None:
            path = self.config.get("records_path") or DEFAULT_RECORDS_PATH
            resolved = Path(path)
            if not resolved.is_absolute():
                resolved = self.workspace / path
            self._writer = DecisionRecordWriter(str(resolved))
        return self._writer

    def _events_for(self) -> EventLog:
        if self._events is None:
            path = self.config.get("events_path") or DEFAULT_EVENTS_PATH
            resolved = Path(path)
            if not resolved.is_absolute():
                resolved = self.workspace / path
            self._events = EventLog(str(resolved))
        return self._events

    def log_event(self, kind: str, **fields: Any) -> None:
        if self.mode == "off":
            return
        fields.setdefault("run_id", self._run_id)
        try:
            self._events_for().log(kind, **fields)
        except Exception:
            pass                      # the event log never breaks decisions

    # -- recording ----------------------------------------------------------
    def record_decision(self, *, episode_id: str,
                        scene_config_id: Optional[str],
                        scene_config_source: Optional[str],
                        seed: Optional[int], map_id: Optional[str],
                        snapshot, candidates,
                        teacher_choice: Optional[str],
                        model_scores: List[Dict[str, Any]],
                        selected_candidate: Optional[str],
                        final_actions: Dict[str, int],
                        execution_status: str,
                        skill_proposals: Optional[Dict[str, Any]] = None
                        ) -> Optional[Tuple]:
        record = build_decision_record(
            run_id=self._run_id, episode_id=episode_id,
            scene_config_id=scene_config_id, seed=seed, map_id=map_id,
            snapshot=snapshot, candidates=candidates,
            global_features=self._last_state_features or {},
            teacher_choice=teacher_choice, model_scores=model_scores,
            selected_candidate=selected_candidate,
            final_actions=final_actions, execution_status=execution_status,
            source_hash=self._source_fingerprint or "",
            execution_context_hash=self._context_hash or "",
            schema=self.schema)
        record["scene_config_source"] = scene_config_source
        if skill_proposals:
            # A-class section (Task 4.3): same decision key, own family and
            # candidate groups; never merged into the joint candidate list.
            record["skill_proposals"] = skill_proposals
        key = record_key(record)
        self._buffer[key] = record
        self.stats["decisions_seen"] += 1
        return key

    def finalize_episode(self, *, outcome: str,
                         episode_return: Optional[float],
                         ally_losses: int, enemy_losses: int,
                         damage_dealt: Optional[float] = None,
                         damage_received: Optional[float] = None,
                         episode_id: Optional[str] = None,
                         baseline_return: Optional[float] = None
                         ) -> int:
        if self.mode == "off":
            return 0
        writer = self._writer_for()
        written = 0
        for key, record in sorted(self._buffer.items()):
            if record["run_id"] != self._run_id:
                continue
            if episode_id is not None and record["episode_id"] != episode_id:
                continue
            finalise_outcome(record, outcome=outcome,
                             episode_return=episode_return,
                             ally_losses=ally_losses,
                             enemy_losses=enemy_losses,
                             damage_dealt=damage_dealt,
                             damage_received=damage_received,
                             baseline_return=baseline_return)
            if writer.write(record):
                written += 1
                self.stats["decisions_recorded"] += 1
        for key in [k for k, r in self._buffer.items()
                    if r["run_id"] == self._run_id
                    and (episode_id is None or r["episode_id"] == episode_id)]:
            del self._buffer[key]
        self._episode_closed = True       # explicit new-episode boundary
        self.log_event("episode_finished", episode_id=episode_id,
                       outcome=outcome, episode_return=episode_return,
                       baseline_return=baseline_return,
                       ally_losses=ally_losses, enemy_losses=enemy_losses,
                       damage_dealt=damage_dealt,
                       damage_received=damage_received,
                       records_written=written,
                       execution_context_hash=self._context_hash)
        return written

    # -- internals ----------------------------------------------------------
    _last_state_features: Optional[Dict[str, Any]] = None
    _pending_identity: Optional[Tuple[str, str, str]] = None

    def prepare_decision(self, payload: Dict[str, Any], manager: Any
                         ) -> Tuple[str, str, str]:
        """Resolve the (run_id, episode_id, scene identity) BEFORE the
        manager steps, so propose-time recording and collector episode
        wipes happen under the correct identity. Idempotent per decision:
        the result is cached for observe_step."""
        state = payload.get("state", {}) or {}
        raw_episode = str(state.get("episode", 0))
        game_loop = int(state.get("game_loop", 0) or 0)
        expected_snapshot = int(getattr(manager, "_snapshot_counter", 0)) + 1
        key = str(raw_episode)
        is_new_game = self._episode_closed \
            or (self._last_snapshot_id is not None
                and expected_snapshot < self._last_snapshot_id) \
            or (self._last_loop is not None and game_loop < self._last_loop)
        self._episode_closed = False
        if is_new_game:
            count = self._episode_seen.get(key, 0)
            self._episode_seen[key] = count + 1
            if count:
                key = f"{key}_r{count + 1}"
        elif key not in self._episode_seen:
            self._episode_seen[key] = 1
        self._last_loop = game_loop
        self._last_snapshot_id = expected_snapshot
        if self._active_collector is not None \
                and self._active_collector.episode_id != key:
            self._active_collector.start_episode(self._run_id, key)
        self._pending_identity = (self._run_id, key, str(state.get("map",
                                   "unknown")))
        return self._pending_identity

    def observe_step(self, payload: Dict[str, Any], manager: Any,
                     actions: List[int], summary: Dict[str, Any]
                     ) -> Tuple[List[int], Dict[str, Any], Optional[Dict]]:
        state = payload.get("state", {}) or {}
        snapshot = self._read_snapshot(manager)
        context = self._read_context(manager)
        self._last_state_features = self.extractor.extract_state(
            snapshot, context=context) if snapshot is not None else None
        prepared = self._pending_identity or self.prepare_decision(payload,
                                                                   manager)
        self._pending_identity = None
        _, episode, _ = prepared
        map_field = state.get("map", "unknown")
        if isinstance(map_field, dict):
            map_field = map_field.get("name") or "unknown"
        scene_config_id, scene_config_source = self._scene_config(
            snapshot, map_field)
        seed = state.get("seed")
        seed = int(seed) if isinstance(seed, int) else None

        candidates = joint_candidates_from_summary(
            summary, snapshot, context=context, payload=payload,
            schema=self.schema) if snapshot is not None else []
        teacher = candidates[0].candidate_id if candidates else None
        rule_actions = list(actions)
        meta: Dict[str, Any] = {"mode": self.mode, "run_id": self._run_id,
                                "episode_id": episode}

        models = {name: self.scorer_model(name)
                  for name in ("zergling_scorer", "hydralisk_scorer",
                               "baneling_scorer", "joint_scorer")}
        scores, route_counts = (score_routed(
            self._last_state_features or {}, candidates, models,
            schema=self.schema) if candidates else ([], {}))
        self.stats.setdefault("routed_candidates", {})
        if isinstance(self.stats.get("routed_candidates"), dict):
            for slot, count in route_counts.items():
                if count:
                    self.stats["routed_candidates"][slot] = \
                        self.stats["routed_candidates"].get(slot, 0) + count
        meta["route_counts"] = route_counts
        meta["scores"] = [row.to_record() for row in scores]

        # ---- Phase 1: A-class (skill-internal) shadow scoring -------------
        # The collector holds THIS decision step's proposals keyed by
        # (run_id, episode_id, snapshot_id, game_loop); candidates_for
        # validates the identity, so a stale step's plans can never be
        # scored. propose() is NOT called again; candidates and actions are
        # untouched (shadow).
        snapshot_id = int(getattr(snapshot, "snapshot_id", 0) or 0)
        game_loop = int(getattr(snapshot, "game_loop", 0) or 0)
        internal_by_skill = {}
        if self._active_collector is not None and snapshot is not None:
            try:
                internal_by_skill = self._active_collector.candidates_for(
                    snapshot_id, game_loop)
            except Exception:
                internal_by_skill = {}
        internal_flat = [c for items in internal_by_skill.values()
                         for c in items]
        internal_scores, internal_routes = (score_routed(
            self._last_state_features or {}, internal_flat, models,
            schema=self.schema) if internal_flat else ([], {}))
        if isinstance(self.stats.get("routed_candidates"), dict):
            for slot, count in internal_routes.items():
                if count:
                    self.stats["routed_candidates"][slot] = \
                        self.stats["routed_candidates"].get(slot, 0) + count
        meta["internal_route_counts"] = internal_routes
        meta["internal_scores"] = [row.to_record() for row in internal_scores]
        meta["internal_candidates"] = {
            skill: [c.to_record() for c in items]
            for skill, items in internal_by_skill.items()}
        supported = [row for row in scores if row.supported]
        predicted = None
        if supported:
            order = {candidate.candidate_id: index
                     for index, candidate in enumerate(candidates)}
            predicted = min(supported,
                            key=lambda row: (-row.score,
                                             order.get(row.candidate_id, 0))
                            ).candidate_id
        meta["rule_choice"] = teacher
        meta["model_choice"] = predicted
        meta["agree"] = (predicted == teacher) if predicted else None
        if predicted is not None and predicted != teacher:
            self.stats["shadow_disagreements"] += 1
        self.log_event("proposed", episode_id=episode,
                       scene_config_id=scene_config_id,
                       snapshot_id=getattr(snapshot, "snapshot_id", None),
                       game_loop=getattr(snapshot, "game_loop", None),
                       teacher_choice=teacher, model_choice=predicted,
                       n_candidates=len(candidates),
                       execution_context_hash=self._context_hash)

        # ---- Phase 2: complete branches from the Coordinator's own
        # validated cache (real CandidatePlan objects; BASE first).
        branches = []
        if snapshot is not None:
            try:
                branches = build_branch_candidates(manager, snapshot, summary)
                branch_scoring_features(branches, summary)
            except Exception:
                branches = []
        base_complete = branches[0] if branches else None

        selected = teacher
        final_actions_map = {f"u{i}": action for i, action
                             in enumerate(rule_actions)}
        execution_status = "pending"
        if self.mode == "tree_score":
            selector_config = self.config.get("selector", {})
            model = self.scorer_model("joint_scorer")
            ranking_model = None
            from .scorer import PairwiseRankModel
            if isinstance(model, PairwiseRankModel):
                ranking_model = model       # Task 4.3: ranking -> tournament
            if not self._tree_score_ready(manager, model, base_complete,
                                          branches):
                self.stats["tree_score_blocked"] += 1
                self.log_event("rejected", episode_id=episode,
                               reason="tree_score_gate_failed",
                               gate=dict(self.last_tree_score_gate),
                               fallback="base")
            else:
                # Task 2.3/2.4: the OFFICIAL B-class replacement path is
                # complete branches only — summary-level action lists are
                # never used to bypass the plan checks.
                decision = select_complete_plan(
                    branches, self._last_state_features or {},
                    model=(ranking_model or model),
                    min_gain=float(selector_config.get("min_gain", 0.0)),
                    tie_tolerance=float(selector_config.get(
                        "tie_tolerance", 1e-9)),
                    schema=self.schema)
                meta["selection"] = {k: v for k, v in decision.items()
                                     if k != "scores"}
                selected_branch = decision.get("selected")
                if decision.get("changed") and selected_branch is not None:
                    replacement = apply_complete_plan(
                        selected_branch, manager, context=context,
                        memory=getattr(manager, "memory", None))
                    meta["replacement"] = {
                        k: v for k, v in replacement.items()
                        if k in ("applied", "reason", "replaced_plan_ids",
                                 "validator_problems", "repair_log",
                                 "commit_status")}
                    if replacement.get("applied"):
                        swapped = self._actions_list(
                            replacement["final_actions"], len(rule_actions))
                        if swapped is not None:
                            self._update_manager_plan_state(
                                manager, replacement["final_plans"],
                                swapped, summary)
                            rule_actions = swapped
                            selected = selected_branch.candidate_id
                            self.stats["tree_score_changes"] += 1
                            final_actions_map = dict(
                                replacement["final_actions"])
                            self.log_event(
                                "selected", episode_id=episode,
                                selected=selected, replaced=True,
                                replaced_plan_ids=replacement[
                                    "replaced_plan_ids"],
                                validator_problems=[],
                                rule_actions_hash=actions_hash(
                                    summary.get("actions", {})),
                                final_actions_hash=actions_hash(swapped))
                        else:
                            selected = teacher
                            self.stats["tree_score_fallbacks"] += 1
                            self.log_event("rejected", episode_id=episode,
                                           reason="action_list_incomplete",
                                           fallback="base")
                    else:
                        # Replacement failed: record the ACTUAL BASE as the
                        # final selection.
                        selected = teacher
                        self.stats["tree_score_fallbacks"] += 1
                        self.log_event("rejected", episode_id=episode,
                                       reason=replacement.get("reason"),
                                       fallback="base")
        self.record_decision(
            episode_id=episode, scene_config_id=scene_config_id,
            scene_config_source=scene_config_source, seed=seed,
            map_id=str(map_field), snapshot=snapshot, candidates=candidates,
            teacher_choice=teacher, model_scores=meta["scores"],
            selected_candidate=selected, final_actions=final_actions_map,
            execution_status=execution_status,
            skill_proposals=self._enriched_skill_proposals(
                snapshot, meta, final_actions_map, execution_status))
        meta["selected_candidate"] = selected
        return rule_actions, summary, meta

    # -- complete-plan candidates (Task 4.1) --------------------------------
    def _base_complete_candidate(self, manager, snapshot, summary,
                                 candidates, teacher
                                 ) -> Optional[CompletePlanCandidate]:
        pending = getattr(manager, "_pending_commit", None)
        if not pending:
            return None
        plans = list(pending[1] or ())
        if not plans:
            return None
        chosen = (summary.get("chosen_plans") or [{}])[0] \
            if summary.get("chosen_plans") else {}
        score = chosen.get("score")
        actions_by_actor = {f"u{i}": action for i, action
                            in enumerate(summary.get("actions", ()) or ())}
        claims = {}
        dependencies = []
        for plan in plans:
            for control in getattr(getattr(plan, "claims", None),
                                   "controls", ()) or ():
                claims.setdefault("controls", []).append(control)
            dependencies.extend(
                {"kind": getattr(d, "kind", ""),
                 "refs": list(getattr(d, "refs", ()) or ())}
                for d in getattr(plan, "dependencies", ()) or ())
        return CompletePlanCandidate(
            candidate_id=teacher or "joint_rule_choice",
            snapshot_id=int(getattr(snapshot, "snapshot_id", 0) or 0),
            plans=plans,
            actions=actions_by_actor,
            resource_claims=claims,
            dependencies=dependencies,
            execution_duties=dict(
                (summary.get("execution_duties") or {}).get("actors") or {}),
            rule_score=float(score) if isinstance(score, (int, float))
            else None,
            candidate_features={},
            legal=True,
            source_plan_ids=[str(getattr(p, "proposal_id", ""))
                             for p in plans],
            candidate_skill=str(chosen.get("skill") or "joint"),
            plan_objects_available=True)

    # -- tree_score gate (V1.3 Task 2.6) ------------------------------------
    def _tree_score_ready(self, manager, model,
                          base_complete: Optional[CompletePlanCandidate],
                          branches: Optional[List[CompletePlanCandidate]] = None
                          ) -> bool:
        """Every condition must hold; any failure means Tree Score falls
        back to BASE (or stays blocked). Correction models are veto
        instruments, not ranking instruments for plan replacement."""
        snapshot_id = getattr(self._read_snapshot(manager),
                              "snapshot_id", None)
        checks = {
            "model_loaded": model is not None,
            "model_target_sortable": model is not None and
            model.record.get("target") in ("gain", "ranking", "imitation"),
            "model_kind_supported": model is not None
            and type(model).__name__ in ("ScorerModel", "PairwiseRankModel"),
            "source_fingerprint_verified": self._source_fingerprint is not None,
            "feature_schema_verified": True,   # enforced at model load time
            "complete_plan_objects": bool(branches)
            and all(b.plan_objects_available and b.plans for b in branches),
            "branches_current_snapshot": bool(branches) and all(
                b.snapshot_id == snapshot_id for b in branches),
            "base_branch_present": bool(branches)
            and branches[0].candidate_id == "joint_base",
            "validator_available":
                getattr(manager, "evaluator", None) is not None,
            "manager_state_updatable": getattr(manager, "_pending_commit",
                                               None) is not None
            and getattr(manager, "_cache", None) is not None,
            "replacement_interface_available": True,   # apply_complete_plan
        }
        self.last_tree_score_gate = checks
        return all(checks.values())

    last_tree_score_gate: Dict[str, bool] = {}

    # -- manager state (Task 4.3) -------------------------------------------
    @staticmethod
    def _update_manager_plan_state(manager: Any, plans, actions_list,
                                   summary: Dict[str, Any]) -> None:
        if isinstance(actions_list, dict):
            n = max((int(k[1:]) for k in actions_list
                     if isinstance(k, str) and k.startswith("u")
                     and k[1:].isdigit()), default=-1) + 1
            actions_list = [actions_list.get(f"u{i}") for i in range(n)]
        pending = getattr(manager, "_pending_commit", None)
        if pending:
            snapshot, _, micro_actions, risk_now, context = pending
            manager._pending_commit = (snapshot, list(plans), micro_actions,
                                       risk_now, context)
        cache = getattr(manager, "_cache", None)
        if cache:
            manager._cache = (cache[0], list(actions_list), cache[2],
                              list(plans), cache[4])
        if summary is not None:
            summary["chosen_plans"] = [
                {"id": str(getattr(p, "proposal_id", "")),
                 "skill": getattr(p, "primary_skill", ""),
                 "provider": getattr(p, "specialization_provider", None),
                 "actors": list(getattr(p, "controlled_actors", ()) or ()),
                 "goal": getattr(p, "goal", None),
                 "score": None,
                 "rationale": getattr(p, "rationale", "")
                 or "tree_score_replacement"}
                for p in plans]

    def _enriched_skill_proposals(self, snapshot, meta,
                                  final_actions: Dict[str, int],
                                  execution_status: str) -> Optional[Dict]:
        """Task 1.3: per-A-candidate shadow log. Candidates keep their
        decision-group identity; model scores, in-group predicted rank,
        teacher (unique final-action match) and execution facts are added —
        an unselected candidate is recorded as unselected, never as an
        error."""
        collector = self._active_collector
        if collector is None or snapshot is None:
            return None
        try:
            proposals = collector.records_for(int(snapshot.snapshot_id),
                                              int(snapshot.game_loop))
        except Exception:
            return None
        if not proposals:
            return proposals
        score_by_id = {row["candidate_id"]: row
                       for row in meta.get("internal_scores", [])}
        episode = meta.get("episode_id") or ""
        run_id = self._run_id
        snapshot_id = int(getattr(snapshot, "snapshot_id", 0) or 0)
        for skill, candidates in proposals.items():
            # group candidates by proposal task, then rank inside groups
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for candidate in candidates:
                groups.setdefault(proposal_group_id(candidate),
                                  []).append(candidate)
            for members in groups.values():
                scored = []
                for candidate in members:
                    row = score_by_id.get(candidate["candidate_id"])
                    supported = bool(row and row.get("supported"))
                    raw_score = row.get("score") if row else None
                    score = (float(raw_score)
                             if isinstance(raw_score, (int, float))
                             else None)
                    matched = _matches_final_actions(candidate, final_actions)
                    candidate["final_action_match"] = matched
                    candidate["teacher"] = None        # resolved below
                    candidate["execution_status"] = execution_status
                    candidate["shadow"] = {
                        "model_score": score,
                        "model_supported": supported,
                        "model_confidence": float(row["confidence"])
                        if row and isinstance(row.get("confidence"),
                                              (int, float)) else None,
                        "rule_score": candidate.get("rule_score"),
                        "reason": (row or {}).get("reason",
                                                  "no_model_rule_fallback"),
                    }
                    scored.append((supported, score, candidate))
                # predicted rank: supported first, score desc, rule order tie
                order = sorted(
                    range(len(scored)),
                    key=lambda i: (not scored[i][0],
                                   -(scored[i][1] if scored[i][1] is not None
                                     else -1e18),
                                   i))
                for rank_position, index in enumerate(order):
                    scored[index][2]["shadow"]["predicted_rank"] = \
                        rank_position
                matched_ids = [scored[i][2]["candidate_id"] for i
                               in range(len(scored))
                               if scored[i][2]["final_action_match"]]
                teacher_id = matched_ids[0] if len(matched_ids) == 1 else None
                for _, _, candidate in scored:
                    candidate["teacher"] = teacher_id == \
                        candidate["candidate_id"] if teacher_id else None
                    candidate["decision_group_id"] = (
                        f"{run_id}:{episode}:{snapshot_id}:{skill}:"
                        f"{proposal_group_id(candidate)}")
        return proposals

    def _skill_proposals(self, snapshot) -> Optional[Dict]:
        collector = self._active_collector
        if collector is None or snapshot is None:
            return None
        try:
            return collector.records_for(int(snapshot.snapshot_id),
                                         int(snapshot.game_loop))
        except Exception:
            return None

    @staticmethod
    def _actions_list(actions_by_actor: Dict[str, int],
                      n_agents: int) -> Optional[List[int]]:
        actions = []
        for i in range(n_agents):
            action = actions_by_actor.get(f"u{i}")
            if not isinstance(action, int) or isinstance(action, bool):
                return None
            actions.append(action)
        return actions

    @staticmethod
    def _read_snapshot(manager: Any):
        pending = getattr(manager, "_pending_commit", None)
        if pending:
            return pending[0]
        cache = getattr(manager, "_cache", None)
        if cache:
            return cache[4]
        return None

    @staticmethod
    def _read_context(manager: Any):
        pending = getattr(manager, "_pending_commit", None)
        if pending and len(pending) >= 5:
            return pending[4]
        return None


def import_module_of(manager: Any):
    import importlib
    return importlib.import_module(type(manager).__module__)


def sys_module(manager: Any):        # kept for interface compatibility
    return import_module_of(manager)


class _RuntimeController:
    """Wraps a SkillManager without touching its class; commit_final_actions
    delegates straight through so the ORIGINAL memory-commit conditions keep
    control."""

    def __init__(self, manager: Any, runtime: ZergTreeRuntime):
        if hasattr(manager, "source_fingerprint"):
            manager = manager.manager       # unwrap a factory bundle
        self.manager = manager
        self.runtime = runtime
        self.collector: Optional[SkillProposalCollector] = None
        self.last_meta: Optional[Dict[str, Any]] = None
        self._pending_actions: Optional[List[int]] = None

    def __getattr__(self, name):
        return getattr(self.manager, name)

    def reset(self) -> None:
        self.manager.reset()
        # Task 1.1.6 / 1.4: run identity survives; episode state drops.
        self.runtime.reset()
        if self.collector is not None:
            self.collector.reset()

    def step(self, payload: Dict[str, Any], n_agents: int):
        runtime = self.runtime
        if runtime.mode != "off":
            try:
                runtime.prepare_decision(payload, self.manager)
            except Exception as exc:
                runtime.last_error = f"identity:{type(exc).__name__}: {exc}"
        actions, summary = self.manager.step(payload, n_agents)
        try:
            processed, annotated, meta = runtime.observe_step(
                payload, self.manager, actions, summary)
        except Exception as exc:          # never break the original skill
            runtime.stats["model_errors"] += 1
            runtime.last_error = f"{type(exc).__name__}: {exc}"
            return actions, summary
        self.last_meta = meta
        self._pending_actions = list(processed)
        if runtime.mode == "tree_score" and processed != list(actions):
            annotated = copy.deepcopy(annotated)
            annotated["zerg_tree"] = {
                "mode": runtime.mode,
                "changed": True,
                "rule_actions": [int(a) for a in actions],
                "tree_actions": [int(a) for a in processed],
                "execution_context_hash": runtime.execution_context_hash,
            }
            runtime.log_event("submitted",
                              episode_id=meta.get("episode_id"),
                              rule_actions_hash=actions_hash(actions),
                              final_actions_hash=actions_hash(processed),
                              replaced_units=[
                                  f"u{i}" for i, (a, b)
                                  in enumerate(zip(actions, processed))
                                  if a != b],
                              reason="tree_score_plan_replacement")
            return processed, annotated
        if runtime.mode == "shadow":
            annotated = copy.deepcopy(annotated)
            annotated["zerg_tree"] = {"mode": runtime.mode,
                                      "changed": False}
            runtime.log_event("submitted",
                              episode_id=meta.get("episode_id"),
                              rule_actions_hash=actions_hash(actions),
                              final_actions_hash=actions_hash(actions),
                              replaced_units=[], reason=None)
            return actions, annotated
        return actions, summary

    def commit_final_actions(self, final_actions: List[int]) -> None:
        """Phase 7: distinguish (a) the commit call completing, (b) the
        manager's memory actually committing new plan state, (c) nothing new
        committed (idempotent re-read or plans dropped by arbitration). The
        decision tree NEVER controls this — full delegation to the manager's
        own surviving-plan logic."""
        runtime = self.runtime
        before_actions = getattr(self.manager.memory,
                                 "last_committed_actions", None)
        before_flag = bool(getattr(self.manager.memory, "committed", False))
        self.manager.commit_final_actions(final_actions)
        after_actions = getattr(self.manager.memory,
                                "last_committed_actions", None)
        final_map = {f"u{i}": a for i, a in enumerate(final_actions)}
        if after_actions is None:
            memory_commit = "none"
        elif before_actions is not None and after_actions == before_actions:
            memory_commit = ("idempotent" if before_flag
                             else "none")
        elif all(after_actions.get(actor) == action
                 for actor, action in final_map.items()
                 if actor in after_actions):
            memory_commit = "new_commit"
        else:
            memory_commit = "partial_or_arbitrated"
        if runtime.mode != "off":
            runtime.log_event(
                "committed",
                episode_id=(self.last_meta or {}).get("episode_id"),
                rule_actions_hash=actions_hash(
                    self._pending_actions or final_actions),
                final_actions_hash=actions_hash(final_actions),
                commit_call_completed=True,
                memory_commit=memory_commit,
                note=("original surviving-plan memory commit passed; NOT an "
                      "SC2 engine acknowledgement"))
