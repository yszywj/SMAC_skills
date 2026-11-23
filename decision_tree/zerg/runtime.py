"""Runtime integration: off / shadow / tree_score (V1.1).

Safety contract (enforced structurally here):

- ``off``: the runtime is inert. ``wrap(manager)`` returns the manager
  itself; no models are created, nothing is recorded, behaviour is
  bit-identical to the original skill.
- ``shadow``: candidates from the manager's own decision are scored and the
  disagreement with the rule choice is recorded to a separate zerg JSONL
  log. Actions, memory and the tactical Telemetry are never touched.
- ``tree_score``: the selector may replace the complete plan ONLY through
  ``selector.apply_selected_plan`` — real CandidatePlan objects, full
  Validator re-run, bounded repair, full BASE fallback on any failure.
  Summary-level candidates without plan objects can never trigger a swap.
  After any swap the runner still calls ``commit_final_actions``; the
  manager's own surviving-plan logic keeps full control of TacticalMemory.

Event log (Phase 9): proposed / selected / submitted / committed /
rejected / episode_finished events go to a separate zerg events JSONL with
rule and final action hashes, replaced units, validator results and model
support. ``committed`` means the ORIGINAL skill memory-commit condition
passed — it is NOT an SC2 engine acknowledgement. Shadow never changes
actions or memory; any model error falls back to the rule behaviour.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .candidates import joint_candidates_from_summary
from .dataset import DecisionRecordWriter, build_decision_record, \
    dumps_key, finalise_outcome, record_key
from .features import DEFAULT_SCHEMA, ZergFeatureSchema, StateFeatureExtractor
from .fingerprints import assert_model_source, execution_context, \
    execution_context_hash, runtime_policy_contract, source_hash
from .proposal_collector import PRIORITY_SKILLS, SkillProposalCollector
from .schema import load_json
from .scorer import ScorerModel, score_candidates
from .selector import apply_selected_plan, select_skill_plan

MODES = ("off", "shadow", "tree_score")
RUNTIME_CONFIG_KIND = "zerg_tree_runtime_config"
DEFAULT_CONFIG_PATH = "decision_tree/zerg/configs/runtime.json"
DEFAULT_RECORDS_PATH = "decision_tree/zerg/artifacts/records/shadow.jsonl"
DEFAULT_EVENTS_PATH = "decision_tree/zerg/artifacts/events/events.jsonl"
DEFAULT_SCORER_SLOTS = ("zergling_scorer", "hydralisk_scorer",
                        "baneling_scorer", "joint_scorer")
EVENT_KINDS = ("proposed", "selected", "submitted", "committed",
               "rejected", "episode_finished")


def default_config(workspace: str = ".") -> Dict[str, Any]:
    return {
        "kind": RUNTIME_CONFIG_KIND,
        "mode": "off",
        "workspace": workspace,
        "source_root": workspace or ".",
        "run_id": None,
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
    return config


def runtime_policy_for_scorers(mode: str, scorers) -> Dict[str, Any]:
    """Policy contract used when exporting a model outside a live runtime."""
    return runtime_policy_contract(mode, scorers, {}, DEFAULT_SCHEMA)


def actions_hash(actions: Any) -> str:
    material = json.dumps(list(actions) if isinstance(actions, (list, tuple))
                          else dict(actions), sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()[:16]


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
        self.extractor = StateFeatureExtractor(schema)
        self._models: Dict[str, Optional[ScorerModel]] = {}
        self._context_hash: Optional[str] = None
        self._source_fingerprint: Optional[str] = None
        self._writer: Optional[DecisionRecordWriter] = None
        self._events: Optional[EventLog] = None
        self._buffer: Dict[Tuple, Dict[str, Any]] = {}
        self._open_decisions: Dict[str, Tuple] = {}
        self.last_error: Optional[str] = None
        self.stats: Dict[str, int] = {
            "decisions_seen": 0, "decisions_recorded": 0,
            "shadow_disagreements": 0, "tree_score_changes": 0,
            "tree_score_fallbacks": 0, "model_errors": 0}
        if self.mode != "off":
            self._load_models()
            self._policy = runtime_policy_contract(
                self.mode,
                [name for name, slot in self.config["scorers"].items()
                 if slot.get("enabled")],
                {"selector": self.config.get("selector", {})},
                schema)
            context = execution_context(
                source_hash(str(self.config.get("source_root")
                                or self.workspace),
                            self.config.get("base_dir"),
                            self.config.get("overlay_dir")),
                self.tactical_config, self._policy, schema)
            self._context = context
            self._context_hash = execution_context_hash(context)
            self._source_fingerprint = context["source_fingerprint"]
            for name, model in self._models.items():
                if model is None:
                    continue
                from .fingerprints import validate_deployment
                validate_deployment(model.record, context)
                # Task 3.2: model source fingerprint must equal the live one.
                assert_model_source(model.record, self._source_fingerprint,
                                    allow_missing=False)

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
                # An unfitted slot is a valid pure-fallback scorer; a wrong
                # model FILE would be a hard error, an absent one is not.
                self._models[name] = None
                continue
            record = load_json(str(resolved))
            if record.get("scorer") != name:
                raise ValueError(f"model file {path} declares scorer "
                                 f"{record.get('scorer')!r}, expected {name!r}")
            self._models[name] = ScorerModel(record)

    # -- integration --------------------------------------------------------
    def wrap(self, manager: Any) -> Any:
        """Return a controller around ``manager``. In ``off`` mode the
        manager itself is returned untouched."""
        if self.mode == "off":
            return manager
        controller = _RuntimeController(manager, self)
        if self.config.get("collect_skill_proposals", True):
            controller.collector = SkillProposalCollector(
                enabled=self.config.get("skill_proposal_skills",
                                        PRIORITY_SKILLS),
                schema=self.schema).install(manager)
            self._active_collector = controller.collector
        return controller

    def scorer_model(self, name: str) -> Optional[ScorerModel]:
        return self._models.get(name)

    @property
    def execution_context_hash(self) -> Optional[str]:
        return self._context_hash

    @property
    def source_fingerprint(self) -> Optional[str]:
        return self._source_fingerprint

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
        try:
            self._events_for().log(kind, **fields)
        except Exception:
            pass                      # the event log never breaks decisions

    # -- recording ----------------------------------------------------------
    def record_decision(self, *, run_id: str, episode_id: str,
                        scene_config_id: Optional[str],
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
            run_id=run_id, episode_id=episode_id,
            scene_config_id=scene_config_id, seed=seed, map_id=map_id,
            snapshot=snapshot, candidates=candidates,
            global_features=self._last_state_features or {},
            teacher_choice=teacher_choice, model_scores=model_scores,
            selected_candidate=selected_candidate,
            final_actions=final_actions, execution_status=execution_status,
            source_hash=self._source_fingerprint or "",
            execution_context_hash=self._context_hash or "",
            schema=self.schema)
        if skill_proposals:
            # A-class section (Task 4.3): same decision key, own family and
            # candidate groups; never merged into the joint candidate list.
            record["skill_proposals"] = skill_proposals
        key = record_key(record)
        self._buffer[key] = record        # last write per decision wins
        decision_key = dumps_key(key)
        self._open_decisions[decision_key] = key
        self.stats["decisions_seen"] += 1
        return key

    def finalize_episode(self, *, outcome: str,
                         episode_return: Optional[float],
                         ally_losses: int, enemy_losses: int,
                         damage_dealt: Optional[float] = None,
                         damage_received: Optional[float] = None,
                         run_id: Optional[str] = None,
                         episode_id: Optional[str] = None,
                         baseline_return: Optional[float] = None
                         ) -> int:
        """Attach the terminal outcome to this episode's buffered records and
        flush them to the JSONL log. Returns the number of records written."""
        if self.mode == "off":
            return 0
        writer = self._writer_for()
        written = 0
        for key, record in sorted(self._buffer.items()):
            if run_id is not None and record["run_id"] != run_id:
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
                    if (run_id is None or r["run_id"] == run_id)
                    and (episode_id is None or r["episode_id"] == episode_id)]:
            del self._buffer[key]
        self.log_event("episode_finished", run_id=run_id,
                       episode_id=episode_id, outcome=outcome,
                       episode_return=episode_return,
                       baseline_return=baseline_return,
                       ally_losses=ally_losses, enemy_losses=enemy_losses,
                       damage_dealt=damage_dealt,
                       damage_received=damage_received,
                       records_written=written,
                       execution_context_hash=self._context_hash)
        return written

    # -- internals ----------------------------------------------------------
    _last_state_features: Optional[Dict[str, Any]] = None

    def observe_step(self, payload: Dict[str, Any], manager: Any,
                     actions: List[int], summary: Dict[str, Any]
                     ) -> Tuple[List[int], Dict[str, Any], Optional[Dict]]:
        """Shared shadow/tree_score path. Returns possibly-swapped actions,
        the (possibly annotated) summary, and the decision meta."""
        state = payload.get("state", {}) or {}
        snapshot = self._read_snapshot(manager)
        context = self._read_context(manager)
        self._last_state_features = self.extractor.extract_state(
            snapshot, context=context) if snapshot is not None else None
        run_id = str(self.config.get("run_id")
                     or f"run_{state.get('episode', 0)}_{state.get('game_loop', 0)}")
        episode = str(state.get("episode", 0))
        map_field = state.get("map", "unknown")
        if isinstance(map_field, dict):       # adapter maps carry {"name": …}
            map_field = map_field.get("name") or "unknown"
        scene_config_id = str(state.get("scene_config_id")
                              or f"roster:{len(state.get('allies', {}) or {})}"
                              f":{map_field}")
        seed = state.get("seed")
        seed = int(seed) if isinstance(seed, int) else None

        candidates = joint_candidates_from_summary(
            summary, snapshot, context=context, payload=payload,
            schema=self.schema) if snapshot is not None else []
        teacher = candidates[0].candidate_id if candidates else None
        rule_actions = list(actions)
        meta: Dict[str, Any] = {"mode": self.mode, "run_id": run_id}

        model = self.scorer_model("joint_scorer")
        scores = score_candidates(self._last_state_features or {}, candidates,
                                  model=model) if candidates else []
        meta["scores"] = [row.to_record() for row in scores]
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
        self.log_event("proposed", run_id=run_id, episode_id=episode,
                       snapshot_id=getattr(snapshot, "snapshot_id", None),
                       game_loop=getattr(snapshot, "game_loop", None),
                       teacher_choice=teacher, model_choice=predicted,
                       n_candidates=len(candidates),
                       execution_context_hash=self._context_hash)

        selected = teacher
        final_actions_map = {f"u{i}": action for i, action
                             in enumerate(rule_actions)}
        execution_status = "pending"
        tree_events: List[Dict[str, Any]] = []
        if self.mode == "tree_score":
            selector_config = self.config.get("selector", {})
            decision = select_skill_plan(
                self._last_state_features or {}, candidates, model=model,
                min_gain=float(selector_config.get("min_gain", 0.0)),
                tie_tolerance=float(selector_config.get("tie_tolerance", 1e-9)),
                schema=self.schema)
            meta["selection"] = {k: v for k, v in decision.items()
                                 if k != "scores"}
            selected = decision.get("selected_candidate_id") or teacher
            if decision.get("changed") and decision.get("selected_actions"):
                selected_candidate = next(
                    (c for c in candidates
                     if c.candidate_id == decision["selected_candidate_id"]),
                    None)
                replacement = apply_selected_plan(
                    selected_candidate, manager, context=context,
                    memory=getattr(manager, "memory", None))
                meta["replacement"] = {k: v for k, v in replacement.items()
                                       if k in ("applied", "reason",
                                                "replaced_plan_ids",
                                                "validator_problems",
                                                "repair_log",
                                                "commit_status")}
                if replacement.get("applied"):
                    swapped = self._actions_list(replacement["final_actions"],
                                                 len(rule_actions))
                    if swapped is not None:
                        self._update_manager_plan_state(
                            manager, replacement["final_plans"],
                            swapped, summary)
                        rule_actions = swapped
                        self.stats["tree_score_changes"] += 1
                        execution_status = "pending"
                        final_actions_map = dict(replacement["final_actions"])
                        self.log_event(
                            "selected", run_id=run_id, episode_id=episode,
                            selected=selected, replaced=True,
                            replaced_plan_ids=replacement["replaced_plan_ids"],
                            validator_problems=[],
                            rule_actions_hash=actions_hash(
                                summary.get("actions", {})),
                            final_actions_hash=actions_hash(swapped))
                    else:
                        self.stats["tree_score_fallbacks"] += 1
                        self.log_event("rejected", run_id=run_id,
                                       episode_id=episode,
                                       reason="action_list_incomplete",
                                       fallback="base")
                else:
                    # Task 5.2: replacement failed -> complete BASE fallback.
                    self.stats["tree_score_fallbacks"] += 1
                    self.log_event("rejected", run_id=run_id,
                                   episode_id=episode,
                                   reason=replacement.get("reason"),
                                   fallback="base")
        self.record_decision(
            run_id=run_id, episode_id=episode,
            scene_config_id=scene_config_id, seed=seed, map_id=str(map_field),
            snapshot=snapshot, candidates=candidates,
            teacher_choice=teacher, model_scores=meta["scores"],
            selected_candidate=selected, final_actions=final_actions_map,
            execution_status=execution_status,
            skill_proposals=self._skill_proposals(manager, snapshot))
        return rule_actions, summary, meta

    def _skill_proposals(self, manager: Any, snapshot) -> Optional[Dict]:
        collector = getattr(self, "_active_collector", None) or \
            getattr(manager, "_zerg_collector", None)
        if collector is None or snapshot is None:
            return None
        try:
            return collector.records_for(int(snapshot.snapshot_id),
                                         int(snapshot.game_loop))
        except Exception:
            return None

    @staticmethod
    def _update_manager_plan_state(manager: Any, plans, actions_list,
                                   summary: Dict[str, Any]) -> None:
        """Task 5.2 (6/7): refresh the manager's final plan state so the
        ORIGINAL commit logic sees the replaced plan set. Memory commit
        conditions themselves are untouched."""
        pending = getattr(manager, "_pending_commit", None)
        if pending:
            snapshot, _, micro_actions, risk_now, context = pending
            manager._pending_commit = (snapshot, list(plans), micro_actions,
                                       risk_now, context)
        cache = getattr(manager, "_cache", None)
        if cache:
            manager._cache = (cache[0], list(actions_list), cache[2],
                              list(plans), cache[4])

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


class _RuntimeController:
    """Wraps a SkillManager without touching its class: step -> optional
    shadow/tree_score processing; commit_final_actions passes straight
    through so the ORIGINAL memory-commit conditions keep control."""

    def __init__(self, manager: Any, runtime: ZergTreeRuntime):
        self.manager = manager
        self.runtime = runtime
        self.collector: Optional[SkillProposalCollector] = None
        self.last_meta: Optional[Dict[str, Any]] = None
        self._pending_actions: Optional[List[int]] = None

    def __getattr__(self, name):          # delegate everything else
        return getattr(self.manager, name)

    def reset(self) -> None:
        self.manager.reset()

    def step(self, payload: Dict[str, Any], n_agents: int):
        actions, summary = self.manager.step(payload, n_agents)
        runtime = self.runtime
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
            runtime.log_event("submitted", run_id=meta.get("run_id"),
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
            runtime.log_event("submitted", run_id=meta.get("run_id"),
                              rule_actions_hash=actions_hash(actions),
                              final_actions_hash=actions_hash(actions),
                              replaced_units=[], reason=None)
            return actions, annotated
        return actions, summary

    def commit_final_actions(self, final_actions: List[int]) -> None:
        # Straight delegation: TacticalMemory commits remain governed by the
        # manager's own surviving-plan checks against the FINAL action list.
        self.manager.commit_final_actions(final_actions)
        runtime = self.runtime
        if runtime.mode != "off":
            runtime.log_event(
                "committed",
                run_id=self.last_meta.get("run_id") if self.last_meta else None,
                rule_actions_hash=actions_hash(
                    self._pending_actions or final_actions),
                final_actions_hash=actions_hash(final_actions),
                note=("original surviving-plan memory commit passed; NOT an "
                      "SC2 engine acknowledgement"))
