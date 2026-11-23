"""Runtime integration: off / shadow / tree_score.

Safety contract (enforced structurally here):

- ``off``: the runtime is inert. ``wrap(manager)`` returns the manager
  itself; no models are created, nothing is recorded, behaviour is
  bit-identical to the original skill.
- ``shadow``: candidates from the manager's own decision are scored and the
  disagreement with the rule choice is recorded to a separate zerg JSONL
  log. Actions, memory and the tactical Telemetry are never touched.
- ``tree_score``: the selector may return a DIFFERENT complete legal plan
  (one the Coordinator itself produced and recorded as an alternative).
  Structural legality is re-checked before the swap; after the swap the
  runner still calls ``commit_final_actions(final_actions)``, and the
  manager's own commit logic decides which plans survive — the decision
  tree never submits TacticalMemory and never relaxes the original commit
  conditions.

Decision records are appended only once per decision (idempotent against
repeated observations of the same snapshot), and episode outcomes are
attached when the caller finalises the episode.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .candidates import joint_candidates_from_summary
from .dataset import DecisionRecordWriter, build_decision_record, \
    finalise_outcome, record_key
from .features import DEFAULT_SCHEMA, ZergFeatureSchema, StateFeatureExtractor
from .fingerprints import execution_context, execution_context_hash, \
    runtime_policy_contract, source_hash
from .schema import load_json
from .scorer import ScorerModel, score_candidates
from .selector import select_skill_plan

MODES = ("off", "shadow", "tree_score")
RUNTIME_CONFIG_KIND = "zerg_tree_runtime_config"
DEFAULT_CONFIG_PATH = "decision_tree/zerg/configs/runtime.json"
DEFAULT_RECORDS_PATH = "decision_tree/zerg/artifacts/records/shadow.jsonl"
DEFAULT_SCORER_SLOTS = ("zergling_scorer", "hydralisk_scorer",
                        "baneling_scorer", "joint_scorer")


def default_config(workspace: str = ".") -> Dict[str, Any]:
    return {
        "kind": RUNTIME_CONFIG_KIND,
        "mode": "off",
        "workspace": workspace,
        "source_root": workspace or ".",
        "base_dir": "skills/tactical_v1",
        "overlay_dir": ("runs/contact_all_attack_20261122_230107_662982/"
                        "marginal_near/skills/tactical_v1"),
        "records_path": DEFAULT_RECORDS_PATH,
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
    scorers = config.get("scorers", {})
    for name in scorers:
        if name not in DEFAULT_SCORER_SLOTS:
            raise ValueError(f"unknown scorer slot {name}")
    return config


def runtime_policy_for_scorers(mode: str, scorers) -> Dict[str, Any]:
    """Policy contract used when exporting a model outside a live runtime."""
    return runtime_policy_contract(mode, scorers, {}, DEFAULT_SCHEMA)


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
        self._writer: Optional[DecisionRecordWriter] = None
        self._buffer: Dict[Tuple, Dict[str, Any]] = {}
        self.last_error: Optional[str] = None
        self.stats: Dict[str, int] = {
            "decisions_seen": 0, "decisions_recorded": 0,
            "shadow_disagreements": 0, "tree_score_changes": 0,
            "model_errors": 0}
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
            self._source = context["source_fingerprint"]
            for name, model in self._models.items():
                if model is None:
                    continue
                from .fingerprints import validate_deployment
                validate_deployment(model.record, context)

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
        return _RuntimeController(manager, self)

    def scorer_model(self, name: str) -> Optional[ScorerModel]:
        return self._models.get(name)

    @property
    def execution_context_hash(self) -> Optional[str]:
        return self._context_hash

    # -- recording ----------------------------------------------------------
    def _writer_for(self) -> DecisionRecordWriter:
        if self._writer is None:
            path = self.config.get("records_path") or DEFAULT_RECORDS_PATH
            resolved = Path(path)
            if not resolved.is_absolute():
                resolved = self.workspace / path
            self._writer = DecisionRecordWriter(str(resolved))
        return self._writer

    def record_decision(self, *, episode_id: str, scene_id: str, snapshot,
                        summary: Dict[str, Any], candidates,
                        teacher_choice: Optional[str],
                        model_scores: List[Dict[str, Any]],
                        selected_candidate: Optional[str],
                        final_actions: Dict[str, int],
                        execution_status: str) -> Optional[Tuple]:
        record = build_decision_record(
            episode_id=episode_id, scene_id=scene_id, snapshot=snapshot,
            summary=summary, candidates=candidates,
            global_features=self._last_state_features or {},
            teacher_choice=teacher_choice, model_scores=model_scores,
            selected_candidate=selected_candidate,
            final_actions=final_actions, execution_status=execution_status,
            source_hash=self._source or "",
            execution_context_hash=self._context_hash or "",
            schema=self.schema)
        key = record_key(record)
        self._buffer[key] = record        # last write per decision wins;
        self.stats["decisions_seen"] += 1
        return key

    def finalize_episode(self, *, outcome: str, episode_return: Optional[float],
                         ally_losses: int, enemy_losses: int,
                         damage_dealt: Optional[float] = None,
                         damage_received: Optional[float] = None,
                         episode_id: Optional[str] = None) -> int:
        """Attach the terminal outcome to this episode's buffered records and
        flush them to the JSONL log. Returns the number of records written."""
        if self.mode == "off":
            return 0
        writer = self._writer_for()
        written = 0
        for key, record in sorted(self._buffer.items()):
            if episode_id is not None and record["episode_id"] != episode_id:
                continue
            finalise_outcome(record, outcome=outcome,
                             episode_return=episode_return,
                             ally_losses=ally_losses,
                             enemy_losses=enemy_losses,
                             damage_dealt=damage_dealt,
                             damage_received=damage_received)
            if writer.write(record):
                written += 1
                self.stats["decisions_recorded"] += 1
        for key in [k for k, r in self._buffer.items()
                    if episode_id is None or r["episode_id"] == episode_id]:
            del self._buffer[key]
        return written

    # -- internals ----------------------------------------------------------
    _last_state_features: Optional[Dict[str, Any]] = None

    def observe_step(self, payload: Dict[str, Any], manager: Any,
                     actions: List[int], summary: Dict[str, Any]
                     ) -> Tuple[List[int], Dict[str, Any], Optional[Dict]]:
        """Shared shadow/tree_score path. Returns possibly-swapped actions,
        the (possibly annotated) summary, and the decision record meta."""
        snapshot = self._read_snapshot(manager)
        context = self._read_context(manager)
        self._last_state_features = self.extractor.extract_state(
            snapshot, context=context) if snapshot is not None else None
        episode = str((payload.get("state", {}) or {}).get("episode", 0))
        map_field = (payload.get("state", {}) or {}).get("map", "unknown")
        if isinstance(map_field, dict):       # adapter maps carry {"name": …}
            map_field = map_field.get("name") or "unknown"
        scene_id = str(map_field)
        candidates = joint_candidates_from_summary(
            summary, snapshot, context=context, payload=payload,
            schema=self.schema) if snapshot is not None else []
        teacher = candidates[0].candidate_id if candidates else None
        rule_actions = list(actions)
        meta: Dict[str, Any] = {"mode": self.mode}

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

        selected = teacher
        final_actions = {f"u{i}": action for i, action
                         in enumerate(rule_actions)}
        execution_status = "pending"
        if self.mode == "tree_score":
            selector_config = self.config.get("selector", {})
            decision = select_skill_plan(
                self._last_state_features or {}, candidates, model=model,
                min_gain=float(selector_config.get("min_gain", 0.0)),
                tie_tolerance=float(selector_config.get("tie_tolerance", 1e-9)),
                schema=self.schema)
            meta["selection"] = {k: v for k, v in decision.items()
                                 if k != "scores"}
            selected = decision["selected_candidate_id"]
            if decision.get("changed") and decision.get("selected_actions"):
                swapped = self._actions_list(decision["selected_actions"],
                                             len(rule_actions))
                if swapped is not None:
                    rule_actions = swapped
                    execution_status = "pending"
                    self.stats["tree_score_changes"] += 1
                    final_actions = dict(decision["selected_actions"])
        self.record_decision(
            episode_id=episode, scene_id=scene_id, snapshot=snapshot,
            summary=summary, candidates=candidates, teacher_choice=teacher,
            model_scores=meta["scores"], selected_candidate=selected,
            final_actions=final_actions, execution_status=execution_status)
        return rule_actions, summary, meta

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
        self.last_meta: Optional[Dict[str, Any]] = None

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
        if runtime.mode == "tree_score" and processed != list(actions):
            summary = copy.deepcopy(summary)
            summary["zerg_tree"] = {
                "mode": runtime.mode,
                "changed": True,
                "rule_actions": [int(a) for a in actions],
                "tree_actions": [int(a) for a in processed],
                "execution_context_hash": runtime.execution_context_hash,
            }
            return processed, summary
        if runtime.mode == "shadow":
            summary = copy.deepcopy(summary)
            summary["zerg_tree"] = {"mode": runtime.mode,
                                    "changed": False}
            return actions, summary
        return actions, summary

    def commit_final_actions(self, final_actions: List[int]) -> None:
        # Straight delegation: TacticalMemory commits remain governed by the
        # manager's own surviving-plan checks against the FINAL action list.
        self.manager.commit_final_actions(final_actions)
