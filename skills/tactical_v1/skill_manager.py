"""tactical_v1 manager facade.

Implements the exact interface the adapter runner calls:
    __init__(**manager_kwargs), reset(), select(skill, posture=...),
    step(payload, n_agents) -> (actions, telemetry),
    commit_final_actions(actions), stats, active.

One submission per unit per environment step; a repeated step() at the same
game loop returns the cached result without consuming budget or advancing
any instance.
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve()
_WORKSPACE = _HERE.parents[2]
import sys
if str(_WORKSPACE) not in sys.path:  # reuse of skills/current pure helpers
    sys.path.insert(0, str(_WORKSPACE))

from .context import build_snapshot, Context, DEAD_NOOP
from .coordinator import Coordinator
from .evaluator import Evaluator
from .memory import TacticalMemory
from .registry import build_skills, GENERIC, SPECIALIZED
from .telemetry import Telemetry

DEFAULTS_PATH = _HERE.parent / "defaults.json"


def _state_digest(payload: Dict) -> str:
    import json as _json
    return hashlib.sha256(_json.dumps(
        payload.get("state", {}), sort_keys=True, default=str).encode()).hexdigest()[:16]


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    with open(DEFAULTS_PATH, encoding="utf-8") as handle:
        config = json.load(handle)
    if path:
        override = json.loads(Path(path).read_text(encoding="utf-8"))
        _deep_merge(config, override)
    return config


def _deep_merge(base: Dict, override: Dict) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def config_fingerprint(config: Dict[str, Any]) -> str:
    """Covers the whole package source (recursive), the shared reused code
    and the effective configuration."""
    parts = []
    for path in sorted((_HERE.parent).rglob("*.py")):
        if "tests" in path.parts:
            continue
        parts.append(path.read_bytes())
    shared = _WORKSPACE / "skills/current/skills"
    for name in ("damage_model.py", "combat_timing.py", "group_geometry.py"):
        target = shared / name
        if target.is_file():
            parts.append(target.read_bytes())
    parts.append(json.dumps(config, sort_keys=True).encode())
    return hashlib.sha256(b"".join(parts)).hexdigest()[:16]


class SkillManager:
    def __init__(self, max_steps: int = 4, posture: str = "balanced",
                 tactical_config: Optional[str] = None, **compat):
        # Explicit compatibility surface: unknown kwargs are NOT silently
        # behaviour-changing; they are recorded. The runner never passes the
        # ones below to tactical_v1 today (they are rejected at the arg
        # resolver); programmatic callers may pass None-valued legacy fields.
        self.compatibility_notes = {k: v for k, v in compat.items()}
        self.config = load_config(tactical_config)
        self.config["_fingerprint"] = config_fingerprint(self.config)
        self.posture = posture
        self.skills = build_skills(self.config)
        self.evaluator = Evaluator(self.config.get("weights"))
        self.coordinator = Coordinator(self.config, self.evaluator)
        self.memory = TacticalMemory(
            observation_ttl_loops=int(
                self.config.get("memory", {}).get("observation_ttl_loops", 240)))
        self.telemetry = Telemetry(
            top_k_candidates=int(self.config.get("telemetry", {}).get(
                "top_k_candidates", 3)))
        self._snapshot_counter = 0
        self._cache = None           # (game_loop, episode, actions, summary, plans)
        self._pending_commit = None
        self.stats = {"steps": 0, "episodes": 0, "search_budget_events": 0,
                      "repairs": 0, "fallback_units": 0}
        self.current_skill = "tactical_v1"
        self.reset()  # construction resets like the current manager does

    # -- runner interface ---------------------------------------------------
    @property
    def active(self) -> bool:
        return True  # the joint controller is always the acting policy

    def reset(self) -> None:
        self.memory.reset()
        self.telemetry = Telemetry(top_k_candidates=self.telemetry.top_k)
        self._snapshot_counter = 0
        self._cache = None
        self._pending_commit = None

    def select(self, skill: Optional[str] = None, priority_target=None,
               posture: Optional[str] = None) -> None:
        """Accepted for interface compatibility. The joint coordinator does
        not hand control to one exclusive skill; posture is recorded."""
        if posture:
            self.posture = posture

    def step(self, payload: Dict, n_agents: int):
        snapshot = build_snapshot(payload, self._snapshot_counter + 1)
        cached = self._cache
        if cached is not None and cached[0] == snapshot.game_loop \
                and cached[1] == snapshot.episode \
                and cached[4] == _state_digest(payload):
            # Same clock AND same observed state: idempotent re-read, no
            # budget or phase advance.
            return cached[2], dict(cached[3])
        self._snapshot_counter += 1
        fresh = self.memory.begin_step(snapshot)
        context = Context(snapshot)
        self.memory.retire_invalid(snapshot)
        self.memory.observe(snapshot)
        actions, summary, plans = self.coordinator.decide(
            context, self.skills, self.memory, self.telemetry, n_agents)
        self.stats["steps"] += 1
        self.stats["search_budget_events"] = self.telemetry.search_budget_events
        self._cache = (snapshot.game_loop, snapshot.episode, actions, summary,
                       _state_digest(payload), plans, context)
        self._pending_commit = (snapshot, plans, actions)
        return actions, summary

    def commit_final_actions(self, final_actions: List[int]) -> None:
        """The runner validated the submitted list; commit memory deltas NOW,
        once, for the executed plan only."""
        pending = self._pending_commit
        self._pending_commit = None
        if pending is None:
            return
        snapshot, plans, actions = pending
        actions_by_actor = {f"u{i}": a for i, a in enumerate(final_actions)}
        deltas: List[Dict[str, Any]] = []
        for plan in plans:
            delta = dict(plan.proposed_memory_delta)
            if delta:
                deltas.append(delta)
            actors = list(plan.controlled_actors)
            if plan.instance_id:
                deltas.append({"kind": "advance", "instance_id": plan.instance_id,
                               "phase": plan.commitment.phase,
                               "progress": True, "goal": plan.goal})
            elif plan.commitment.max_steps is not None or plan.commitment.min_steps:
                deltas.append({"kind": "create",
                               "skill": plan.primary_skill,
                               "actors": actors,
                               "goal": plan.goal,
                               "phase": plan.commitment.phase,
                               "max_steps": plan.commitment.max_steps,
                               "min_steps": plan.commitment.min_steps,
                               "extra": {"provider": plan.specialization_provider}})
        # Commit-time instance retirement: actors whose selected skill changed
        # away from a held instance release it (e.g. disengage -> re-enter).
        selected_skill = {p.primary_skill: None for p in plans}
        for plan in plans:
            for actor in plan.controlled_actors:
                selected_skill[actor] = plan.primary_skill
        for instance in list(self.memory.instances.values()):
            for actor in instance.actors:
                chosen = selected_skill.get(actor)
                if chosen is not None and chosen != instance.skill:
                    deltas.append({"kind": "drop",
                                   "instance_id": instance.instance_id})
                    break
        self.memory.commit(snapshot, actions_by_actor, deltas)
