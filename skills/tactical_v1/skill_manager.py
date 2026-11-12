"""tactical_v1 manager facade (runner contract unchanged).

step(payload, n_agents) -> (actions, telemetry); commit_final_actions(actions)
commits memory ONLY for plans whose final actions survived untouched. The
step cache is keyed by the decision-relevant content (state + legal menus +
n_agents + config); repeated identical calls return safe copies and never
double-count observation history.
"""
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve()
_WORKSPACE = _HERE.parents[2]
if str(_WORKSPACE) not in sys.path:  # reuse of skills/current pure helpers
    sys.path.insert(0, str(_WORKSPACE))

from . import config as config_module
from .context import build_snapshot, Context
from .coordinator import Coordinator
from .evaluator import Evaluator
from .memory import TacticalMemory
from .registry import build_skills, GENERIC, SPECIALIZED
from .telemetry import Telemetry


def _digest(payload: Dict, n_agents: int, fingerprint: str) -> str:
    material = json.dumps(
        {"state": payload.get("state", {}),
         "questions": payload.get("questions", {}),
         "n_agents": n_agents, "fingerprint": fingerprint},
        sort_keys=True, default=str)
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def config_fingerprint(config: Dict[str, Any]) -> str:
    """Whole package source (recursive, tests excluded) + the actually
    reused shared modules + the effective configuration."""
    parts = []
    for path in sorted(_HERE.parent.rglob("*.py")):
        if "tests" in path.parts or "__pycache__" in path.parts:
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
        # Explicit compatibility surface: unknown kwargs are recorded, never
        # silently behaviour-changing. The arg resolver rejects the legacy
        # flag surface for tactical_v1 before we ever get here.
        self.compatibility_notes = {k: v for k, v in compat.items()}
        self.config = config_module.load(tactical_config)
        self.config["_fingerprint"] = config_fingerprint(self.config)
        self.posture = posture
        self.skills = build_skills(self.config)
        enabled = self.config.get("enabled", {})
        # real mechanism switches reach the FINAL effect/scoring path (B3)
        self.evaluator = Evaluator(
            self.config.get("weights"),
            mechanisms={"splash": bool(enabled.get("splash_fire", True)),
                        "burst": bool(enabled.get("burst_engage", True))},
            splash_radius=self.config.get("splash_fire", {}).get(
                "coverage_radius", 3.0),
            splash_weight=self.config.get("splash_fire", {}).get(
                "coverage_weight", 0.5),
            burst_radius=self.config.get("burst_engage", {}).get(
                "proxy_radius", 2.5),
            burst_weight=0.5,
            burst_self_cost=self.config.get("burst_engage", {}).get(
                "self_cost", 1.0))
        self.coordinator = Coordinator(self.config, self.evaluator)
        lifecycle = self.config.get("lifecycle", {})
        self.memory = TacticalMemory(
            observation_ttl_loops=int(lifecycle.get("observation_ttl_loops", 240)),
            search_revisit_cooldown_loops=int(
                lifecycle.get("search_revisit_cooldown_loops", 160)))
        self.telemetry = Telemetry(top_k_candidates=int(
            self.config.get("telemetry", {}).get("top_k_candidates", 3)))
        self._snapshot_counter = 0
        self._cache = None           # (digest, actions, summary, plans, snapshot)
        self._pending_commit = None
        self.stats = {"steps": 0, "episodes": 0, "search_budget_events": 0,
                      "repairs": 0, "baseline_units": 0}
        self.current_skill = "tactical_v1"
        self.reset()  # construction resets like the current manager does

    # -- runner interface ---------------------------------------------------
    @property
    def active(self) -> bool:
        return True  # the joint controller is always the acting policy

    def reset(self) -> None:
        self.memory = TacticalMemory(
            observation_ttl_loops=self.memory.observation_ttl_loops,
            search_revisit_cooldown_loops=self.memory.search_revisit_cooldown_loops)
        self.telemetry = Telemetry(top_k_candidates=self.telemetry.top_k)
        self._snapshot_counter = 0
        self._cache = None
        self._pending_commit = None

    def select(self, skill: Optional[str] = None, priority_target=None,
               posture: Optional[str] = None) -> None:
        """Accepted for interface compatibility. The joint coordinator never
        hands exclusive control to one skill; posture is recorded."""
        if posture:
            self.posture = posture

    def step(self, payload: Dict, n_agents: int):
        digest = _digest(payload, n_agents, self.config["_fingerprint"])
        cached = self._cache
        if cached is not None and cached[0] == digest:
            # Same decision content: idempotent re-read, safe copies, no
            # observation-history appends, no budget advance.
            return list(cached[1]), copy.deepcopy(cached[2])
        snapshot = build_snapshot(payload, self._snapshot_counter + 1)
        self._snapshot_counter += 1
        self.memory.begin_step(snapshot)
        context = Context(snapshot)
        risk_now = {a: len(context.risk_at(a, snapshot.allies[a].position).threats)
                    for a in snapshot.allies if snapshot.allies[a].alive}
        # observation phase FIRST: compare with the last committed record,
        # then update confirmed facts (arrivals, timeouts, invalidations).
        # Only after this may read-only candidates be generated.
        self.memory.retire_invalid(snapshot)
        self.memory.observe(snapshot, risk_now)
        for event in self.memory.update_task_facts(snapshot, context):
            self.telemetry.log(event)
        actions, summary, plans = self.coordinator.decide(
            context, self.skills, self.memory, self.telemetry, n_agents)
        self.stats["steps"] += 1
        self.stats["search_budget_events"] = self.telemetry.search_budget_events
        self._cache = (digest, list(actions), copy.deepcopy(summary), plans,
                       snapshot)
        self._pending_commit = (snapshot, plans, list(actions), risk_now)
        return list(actions), copy.deepcopy(summary)

    def commit_final_actions(self, final_actions: List[int]) -> None:
        """Commit memory for exactly the plans whose actions survived."""
        pending = self._pending_commit
        self._pending_commit = None
        if pending is None:
            return
        snapshot, plans, actions, risk_now = pending
        actions_by_actor = {f"u{i}": a for i, a in enumerate(final_actions)}
        self.memory.commit(snapshot, actions_by_actor, plans, risk_now,
                           context=Context(snapshot))
