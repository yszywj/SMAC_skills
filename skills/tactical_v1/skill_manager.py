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
from .micro_approach import extra_approaches, merge_approaches, record_approach
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
    for path in sorted(p for p in _HERE.parent.rglob("*")
                       if p.suffix in (".py", ".json")):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        parts.append(path.read_bytes())
    shared = _WORKSPACE / "skills/current/skills"
    for name in ("damage_model.py", "combat_timing.py", "group_geometry.py"):
        target = shared / name
        if target.is_file():
            parts.append(target.read_bytes())
    # The reused legacy object-value evaluator changes behaviour with us.
    legacy_values = _WORKSPACE / "skills/current/unit_value.py"
    if legacy_values.is_file():
        parts.append(legacy_values.read_bytes())
    parts.append(json.dumps(config, sort_keys=True).encode())
    return hashlib.sha256(b"".join(parts)).hexdigest()[:16]


class SkillManager:
    micro_all_attackers = False  # Opt-in runner mode; reset preserves it.

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
                "self_cost", 1.0),
            forecast_config=self.config.get("forecasting"),
            urgency_value=self.config.get("focus_fire", {}).get(
                "urgency_value", 0.5),
            target_planning_enabled=self.config.get("focus_fire", {}).get(
                "target_planning_enabled", True),
            common_combat=self.config.get("common_combat"),
            splash_reliability=self.config.get("splash_fire", {}).get(
                "coverage_reliability", 0.5))
        self.coordinator = Coordinator(self.config, self.evaluator)
        lifecycle = self.config.get("lifecycle", {})
        self.memory = TacticalMemory(
            observation_ttl_loops=int(lifecycle.get("observation_ttl_loops", 240)),
            search_revisit_cooldown_loops=int(
                lifecycle.get("search_revisit_cooldown_loops", 160)),
            retry_cooldown_h=lifecycle.get("retry_cooldown_h", 2.0),
            no_progress_window_h=lifecycle.get("no_progress_window_h", 3.0),
            encounter_config=self.config)
        self.telemetry = Telemetry(top_k_candidates=int(
            self.config.get("telemetry", {}).get("top_k_candidates", 3)))
        self._snapshot_counter = 0
        self._cache = None           # (digest, actions, summary, plans, snapshot)
        self._pending_commit = None
        self.last_micro_intents = []
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
            search_revisit_cooldown_loops=self.memory.search_revisit_cooldown_loops,
            retry_ttl_loops=self.memory.retry_ttl_loops,
            retry_cooldown_h=self.memory.retry_cooldown_h,
            no_progress_window_h=self.memory.no_progress_window_h,
            encounter_config=self.config)
        self.telemetry = Telemetry(top_k_candidates=self.telemetry.top_k)
        self._snapshot_counter = 0
        self._cache = None
        self._pending_commit = None
        self.last_micro_intents = []

    def select(self, skill: Optional[str] = None, priority_target=None,
               posture: Optional[str] = None) -> None:
        """Accepted for interface compatibility. The joint coordinator never
        hands exclusive control to one skill; posture is recorded."""
        if posture:
            self.posture = posture

    def step(self, payload: Dict, n_agents: int):
        self.last_micro_intents = []  # No authority until this decision is committed.
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
        context.deployment_memory = self.memory
        # Shared per-snapshot object-value configuration (unit_value_view);
        # queries read it lazily and never mutate tasks.
        context.unit_value_parameters = dict(self.config.get("unit_value", {})) \
            or {"enabled": False}
        context.intercept_config = dict(self.config.get("intercept", {}))
        context.common_config = dict(self.config.get("common_combat", {}))
        context.burst_config = dict(self.config.get("burst_engage", {}))
        risk_now = {a: len(context.risk_at(a, snapshot.allies[a].position).threats)
                    for a in snapshot.allies if snapshot.allies[a].alive}
        # observation phase FIRST (single ordered entry, 3.1): record the
        # observation, then judge identity invalid -> progress -> completion
        # -> deadline. No earlier retire call may delete timed-out tasks.
        self.memory.observe(snapshot, risk_now)
        for event in self.memory.update_task_facts(snapshot, context):
            self.telemetry.log(event)
        previous_budget_events = self.telemetry.search_budget_events
        actions, summary, plans = self.coordinator.decide(
            context, self.skills, self.memory, self.telemetry, n_agents)
        summary["battle_intent"] = self.memory.battle_state(context)
        provider = self.skills.get("splash_fire")
        if provider is not None and hasattr(provider, "coverage_model"):
            summary["splash_model"] = provider.coverage_model()
        self.stats["steps"] += 1
        self.stats["episodes"] += self._snapshot_counter == 1
        # Telemetry resets per episode; run statistics accumulate deltas.
        self.stats["search_budget_events"] += (
            self.telemetry.search_budget_events - previous_budget_events)
        self.stats["repairs"] += len(summary.get("repairs", ()))
        self.stats["baseline_units"] += sum(
            row["skill"] == "baseline" and snapshot.allies[actor].alive
            for actor, row in summary.get("units", {}).items())
        self._cache = (digest, list(actions), copy.deepcopy(summary), plans,
                       snapshot)
        # A rescue can share the exact cardinal move of an approach. Do not
        # interrupt that route, even if the attack plan's action also matches.
        duties = summary['execution_duties']['actors']
        micro_actions = {f'u{i}': a for i, a in enumerate(actions)
                         if duties.get(f'u{i}', {}).get('kind') not in ('rescue', 'intercept')}
        self._pending_commit = (snapshot, plans, micro_actions, risk_now, context)
        if self.micro_all_attackers is True:
            context.micro_approaches = extra_approaches(context, summary, plans, actions)
        return list(actions), copy.deepcopy(summary)

    def record_micro_action(self, actor: str, action: int) -> None:
        """Record an acknowledged interior command without advancing plans or timers."""
        record_approach(self.memory, self.last_micro_intents, actor, action)

    def commit_final_actions(self, final_actions: List[int]) -> None:
        """Commit memory for exactly the plans whose actions survived."""
        pending = self._pending_commit
        self._pending_commit = None
        self.last_micro_intents = []
        if pending is None:
            return
        snapshot, plans, micro_actions, risk_now, context = pending
        actions_by_actor = {f"u{i}": a for i, a in enumerate(final_actions)}
        # Execution-layer micro-trigger authorizations are derived here, from
        # the SAME post-arbitration final actions the memory commit verifies.
        # Both micro modes read the identical list; sending is decided later.
        from .zerg_hydra_policy import approach_then_attack_intents
        surviving = {actor: final for actor, final in actions_by_actor.items()
                     if micro_actions.get(actor) == final}
        intents = approach_then_attack_intents(snapshot, plans, surviving)
        if self.micro_all_attackers is True and final_actions == self._cache[1]:
            intents = merge_approaches(intents, getattr(context, 'micro_approaches', ()))
        self.memory.commit(snapshot, actions_by_actor, plans, risk_now,
                           context=context)
        self.last_micro_intents = intents
