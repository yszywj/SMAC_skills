"""Adaptive joint coordinator (P4).

Fixed pipeline: observe -> generate candidates (continue / generic / special /
fallback) -> unified evaluation -> beam-style joint selection over plans
(single-unit AND atomic multi-unit candidates) -> bounded validation and
repair -> ONE submission per unit -> commit memory. Never: old-Adaptive
decides everything first, specializations overwrite afterwards.
"""
import time
from typing import Dict, List, Optional, Tuple

from .context import Context, DEAD_NOOP, STOP
from .contracts import CandidatePlan, Snapshot
from .evaluator import Evaluator
from .ledger import Ledger
from .memory import TacticalMemory
from .telemetry import Telemetry
from .validator import fallback_action, validate_actions, validate_joint


class _Branch:
    """One beam state: accepted plans + their ledger."""
    __slots__ = ("plans", "ledger", "score")

    def __init__(self, plans: List[CandidatePlan], ledger: Ledger, score: float):
        self.plans = plans
        self.ledger = ledger
        self.score = score


class Coordinator:
    def __init__(self, config: Dict, evaluator: Optional[Evaluator] = None):
        self.config = config
        search = config.get("search", {})
        self.beam_width = int(search.get("beam_width", 16))
        self.max_expansions = int(search.get("max_expansions", 400))
        self.repair_rounds = int(search.get("repair_rounds", 2))
        self.evaluator = evaluator or Evaluator(config.get("weights"))

    def decide(self, context: Context, skills: Dict[str, object],
               memory: TacticalMemory, telemetry: Telemetry,
               n_agents: int) -> Tuple[List[int], Dict[str, object], List[dict]]:
        started = time.perf_counter()
        snapshot = context.snapshot
        # 1. Candidate generation (shared, read-only).
        seed_ledger = Ledger()
        candidates: List[CandidatePlan] = []
        for name in sorted(skills):
            skill = skills[name]
            if not getattr(skill, "enabled", True):
                continue
            plans = skill.propose(context, memory, telemetry, seed_ledger)
            candidates.extend(p for p in plans if p.snapshot_id == snapshot.snapshot_id)
        # Deterministic global order: high standalone score first, stable ids.
        scored = [(self.evaluator.score_plan(p, context, seed_ledger), p)
                  for p in candidates]
        scored.sort(key=lambda item: (-item[0], item[1].proposal_id))
        # 2. Bounded joint search: accept/reject beam over plan order.
        beam = [_Branch([], Ledger(), 0.0)]
        expansions = 0
        for score, plan in scored:
            if expansions >= self.max_expansions:
                telemetry.search_budget_events += 1
                break
            next_beam: List[_Branch] = []
            for branch in beam:
                expansions += 1
                if branch.ledger.conflicts(plan):
                    next_beam.append(branch)  # reject: keep branch unchanged
                    continue
                kept = branch.score
                added = self.evaluator.score_plan(plan, context, branch.ledger)
                accepted = _Branch([*branch.plans, plan],
                                   _clone_ledger(branch.ledger), kept + added)
                accepted.ledger.reserve(plan)
                next_beam.append(accepted)
                next_beam.append(branch)  # also keep the rejection branch
            next_beam.sort(key=lambda b: (-b.score, tuple(
                p.proposal_id for p in b.plans)))
            beam = next_beam[:self.beam_width]
        best = beam[0]
        # 3. Complete the plan: every uncontrolled unit gets the deterministic
        #    legal fallback. Only complete joint plans are ever submitted.
        plans = list(best.plans)
        repairs: List[str] = []
        actions, per_plan_actors = _actions_from(plans, context, n_agents)
        for i in range(n_agents):
            actor = f"u{i}"
            if actor in per_plan_actors:
                continue
            action = fallback_action(actor, context)
            unit = snapshot.allies.get(actor)
            if unit is None or not unit.alive:
                action = DEAD_NOOP
            actions[i] = action
            repairs.append(f"fallback:{actor}:{action}")
        # 4. Validate + bounded repair.
        for round_index in range(self.repair_rounds + 1):
            problems = validate_actions(actions, context, n_agents)
            if not problems and not validate_joint(plans, context):
                break
            actions = _repair(actions, plans, context, n_agents,
                              per_plan_actors)
        # 5. Telemetry summary (chosen vs alternates).
        chosen_ids = {p.proposal_id for p in plans}
        alternates = [{"id": p.proposal_id, "skill": p.primary_skill,
                       "score": round(s, 4), "rationale": p.rationale}
                      for s, p in scored if p.proposal_id not in chosen_ids]
        per_unit = {}
        for plan in plans:
            for actor in plan.controlled_actors:
                per_unit[actor] = {
                    "skill": plan.primary_skill,
                    "provider": plan.specialization_provider,
                    "goal": plan.goal,
                    "action": actions[int(actor[1:])],
                    "plan": plan.proposal_id}
        for i in range(n_agents):
            actor = f"u{i}"
            if actor not in per_unit:
                per_unit[actor] = {"skill": "fallback", "provider": None,
                                   "goal": "", "action": actions[i], "plan": None}
            telemetry.count(per_unit[actor]["skill"], "selected")
            if per_unit[actor]["provider"]:
                telemetry.specialization_stats.setdefault(
                    per_unit[actor]["provider"], {}).setdefault(
                    "plans_selected", 0)
                telemetry.specialization_stats[
                    per_unit[actor]["provider"]]["plans_selected"] += 1
        for name, cap in context.capabilities.items():
            if cap.specialization:
                telemetry.specialization_stats[cap.specialization][
                    "providers_active"] = max(
                    telemetry.specialization_stats[cap.specialization].get(
                        "providers_active", 0), 1)
        summary = telemetry.step_summary(
            snapshot.snapshot_id, snapshot.game_loop,
            {f"u{i}": a for i, a in enumerate(actions)},
            per_unit,
            [{"id": p.proposal_id, "skill": p.primary_skill,
              "provider": p.specialization_provider,
              "actors": list(p.controlled_actors), "goal": p.goal,
              "score": 0.0, "rationale": p.rationale} for p in plans],
            alternates, repairs,
            (time.perf_counter() - started) * 1000.0,
            self.config.get("_fingerprint", ""))
        summary["skill_usage"] = {k: dict(v) for k, v in telemetry.usage.items()}
        summary["specializations"] = {k: dict(v) for k, v in
                                      telemetry.specialization_stats.items()}
        return actions, summary, plans


def _clone_ledger(ledger: Ledger) -> Ledger:
    return ledger.clone()


def _actions_from(plans: List[CandidatePlan], context: Context,
                  n_agents: int) -> Tuple[List[int], Dict[str, str]]:
    actions = [0] * n_agents
    actors: Dict[str, str] = {}
    for plan in plans:
        for actor, action in plan.current_actions.items():
            index = int(actor[1:])
            if 0 <= index < n_agents:
                actions[index] = action
            actors[actor] = plan.proposal_id
    return actions, actors


def _repair(actions: List[int], plans: List[CandidatePlan], context: Context,
            n_agents: int, per_plan_actors: Dict[str, str]) -> List[int]:
    """Drop any plan with an illegal action (atomically: all its units) and
    refill from the deterministic fallback."""
    broken = set()
    for plan in plans:
        if any(action not in context.legal_actions(actor)
               for actor, action in plan.current_actions.items()):
            broken.add(plan.proposal_id)
    if not broken:
        for i, action in enumerate(actions):
            actor = f"u{i}"
            if action not in context.legal_actions(actor):
                actions[i] = fallback_action(actor, context)
        return actions
    plans[:] = [p for p in plans if p.proposal_id not in broken]
    fresh, actors = _actions_from(plans, context, n_agents)
    for i in range(n_agents):
        actor = f"u{i}"
        if actor not in actors:
            fresh[i] = fallback_action(actor, context)
    return fresh
