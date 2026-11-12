"""Adaptive joint coordinator (round 4: D1-D3 frontier + budgeted evaluate).

Every frontier entry is a COMPLETE, validated, fully-scored plan. The
frontier is re-sorted, semantically deduped and trimmed after EVERY
candidate, so the next candidate always extends the currently best plans —
never a fixed insertion prefix. All full scorings go through one counted,
cached entry point (evaluate_once); a small budget is reserved for final
repair. Tie order (B6): higher U -> keeps live instance tasks -> fewer
action changes -> stable semantic key (NEVER 'more plans').
"""
import time
from typing import Dict, List, Optional, Tuple

from .baseline import baseline_actions, complete_plan
from .context import Context, DEAD_NOOP, distance
from .contracts import CandidatePlan
from .evaluator import Evaluator
from .ledger import Ledger
from .telemetry import Telemetry
from .validator import repair, validate_actions, validate_plans

REPAIR_RESERVE = 2


class _CompletePlan:
    __slots__ = ("plans", "actions", "score", "breakdown")

    def __init__(self, plans, actions, score, breakdown):
        self.plans = plans
        self.actions = actions
        self.score = score
        self.breakdown = breakdown

    def semantic_key(self):
        """A2/D2: actions + per-plan (skill, provider, goal, goal_kind,
        goal_data CONTENT, phase, atomic, actors, dependency content)."""
        return (
            tuple(sorted(self.actions.items())),
            tuple(sorted(
                (p.primary_skill, p.specialization_provider or "", p.goal,
                 p.goal_kind,
                 tuple(sorted((k, str(v)) for k, v
                              in (p.goal_data or {}).items())),
                 p.commitment.phase, p.atomic,
                 tuple(sorted(p.controlled_actors)),
                 tuple(sorted((d.kind, d.refs) for d in p.dependencies)))
                for p in self.plans)))


class Coordinator:
    def __init__(self, config: Dict, evaluator: Optional[Evaluator] = None):
        self.config = config
        search = config.get("search", {})
        self.beam_width = int(search.get("beam_width", 8))
        self.max_evaluations = int(search.get("max_plan_evaluations", 512))
        self.rounds = int(search.get("rounds", 2))
        self.evaluator = evaluator or Evaluator(config.get("weights"))
        self._cache: Dict[tuple, _CompletePlan] = {}

    # ------------------------------------------------------------ evaluation
    def evaluate_once(self, context: Context, plans: List[CandidatePlan],
                      actions: Dict[str, int], memory, budget: List[int],
                      n_agents: int) -> Optional[_CompletePlan]:
        """The ONLY full-scoring entry point: counted, cached, validated.
        budget[0] decrements before every real score_joint (D3)."""
        probe = _CompletePlan(plans, actions, 0.0, {})
        key = probe.semantic_key()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if budget[0] <= 0:
            return None                      # BUDGET_EXHAUSTED
        budget[0] -= 1
        score, breakdown = self.evaluator.score_joint(context, plans, actions,
                                                      memory)
        entry = _CompletePlan(list(plans), dict(actions), score, breakdown)
        if len(entry.actions) != n_agents \
                or validate_actions([entry.actions[f"u{i}"]
                                     for i in range(n_agents)], context,
                                    n_agents) \
                or validate_plans(entry.plans, context, entry.actions,
                                  self.evaluator.effects(context,
                                                         entry.actions)):
            return None
        self._cache[key] = entry
        return entry

    # -------------------------------------------------------------- ranking
    def _rank(self, entry: _CompletePlan, memory) -> tuple:
        """B6 tie order: higher U; keep live instance tasks (fewer switches);
        fewer action changes vs the last committed actions; stable key."""
        live = {i.instance_id for i in (memory.instances or {}).values()
                if i.termination_reason is None}
        keeps_tasks = sum(1 for p in entry.plans if p.instance_id in live)
        switches = sum(1 for p in entry.plans
                       if p.features.get("switch_cost", 0.0) > 0.0)
        previous = getattr(memory, "last_committed_actions", {}) or {}
        changes = sum(1 for actor, action in entry.actions.items()
                      if actor in previous and previous[actor] != action)
        covered = len({a for p in entry.plans for a in p.controlled_actors})
        return (-entry.score, switches, -keeps_tasks, changes,
                -covered,                      # explicit task semantics over
                entry.semantic_key())          # unattributed baseline rows

    def _dedup_trim(self, entries: List[_CompletePlan], memory
                    ) -> List[_CompletePlan]:
        seen, result = set(), []
        for entry in sorted(entries, key=lambda e: self._rank(e, memory)):
            key = entry.semantic_key()
            if key in seen:
                continue
            seen.add(key)
            result.append(entry)
            if len(result) >= self.beam_width:
                break
        return result

    # --------------------------------------------------------------- decide
    def decide(self, context: Context, skills: Dict[str, object],
               memory, telemetry: Telemetry,
               n_agents: int) -> Tuple[List[int], Dict[str, object], List]:
        started = time.perf_counter()
        self._cache = {}
        budget = [max(self.max_evaluations - REPAIR_RESERVE, 1)]
        # ---- A3/D1: P0 built ONCE through the shared completion ----------
        p0_actions = complete_plan(context, memory, skills)
        p0 = self.evaluate_once(context, [], p0_actions, memory, budget,
                                n_agents)
        if p0 is None:
            raise RuntimeError(
                "baseline plan P0 failed validation: "
                f"{validate_actions([p0_actions[f'u{i}'] for i in range(len(p0_actions))], context, len(p0_actions))}")
        # ---- candidates ---------------------------------------------------
        candidates: List[CandidatePlan] = []
        for name in sorted(skills):
            skill = skills[name]
            if not getattr(skill, "enabled", True):
                continue
            proposals = skill.propose(context, memory, telemetry, None)
            candidates.extend(p for p in proposals
                              if p.snapshot_id == context.snapshot.snapshot_id)
        by_unit: Dict[str, List[CandidatePlan]] = {}
        atomics: List[CandidatePlan] = []
        for plan in candidates:
            if plan.atomic:
                atomics.append(plan)
            else:
                for actor in plan.controlled_actors:
                    by_unit.setdefault(actor, []).append(plan)
        # ---- frontier: P0 + validated continuation seeds (D1) -------------
        frontier: List[_CompletePlan] = [p0]
        continuation = self._continuation_plan(context, memory, candidates,
                                               p0_actions, budget,
                                               len(p0_actions))
        if continuation is not None:
            frontier.append(continuation)
        frontier = self._dedup_trim(frontier, memory)
        best = min(frontier, key=lambda e: self._rank(e, memory))
        # ---- bounded search over the CURRENT frontier (D1/D2) -------------
        unit_rotation = sorted(by_unit)
        for _round in range(self.rounds):
            round_start_keys = {e.semantic_key() for e in frontier}
            index = 0
            max_index = max([len(b) for b in by_unit.values()]
                            + [len(atomics)] + [0])
            while index < max_index and budget[0] > 0:
                # fair interleave: A's first, B's first, an atomic, then the
                # next index — later units are never starved (D2)
                for owner in unit_rotation:
                    bucket = by_unit[owner]
                    if index < len(bucket) and budget[0] > 0:
                        self._expand(frontier, bucket[index], context,
                                     memory, budget, len(p0_actions))
                if index < len(atomics) and budget[0] > 0:
                    self._expand(frontier, atomics[index], context, memory,
                                 budget, len(p0_actions))
                index += 1
            if {e.semantic_key() for e in frontier} == round_start_keys:
                break        # no improvement this round: stop early
        best = min(frontier, key=lambda e: self._rank(e, memory))
        if budget[0] <= 0:
            telemetry.search_budget_events += 1
        # ---- A4: final validation & bounded repair ------------------------
        repair_budget = [REPAIR_RESERVE]
        actions = dict(best.actions)
        plans = list(best.plans)
        plans, actions, repair_log = repair(plans, context, actions, memory,
                                            self.evaluator, self.rounds,
                                            skills)
        final = self.evaluate_once(context, plans, actions, memory,
                                   repair_budget, len(actions)) or p0
        summary = self._summary(final, context, telemetry, candidates,
                                repair_log,
                                (time.perf_counter() - started) * 1000.0,
                                len(actions), budget)
        return [final.actions[f"u{i}"] for i in range(len(actions))], \
            summary, plans

    def _expand(self, frontier: List[_CompletePlan],
                candidate: CandidatePlan, context: Context, memory,
                budget: List[int], n_agents: int) -> None:
        """Replace + complete against the CURRENT sorted frontier; the
        frontier list is re-trimmed in place after this candidate (D1)."""
        parents = tuple(frontier)
        contenders = list(frontier)           # parents are always retained
        for entry in parents:
            if budget[0] <= 0:
                break
            q = self._substitute(entry, candidate, context, memory, budget,
                                 n_agents)
            if q is not None:
                contenders.append(q)
        frontier[:] = self._dedup_trim(contenders, memory)

    def _continuation_plan(self, context, memory, candidates, base_actions,
                           budget, n_agents):
        """Continue live instance tasks with fresh candidates; the BEST
        continuation per instance is chosen GEOMETRICALLY (predicted
        closeness to the instance goal) — no hidden uncounted score_joint."""
        actions = dict(base_actions)
        plans: List[CandidatePlan] = []
        used = set()
        best_per_instance: Dict[str, CandidatePlan] = {}
        for plan in candidates:
            if not plan.instance_id:
                continue
            instance = memory.instance_by_id(plan.instance_id)
            if instance is None or instance.termination_reason:
                continue
            if any(a in used for a in plan.controlled_actors):
                continue
            current = best_per_instance.get(plan.instance_id)
            if current is None or _closeness(context, plan) > \
                    _closeness(context, current):
                best_per_instance[plan.instance_id] = plan
        for instance_id, plan in sorted(best_per_instance.items()):
            if any(a in used for a in plan.controlled_actors):
                continue
            if any(actions.get(a) not in context.legal_actions(a)
                   for a in plan.controlled_actors):
                continue
            for actor in plan.controlled_actors:
                actions[actor] = plan.current_actions[actor]
                used.add(actor)
            plans.append(plan)
        if not plans:
            return None
        return self.evaluate_once(context, plans, actions, memory, budget,
                                  n_agents)

    def _substitute(self, entry: _CompletePlan, candidate: CandidatePlan,
                    context: Context, memory, budget: List[int],
                    n_agents) -> Optional[_CompletePlan]:
        """A3: release conflicting plans (atomics whole), protect kept plans
        verbatim, complete ALL free actors in ONE shared-resource call, then
        count-validate-score exactly once."""
        targets = set(candidate.controlled_actors)
        kept, released = [], set()
        for plan in entry.plans:
            if set(plan.controlled_actors) & targets:
                released.update(plan.controlled_actors)   # atomic goes whole
            else:
                kept.append(plan)
        actions = {a: p.current_actions[a] for p in kept
                   for a in p.current_actions}
        for actor in released:
            actions[actor] = None
        fill = complete_plan(context, memory, None,
                             locked_plans=[*kept, candidate])
        for actor in released:
            actions[actor] = fill.get(actor, DEAD_NOOP)
        for actor, action in candidate.current_actions.items():
            actions[actor] = action
        for i in range(n_agents):
            actions.setdefault(f"u{i}", fill.get(f"u{i}", DEAD_NOOP))
        # joint ledger check-then-reserve (A1)
        ledger = Ledger()
        for plan in [*kept, candidate]:
            if ledger.conflicts(plan):
                return None
            ledger.reserve(plan)
        return self.evaluate_once(context, [*kept, candidate], actions,
                                  memory, budget, n_agents)

    def _summary(self, final: _CompletePlan, context, telemetry, candidates,
                 repair_log, duration_ms, n_agents, budget):
        chosen_ids = {p.proposal_id for p in final.plans}
        per_unit = {}
        for plan in final.plans:
            for actor in plan.controlled_actors:
                per_unit[actor] = {
                    "skill": plan.primary_skill,
                    "provider": plan.specialization_provider,
                    "goal": plan.goal, "goal_kind": plan.goal_kind,
                    "action": final.actions[actor],
                    "plan": plan.proposal_id,
                    "phase": plan.commitment.phase}
        for i in range(n_agents):
            actor = f"u{i}"
            if actor not in per_unit:
                per_unit[actor] = {"skill": "baseline", "provider": None,
                                   "goal": "", "goal_kind": "",
                                   "action": final.actions[actor],
                                   "plan": None, "phase": ""}
            telemetry.count(per_unit[actor]["skill"], "selected")
        for actor, info in per_unit.items():
            if info["provider"]:
                stats = telemetry.specialization_stats.setdefault(
                    info["provider"], {"providers_active": 0,
                                       "plans_selected": 0})
                stats["plans_selected"] += 1
        for actor, cap in context.capabilities.items():
            if cap.specialization:
                stats = telemetry.specialization_stats.setdefault(
                    cap.specialization, {"providers_active": 0,
                                         "plans_selected": 0})
                stats["providers_active"] = max(stats.get(
                    "providers_active", 0), 1)
        summary = telemetry.step_summary(
            context.snapshot.snapshot_id, context.snapshot.game_loop,
            dict(final.actions), per_unit,
            [{"id": p.proposal_id, "skill": p.primary_skill,
              "provider": p.specialization_provider,
              "actors": list(p.controlled_actors), "goal": p.goal,
              "rationale": p.rationale} for p in final.plans],
            [{"id": p.proposal_id, "skill": p.primary_skill,
              "rationale": p.rationale}
             for p in candidates if p.proposal_id not in chosen_ids],
            list(repair_log), duration_ms,
            self.config.get("_fingerprint", ""))
        summary["joint_score"] = round(final.score, 4)
        summary["score_breakdown"] = {k: round(v, 4)
                                      for k, v in final.breakdown.items()}
        summary["plan_evaluations_used"] = self.max_evaluations - budget[0]
        summary["cache_entries"] = len(self._cache)
        summary["skill_usage"] = {k: dict(v) for k, v in
                                  telemetry.usage.items()}
        summary["specializations"] = {k: dict(v) for k, v in
                                      telemetry.specialization_stats.items()}
        return summary


def _closeness(context: Context, plan: CandidatePlan) -> float:
    """Geometric continuation preference: negative predicted distance from
    each controlled actor to the plan's goal position (pure geometry)."""
    goal = (plan.goal_data or {}).get("position") \
        or (plan.goal_data or {}).get("waypoint")
    if goal is None and plan.goal_kind == "enemy" \
            and plan.goal in context.snapshot.enemies:
        goal = context.snapshot.enemies[plan.goal].position
    if goal is None and plan.goal_kind == "ally" \
            and plan.goal in context.snapshot.allies:
        goal = context.snapshot.allies[plan.goal].position
    if goal is None:
        return 0.0
    total = 0.0
    for actor in plan.controlled_actors:
        prediction = context.predict_position(
            actor, plan.current_actions.get(actor, DEAD_NOOP))
        total -= distance(prediction.position, tuple(goal))
    return total
