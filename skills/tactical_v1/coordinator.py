"""Adaptive joint coordinator (round 4: D1-D3 frontier + budgeted evaluate).

Every frontier entry is a COMPLETE, validated, fully-scored plan. Execution
duties precede utility in every ranking, including baseline and repair. The
frontier is re-sorted, semantically deduped and trimmed after EVERY
candidate, so the next candidate always extends the currently best plans —
never a fixed insertion prefix. All full scorings go through one counted,
cached entry point (evaluate_once); a small budget is reserved for final
repair. Within equal execution priority: higher U -> live tasks -> fewer
action changes -> stable semantic key (NEVER 'more plans').
"""
import copy
import json
import time
from dataclasses import asdict
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
    __slots__ = ("plans", "actions", "score", "breakdown", "execution")

    def __init__(self, plans, actions, score, breakdown, execution=None):
        self.plans = plans
        self.actions = actions
        self.score = score
        self.breakdown = breakdown
        self.execution = execution or {'priority': (0, 0, 0, 0), 'actors': {}, 'violations': {}}

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
                 tuple((name, float(p.features.get(name, 0.0)))
                       for name in ("hold_value", "switch_cost", "reentry_cost")),
                 tuple(sorted(p.controlled_actors)),
                 tuple(sorted((d.kind, d.refs) for d in p.dependencies)))
                for p in self.plans)))


class _BudgetSlice:
    """Remaining-count list wrapper sharing the owner's real used counter."""

    def __init__(self, remaining: int, owner: "Budget"):
        self._remaining = remaining
        self._owner = owner

    def __getitem__(self, _index):
        return self._remaining

    def __setitem__(self, _index, value):
        self._remaining = value


class Budget:
    """4.3: one counted evaluation budget covering P0, seeds, search and
    repair. limit=1 -> exactly one full scoring (P0) and no repair reserve."""

    def __init__(self, total: int):
        self.total = max(total, 1)
        # With >=2 evaluations, preserve one group-candidate comparison
        # after P0; a valid cached winner needs no extra final scoring.
        self.reserve = min(2, max(0, self.total - 2))
        self.used = 0
        remaining = (self.total if self.total == 1
                     else max(self.total - self.reserve, 0))
        self.search_budget = _BudgetSlice(remaining, self)
        self.repair_budget = _BudgetSlice(self.reserve, self)


class Coordinator:
    def __init__(self, config: Dict, evaluator: Optional[Evaluator] = None):
        self.config = config
        search = config.get("search", {})
        self.beam_width = int(search.get("beam_width", 8))
        self.max_evaluations = int(search.get("max_plan_evaluations", 512))
        self.rounds = int(search.get("rounds", 2))
        self.evaluator = evaluator or Evaluator(config.get("weights"))
        self._cache: Dict[tuple, _CompletePlan] = {}
        self._search_coverage = {}
        self._screened = set()
        self._contract_keys = {}

    # ------------------------------------------------------------ evaluation
    def evaluate_once(self, context: Context, plans: List[CandidatePlan],
                      actions: Dict[str, int], memory, budget: List[int],
                      n_agents: int) -> Optional[_CompletePlan]:
        """The ONLY full-scoring entry point: counted, cached, validated.
        budget[0] decrements before every real score_joint (D3)."""
        probe = _CompletePlan(plans, actions, 0.0, {})
        # Retaining a live instance changes the minimum-commitment cost.
        # Unknown/stale IDs still share a score cache entry, while the
        # current call always keeps its own plan objects and memory deltas.
        retained = tuple(sorted({p.instance_id for p in plans
                                 if p.instance_id in memory.instances}))
        key = (probe.semantic_key(), retained)
        cached = self._cache.get(key)
        if cached is not None:
            # The cache stores score/breakdown and snapshot execution facts;
            # entries
            # are rebuilt from THIS step's plans so a hit never carries a
            # stale instance_id or memory_delta (4.2).
            return _CompletePlan(list(plans), dict(actions), cached.score,
                                 cached.breakdown, cached.execution)
        if budget[0] <= 0:
            return None                      # BUDGET_EXHAUSTED
        if self.config.get('common_combat', {}).get('joint_assessment_enabled', True):
            # A STABILIZE contract must be backed by this branch's service,
            # not by a healer that another candidate has sent elsewhere.
            from .joint_combat import assess_joint
            facts = assess_joint(context, actions)
            for plan in plans:
                if plan.primary_skill == 'disengage_reenter' and plan.commitment.phase == 'STABILIZE':
                    if any(a not in facts['healing_services'] for a in plan.controlled_actors):
                        return None
        budget[0] = budget[0] - 1
        score, breakdown = self.evaluator.score_joint(context, plans, actions,
                                                      memory)
        owner = getattr(budget, "_owner", None)
        if owner is not None:
            owner.used += 1
        entry = _CompletePlan(list(plans), dict(actions), score, breakdown)
        if len(entry.actions) != n_agents \
                or validate_actions([entry.actions[f"u{i}"]
                                     for i in range(n_agents)], context,
                                    n_agents) \
                or validate_plans(entry.plans, context, entry.actions,
                                  self.evaluator.effects(context,
                                                         entry.actions)):
            return None
        if self.config.get('common_combat', {}).get('joint_assessment_enabled', True):
            from .action_duties import assess
            entry.execution = assess(context, actions, memory)
        self._cache[key] = entry
        return entry

    # -------------------------------------------------------------- ranking
    def _rank(self, entry: _CompletePlan, memory) -> tuple:
        """Execution duties precede utility; legal least-violation fallback.

        Remaining ties keep live tasks, continuity and a stable semantic key.
        """
        live = {i.instance_id for i in (memory.instances or {}).values()
                if i.termination_reason is None}
        keeps_tasks = sum(1 for p in entry.plans if p.instance_id in live)
        switches = sum(1 for p in entry.plans
                       if p.features.get("switch_cost", 0.0) > 0.0)
        previous = getattr(memory, "last_committed_actions", {}) or {}
        changes = sum(1 for actor, action in entry.actions.items()
                      if actor in previous and previous[actor] != action)
        covered = len({a for p in entry.plans for a in p.controlled_actors})
        return (*entry.execution['priority'], -entry.score, switches, -keeps_tasks, changes,
                -covered,                      # explicit task semantics over
                entry.semantic_key())          # unattributed baseline rows

    def _dedup_trim(self, entries: List[_CompletePlan], memory
                    ) -> List[_CompletePlan]:
        seen, result = set(), []
        for entry in sorted(entries, key=lambda e: self._rank(e, memory)):
            key = self._frontier_key(entry)
            if key in seen:
                continue
            seen.add(key)
            result.append(entry)
            if len(result) >= self.beam_width:
                break
        return result

    def _frontier_key(self, entry):
        # Same present score can lead to different future task obligations.
        # Keep those alternatives distinct while sharing their score cache.
        contracts = []
        for plan in entry.plans:
            cached = self._contract_keys.get(id(plan))
            if cached is None:
                cached = (plan, self._candidate_key(plan))
                self._contract_keys[id(plan)] = cached
            contracts.append(cached[1])
        return tuple(sorted(entry.actions.items())), tuple(sorted(contracts))

    # --------------------------------------------------------------- decide
    @staticmethod
    def _candidate_key(plan):
        """Only exact contracts are duplicates; same actions can serve
        different objectives, resources or live task identities."""
        data = asdict(plan)
        for field in ("proposal_id", "snapshot_id", "config_fingerprint", "rationale"):
            data.pop(field, None)
        return json.dumps(data, sort_keys=True, default=str)

    def _schedule_candidates(self, candidates, context):
        """Interleave skill families within each actor before their second
        options. Rotate the first actor/family deterministically each window.
        No scoring happens in ordering or deduplication."""
        unique, seen = [], set()
        for plan in candidates:
            key = self._candidate_key(plan)
            if key not in seen:
                seen.add(key)
                unique.append(plan)
        families = {}
        for plan in unique:
            owner = "@joint" if plan.atomic else "+".join(plan.controlled_actors)
            families.setdefault(owner, {}).setdefault(plan.primary_skill, []).append(plan)
        offset = context.snapshot.game_loop // max(context.H, 1)

        def rotated(items):
            if not items:
                return items
            start = offset % len(items)
            return items[start:] + items[:start]

        owners = rotated(sorted(families))
        names = {owner: rotated(sorted(families[owner])) for owner in owners}
        schedule = []
        depth = max((len(bucket) for skills in families.values()
                     for bucket in skills.values()), default=0)
        width = max((len(skills) for skills in families.values()), default=0)
        for rank in range(depth):
            for skill_index in range(width):
                for owner in owners:
                    if skill_index >= len(names[owner]):
                        continue
                    bucket = families[owner][names[owner][skill_index]]
                    if rank < len(bucket):
                        schedule.append(bucket[rank])
        return schedule

    def _mark_screened(self, plan):
        if id(plan) in self._screened:
            return
        self._screened.add(id(plan))
        self._search_coverage["screened"] += 1
        self._search_coverage["by_skill"][plan.primary_skill]["screened"] += 1
        for actor in plan.controlled_actors:
            self._search_coverage["by_actor"][actor]["screened"] += 1

    def decide(self, context: Context, skills: Dict[str, object],
               memory, telemetry: Telemetry,
               n_agents: int) -> Tuple[List[int], Dict[str, object], List]:
        started = time.perf_counter()
        context.deployment_memory = memory
        self._cache = {}
        self._screened = set()
        self._contract_keys = {}
        self._execution_seeds = 0
        budget = Budget(self.max_evaluations)
        # ---- A3/D1: P0 built ONCE through the single completion entry ----
        p0_actions = complete_plan(context, memory, skills)
        p0 = self.evaluate_once(context, [], p0_actions, memory,
                                budget.search_budget,
                                len(p0_actions))
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
        generated = len(candidates)
        candidates = self._schedule_candidates(candidates, context)
        self._search_coverage = {
            "generated": generated, "deduplicated": len(candidates), "screened": 0,
            "by_skill": {}, "by_actor": {},
            "semantics": "screened means one complete baseline substitution attempted; invalid branches rejected"}
        for plan in candidates:
            self._search_coverage["by_skill"].setdefault(
                plan.primary_skill, {"proposed": 0, "screened": 0})["proposed"] += 1
            for actor in plan.controlled_actors:
                self._search_coverage["by_actor"].setdefault(
                    actor, {"proposed": 0, "screened": 0})["proposed"] += 1
        atomics = sorted((plan for plan in candidates if plan.atomic), key=lambda plan: (
            0 if plan.instance_id in memory.instances else
            1 if (plan.goal_data.get("objective") or {}).get("kind") in
                 ("rally", "combat_rejoin") else 2))
        # ---- frontier: P0 + validated continuation seeds (D1) -------------
        frontier: List[_CompletePlan] = [p0]
        # A strong duty needs an actual complete alternative. This seed is
        # scored/validated within the same budget, including all free actors.
        if self.config.get('common_combat', {}).get('joint_assessment_enabled', True):
            from .action_duties import seed_actions
            duty_actions = seed_actions(context, p0_actions, memory)
            if duty_actions != p0_actions:
                duty_seed = self.evaluate_once(context, [], duty_actions, memory,
                    budget.search_budget, len(p0_actions))
                if duty_seed is not None:
                    frontier.append(duty_seed)
                    self._execution_seeds += 1
        # Important joint candidates get a full comparison even with a tiny
        # budget. They still compete on the same objective as other actions.
        for plan in atomics:
            if budget.search_budget[0] <= 0:
                break
            self._mark_screened(plan)
            seed = self._substitute(p0, plan, context, memory,
                                    budget.search_budget, len(p0_actions))
            if seed is not None:
                frontier.append(seed)
            if budget.search_budget[0] <= 0:
                break
        continuation = self._continuation_plan(context, memory, candidates,
                                               p0_actions,
                                               budget.search_budget,
                                               len(p0_actions))
        if continuation is not None:
            frontier.append(continuation)
        frontier = self._dedup_trim(frontier, memory)
        # One baseline comparison per distinct contract before any candidate
        # spends a beam's worth of evaluations. Later skills/units get a
        # real opportunity; their resources are not blocked by an earlier
        # candidate during this screening pass.
        for plan in candidates:
            if id(plan) in self._screened:
                continue
            if budget.search_budget[0] <= 0:
                break
            self._mark_screened(plan)
            seed = self._substitute(p0, plan, context, memory,
                                    budget.search_budget, len(p0_actions))
            if seed is not None:
                frontier = self._dedup_trim([*frontier, seed], memory)
        # ---- bounded search over the CURRENT frontier (D1/D2) -------------
        for _round in range(self.rounds):
            round_start_keys = {self._frontier_key(e) for e in frontier}
            for parent_rank in range(self.beam_width):
                for plan in candidates:
                    if budget.search_budget[0] <= 0:
                        break
                    self._expand(frontier, plan, context, memory,
                                 budget.search_budget, len(p0_actions), parent_rank)
                if budget.search_budget[0] <= 0:
                    break
            if {self._frontier_key(e) for e in frontier} == round_start_keys:
                break        # no improvement this round: stop early
        best = min(frontier, key=lambda e: self._rank(e, memory))
        if budget.search_budget[0] <= 0 and self.max_evaluations > 1:
            telemetry.search_budget_events += 1
        self._search_coverage["unexamined"] = len(candidates) - len(self._screened)
        # ---- A4: final validation & bounded repair ------------------------
        repair_budget = budget.repair_budget
        actions = dict(best.actions)
        plans = list(best.plans)
        plans, actions, repair_log = repair(plans, context, actions, memory,
                                            self.evaluator, self.rounds,
                                            skills)
        final = self.evaluate_once(context, plans, actions, memory,
                                   repair_budget, len(actions)) or best
        # Repair or a newly combined plan must not bypass execution duties.
        # Changed atomic plans are released whole; unchanged members keep
        # their legal commands as baseline actions, not stale task claims.
        if self.config.get('common_combat', {}).get('joint_assessment_enabled', True):
            from .action_duties import seed_actions
            duty_actions = seed_actions(context, final.actions, memory)
            changed = {a for a in duty_actions if duty_actions[a] != final.actions[a]}
            if changed:
                kept = [p for p in final.plans if not changed.intersection(p.controlled_actors)]
                duty_final = self.evaluate_once(context, kept, duty_actions, memory,
                    repair_budget, len(duty_actions))
                if duty_final is not None:
                    self._execution_seeds += 1
                    final = min((final, duty_final), key=lambda e: self._rank(e, memory))
        # 4.1: ONE final object — actions, plans, score, summary and the
        # manager's pending commit all come from this single entry.
        summary = self._summary(final, context, telemetry, candidates,
                                repair_log,
                                (time.perf_counter() - started) * 1000.0,
                                len(actions), budget, memory)
        return ([final.actions[f"u{i}"] for i in range(len(actions))],
                summary, list(final.plans))

    def _expand(self, frontier: List[_CompletePlan],
                candidate: CandidatePlan, context: Context, memory,
                budget: List[int], n_agents: int, parent_rank=None) -> None:
        """Replace + complete against the CURRENT sorted frontier; the
        frontier list is re-trimmed in place after this candidate (D1)."""
        parents = (tuple(frontier) if parent_rank is None
                   else tuple(frontier[parent_rank:parent_rank + 1]))
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
        from .baseline import LockedPlanConflict
        try:
            fill = complete_plan(context, memory, None,
                                 locked_plans=[*kept, candidate])
        except LockedPlanConflict:
            return None        # reject the branch; never rewrite a locked task
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
                 repair_log, duration_ms, n_agents, budget, memory=None):
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
        # Report actual scored, valid COMPLETE alternatives. Candidate
        # generation order is not an ordering by quality.
        alternatives, seen_actions = [], {tuple(sorted(final.actions.items()))}
        for entry in sorted(self._cache.values(), key=lambda e: self._rank(e, memory)):
            key = tuple(sorted(entry.actions.items()))
            if key in seen_actions:
                continue
            seen_actions.add(key)
            alternatives.append({
                "id": f"joint-alternative-{len(alternatives) + 1}",
                "skill": "joint_plan", "rationale": "scored complete alternative",
                "score": round(entry.score, 4), "actions": dict(entry.actions),
                "execution_priority": entry.execution['priority'],
                "execution_violations": entry.execution['violations'],
                "score_breakdown": {k: round(v, 4) for k, v in entry.breakdown.items()},
                "plans": [{"id": p.proposal_id, "skill": p.primary_skill,
                           "actors": list(p.controlled_actors), "goal": p.goal}
                          for p in entry.plans]})
            if len(alternatives) >= telemetry.top_k:
                break
        telemetry.sync_lifecycle(getattr(memory, "lifecycle_counts", {}))
        summary = telemetry.step_summary(
            context.snapshot.snapshot_id, context.snapshot.game_loop,
            dict(final.actions), per_unit,
            [{"id": p.proposal_id, "skill": p.primary_skill,
              "provider": p.specialization_provider,
              "actors": list(p.controlled_actors), "goal": p.goal,
              "score": round(final.score, 4),
              "rationale": p.rationale} for p in final.plans],
            alternatives,
            list(repair_log), duration_ms,
            self.config.get("_fingerprint", ""))
        summary["joint_score"] = round(final.score, 4)
        summary["score_breakdown"] = {k: round(v, 4)
                                      for k, v in final.breakdown.items()}
        summary["plan_evaluations_used"] = budget.used
        summary["cache_entries"] = len(self._cache)
        summary["candidate_search"] = copy.deepcopy(self._search_coverage)
        summary['execution_duties'] = final.execution
        summary['execution_seed_candidates'] = self._execution_seeds
        # --- shared diagnostics (predictions, never observed damage) -------
        try:
            from .unit_value_view import summary as value_summary
            summary['unit_value'] = value_summary(context)
        except Exception:
            summary['unit_value'] = {"enabled": False, "error": "summary_failed"}
        try:
            from .intercept import summary as intercept_summary
            summary['enemy_bomber_intercept'] = intercept_summary(context, memory)
        except Exception:
            summary['enemy_bomber_intercept'] = {"enabled": False,
                                                 "error": "summary_failed"}
        summary['burst_defer_reasons'] = dict(
            getattr(context, '_burst_defer_reasons', {}))
        summary['burst_decline_reasons'] = dict(
            getattr(context, '_burst_decline_log', {}))
        try:
            from .deployment import summary as deployment_summary
            summary['deployment'] = deployment_summary(context)
        except Exception:
            summary['deployment'] = {"error": "summary_failed"}
        if self.config.get('common_combat', {}).get('joint_assessment_enabled', True):
            from .joint_combat import summary as combat_summary
            from .local_entry import assess_local_entry
            from .battle_line import assess_battle_line
            joint = self.evaluator.joint_facts(context, final.actions)
            summary['joint_combat'] = combat_summary(joint)
            summary['local_entry'] = assess_local_entry(
                context, final.actions, joint, final.plans, memory)
            summary['battle_line'] = assess_battle_line(
                context, final.actions, joint, memory)
            summary['output_duties'] = {a: duty for a in final.actions
                if (duty := getattr(memory.combat_progress, 'output_duty', lambda *_: None)(context, a))}
            summary['contact_recovery'] = {a: duty for a in final.actions
                if (duty := getattr(memory, 'contact_recovery', lambda *_: None)(context, a))}
            progress = getattr(memory, 'contact_progress', None)
            if progress is not None:
                summary['contact_outcomes'] = progress.summary(context)
                summary['contact_decisions'] = {a: progress.action_reason(context, a, action, memory)
                    for a, action in final.actions.items()}
            summary['burst_progress'] = getattr(
                getattr(memory, 'burst_progress', None), 'summary', lambda: {})()
        summary["lifecycle_counts"] = {
            name: dict(counts) for name, counts in
            getattr(memory, "lifecycle_counts", {}).items()}
        summary["lifecycle_last_committed_loop"] = getattr(
            memory, "last_committed_loop", None)
        summary["lifecycle_semantics"] = "observed exits and completed commits; current selection pending commit"
        summary["skill_usage"] = {k: dict(v) for k, v in
                                  telemetry.usage.items()}
        summary["specializations"] = {k: dict(v) for k, v in
                                      telemetry.specialization_stats.items()}
        return summary


def _closeness(context: Context, plan: CandidatePlan) -> float:
    """Geometric continuation preference: negative predicted distance from
    each controlled actor to the plan's goal position (pure geometry)."""
    goal = (plan.goal_data or {}).get("anchor") \
        or (plan.goal_data or {}).get("position") \
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
