"""Adaptive joint coordinator (A2/A3).

Every beam entry is a COMPLETE, validated, fully-scored plan with explicit
action provenance. No post-hoc attribution: an action enters the final plan
only through its own plan entry (a skill candidate or an honest baseline
row). Dedup is semantic (actions + actors + goal + skill/provider + phase +
atomic members + dependencies), never by action list alone. The refill used
by substitution/repair is context-aware (kept fire, heal capacity, slots).
"""
import time
from typing import Dict, List, Optional, Tuple

from .baseline import baseline_actions
from .context import Context, DEAD_NOOP
from .contracts import CandidatePlan
from .evaluator import Evaluator
from .ledger import Ledger
from .telemetry import Telemetry
from .validator import repair, validate_actions, validate_plans


class _CompletePlan:
    __slots__ = ("plans", "actions", "score", "breakdown")

    def __init__(self, plans, actions, score, breakdown):
        self.plans = plans
        self.actions = actions
        self.score = score
        self.breakdown = breakdown

    def semantic_key(self):
        """A2 dedup: same actions alone is NOT identity — goal/skill/phase/
        members/deps distinguish genuinely different tasks."""
        return (
            tuple(sorted(self.actions.items())),
            tuple(sorted((p.primary_skill, p.specialization_provider or "",
                          p.goal, p.commitment.phase, p.atomic,
                          tuple(sorted(p.controlled_actors)),
                          tuple(sorted(d.kind for d in p.dependencies)))
                         for p in self.plans)))


class Coordinator:
    def __init__(self, config: Dict, evaluator: Optional[Evaluator] = None):
        self.config = config
        search = config.get("search", {})
        self.beam_width = int(search.get("beam_width", 8))
        self.max_evaluations = int(search.get("max_plan_evaluations", 512))
        self.rounds = int(search.get("rounds", 2))
        self.evaluator = evaluator or Evaluator(config.get("weights"))

    # ------------------------------------------------------------------ core
    def _score(self, context: Context, plans: List[CandidatePlan],
               actions: Dict[str, int], memory) -> _CompletePlan:
        score, breakdown = self.evaluator.score_joint(context, plans, actions,
                                                      memory)
        return _CompletePlan(plans, dict(actions), score, breakdown)

    def _validate_complete(self, context: Context, entry: _CompletePlan,
                           n_agents: int) -> bool:
        if len(entry.actions) != n_agents:
            return False
        ordered = [entry.actions[f"u{i}"] for i in range(n_agents)]
        if validate_actions(ordered, context, n_agents):
            return False
        report = self.evaluator.effects(context, entry.actions)
        return not validate_plans(entry.plans, context, entry.actions, report)

    def decide(self, context: Context, skills: Dict[str, object],
               memory, telemetry: Telemetry,
               n_agents: int) -> Tuple[List[int], Dict[str, object], List]:
        started = time.perf_counter()
        snapshot = context.snapshot
        actors = [f"u{i}" for i in range(n_agents)]
        # ---- A1: complete baseline P0 (explicit provenance) ---------------
        base_actions = {a: baseline_actions(context, memory, skills).get(
            a, DEAD_NOOP) for a in actors}
        p0 = self._score(context, [], base_actions, memory)
        if not self._validate_complete(context, p0, n_agents):
            raise RuntimeError(
                "baseline plan P0 failed validation: "
                f"{validate_actions([base_actions[f'u{i}'] for i in range(n_agents)], context, n_agents)}")
        # ---- candidates ----------------------------------------------------
        candidates: List[CandidatePlan] = []
        for name in sorted(skills):
            skill = skills[name]
            if not getattr(skill, "enabled", True):
                continue
            proposals = skill.propose(context, memory, telemetry, None)
            candidates.extend(p for p in proposals
                              if p.snapshot_id == snapshot.snapshot_id)
        by_unit: Dict[str, List[CandidatePlan]] = {}
        atomics: List[CandidatePlan] = []
        for plan in candidates:
            if plan.atomic:
                atomics.append(plan)
            else:
                for actor in plan.controlled_actors:
                    by_unit.setdefault(actor, []).append(plan)
        for plan_list in by_unit.values():
            plan_list.sort(key=lambda p: p.proposal_id)
        atomics.sort(key=lambda p: p.proposal_id)
        # ---- seeds: P0 + validated instance continuation --------------------
        beam: List[_CompletePlan] = [p0]
        continuation = self._continuation_plan(context, memory, candidates,
                                               base_actions, n_agents)
        if continuation is not None \
                and self._validate_complete(context, continuation, n_agents):
            beam.append(continuation)
        beam = self._dedup(beam)
        # on ties prefer entries that CARRY plans: identical physics plus a
        # real task must not lose to a bare baseline row (A2 provenance)
        best = max(beam, key=lambda e: (e.score, len(e.plans),
                                        e.semantic_key()))
        # ---- bounded neighborhood search (A3) -------------------------------
        evaluations = 1 + len(beam)
        unit_rotation = sorted(by_unit)
        stalled = False
        for _round in range(self.rounds):
            if stalled or evaluations >= self.max_evaluations:
                break
            stalled = True
            next_beam = list(beam)
            # fair interleave: candidate index outer, buckets inner, so later
            # units get their first candidate before early units exhaust theirs
            index = 0
            while index < max([len(b) for b in by_unit.values()]
                              + [len(atomics)] + [0]) \
                    and evaluations < self.max_evaluations:
                for owner in unit_rotation:
                    bucket = by_unit[owner]
                    if index >= len(bucket) or evaluations >= self.max_evaluations:
                        continue
                    entry_added = self._try_substitutions(
                        beam, next_beam, bucket[index], context, memory,
                        n_agents)
                    evaluations += entry_added
                    if entry_added:
                        stalled = False
                if index < len(atomics) \
                        and evaluations < self.max_evaluations:
                    entry_added = self._try_substitutions(
                        beam, next_beam, atomics[index], context, memory,
                        n_agents)
                    evaluations += entry_added
                    if entry_added:
                        stalled = False
                index += 1
            next_beam.sort(key=lambda e: (-e.score, -len(e.plans),
                                           e.semantic_key()))
            beam = next_beam[:self.beam_width]
            top = beam[0]
            if (top.score, len(top.plans)) > (best.score, len(best.plans)):
                best = top
        if evaluations >= self.max_evaluations:
            telemetry.search_budget_events += 1
        # ---- A4: final validation & bounded repair --------------------------
        actions = dict(best.actions)
        plans = list(best.plans)
        plans, actions, repair_log = repair(plans, context, actions, memory,
                                            self.evaluator, self.rounds,
                                            skills)
        final = self._score(context, plans, actions, memory)
        if not self._validate_complete(context, final, n_agents):
            raise RuntimeError("post-repair plan failed validation")
        summary = self._summary(final, context, telemetry, candidates,
                                repair_log,
                                (time.perf_counter() - started) * 1000.0,
                                n_agents)
        return [final.actions[f"u{i}"] for i in range(n_agents)], summary, plans

    def _try_substitutions(self, beam, next_beam, candidate, context, memory,
                           n_agents) -> int:
        """Try ``candidate`` against the EVOLVING entry set (the round's
        next_beam, seeded by beam), so cross-unit combinations compose within
        a single round; returns evaluations used."""
        used = 0
        for entry in list(next_beam)[:self.beam_width * 2]:
            q = self._substitute(entry, candidate, context, memory, n_agents)
            used += 1
            if q is None:
                continue
            if self._validate_complete(context, q, n_agents) \
                    and not any(q.semantic_key() == e.semantic_key()
                                for e in next_beam):
                next_beam.append(q)
        return used

    def _continuation_plan(self, context, memory, candidates, base_actions,
                           n_agents):
        """Continue live instance tasks with THIS step's fresh candidates —
        the BEST continuation candidate per instance (never an arbitrary
        hash-ordered one), so a valid task is not lost to its own bad move."""
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
            if current is None:
                best_per_instance[plan.instance_id] = plan
                continue
            trial = dict(actions)
            trial.update(plan.current_actions)
            trial_actions = dict(actions)
            trial_actions.update(current.current_actions)
            if self.evaluator.score_joint(
                    context, [plan], trial, memory)[0] >=                     self.evaluator.score_joint(
                        context, [current], trial_actions, memory)[0]:
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
        return self._score(context, plans, actions, memory)

    def _substitute(self, entry: _CompletePlan, candidate: CandidatePlan,
                    context: Context, memory, n_agents) -> Optional[_CompletePlan]:
        """Replace plans conflicting with ``candidate`` (atomics whole),
        refill the released units with the CONTEXT-AWARE baseline, apply the
        candidate, check the joint ledger, rescore from final actions."""
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
        fill = baseline_actions(context, memory, None, reserved=entry.actions,
                                reserved_plans=kept, only=sorted(released))
        for actor in released:
            actions[actor] = fill.get(actor, DEAD_NOOP)
        for actor, action in candidate.current_actions.items():
            actions[actor] = action
        for i in range(n_agents):
            actions.setdefault(f"u{i}", base_default(context, memory,
                                                     f"u{i}"))
        # joint ledger check-then-reserve (A1)
        ledger = Ledger()
        for plan in [*kept, candidate]:
            if ledger.conflicts(plan):
                return None
            ledger.reserve(plan)
        return self._score(context, [*kept, candidate], actions, memory)

    def _dedup(self, entries: List[_CompletePlan]) -> List[_CompletePlan]:
        seen, result = set(), []
        for entry in entries:
            key = entry.semantic_key()
            if key not in seen:
                seen.add(key)
                result.append(entry)
        return result

    def _summary(self, final: _CompletePlan, context, telemetry, candidates,
                 repair_log, duration_ms, n_agents):
        chosen_ids = {p.proposal_id for p in final.plans}
        per_unit = {}
        for plan in final.plans:
            for actor in plan.controlled_actors:
                per_unit[actor] = {
                    "skill": plan.primary_skill,
                    "provider": plan.specialization_provider,
                    "goal": plan.goal,
                    "action": final.actions[actor],
                    "plan": plan.proposal_id,
                    "phase": plan.commitment.phase}
        for i in range(n_agents):
            actor = f"u{i}"
            if actor not in per_unit:
                per_unit[actor] = {"skill": "baseline", "provider": None,
                                   "goal": "", "action": final.actions[actor],
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
        # E2: one joint score, clearly named; never a per-candidate total
        summary["joint_score"] = round(final.score, 4)
        summary["score_breakdown"] = {k: round(v, 4)
                                      for k, v in final.breakdown.items()}
        summary["skill_usage"] = {k: dict(v) for k, v in
                                  telemetry.usage.items()}
        summary["specializations"] = {k: dict(v) for k, v in
                                      telemetry.specialization_stats.items()}
        return summary


def base_default(context, memory, actor):
    return baseline_actions(context, memory, None, only=[actor]).get(
        actor, DEAD_NOOP)
