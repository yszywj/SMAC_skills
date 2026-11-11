"""Adaptive joint coordinator (A1/A2).

Every beam entry is a COMPLETE, validated, fully-scored plan. The search is
a bounded neighborhood walk over complete plans: replace the plans
conflicting with a candidate (atomic plans whole), refill released units
from the baseline, re-validate, rescore from scratch. Deterministic
evaluation budget; truncated search returns the best COMPLETE plan.
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

    def key(self):
        return tuple(sorted(self.actions.items()))


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
        return not validate_plans(entry.plans, context, entry.actions)

    def decide(self, context: Context, skills: Dict[str, object],
               memory, telemetry: Telemetry,
               n_agents: int) -> Tuple[List[int], Dict[str, object], List]:
        started = time.perf_counter()
        snapshot = context.snapshot
        actors = [f"u{i}" for i in range(n_agents)]
        # ---- A1: complete baseline P0 ----------------------------------
        base_actions = baseline_actions(context, memory, skills)
        base_actions = {a: base_actions.get(a, DEAD_NOOP) for a in actors}
        p0 = self._score(context, [], base_actions, memory)
        if not self._validate_complete(context, p0, n_agents):
            raise RuntimeError(
                f"baseline plan P0 failed validation: "
                f"{validate_actions([base_actions[f'u{i}'] for i in range(n_agents)], context, n_agents)}"
                f"{validate_plans([], context, base_actions)}")
        # ---- candidates -------------------------------------------------
        candidates: List[CandidatePlan] = []
        for name in sorted(skills):
            skill = skills[name]
            if not getattr(skill, "enabled", True):
                continue
            proposals = skill.propose(context, memory, telemetry, None)
            candidates.extend(p for p in proposals
                              if p.snapshot_id == snapshot.snapshot_id)
        # per-unit candidate buckets + atomic list (fair rotation, A2)
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
        # ---- seeds: P0 + instance-continuation plan ----------------------
        seeds = [p0]
        continuation = self._continuation_plan(
            context, memory, candidates, base_actions)
        if continuation is not None:
            seeds.append(continuation)
        beam: List[_CompletePlan] = self._dedup(seeds)
        best = max(beam, key=lambda e: (e.score, e.key()))
        # ---- bounded neighborhood search --------------------------------
        evaluations = 1 + len(beam)
        unit_rotation = sorted(by_unit)
        stalled = False
        for _round in range(self.rounds):
            if stalled or evaluations >= self.max_evaluations:
                break
            stalled = True
            next_beam = list(beam)
            # fair rotation: units first, then atomic candidates
            tasks = [(u, by_unit[u]) for u in unit_rotation] + \
                    [("*", atomics)]
            for bucket_owner, bucket in tasks:
                if evaluations >= self.max_evaluations:
                    telemetry.search_budget_events += 1
                    break
                for candidate in bucket:
                    if evaluations >= self.max_evaluations:
                        telemetry.search_budget_events += 1
                        break
                    for entry in list(beam):
                        q = self._substitute(entry, candidate, context,
                                             memory, n_agents)
                        if q is None:
                            continue
                        evaluations += 1
                        if self._validate_complete(context, q, n_agents) \
                                and not any(q.key() == e.key() for e in next_beam):
                            next_beam.append(q)
                            stalled = False
                next_beam.sort(key=lambda e: (-e.score, e.key()))
                beam = next_beam[:self.beam_width]
            top = beam[0]
            if top.score > best.score:
                best = top
        # ---- A4: final validation & bounded repair -----------------------
        actions = dict(best.actions)
        plans = list(best.plans)
        plans, actions, repair_log = repair(plans, context, actions, memory,
                                            self.evaluator, self.rounds)
        # Attribute baseline-equal behaviour: candidates whose actions match
        # the final actions exactly still represent the executed task (their
        # instances and telemetry must exist); scores are unchanged because
        # the PHYSICAL actions are identical.
        covered = {a for p in plans for a in p.controlled_actors}
        for candidate in sorted(candidates, key=lambda p: p.proposal_id):
            if any(a in covered for a in candidate.controlled_actors):
                continue
            if all(actions.get(a) == candidate.current_actions.get(a)
                   for a in candidate.controlled_actors):
                plans.append(candidate)
                covered.update(candidate.controlled_actors)
        final = self._score(context, plans, actions, memory)
        if not self._validate_complete(context, final, n_agents):
            raise RuntimeError("post-repair plan failed validation")
        summary = self._summary(final, context, telemetry, candidates,
                                repair_log, (time.perf_counter() - started) * 1000.0,
                                n_agents, base_actions)
        return [final.actions[f"u{i}"] for i in range(n_agents)], summary, plans

    # ------------------------------------------------------------ helpers
    def _continuation_plan(self, context, memory, candidates, base_actions):
        """Continue live instance tasks with THIS step's fresh candidates."""
        actions = dict(base_actions)
        plans: List[CandidatePlan] = []
        used = set()
        for plan in sorted(candidates, key=lambda p: p.proposal_id):
            if not plan.instance_id:
                continue
            instance = memory.instance_by_id(plan.instance_id)
            if instance is None or instance.termination_reason:
                continue
            if any(a in used for a in plan.controlled_actors):
                continue
            ok = all(actions.get(a) in context.legal_actions(a)
                     for a in plan.controlled_actors)
            if not ok:
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
        """Replace every plan conflicting with `candidate` (atomic plans
        whole), refill released units from the baseline, rescore fully."""
        from .baseline import baseline_actions
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
            actions.setdefault(actor, None)
        fill = baseline_actions(context, memory, None)
        for actor in list(actions):
            if actions[actor] is None:
                actions[actor] = fill.get(actor, DEAD_NOOP)
        for actor, action in candidate.current_actions.items():
            actions[actor] = action
        if len(actions) != n_agents:
            for i in range(n_agents):
                actions.setdefault(f"u{i}", fill.get(f"u{i}", DEAD_NOOP))
        ledger = Ledger.from_plans([*kept, candidate])
        for plan in [*kept, candidate]:
            if ledger.conflicts(plan) and plan is not candidate:
                return None
        return self._score(context, [*kept, candidate], actions, memory)

    def _dedup(self, entries: List[_CompletePlan]) -> List[_CompletePlan]:
        seen, result = set(), []
        for entry in entries:
            if entry.key() not in seen:
                seen.add(entry.key())
                result.append(entry)
        return result

    def _summary(self, final: _CompletePlan, context, telemetry, candidates,
                 repair_log, duration_ms, n_agents, base_actions):
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
                telemetry.specialization_stats.setdefault(
                    info["provider"], {"providers_active": 0,
                                       "plans_selected": 0})
                telemetry.specialization_stats[info["provider"]][
                    "plans_selected"] += 1
        for actor, cap in context.capabilities.items():
            if cap.specialization:
                stats = telemetry.specialization_stats.setdefault(
                    cap.specialization, {"providers_active": 0,
                                         "plans_selected": 0})
                stats["providers_active"] = max(stats.get(
                    "providers_active", 0), 1)
        alternates = [{"id": p.proposal_id, "skill": p.primary_skill,
                       "rationale": p.rationale}
                      for p in candidates if p.proposal_id not in chosen_ids]
        summary = telemetry.step_summary(
            context.snapshot.snapshot_id, context.snapshot.game_loop,
            dict(final.actions), per_unit,
            [{"id": p.proposal_id, "skill": p.primary_skill,
              "provider": p.specialization_provider,
              "actors": list(p.controlled_actors), "goal": p.goal,
              "score": round(final.score, 4),   # joint score, never a fixed 0
              "rationale": p.rationale} for p in final.plans],
            alternates, list(repair_log), duration_ms,
            self.config.get("_fingerprint", ""))
        summary["score_breakdown"] = {k: round(v, 4)
                                      for k, v in final.breakdown.items()}
        summary["score_total"] = round(final.score, 4)
        summary["skill_usage"] = {k: dict(v) for k, v in
                                  telemetry.usage.items()}
        summary["specializations"] = {k: dict(v) for k, v in
                                      telemetry.specialization_stats.items()}
        return summary
