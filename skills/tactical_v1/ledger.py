"""The step resource ledger (R5).

Four separate resource classes. Space claims are structured (actor,
predicted position, window, collision class, radius, optional hard slot id):
congestion between DIFFERENT collision classes or without a shared slot is a
soft penalty evaluated by the evaluator, not a hard exclusion. Release is
done by REBUILDING from the remaining plans — no error-prone incremental
subtraction.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .contracts import CandidatePlan, SpaceClaim


@dataclass
class Ledger:
    controls: Dict[str, str] = field(default_factory=dict)     # actor -> proposal_id
    fire: Dict[str, float] = field(default_factory=dict)       # enemy -> window damage
    healing: Dict[str, float] = field(default_factory=dict)    # ally -> heal proxy
    plans: Dict[str, CandidatePlan] = field(default_factory=dict)

    @classmethod
    def from_plans(cls, plans: List[CandidatePlan]) -> "Ledger":
        ledger = cls()
        for plan in sorted(plans, key=lambda p: p.proposal_id):
            ledger.reserve(plan)
        return ledger

    def clone(self) -> "Ledger":
        return Ledger(controls=dict(self.controls), fire=dict(self.fire),
                      healing=dict(self.healing), plans=dict(self.plans))

    # -- conflict checking --------------------------------------------------
    def conflicts(self, plan: CandidatePlan) -> List[str]:
        problems = []
        for actor in plan.controlled_actors:
            if actor in self.controls:
                problems.append(f"control:{actor}:{self.controls[actor]}")
        # Hard space exclusions: only identical contact-slot ids clash.
        claimed_slots = {}
        for existing in self.plans.values():
            for claim in existing.claims.space:
                if claim.slot_id:
                    claimed_slots.setdefault(claim.slot_id, set()).add(claim.actor)
        for claim in plan.claims.space:
            if claim.slot_id:
                holders = claimed_slots.get(claim.slot_id, set())
                outsiders = holders - set(plan.controlled_actors)
                if outsiders:
                    problems.append(f"slot:{claim.slot_id}:{sorted(outsiders)}")
        # Dependencies: every member dependency must be inside this plan or
        # already controlled by a plan listed as a co-dependency.
        for dependency in plan.dependencies:
            if dependency.kind == "member":
                missing = [a for a in dependency.refs
                           if a not in plan.controlled_actors
                           and a not in self.controls]
                if missing:
                    problems.append(f"member_missing:{missing}")
        return problems

    def reserve(self, plan: CandidatePlan) -> None:
        for actor in plan.controlled_actors:
            self.controls[actor] = plan.proposal_id
        for enemy_id, amount in plan.claims.fire:
            if amount:
                self.fire[enemy_id] = self.fire.get(enemy_id, 0.0) + amount
        for ally_id, amount in plan.claims.healing:
            if amount:
                self.healing[ally_id] = self.healing.get(ally_id, 0.0) + amount
        self.plans[plan.proposal_id] = plan

    def release(self, proposal_id: str) -> None:
        """Release by rebuilding from the remaining plans (R5)."""
        if proposal_id in self.plans:
            remaining = [p for pid, p in self.plans.items() if pid != proposal_id]
            fresh = Ledger.from_plans(remaining)
            self.controls = fresh.controls
            self.fire = fresh.fire
            self.healing = fresh.healing
            self.plans = fresh.plans

    def fire_on(self, enemy_id: str) -> float:
        return self.fire.get(enemy_id, 0.0)

    def healing_on(self, ally_id: str) -> float:
        return self.healing.get(ally_id, 0.0)
