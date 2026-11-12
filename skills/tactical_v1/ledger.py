"""The step resource ledger (A1).

Four separate resource classes. Insertions always use check-then-reserve in
a deterministic order: a plan is validated against every OTHER plan already
in the ledger, never against itself. Space claims are structured; only a
shared contact slot id is a hard exclusion, predicted congestion is a soft
evaluator penalty. Release rebuilds from the remaining plans.
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
        """Check-then-reserve insertion in deterministic order (A1): a plan
        conflicting with any OTHER already-reserved plan is skipped, never
        silently overwriting the occupant table."""
        ledger = cls()
        for plan in sorted(plans, key=lambda p: p.proposal_id):
            if not ledger.conflicts(plan):
                ledger.reserve(plan)
        return ledger

    def clone(self) -> "Ledger":
        return Ledger(controls=dict(self.controls), fire=dict(self.fire),
                      healing=dict(self.healing), plans=dict(self.plans))

    # -- conflict checking --------------------------------------------------
    def conflicts(self, plan: CandidatePlan) -> List[str]:
        """Conflicts of ``plan`` with every OTHER reserved plan. A plan never
        conflicts with itself (own proposal_id is excluded)."""
        problems = []
        for actor in plan.controlled_actors:
            holder = self.controls.get(actor)
            if holder is not None and holder != plan.proposal_id:
                problems.append(f"control:{actor}:{holder}")
        # Hard space exclusions: only identical contact-slot ids clash.
        claimed_slots: Dict[str, set] = {}
        for existing in self.plans.values():
            if existing.proposal_id == plan.proposal_id:
                continue
            for claim in existing.claims.space:
                if claim.slot_id:
                    claimed_slots.setdefault(claim.slot_id, set()).add(claim.actor)
        for claim in plan.claims.space:
            if claim.slot_id:
                outsiders = claimed_slots.get(claim.slot_id, set()) \
                    - set(plan.controlled_actors)
                if outsiders:
                    problems.append(f"slot:{claim.slot_id}:{sorted(outsiders)}")
        # Structured dependencies: members must be inside this plan or
        # already controlled by a co-reserved plan.
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
        """Release by rebuilding from the remaining plans."""
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
