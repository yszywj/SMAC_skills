"""The step resource ledger.

Four resource classes are tracked separately (P1.4): action control, fire,
healing and coarse space. A plan's claims are checked and then reserved; a
withdrawn plan releases everything it claimed.
"""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .contracts import CandidatePlan


@dataclass
class Ledger:
    controls: Dict[str, str] = field(default_factory=dict)     # actor -> proposal_id
    fire: Dict[str, float] = field(default_factory=dict)       # enemy -> expected damage
    fire_sources: Dict[str, List[str]] = field(default_factory=dict)
    healing: Dict[str, float] = field(default_factory=dict)    # ally -> expected heal
    heal_sources: Dict[str, List[str]] = field(default_factory=dict)
    space: Dict[str, str] = field(default_factory=dict)        # destination key -> actor
    plans: Dict[str, CandidatePlan] = field(default_factory=dict)

    def clone(self) -> "Ledger":
        return Ledger(controls=dict(self.controls),
                      fire=dict(self.fire),
                      fire_sources={k: list(v) for k, v in self.fire_sources.items()},
                      healing=dict(self.healing),
                      heal_sources={k: list(v) for k, v in self.heal_sources.items()},
                      space=dict(self.space),
                      plans=dict(self.plans))

    # -- conflict checking --------------------------------------------------
    def conflicts(self, plan: CandidatePlan) -> List[str]:
        problems = []
        for actor in plan.controlled_actors:
            if actor in self.controls:
                problems.append(f"control:{actor}:{self.controls[actor]}")
        for enemy_id, amount in plan.claims.fire:
            # Fire is sharable; only the marginal budget is applied later.
            # No conflict here by design.
            pass
        for ally_id, amount in plan.claims.healing:
            pass  # heal capacity depends on the env protocol; not pre-clashed
        for key in plan.claims.space:
            holder = self.space.get(key)
            if holder is not None and holder not in plan.controlled_actors:
                problems.append(f"space:{key}:{holder}")
        return problems

    def reserve(self, plan: CandidatePlan) -> None:
        for actor in plan.controlled_actors:
            self.controls[actor] = plan.proposal_id
        for enemy_id, amount in plan.claims.fire:
            self.fire[enemy_id] = self.fire.get(enemy_id, 0.0) + amount
            self.fire_sources.setdefault(enemy_id, []).append(plan.proposal_id)
        for ally_id, amount in plan.claims.healing:
            self.healing[ally_id] = self.healing.get(ally_id, 0.0) + amount
            self.heal_sources.setdefault(ally_id, []).append(plan.proposal_id)
        for key in plan.claims.space:
            self.space[key] = next((a for a in plan.controlled_actors), "?")
        self.plans[plan.proposal_id] = plan

    def release(self, proposal_id: str) -> None:
        plan = self.plans.pop(proposal_id, None)
        if plan is None:
            return
        for actor in plan.controlled_actors:
            if self.controls.get(actor) == proposal_id:
                del self.controls[actor]
        for enemy_id in list(self.fire_sources):
            sources = self.fire_sources[enemy_id]
            if proposal_id in sources:
                sources.remove(proposal_id)
                plan_amount = next((a for e, a in plan.claims.fire if e == enemy_id), 0.0)
                self.fire[enemy_id] = self.fire.get(enemy_id, 0.0) - plan_amount
                if not sources:
                    del self.fire_sources[enemy_id]
                    self.fire.pop(enemy_id, None)
        for ally_id in list(self.heal_sources):
            sources = self.heal_sources[ally_id]
            if proposal_id in sources:
                sources.remove(proposal_id)
                plan_amount = next((a for e, a in plan.claims.healing if e == ally_id), 0.0)
                self.healing[ally_id] = self.healing.get(ally_id, 0.0) - plan_amount
                if not sources:
                    del self.heal_sources[ally_id]
                    self.healing.pop(ally_id, None)
        for key in list(self.space):
            if self.plans.get(self.space.get(key, ""), None) is None \
                    and self.space.get(key) not in self.controls.values():
                pass  # coarse keys are advisory; keep bookkeeping simple

    def remaining_damage_budget(self, enemy_id: str, effective_health: float) -> float:
        return max(effective_health - self.fire.get(enemy_id, 0.0), 0.0)

    def fire_on(self, enemy_id: str) -> float:
        return self.fire.get(enemy_id, 0.0)
