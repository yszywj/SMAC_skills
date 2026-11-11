"""Unified rule-based plan evaluation (P4.3).

One evaluation time window: the current control window (the loops until this
unit could next act meaningfully, approximated by step_mul). All feature
terms are normalized by the weights below; heuristic effects are discounted
by their confidence and never enter "certain kill" accounting. Damage on the
same target is credited marginally against the remaining effective-health
budget, so overlapping plans cannot double-count.
"""
from typing import Dict, List, Optional

from .context import Context
from .contracts import CALIBRATED, CandidatePlan, HEURISTIC, VERIFIED
from .ledger import Ledger

DEFAULT_WEIGHTS = {
    "damage": 1.0,
    "kill": 6.0,
    "heal": 0.9,
    "position_progress": 0.35,
    "exposure_relief": 0.5,
    "exposure_cost": 0.45,
    "expected_loss": 0.8,
    "move_opportunity_cost": 0.3,
    "switch_cost": 0.25,
    "commitment_break_cost": 0.4,
    "suicide_cost": 1.2,
    "support_follow": 0.2,
    "attack_intent": 0.3,   # verified-legal attack without a damage estimate
    "overkill_credit": 0.25,  # diminishing insurance value of redundant fire
}

EVIDENCE_DISCOUNT = {VERIFIED: 1.0, CALIBRATED: 0.85, HEURISTIC: 0.4}


class Evaluator:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update({k: float(v) for k, v in weights.items()})

    def score_plan(self, plan: CandidatePlan, context: Context,
                   ledger: Ledger) -> float:
        """Score one plan against the CURRENT ledger (marginal credit)."""
        w = self.weights
        score = 0.0
        snapshot = context.snapshot
        for effect in plan.expected_effects:
            discount = EVIDENCE_DISCOUNT.get(effect.evidence, 0.4) * max(min(effect.confidence, 1.0), 0.0)
            amount = effect.amount
            if effect.kind == "damage":
                enemy = snapshot.enemies.get(effect.target)
                budget = (ledger.remaining_damage_budget(effect.target, enemy.effective_health)
                          if enemy else amount)
                in_budget = min(amount, budget)
                # Redundant fire beyond the kill budget still has diminishing
                # insurance value (estimates can be wrong); it is credited at
                # a discount, never as additional certain damage.
                overkill = max(amount - in_budget, 0.0)
                credited = in_budget + w["overkill_credit"] * overkill
                credited *= discount if budget > 0 else discount * 0.5
                score += w["damage"] * credited
                if enemy and budget > 0 and amount >= budget:
                    score += w["kill"] * discount  # this plan tips the target over
            elif effect.kind == "heal":
                ally = snapshot.allies.get(effect.target)
                if ally is not None:
                    missing = ally.max_health - ally.health
                    already = ledger.healing.get(effect.target, 0.0)
                    credited = max(min(amount, max(missing - already, 0.0)), 0.0)
                    score += w["heal"] * credited * discount
            elif effect.kind == "position":
                score += w["position_progress"] * amount * discount
            elif effect.kind == "exposure":
                score += (w["exposure_relief"] * amount if amount > 0
                          else -w["exposure_cost"] * -amount) * discount
            elif effect.kind == "loss":
                score -= w["expected_loss"] * amount * discount
            elif effect.kind == "suicide":
                score -= w["suicide_cost"] * amount * discount
        for key, value in plan.features.items():
            if key == "move_opportunity_cost":
                score -= w["move_opportunity_cost"] * value
            elif key == "switch_cost":
                score -= w["switch_cost"] * value
            elif key == "commitment_break_cost":
                score -= w["commitment_break_cost"] * value
            elif key == "support_follow":
                score += w["support_follow"] * value
            elif key == "attack_intent":
                score += w["attack_intent"] * value
            elif key == "raw_bonus":
                score += value
        return score

    def score_joint(self, plans: List[CandidatePlan], context: Context) -> float:
        """Re-score a COMPLETE joint plan from scratch (P4.4: no adding of
        self-reported subtotals). The ledger is rebuilt in deterministic
        candidate order so marginal credit is reproducible."""
        ledger = Ledger()
        total = 0.0
        for plan in sorted(plans, key=lambda p: p.proposal_id):
            total += self.score_plan(plan, context, ledger)
            ledger.reserve(plan)
        return total
