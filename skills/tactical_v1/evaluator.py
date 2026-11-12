"""Unified full-plan evaluation (A3/B5).

score_joint reads the FINAL action list, rebuilds physical effects, and
scores them on shared normalized scales. Per-source damage aggregation: a
cooling or unknown source never cancels another source's known
contribution. Intent value (signed progress toward the task goal) is a
separate capped term; physical value never depends on the skill label.
"""
from typing import Dict, List, Optional, Tuple

from .context import Context, distance
from .contracts import CandidatePlan
from .effects import EffectsReport, physical_effects
from .ledger import Ledger

DEFAULT_WEIGHTS = {
    "useful_damage": 3.0,   # per target-fraction of effective health removed
    "kill": 2.0,            # once per target, conservative feasibility only
    "heal": 2.0,            # per bounded heal OPPORTUNITY (uncalibrated proxy)
    "coverage": 1.0,        # saturated opportunity proxy per enemy
    "progress": 0.6,        # signed intent progress toward the task goal
    "risk": 0.8,            # threats able to reach the FINAL position
    "congestion": 0.2,      # predicted overlaps between same-class movers
    "switch": 0.3,          # abandoning a live instance task
    "reentry_cost": 0.2,
    "planned_loss": 1.5,    # planned in-contact self-consumption, once
    "hold_value": 0.3,      # bounded wait/hold intent value
}

RISK_NORM = 3.0


class Evaluator:
    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 mechanisms: Optional[Dict[str, bool]] = None,
                 splash_radius: float = 3.0, splash_weight: float = 0.5,
                 burst_radius: float = 2.5, burst_weight: float = 0.5,
                 burst_self_cost: float = 1.0):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.mechanisms = mechanisms or {"splash": True, "burst": True}
        self.effect_params = dict(splash_radius=splash_radius,
                                  splash_weight=splash_weight,
                                  burst_radius=burst_radius,
                                  burst_weight=burst_weight,
                                  burst_self_cost=burst_self_cost)

    def effects(self, context: Context,
                actions: Dict[str, int]) -> EffectsReport:
        return physical_effects(context, actions, self.mechanisms,
                                **self.effect_params)

    def score_joint(self, context: Context, plans: List[CandidatePlan],
                    actions: Dict[str, int],
                    memory=None) -> Tuple[float, Dict[str, float]]:
        w = self.weights
        snapshot = context.snapshot
        report = self.effects(context, actions)
        breakdown: Dict[str, float] = {}

        # -- useful damage & kills: aggregate KNOWN sources per target ----
        damage_score = kill_score = 0.0
        for enemy_id, total in report.damage_known.items():
            enemy = snapshot.enemies.get(enemy_id)
            if enemy is None:
                continue
            useful = min(total, enemy.effective_health)
            damage_score += w["useful_damage"] * useful / max(enemy.effective_health, 1.0)
            if total >= enemy.effective_health:
                kill_score += w["kill"]      # EXPECTED kill, never a real one
        breakdown["useful_damage"] = damage_score
        breakdown["expected_kills"] = kill_score
        breakdown["unknown_damage_sources"] = float(sum(
            len(v) for v in report.unknown_sources.values()))

        # -- heal: bounded opportunity proxy (need x sustain x urgency) -----
        heal_score = 0.0
        for opportunity in report.heal.values():
            heal_score += w["heal"] * min(opportunity.opportunity, 1.0)
        breakdown["heal"] = heal_score

        # -- coverage opportunity (window-qualified, saturating) ------------
        breakdown["coverage"] = w["coverage"] * sum(report.coverage.values())
        breakdown["coverage_sources"] = float(sum(
            len(v) for v in report.coverage_sources.values()))

        # -- risk at FINAL positions ----------------------------------------
        risk_score = 0.0
        for actor, (position, _certain) in report.positions.items():
            if actor not in snapshot.allies or not snapshot.allies[actor].alive:
                continue
            risk = context.risk_at(actor, position)
            normalized = min(len(risk.threats) / RISK_NORM, 1.0)
            if risk.effect_weapon_near:
                normalized = min(normalized + 0.3, 1.0)
            risk_score += w["risk"] * normalized
        breakdown["risk"] = -risk_score

        # -- congestion between predicted same-class positions --------------
        congestion = 0.0
        movers = [a for a in report.positions
                  if a in snapshot.allies and snapshot.allies[a].alive]
        for i, a in enumerate(movers):
            for b in movers[i + 1:]:
                ua, ub = snapshot.allies[a], snapshot.allies[b]
                if ua.is_flying != ub.is_flying:
                    continue
                if distance(report.positions[a][0], report.positions[b][0]) \
                        < (ua.radius + ub.radius):
                    congestion += 1.0
        breakdown["congestion"] = -w["congestion"] * min(congestion, 4.0)

        # -- intent terms: signed, capped, aggregated per TASK ---------------
        progress = hold = switch = reentry = 0.0
        for plan in plans:
            progress += w["progress"] * _cap(plan.features.get("progress", 0.0))
            hold += w["hold_value"] * _cap(plan.features.get("hold_value", 0.0))
            switch += w["switch"] * _cap(plan.features.get("switch_cost", 0.0))
            reentry += w["reentry_cost"] * _cap(plan.features.get("reentry_cost", 0.0))
        breakdown["progress"] = progress
        breakdown["hold_value"] = hold
        breakdown["switch"] = -switch
        breakdown["reentry_cost"] = -reentry
        planned = sum(report.planned_loss.values()) * w["planned_loss"]
        breakdown["planned_loss"] = -planned

        total = (damage_score + kill_score + heal_score + breakdown["coverage"]
                 + progress + hold - risk_score + breakdown["congestion"]
                 - switch - reentry - planned)
        breakdown["total"] = total
        return total, breakdown


def _cap(value: Optional[float], limit: float = 1.0) -> float:
    if value is None:
        return 0.0
    return max(min(float(value), limit), -limit)
