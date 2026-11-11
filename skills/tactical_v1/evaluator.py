"""Unified full-plan evaluation (A3).

score_joint reads the FINAL action list, rebuilds physical effects through
effects.physical_effects and scores them on shared normalized scales. A
skill label never changes the physics; intent-dependent value (progress
toward a task goal) is a separate, capped term. No raw_bonus channels.
"""
from typing import Dict, List, Optional, Tuple

from .context import Context, distance
from .contracts import CALIBRATED, CandidatePlan, HEURISTIC
from .effects import EffectsReport, physical_effects
from .ledger import Ledger

DEFAULT_WEIGHTS = {
    "useful_damage": 3.0,   # per target-fraction of effective health removed
    "kill": 2.0,            # once per target, conservative feasibility only
    "heal": 1.5,            # per patient-fraction of missing health restored
    "coverage": 1.0,        # saturated opportunity proxy per enemy
    "progress": 0.6,        # capped intent progress toward a task goal
    "risk": 0.8,            # threats able to reach the FINAL position
    "congestion": 0.2,      # predicted overlaps between same-class movers
    "switch": 0.3,          # abandoning a live instance task
    "reentry_cost": 0.2,    # later re-entry value, capped
    "planned_loss": 1.5,    # planned self-consumption (once)
    "hold_value": 0.3,      # bounded wait/hold intent value
}

RISK_NORM = 3.0        # risk counts are normalized by this cap
PROGRESS_MOVE_CAP = 2.0  # progress normalized against this distance scale


class Evaluator:
    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 coverage_radius: float = 3.0):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.coverage_radius = coverage_radius

    def effects(self, context: Context,
                actions: Dict[str, int]) -> EffectsReport:
        return physical_effects(context, actions, self.coverage_radius)

    def score_joint(self, context: Context, plans: List[CandidatePlan],
                    actions: Dict[str, int],
                    memory=None) -> Tuple[float, Dict[str, float]]:
        """Score one COMPLETE joint plan from its final actions."""
        w = self.weights
        snapshot = context.snapshot
        report = self.effects(context, actions)
        breakdown: Dict[str, float] = {}

        # -- useful damage & kills: aggregate per target, then cap ------
        damage_score = kill_score = 0.0
        for enemy_id, total in report.damage.items():
            enemy = snapshot.enemies.get(enemy_id)
            if enemy is None or not report.damage_known.get(enemy_id, False):
                continue
            useful = min(total, enemy.effective_health)
            damage_score += w["useful_damage"] * useful / max(enemy.effective_health, 1.0)
            if total >= enemy.effective_health:
                # conservative kill feasibility: window-qualified damage is
                # sufficient — reported as EXPECTED, never as a real kill
                kill_score += w["kill"]
        breakdown["useful_damage"] = damage_score
        breakdown["expected_kills"] = kill_score

        # -- healing: per patient, capped by real need -------------------
        heal_score = 0.0
        for patient, amount in report.heal.items():
            target = snapshot.allies.get(patient)
            if target is None:
                continue
            missing = max(target.max_health - target.health, 1.0)
            fraction = amount / missing
            if report.heal_blocked_by_move.get(patient):
                fraction *= 0.5    # moving patient interrupts sustained healing
            heal_score += w["heal"] * fraction
        breakdown["heal"] = heal_score

        # -- coverage opportunity (saturating proxy) ----------------------
        breakdown["coverage"] = w["coverage"] * sum(report.coverage.values())

        # -- risk at FINAL positions (same standard for everyone) ---------
        risk_score = 0.0
        for actor, (position, _certain) in report.positions.items():
            if actor not in snapshot.allies or not snapshot.allies[actor].alive:
                continue
            risk = context.risk_at(actor, position)
            normalized = min(len(risk.threats) / RISK_NORM, 1.0)
            if risk.effect_weapon_near:
                normalized = min(normalized + 0.3, 1.0)   # proximity danger proxy
            risk_score += w["risk"] * normalized
        breakdown["risk"] = -risk_score

        # -- congestion between predicted same-class positions ------------
        congestion = 0.0
        movers = [a for a in report.positions
                  if a in snapshot.allies and snapshot.allies[a].alive]
        for i, a in enumerate(movers):
            for b in movers[i + 1:]:
                ua, ub = snapshot.allies[a], snapshot.allies[b]
                if ua.is_flying != ub.is_flying:
                    continue   # different collision classes may overlap
                pa, pb = report.positions[a][0], report.positions[b][0]
                if distance(pa, pb) < (ua.radius + ub.radius):
                    congestion += 1.0
        breakdown["congestion"] = -w["congestion"] * min(congestion, 4.0)

        # -- intent terms: capped, labeled, never physics ------------------
        progress = hold = switch = reentry = planned = 0.0
        for plan in plans:
            progress += w["progress"] * _cap(plan.features.get("progress", 0.0), 1.0)
            hold += w["hold_value"] * _cap(plan.features.get("hold_value", 0.0), 1.0)
            switch += w["switch"] * _cap(plan.features.get("switch_cost", 0.0), 1.0)
            reentry += w["reentry_cost"] * _cap(plan.features.get("reentry_cost", 0.0), 1.0)
        for actor in report.planned_loss:
            planned += w["planned_loss"] * report.planned_loss[actor]
        breakdown["progress"] = progress
        breakdown["hold_value"] = hold
        breakdown["switch"] = -switch
        breakdown["reentry_cost"] = -reentry
        breakdown["planned_loss"] = -planned

        total = (damage_score + kill_score + heal_score + breakdown["coverage"]
                 + progress + hold - risk_score
                 + breakdown["congestion"] - switch - reentry - planned)
        breakdown["total"] = total
        return total, breakdown


def _cap(value: float, limit: float) -> float:
    if value is None:
        return 0.0
    return max(min(float(value), limit), -limit)
