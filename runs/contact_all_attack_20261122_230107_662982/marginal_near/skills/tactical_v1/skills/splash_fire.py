"""S1 SplashFire: Colossus splash mechanism provider (registry-injected).

Provides G1's target value/impact model for splash units. G3 declares its
next-shot target and a spatial objective; the unified evaluator reconstructs
future splash opportunities from final positions. The geometry is NOT
calibrated for this environment, so
secondary effects are unknown-magnitude coverage OPPORTUNITIES (named
feature consumed by the unified evaluator's saturation proxy), never
damage amounts and never a kill budget entry.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..context import Context, distance
from ..contracts import EffectClaim, HEURISTIC


@dataclass(frozen=True)
class CoverageEstimate:
    targets: Tuple[str, ...]
    raw_opportunity: float
    discounted_opportunity: float
    reliability: float
    qualified: bool
    evidence: str = HEURISTIC
    calibration_id: Optional[str] = None


class SplashFireSkill:
    name = "splash_fire"

    def __init__(self, config: Dict):
        self.config = config
        params = config.get("splash_fire", {})
        self.coverage_radius = params.get("coverage_radius", 3.0)   # proxy
        self.coverage_weight = params.get("coverage_weight", 0.5)   # proxy
        # Engineering discount, NOT a measured hit probability or calibration.
        # Zero and one permit a clean offline proxy-off/proxy-on comparison.
        self.coverage_reliability = params.get("coverage_reliability", 0.5)

    @property
    def effective_coverage_weight(self):
        return self.coverage_weight * self.coverage_reliability

    def coverage_model(self):
        """Diagnostics shared with the evaluator; no native calibration yet."""
        return {"geometry": "radial_neighbor_proxy", "calibration_id": None,
                "calibration_status": "uncalibrated", "secondary_damage_known": False,
                "reliability_discount": self.coverage_reliability,
                "raw_weight": self.coverage_weight,
                "effective_weight": self.effective_coverage_weight}

    def coverage_estimate(self, context: Context, actor: str, enemy_id: str,
                          position=None, horizon=None) -> CoverageEstimate:
        """Separate opportunity geometry from its conservative scoring weight.

        Current coverage needs a legal, window-qualified primary attack.
        An explicit future position uses the shared prospective range/mask
        query instead. Neither path invents secondary damage or hit timing.
        """
        if position is None:
            qualified = (context.can_issue_attack(actor, enemy_id)
                         and context.fire_estimate(actor, enemy_id,
                             horizon=horizon).window_shots >= 1)
        else:
            opportunity = context.projected_attack_opportunity(
                actor, enemy_id, position, horizon=horizon)
            qualified = opportunity.available and opportunity.ready
        enemy = context.snapshot.enemies.get(enemy_id)
        targets = tuple(other_id for other_id, other in
                        sorted(context.snapshot.enemies.items())
                        if qualified and enemy is not None and other_id != enemy_id
                        and not other.is_flying
                        and distance(other.position, enemy.position) <= self.coverage_radius)
        raw = self.coverage_weight * len(targets)
        return CoverageEstimate(targets, raw, raw * self.coverage_reliability,
                                self.coverage_reliability, qualified)

    def propose(self, context: Context, memory, telemetry, ledger=None):
        return []   # provider-only: G1/G3 generate the actions

    # -- provider interface (called by G1 when injected and enabled) -------
    def impact_claims(self, context: Context, actor: str,
                      enemy_id: str) -> List[EffectClaim]:
        """Primary target: calibrated single-shot damage if the window
        qualifies. Secondary targets: coverage opportunities, amount None."""
        window = max(context.H, 1)
        estimate = context.fire_estimate(actor, enemy_id)
        claims: List[EffectClaim] = []
        if estimate.window_shots >= 1 and estimate.damage_est is not None:
            claims.append(EffectClaim(
                "damage", enemy_id, estimate.damage_est, window,
                estimate.evidence, "splash_primary_window_shot",
                lower=estimate.damage_lo, upper=estimate.damage_hi))
        else:
            claims.append(EffectClaim(
                "damage", enemy_id, None, window, HEURISTIC,
                "attack_command_without_window_damage"))
        for other_id in self.coverage_estimate(context, actor, enemy_id).targets:
            claims.append(EffectClaim(
                "coverage", other_id, None, window, HEURISTIC,
                "uncalibrated_splash_coverage_opportunity"))
        return claims

    def target_candidates(self, context: Context, actor: str) -> List[str]:
        """Diverse retention for the colossus: original/critical/high-coverage/
        nearby targets, ranked with a bounded coverage discount."""
        issuable = [e for e in sorted(context.snapshot.enemies)
                    if context.can_issue_attack(actor, e)]

        def value(enemy_id: str) -> float:
            base = context.snapshot.enemies[enemy_id].effective_health
            coverage = self.coverage_estimate(
                context, actor, enemy_id, horizon=2 * max(context.H, 1))
            return base - min(coverage.discounted_opportunity,
                              4 * self.effective_coverage_weight) * 10

        return sorted(issuable, key=lambda e: (value(e), e))
