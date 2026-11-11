"""S1 SplashFire: Colossus splash mechanism provider (registry-injected).

Replaces G1's target value/impact model AND G3's attack scoring for splash
units. The splash geometry is NOT calibrated for this environment, so
secondary effects are unknown-magnitude coverage OPPORTUNITIES (named
feature consumed by the unified evaluator's saturation proxy), never
damage amounts and never a kill budget entry.
"""
from typing import Dict, List

from ..context import Context, distance
from ..contracts import CALIBRATED, EffectClaim, HEURISTIC


class SplashFireSkill:
    name = "splash_fire"

    def __init__(self, config: Dict):
        self.config = config
        params = config.get("splash_fire", {})
        self.coverage_radius = params.get("coverage_radius", 3.0)   # proxy
        self.coverage_weight = params.get("coverage_weight", 0.5)   # proxy

    def propose(self, context: Context, memory, telemetry, ledger=None):
        return []   # provider-only: G1/G3 generate the actions

    # -- provider interface (called by G1/G3 when injected and enabled) ----
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
        enemy = context.snapshot.enemies[enemy_id]
        for other_id, other in sorted(context.snapshot.enemies.items()):
            if other_id == enemy_id:
                continue
            if distance(other.position, enemy.position) <= self.coverage_radius:
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
            coverage = sum(1 for o in context.snapshot.enemies
                           if o != enemy_id and distance(
                               context.snapshot.enemies[o].position,
                               context.snapshot.enemies[enemy_id].position)
                           <= self.coverage_radius)
            return base - self.coverage_weight * min(coverage, 4) * 10

        return sorted(issuable, key=lambda e: (value(e), e))

    def reposition_coverage(self, context: Context, actor: str, action: int,
                            target_id: str) -> float:
        """G3 hook: bounded coverage value of a candidate reposition (the
        count of enemies near the primary target from the new position)."""
        prediction = context.predict_position(actor, action)
        target = context.snapshot.enemies.get(target_id)
        if target is None:
            return 0.0
        return sum(1 for o in context.snapshot.enemies.values()
                   if distance(o.position, prediction.position)
                   <= self.coverage_radius + 3.0)
