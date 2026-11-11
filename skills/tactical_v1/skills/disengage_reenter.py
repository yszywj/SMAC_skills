"""G4 DisengageReenter: bounded disengagement and re-entry.

Phases: retreat -> stabilize -> reenter, each with progress conditions,
durations and exits. The skill controls ONLY the retreating unit; covering
fire comes from G1/G3 plans coordinated by Adaptive. Re-entry does not
require full health, and without any heal capability the unit may not wait
for healing forever.
"""
from typing import Dict, List

from ..context import Context, STOP, distance
from ..contracts import Commitment, EffectClaim, HEURISTIC, ResourceClaims
from .common import best_move_away, make_plan, move_progress


class DisengageReenterSkill:
    name = "disengage_reenter"

    def __init__(self, config: Dict):
        self.config = config
        params = config.get("disengage_reenter", {})
        self.critical_fraction = params.get("critical_health_fraction", 0.35)
        self.reenter_fraction = params.get("reenter_health_fraction", 0.6)
        self.max_retreat_steps = params.get("max_retreat_steps", 10)
        self.max_stabilize_steps = params.get("max_stabilize_steps", 20)

    def _critical(self, context: Context, actor: str) -> bool:
        unit = context.snapshot.allies[actor]
        capability = context.capabilities[actor]
        if capability.suicide_attack:
            # S3 owns the exchange judgement for suicide units.
            return False
        recent_loss = unit.health_lost + unit.shield_lost > 0
        threatened = bool(context.threats_to(actor))
        return unit.health_fraction <= self.critical_fraction \
            and threatened and (recent_loss or unit.health_fraction <= 0.2)

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        healers_present = any(cap.healer for cap in context.capabilities.values())
        for actor, unit in sorted(context.snapshot.allies.items()):
            if not unit.alive or context.capabilities[actor].healer:
                continue
            instance = memory.instance_for(self.name, actor)
            if instance is not None and instance.phase == "retreat":
                telemetry.count(self.name, "continued")
                # Retreat continues only while it still makes progress or the
                # budget lasts; then the unit must stabilize or re-enter.
                threats = context.threats_to(actor)
                move = best_move_away(context, actor, threats) if threats else None
                progressed = memory.displacement(actor, 3 * context.snapshot.step_mul,
                                                 context.snapshot) > 0.5
                if move is not None and progressed \
                        and instance.budget_steps < self.max_retreat_steps:
                    allies = [u.position for a, u in sorted(context.snapshot.allies.items())
                              if a != actor and u.alive]
                    target = (min(allies, key=lambda p: distance(
                        context.snapshot.allies[actor].position, p))
                        if allies else context.snapshot.allies[actor].position)
                    plans.append(make_plan(
                        self.name, {actor: move}, "safe", {actor: move},
                        [EffectClaim("position", actor,
                                     move_progress(context, actor, move, target),
                                     max(context.snapshot.step_mul, 1), HEURISTIC,
                                     "retreat_progress", 0.7),
                         EffectClaim("exposure", actor, 2.0 * len(threats),
                                     max(context.snapshot.step_mul, 1), HEURISTIC,
                                     "leaving_threat_range", 0.6)],
                        ResourceClaims(controls=(actor,), space=(f"{actor}@{move}",)),
                        Commitment("retreat", 0, self.max_retreat_steps,
                                   interrupt_on=("illegal", "safe")),
                        {"move_opportunity_cost": 0.5}, snapshot_id, fingerprint,
                        rationale="continue bounded retreat"))
                    continue
                # Stabilize: hold at a safe spot until re-entry conditions.
                can_reenter = (unit.health_fraction >= self.reenter_fraction
                               or (not healers_present
                                   and not context.threats_to(actor))
                               or instance.budget_steps >= self.max_stabilize_steps)
                if can_reenter:
                    telemetry.count(self.name, "exited")
                    # No retreat/stabilize candidate: the unit falls back to
                    # G1/G2 candidates and the coordinator retires this
                    # instance when the selected skill changes (commit-time).
                else:
                    plans.append(make_plan(
                        self.name, {actor: STOP}, "stabilize", {actor: STOP},
                        [EffectClaim("position", actor, 0.0,
                                     max(context.snapshot.step_mul, 1), HEURISTIC,
                                     "stabilize_hold", 0.6)],
                        ResourceClaims(controls=(actor,)),
                        Commitment("stabilize", 0, self.max_stabilize_steps,
                                   interrupt_on=("danger_change", "healed")),
                        {"raw_bonus": 0.0}, snapshot_id, fingerprint,
                        rationale="stabilize before re-entry"))
                    continue
            if self._critical(context, actor):
                telemetry.count(self.name, "applicable")
                threats = context.threats_to(actor)
                move = best_move_away(context, actor, threats) if threats else None
                if move is not None:
                    plans.append(make_plan(
                        self.name, {actor: move}, "safe", {actor: move},
                        [EffectClaim("exposure", actor, 2.0 * len(threats),
                                     max(context.snapshot.step_mul, 1), HEURISTIC,
                                     "leaving_threat_range", 0.6)],
                        ResourceClaims(controls=(actor,), space=(f"{actor}@{move}",)),
                        Commitment("retreat", 0, self.max_retreat_steps,
                                   interrupt_on=("illegal", "safe")),
                        {"move_opportunity_cost": 0.5}, snapshot_id, fingerprint,
                        rationale=f"critical ({unit.health_fraction:.0%}); disengage"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans
