"""G3 CombatReposition: retreat, sidestep, pursuit and hold IN the cycle.

One comparison over several move classes plus keep-attacking (via G1). The
movement decision compares the LOST window output (only when a shot could
actually land) against the risk/congestion change at the predicted
position. Colossus runs the same flow with S1 impact/coverage hooks.
"""
from typing import Dict, List, Optional

from ..context import Context, STOP, distance
from ..contracts import Commitment, EffectClaim, HEURISTIC, ResourceClaims
from .common import best_move_away, best_move_toward, make_plan, move_progress, space_claim


class CombatRepositionSkill:
    name = "combat_reposition"

    def __init__(self, config: Dict, providers: Optional[Dict] = None):
        self.config = config
        self.providers = providers or {}
        self.max_candidates = config.get("search", {}).get(
            "max_candidates_per_unit", 6)

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        if not context.snapshot.enemies:
            return plans
        for actor, unit in sorted(context.snapshot.allies.items()):
            if not unit.alive:
                continue
            capability = context.capabilities[actor]
            if capability.healer or capability.suicide_attack:
                continue
            attackable = context.attackable_enemies(actor)
            if not attackable:
                continue                      # G2 owns out-of-contact units
            telemetry.count(self.name, "applicable")
            estimate = context.fire_estimate(actor, attackable[0])
            shot_available = estimate.window_shots >= 1
            # Opportunity actually lost by moving: a real window shot only
            # (cooldown > H or out of range => nothing lost this window).
            lost = estimate.damage_est if shot_available else 0.0
            lost_norm = min((lost or 0.0) /
                            max(context.snapshot.enemies[attackable[0]].max_health, 1.0), 1.0)
            risk_now = context.risk_at(actor, unit.position)
            under_fire = (unit.health_lost + unit.shield_lost > 0
                          and risk_now.threats) or risk_now.effect_weapon_near
            cooldown_long = unit.weapon_cooldown_loops > context.H
            melee = capability.combat_class == "melee"
            worth_moving = (cooldown_long or not shot_available) or \
                           (under_fire and (melee is False or
                                            unit.health_fraction < 0.5))
            if not worth_moving:
                continue
            target_id = attackable[0]
            enemy = context.snapshot.enemies[target_id]
            # candidate classes: retreat, sidestep, pursuit
            classes: List[tuple] = []
            retreat = best_move_away(context, actor, list(risk_now.threats))
            if retreat is not None:
                classes.append(("retreat", retreat))
            for action in sorted(context.move_destinations(actor)):
                if action not in (c[1] for c in classes):
                    classes.append(("sidestep", action))
            pursuit = best_move_toward(context, actor, enemy.position)
            if pursuit is not None and pursuit not in (c[1] for c in classes):
                classes.append(("pursuit", pursuit))
            provider = self.providers.get("splash_fire")
            for label, move in classes[:self.max_candidates]:
                prediction = context.predict_position(actor, move)
                risk_after = context.risk_at(actor, prediction.position)
                # melee: leaving the contact face must be paid for explicitly
                if melee:
                    after = distance(prediction.position, enemy.position)
                    relation = context.snapshot.relations[actor][target_id]
                    if after > (relation.native_weapon_range or 2.0) + 1.0:
                        continue      # keeps contact or it is a G4 decision
                coverage = 0.0
                if capability.specialization == "splash_fire" \
                        and provider is not None:
                    coverage = min(provider.reposition_coverage(
                        context, actor, move, target_id) / 4.0, 1.0)
                plans.append(make_plan(
                    self.name, {actor: move}, target_id, {actor: move},
                    [EffectClaim("position", actor,
                                 move_progress(context, actor, move, enemy.position),
                                 max(context.H, 1), HEURISTIC,
                                 f"reposition_{label}")],
                    ResourceClaims(controls=(actor,),
                                   space=(space_claim(context, actor, move),)),
                    Commitment("repositioning", 0, max_loops=8 * max(context.H, 1),
                               interrupt_on=("weapon_ready", "danger_change")),
                    {"switch_cost": lost_norm * 0.5,   # bounded lost-output cost
                     "progress": coverage * 0.3},
                    snapshot_id, fingerprint,
                    provider="splash_fire" if capability.specialization == "splash_fire"
                    and provider else None,
                    rationale=f"{label} during weapon cycle"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans
