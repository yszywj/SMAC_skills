"""G3 CombatReposition: retreat, sidestep, pursuit and hold IN the cycle (D3).

Candidates are classified by GEOMETRY against the target/threat directions
(pursuit = closing on the target, retreat = growing min-threat distance,
sidestep = neither), so each class is genuinely non-empty. The lost window
shot is already priced by the physical scoring (attack action earns its
damage, the move does not); it is NOT charged a second time as switch cost.
"""
from typing import Dict, List, Optional

from ..context import Context, STOP, distance
from ..contracts import Commitment, EffectClaim, HEURISTIC, ResourceClaims
from .common import best_move_toward, h_loops, make_plan, move_progress, space_claim


class CombatRepositionSkill:
    name = "combat_reposition"

    def __init__(self, config: Dict, providers: Optional[Dict] = None):
        self.config = config
        self.providers = providers or {}
        self.max_candidates = config.get("search", {}).get(
            "max_candidates_per_unit", 6)

    def _classify(self, context: Context, actor: str, action: int,
                  target_id: str, threats) -> str:
        unit = context.snapshot.allies[actor]
        enemy = context.snapshot.enemies[target_id]
        destination = context.move_destinations(actor).get(action)
        if destination is None:
            return "hold"
        to_target = distance(unit.position, enemy.position) - \
            distance(destination, enemy.position)
        if to_target > 0.15:
            return "pursuit"
        threat_gains = [distance(context.snapshot.enemies[e].position,
                                 destination) -
                        distance(context.snapshot.enemies[e].position,
                                 unit.position) for e in threats]
        if threat_gains and min(threat_gains) > 0.15:
            return "retreat"
        return "sidestep"

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
            risk_now = context.risk_at(actor, unit.position)
            under_pressure = (unit.health_lost + unit.shield_lost > 0
                              and risk_now.threats) or risk_now.effect_weapon_near
            cooldown_window = unit.weapon_cooldown_loops > context.H
            melee = capability.combat_class == "melee"
            # full-health ranged may still reposition inside a valuable
            # cooldown window; melee only when contact can be kept
            worth_moving = cooldown_window or not shot_available or \
                (under_pressure and not melee) or \
                (under_pressure and melee and unit.health_fraction < 0.5)
            if not worth_moving:
                continue
            target_id = attackable[0]
            enemy = context.snapshot.enemies[target_id]
            # one candidate per class, classified by geometry (D3)
            by_class: Dict[str, int] = {}
            for action in sorted(context.move_destinations(actor)):
                label = self._classify(context, actor, action, target_id,
                                       risk_now.threats)
                if label not in by_class:
                    by_class[label] = action
            provider = self.providers.get("splash_fire")
            budget = h_loops(context, 8.0)
            relation = context.snapshot.relations[actor][target_id]
            r_max = relation.native_weapon_range or 5.0
            # keep one candidate per class, but never fewer than the plain
            # four directions for the unified evaluator to compare (2.3)
            seen_actions = set()
            ordered = sorted(by_class.items())
            for label, move in ordered:
                seen_actions.add(move)
            for action in sorted(context.move_destinations(actor)):
                if action not in seen_actions and len(ordered) < \
                        self.max_candidates:
                    ordered.append((by_class.get(action, "sidestep"), action))
                    seen_actions.add(action)
            for label, move in ordered:
                prediction = context.predict_position(actor, move)
                if melee and label != "pursuit":
                    after = distance(prediction.position, enemy.position)
                    relation = context.snapshot.relations[actor][target_id]
                    if after > (relation.native_weapon_range or 2.0) + 1.0:
                        continue      # leaving the contact face is a G4 matter
                coverage = 0.0
                if capability.specialization == "splash_fire" \
                        and provider is not None:
                    coverage = min(provider.reposition_coverage(
                        context, actor, move, target_id) / 4.0, 1.0)
                instance = memory.instance_for(self.name, actor)
                plans.append(make_plan(
                    self.name, {actor: move}, target_id, {actor: move},
                    [EffectClaim("position", actor,
                                 move_progress(context, actor, move,
                                               enemy.position),
                                 max(context.H, 1), HEURISTIC,
                                 f"reposition_{label}")],
                    ResourceClaims(controls=(actor,),
                                   space=(space_claim(context, actor, move),)),
                    Commitment("repositioning", 0, max_loops=budget,
                               interrupt_on=("weapon_ready", "danger_change")),
                    # the lost window shot is priced by physics only
                    {"switch_cost": 0.0},
                    snapshot_id, fingerprint,
                    provider="splash_fire"
                    if capability.specialization == "splash_fire" and provider
                    else None,
                    instance_id=instance.instance_id if instance else None,
                    memory_delta=({"kind": "create", "skill": self.name,
                                   "actors": [actor], "goal": target_id,
                                   "goal_kind": "enemy",
                                   "phase": "repositioning",
                                   "max_loops": budget}
                                  if instance is None else None),
                    goal_kind="enemy",
                    goal_data={"target_id": target_id,
                               "objective": {
                                   "kind": "range_band", "target": target_id,
                                   "r_min": max(r_max - 1.5, 0.0)
                                   if not melee else 0.0,
                                   "r_max": r_max if not melee else 2.0}},
                    rationale=f"{label} during weapon cycle"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans
