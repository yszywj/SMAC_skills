"""G6 LocalFormation: joint position improvement (de-crowding, moderate
spread, role positions). Synchronous moves are evaluated against the
PREDICTED positions of teammates, with soft congestion penalties — a
counter-baneling spread competes against G1 interception instead of
overriding it."""
from typing import Dict, List

from ..context import Context, STOP, distance
from ..contracts import Commitment, EffectClaim, HEURISTIC, ResourceClaims
from .common import h_loops, make_plan, space_claim


class LocalFormationSkill:
    name = "local_formation"

    def __init__(self, config: Dict):
        self.config = config
        lifecycle = config.get("lifecycle", {})
        self.max_h = lifecycle.get("formation_max_h", 3.0)
        self.spread_radius = config.get("local_formation", {}).get(
            "spread_radius", 3.5)
        self.min_pair_distance = config.get("local_formation", {}).get(
            "min_pair_distance", 1.2)

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        splash_threats = [e for e, cap in sorted(context.enemy_capabilities.items())
                          if cap.splash_attack]
        # predicted teammate positions under a SAME-STEP joint adjustment
        predicted = {a: u.position for a, u in context.snapshot.allies.items()
                     if u.alive}
        for actor, unit in sorted(context.snapshot.allies.items()):
            if not unit.alive or context.capabilities[actor].healer:
                continue
            destinations = context.move_destinations(actor)
            if not destinations:
                continue
            crowded = self._crowding(context, actor, predicted)
            under_splash = any(distance(unit.position,
                                        context.snapshot.enemies[e].position)
                               <= self.spread_radius
                               for e in splash_threats)
            if not (crowded >= 1 or under_splash):
                continue
            telemetry.count(self.name, "applicable")
            attackable = context.attackable_enemies(actor)
            # interception competition: a timely joint kill beats scattering;
            # that comparison happens in the coordinator via equal scoring.
            for action in sorted(destinations):
                prediction = context.predict_position(actor, action)
                after = self._crowding_at(context, actor, prediction.position,
                                          predicted)
                new_splash = any(distance(prediction.position,
                                          context.snapshot.enemies[e].position)
                                 <= self.spread_radius for e in splash_threats)
                improves = (after < crowded) or (under_splash and not new_splash)
                if not improves:
                    continue
                plans.append(make_plan(
                    self.name, {actor: action}, "formation", {actor: action},
                    [EffectClaim("position", actor,
                                 float(crowded - after) / 2.0,
                                 max(context.H, 1), HEURISTIC,
                                 "local_position_quality_gain")],
                    ResourceClaims(controls=(actor,),
                                   space=(space_claim(context, actor, action),)),
                    Commitment("adjusting", 0,
                               max_loops=h_loops(context, self.max_h)),
                    {"switch_cost": 0.1 if attackable else 0.0},
                    snapshot_id, fingerprint,
                    rationale=("spread from splash danger" if under_splash
                               else "reduce local crowding")))
        telemetry.count(self.name, "proposed", len(plans))
        return plans

    def _crowding(self, context: Context, actor: str, positions) -> int:
        unit = context.snapshot.allies[actor]
        return sum(1 for other, position in positions.items()
                   if other != actor
                   and distance(unit.position, position) < self.min_pair_distance)

    def _crowding_at(self, context: Context, actor: str, destination,
                     positions) -> int:
        unit = context.snapshot.allies[actor]
        return sum(1 for other, position in positions.items()
                   if other != actor
                   and distance(destination, position) < self.min_pair_distance)
