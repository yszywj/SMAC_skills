"""G6 LocalFormation: local relative-position improvement.

Configurable spacing, spread against enemy splash threat and melee envelop
positions. Distinct from G2 (which establishes contact) and G5 (which repairs
support breaks): this skill only improves local stance, and its candidates
always pay the opportunity cost of the attack they would give up — never a
last unconditional override.
"""
import math
from typing import Dict, List

from ..context import Context, STOP, distance
from ..contracts import Commitment, EffectClaim, HEURISTIC, ResourceClaims
from .common import make_plan, threats_reaching


class LocalFormationSkill:
    name = "local_formation"

    def __init__(self, config: Dict):
        self.config = config
        params = config.get("local_formation", {})
        self.spread_radius = params.get("spread_radius", 3.5)
        self.min_pair_distance = params.get("min_pair_distance", 1.2)

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        splash_threats = [e for e, cap in sorted(context.enemy_capabilities.items())
                          if cap.splash_attack]
        for actor, unit in sorted(context.snapshot.allies.items()):
            if not unit.alive or context.capabilities[actor].healer:
                continue
            attackable = context.attackable_enemies(actor)
            opportunity = 0.0
            if attackable:
                damage, _, _ = context.expected_damage(actor, attackable[0])
                opportunity = damage
            destinations = context.move_destinations(actor)
            if not destinations:
                continue
            # 1. Spread against an observed enemy splash threat.
            if splash_threats:
                telemetry.count(self.name, "applicable")
                crowded = self._crowding(context, actor)
                best = None
                for action in sorted(destinations):
                    destination = destinations[action]
                    near_threat = any(distance(destination,
                                               context.snapshot.enemies[e].position)
                                      <= self.spread_radius
                                      for e in splash_threats)
                    if near_threat:
                        continue
                    score = (self._crowding_at(context, actor, destination), action)
                    if best is None or score < best:
                        best = score
                if best is not None and crowded > 0:
                    action = best[1]
                    endangered = 1 + self._crowding(context, actor)  # self + packed neighbours
                    plans.append(make_plan(
                        self.name, {actor: action}, "spread", {actor: action},
                        [EffectClaim("exposure", actor, endangered * 3.0,
                                     max(context.snapshot.step_mul, 1), HEURISTIC,
                                     "leaving_splash_threat_area", 0.6)],
                        ResourceClaims(controls=(actor,),
                                       space=(f"{actor}@{action}",)),
                        Commitment("spreading", 0, 3,
                                   interrupt_on=("illegal", "danger_change")),
                        {"move_opportunity_cost": opportunity},
                        snapshot_id, fingerprint,
                        rationale="spread from enemy splash threat"))
                    continue
            # 2. De-crowding when stacked on a teammate while in contact.
            if attackable and self._crowding(context, actor) >= 2:
                telemetry.count(self.name, "applicable")
                best = min(sorted(destinations),
                           key=lambda a: (self._crowding_at(
                               context, actor, destinations[a]), a))
                if self._crowding_at(context, actor, destinations[best]) \
                        < self._crowding(context, actor):
                    plans.append(make_plan(
                        self.name, {actor: best}, "spacing", {actor: best},
                        [EffectClaim("position", actor, 0.3,
                                     max(context.snapshot.step_mul, 1), HEURISTIC,
                                     "local_spacing_improvement", 0.5)],
                        ResourceClaims(controls=(actor,),
                                       space=(f"{actor}@{best}",)),
                        Commitment("spacing", 0, 2, interrupt_on=("illegal",)),
                        {"move_opportunity_cost": opportunity},
                        snapshot_id, fingerprint,
                        rationale="reduce local crowding while in contact"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans

    def _crowding(self, context: Context, actor: str) -> int:
        unit = context.snapshot.allies[actor]
        return sum(1 for other, u in context.snapshot.allies.items()
                   if other != actor and u.alive
                   and distance(unit.position, u.position) < self.min_pair_distance)

    def _crowding_at(self, context: Context, actor: str, destination) -> int:
        return sum(1 for other, u in context.snapshot.allies.items()
                   if other != actor and u.alive
                   and distance(destination, u.position) < self.min_pair_distance)
