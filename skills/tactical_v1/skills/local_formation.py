"""G6 LocalFormation: de-crowding, spread and ROLE positions (D5).

Beyond congestion/spread: melee contact positions vs ranged firing
positions, a protected effective fire position for the colossus, and a
support position for the medivac that covers patients without excess
exposure. The rear direction comes from the LOCAL threat distribution, not a
fixed axis. A few two-unit ATOMIC adjustments express "one mover alone
gains nothing". S2 keeps patient pairing; G6 only provides position quality.
"""
from typing import Dict, List, Optional

from ..context import Context, STOP, distance
from ..contracts import (Commitment, Dependency, EffectClaim, HEURISTIC,
                         ResourceClaims, SpaceClaim)
from .common import best_move_toward, h_loops, make_plan, move_progress, space_claim


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

    # -- role targets --------------------------------------------------------
    def _rear_direction(self, context: Context, actor: str) -> tuple:
        """Away from the LOCAL visible threat distribution (multi-direction
        aware), normalised; zero vector when no threat is visible."""
        unit = context.snapshot.allies[actor]
        vx = vy = 0.0
        for enemy_id in context.snapshot.enemies:
            enemy = context.snapshot.enemies[enemy_id]
            dx = unit.position[0] - enemy.position[0]
            dy = unit.position[1] - enemy.position[1]
            norm = math_hypot(dx, dy) or 1.0
            vx += dx / norm
            vy += dy / norm
        return vx, vy

    def _role_goal(self, context: Context, actor: str) -> Optional[tuple]:
        """Soft role position: colossus behind the ground line, medivac over
        the wounded but away from domain-compatible threats; None for others."""
        capability = context.capabilities[actor]
        unit = context.snapshot.allies[actor]
        if capability.specialization == "splash_fire":
            vx, vy = self._rear_direction(context, actor)
            norm = math_hypot(vx, vy)
            if norm < 0.5:
                return None            # encircled: no rear exists
            back = (unit.position[0] + vx / norm * 2.0,
                    unit.position[1] + vy / norm * 2.0)
            front_allies = [u.position for a, u in
                            context.snapshot.allies.items()
                            if a != actor and u.alive and not u.is_flying]
            if front_allies and min(distance(back, p) for p in front_allies) < 1.0:
                return back            # protected spot behind the line
            return None
        if capability.healer:
            wounded = [u for a, u in sorted(context.snapshot.allies.items())
                       if u.alive and not context.capabilities[a].healer
                       and 0 < u.health < u.max_health]
            if not wounded:
                return None
            nearest_wounded = min(wounded, key=lambda u: distance(
                unit.position, u.position))
            risk = context.risk_at(actor, unit.position)
            if risk.threats:
                vx, vy = self._rear_direction(context, actor)
                norm = math_hypot(vx, vy)
                if norm >= 0.5:
                    return (unit.position[0] + vx / norm * 1.0,
                            unit.position[1] + vy / norm * 1.0)
            return nearest_wounded.position
        return None

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        budget = h_loops(context, self.max_h)
        plans: List = []
        splash_threats = [e for e, cap in sorted(context.enemy_capabilities.items())
                          if cap.splash_attack]
        predicted = {a: u.position for a, u in
                     context.snapshot.allies.items() if u.alive}
        atomic_budget = 2
        for actor, unit in sorted(context.snapshot.allies.items()):
            if not unit.alive or context.capabilities[actor].healer:
                continue
            # medivac role goals are produced by S2; G6 skips healers fully
            destinations = context.move_destinations(actor)
            if not destinations:
                continue
            crowded = self._crowding(context, actor, predicted)
            under_splash = any(distance(unit.position,
                                        context.snapshot.enemies[e].position)
                               <= self.spread_radius
                               for e in splash_threats)
            role_goal = self._role_goal(context, actor)
            if not (crowded >= 1 or under_splash or role_goal is not None):
                continue
            telemetry.count(self.name, "applicable")
            attackable = context.attackable_enemies(actor)
            best_action, best_gain = None, 0.0
            for action in sorted(destinations):
                prediction = context.predict_position(actor, action)
                after = self._crowding_at(context, actor, prediction.position,
                                          predicted)
                new_splash = any(distance(prediction.position,
                                          context.snapshot.enemies[e].position)
                                 <= self.spread_radius
                                 for e in splash_threats)
                gain = float(crowded - after)
                if under_splash and not new_splash:
                    gain += 1.0
                if role_goal is not None:
                    gain += min(distance(unit.position, role_goal)
                                - distance(prediction.position, role_goal), 2.0)
                if gain > best_gain:
                    best_action, best_gain = action, gain
            if best_action is not None:
                plans.append(make_plan(
                    self.name, {actor: best_action}, "formation",
                    {actor: best_action},
                    [EffectClaim("position", actor, best_gain / 2.0,
                                 max(context.H, 1), HEURISTIC,
                                 "local_position_quality_gain")],
                    ResourceClaims(controls=(actor,),
                                   space=(space_claim(context, actor,
                                                      best_action),)),
                    Commitment("adjusting", 0, max_loops=budget),
                    {"switch_cost": 0.1 if attackable else 0.0},
                    snapshot_id, fingerprint,
                    rationale=("spread from splash danger" if under_splash
                               else "role/quality position adjustment")))
            # two-unit atomic adjustment: the crowded pair steps apart TOGETHER
            if crowded >= 1 and atomic_budget > 0:
                partner = next((other for other, position in predicted.items()
                                if other != actor
                                and not context.snapshot.allies[other].is_flying
                                and distance(unit.position, position)
                                < self.min_pair_distance), None)
                if partner is not None:
                    atomic_budget -= 1
                    a_move = best_action if best_action is not None else STOP
                    b_move = self._opposite_move(context, partner, actor)
                    if b_move is not None:
                        plans.append(make_plan(
                            self.name, {actor: a_move, partner: b_move},
                            "formation_pair", {actor: a_move, partner: b_move},
                            [EffectClaim("position", actor, 1.0,
                                         max(context.H, 1), HEURISTIC,
                                         "pair_decongestion"),
                             EffectClaim("position", partner, 1.0,
                                         max(context.H, 1), HEURISTIC,
                                         "pair_decongestion")],
                            ResourceClaims(controls=(actor, partner),
                                           space=(space_claim(context, actor,
                                                              a_move),
                                                  space_claim(context, partner,
                                                              b_move))),
                            Commitment("adjusting", 0, max_loops=budget),
                            {}, snapshot_id, fingerprint, atomic=True,
                            dependencies=(Dependency("member", (actor, partner)),),
                            rationale="pair moves apart together"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans

    def _opposite_move(self, context: Context, mover: str, away_from: str):
        destinations = context.move_destinations(mover)
        if not destinations:
            return None
        anchor = context.snapshot.allies[away_from].position
        return max(sorted(destinations),
                   key=lambda a: (distance(destinations[a], anchor), a))

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


def math_hypot(x, y):
    import math
    return math.hypot(x, y)
