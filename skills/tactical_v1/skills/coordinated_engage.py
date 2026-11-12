"""G2 CoordinatedEngage: establish effective shared contact (D2).

A legal attack command is NOT effective contact: units whose command target
is known out of physical range still get engage candidates. Real blockage is
detected from OBSERVATION history (same-target command + no distance or
contact improvement over the no-progress window), never from an unreachable
instance phase. Melee contact slots are shared hard exclusions (slot_id in
the space claim), sized by both bodies, minus currently occupied positions.
"""
import math
from typing import Dict, List, Optional

from ..context import Context, STOP, distance
from ..contracts import (Commitment, Dependency, EffectClaim, HEURISTIC,
                         ResourceClaims, SpaceClaim)
from .common import best_move_toward, h_loops, make_plan, move_progress


class CoordinatedEngageSkill:
    name = "coordinated_engage"

    def __init__(self, config: Dict):
        self.config = config
        lifecycle = config.get("lifecycle", {})
        self.target_max_h = lifecycle.get("engage_target_max_h", 12.0)
        self.wait_max_h = lifecycle.get("engage_wait_max_h", 2.0)
        self.no_progress_h = lifecycle.get("no_progress_window_h", 3.0)
        self.retry_h = lifecycle.get("retry_cooldown_h", 2.0)
        self.slot_spacing = config.get("coordinated_engage", {}).get(
            "slot_spacing", 1.4)
        self.max_candidates = config.get("search", {}).get(
            "max_candidates_per_unit", 6)

    def _squad(self, context: Context, actor: str) -> List[str]:
        for group in context.engagement_groups():
            if actor in group:
                return group
        return [actor]

    def _squad_target(self, context: Context, squad: List[str]) -> Optional[str]:
        if not context.snapshot.enemies:
            return None

        def cost(enemy_id):
            return sum(context.distance(a, enemy_id) for a in squad)

        return min(sorted(context.snapshot.enemies), key=lambda e: (cost(e), e))

    def _blocked(self, context: Context, memory, actor: str,
                 target: Optional[str]) -> bool:
        """Real blockage: the same target has been commanded for the whole
        window with no distance improvement — from OBSERVATION history."""
        if target is None:
            return False
        history = memory.move_history.get(actor, [])
        if len(history) < 2:
            return False           # insufficient observation: never judge stuck
        window = h_loops(context, self.no_progress_h)
        past = [position for loop, position in history
                if loop <= context.snapshot.game_loop - window]
        if not past:
            return False
        enemy = context.snapshot.enemies.get(target)
        if enemy is None:
            return False
        closed = distance(past[0], enemy.position) - distance(
            context.snapshot.allies[actor].position, enemy.position)
        if closed > 0.5:
            return False           # actually closing in: not blocked
        return context.contact_status(actor, target) != "in_range"

    def _slot(self, context: Context, actor: str, target_id: str, index: int):
        target = context.snapshot.enemies[target_id]
        unit = context.snapshot.allies[actor]
        occupied = [u.position for u in context.snapshot.allies.values()
                    if u.alive and distance(u.position, target.position)
                    <= target.radius + u.radius + 0.6]
        offset = target.radius + unit.radius + self.slot_spacing * 0.5
        for step in range(8):
            angle = (index + step) * math.pi / 4.0
            position = (target.position[0] + offset * math.cos(angle),
                        target.position[1] + offset * math.sin(angle))
            if all(distance(position, o) > unit.radius + 0.2 for o in occupied):
                return position, f"slot:{target_id}:{(index + step) % 8}"
        return None, None    # ring full: another target or lateral bypass wins

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        if not context.snapshot.enemies:
            return plans
        melee_index = 0
        for actor, unit in sorted(context.snapshot.allies.items()):
            if not unit.alive:
                continue
            capability = context.capabilities[actor]
            if capability.healer or capability.suicide_attack:
                continue
            if memory.retry_blocked(self.name, [actor], "engage",
                                    context.snapshot.game_loop):
                continue
            instance = memory.instance_for(self.name, actor)
            attackable = context.attackable_enemies(actor)
            squad = self._squad(context, actor)
            target_id = self._squad_target(context, squad)
            original = instance.goal if instance is not None and \
                instance.goal_kind == "enemy" and \
                instance.goal in context.snapshot.enemies else None
            in_contact = any(context.contact_status(actor, e) == "in_range"
                             for e in attackable)
            blocked = self._blocked(context, memory, actor, original)
            if in_contact and not blocked:
                continue                      # G1/G3 own this unit
            telemetry.count(self.name, "applicable")
            enemy = context.snapshot.enemies[target_id]
            goal, slot_id = enemy.position, None
            if capability.combat_class == "melee":
                goal, slot_id = self._slot(context, actor, target_id,
                                           melee_index)
                if goal is None:
                    continue
                melee_index += 1
            else:
                # ranged: approach the effective range BAND edge, not the
                # enemy centre
                band = self._range_band(context, actor, target_id)
                if band is not None and distance(unit.position,
                                                 enemy.position) > band:
                    direction = (unit.position[0] - enemy.position[0],
                                 unit.position[1] - enemy.position[1])
                    norm = math.hypot(*direction) or 1.0
                    goal = (enemy.position[0] + direction[0] / norm * band,
                            enemy.position[1] + direction[1] / norm * band)
            moves = [m for m in [best_move_toward(context, actor, goal)]
                     if m is not None]
            for lateral in sorted(context.move_destinations(actor)):
                if lateral not in moves and len(moves) < self.max_candidates:
                    moves.append(lateral)
            no_progress = instance is not None and memory.displacement(
                actor, h_loops(context, self.no_progress_h),
                context.snapshot) < 0.5
            budget = h_loops(context, self.target_max_h)
            for move in moves:
                claims_space = (SpaceClaim(
                    actor=actor,
                    position=context.predict_position(actor, move).position,
                    window_loops=context.H,
                    collision_class="air" if unit.is_flying else "ground",
                    radius=unit.radius, slot_id=slot_id),)
                plans.append(make_plan(
                    self.name, {actor: move}, target_id, {actor: move},
                    [EffectClaim("position", actor,
                                 move_progress(context, actor, move, goal),
                                 max(context.H, 1), HEURISTIC,
                                 "approach_contact_position")],
                    ResourceClaims(controls=(actor,), space=claims_space),
                    Commitment("approaching", 0, max_loops=budget,
                               interrupt_on=("contact", "no_progress",
                                             "danger_change")),
                    {"switch_cost": 0.1 if no_progress else 0.0,
                     "progress": move_progress(context, actor, move, goal)},
                    snapshot_id, fingerprint,
                    instance_id=instance.instance_id if instance else None,
                    dependencies=(Dependency("target_valid", (target_id,)),),
                    goal_kind="enemy",
                    goal_data={"target_id": target_id, "waypoint": goal},
                    memory_delta=({"kind": "create", "skill": self.name,
                                   "actors": [actor], "goal": target_id,
                                   "goal_kind": "enemy",
                                   "phase": "approaching",
                                   "max_loops": budget,
                                   "extra": {"waypoint": goal}}
                                  if instance is None else None),
                    rationale=f"approach {target_id}"
                              + (f" {slot_id}" if slot_id else "")))
            mates_close = any(
                distance(context.snapshot.allies[m].position, enemy.position)
                <= 2 * (unit.radius + enemy.radius) + 3.0
                for m in squad if m != actor and m in context.snapshot.allies)
            if mates_close and not context.threats_to(actor):
                plans.append(make_plan(
                    self.name, {actor: STOP}, target_id, {actor: STOP},
                    [EffectClaim("position", actor, None, max(context.H, 1),
                                 HEURISTIC, "bounded_wait")],
                    ResourceClaims(controls=(actor,)),
                    Commitment("waiting", 0, max_loops=h_loops(
                        context, self.wait_max_h)),
                    {"hold_value": 0.3}, snapshot_id, fingerprint,
                    goal_kind="enemy",
                    goal_data={"target_id": target_id, "waypoint": goal},
                    memory_delta={"kind": "create", "skill": self.name,
                                  "actors": [actor], "goal": target_id,
                                  "goal_kind": "enemy", "phase": "waiting",
                                  "max_loops": h_loops(context,
                                                       self.wait_max_h)},
                    rationale="bounded wait for squad contact"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans

    def _range_band(self, context: Context, actor: str, target_id: str):
        relation = context.snapshot.relations[actor].get(target_id)
        if relation is not None and relation.native_weapon_range:
            return max(relation.native_weapon_range - 0.5, 0.5)
        return None
