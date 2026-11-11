"""G7 CohesiveSearch: arrival-confirmed clue tracking with bounded coverage.

`visited` is written ONLY when the squad actually REACHES the waypoint
(arrival tolerance), never when a move command is issued. Fresh last-seen
clues first, then legal exploration points; severe outliers go to G5.
Single-point timeout, TTL and revisit cooldowns are loop-based.
"""
from typing import Dict, List, Optional

from ..context import Context, STOP, distance
from ..contracts import Commitment, EffectClaim, HEURISTIC, ResourceClaims
from .common import best_move_toward, h_loops, make_plan, move_progress, space_claim


class CohesiveSearchSkill:
    name = "cohesive_search"

    def __init__(self, config: Dict):
        self.config = config
        lifecycle = config.get("lifecycle", {})
        self.waypoint_max_h = lifecycle.get("search_waypoint_max_h", 12.0)
        self.ttl = lifecycle.get("observation_ttl_loops", 240)
        self.revisit_cooldown = lifecycle.get("search_revisit_cooldown_loops", 160)
        self.arrival_tolerance = config.get("cohesive_search", {}).get(
            "arrival_tolerance", 2.0)
        self.cohesion_radius = config.get("cohesive_search", {}).get(
            "cohesion_radius", 10.0)

    def _waypoint(self, context: Context, memory, squad) -> Optional[tuple]:
        loop = context.snapshot.game_loop
        # 1. fresh unvisited last-seen clue (shared per squad)
        clues = sorted(((loop - seen, position) for enemy_id, (seen, position)
                        in memory.enemy_last_seen.items()), key=lambda i: i[0])
        for age, position in clues:
            if age > self.ttl:
                continue
            cell = self._cell(position)
            if memory.visited_search_cells.get(cell, -1) < loop - self.revisit_cooldown:
                return position
        # 2. map exploration points: search location, then offset variants
        base = context.snapshot.search_location
        for point in (base, (base[0] / 2, base[1] / 2),
                      (base[0], base[1] / 2), (base[0] / 2, base[1])):
            cell = self._cell(point)
            if memory.visited_search_cells.get(cell, -1) < loop - self.revisit_cooldown:
                return point
        return base

    def _cell(self, position) -> str:
        return f"{int(position[0] // 8)}:{int(position[1] // 8)}"

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        if context.snapshot.enemies:
            return plans                      # contact info exists: not search
        telemetry.count(self.name, "applicable", len(context.snapshot.allies))
        loop = context.snapshot.game_loop
        groups = context.engagement_groups(self.cohesion_radius)
        for group in groups:
            squad_positions = [context.snapshot.allies[a].position
                               for a in group if a in context.snapshot.allies]
            if not squad_positions:
                continue
            centre = (sum(p[0] for p in squad_positions) / len(squad_positions),
                      sum(p[1] for p in squad_positions) / len(squad_positions))
            waypoint = self._waypoint(context, memory, group)
            if waypoint is None:
                continue
            for actor in sorted(group):
                unit = context.snapshot.allies[actor]
                if not unit.alive or context.capabilities[actor].healer:
                    continue
                instance = memory.instance_for(self.name, actor)
                # single-point timeout: switch waypoint or mark unreachable
                if instance is not None and instance.phase == "searching":
                    tracked = instance.extra.get("waypoint")
                    if tracked is not None:
                        if distance(unit.position, tracked) <= self.arrival_tolerance:
                            # ARRIVAL CONFIRMED: only now write visited
                            memory.apply_delta(context.snapshot, {
                                "kind": "visit_cell",
                                "cell": self._cell(tracked)}, None)
                            memory.drop_instance(instance.instance_id,
                                                 "waypoint_reached")
                            instance = None
                        elif loop - instance.start_loop > h_loops(
                                context, self.waypoint_max_h):
                            memory.drop_instance(instance.instance_id,
                                                 "waypoint_timeout")
                            instance = None
                goal = waypoint
                if distance(unit.position, centre) > self.cohesion_radius:
                    goal = centre              # outliers pull to the squad (G5 alt)
                # already at the goal -> pick the next exploration move instead
                if distance(unit.position, goal) <= self.arrival_tolerance:
                    continue
                move = best_move_toward(context, actor, goal)
                if move is None:
                    plans.append(make_plan(
                        self.name, {actor: STOP}, "waypoint", {actor: STOP},
                        [EffectClaim("position", actor, None, max(context.H, 1),
                                     HEURISTIC, "search_wait")],
                        ResourceClaims(controls=(actor,)),
                        Commitment("searching", 0, max_loops=h_loops(
                            context, self.waypoint_max_h)),
                        {"hold_value": 0.1}, snapshot_id, fingerprint,
                        rationale="no improving search move"))
                    continue
                plans.append(make_plan(
                    self.name, {actor: move}, "waypoint", {actor: move},
                    [EffectClaim("position", actor,
                                 move_progress(context, actor, move, goal),
                                 max(context.H, 1), HEURISTIC,
                                 "search_waypoint_progress")],
                    ResourceClaims(controls=(actor,),
                                   space=(space_claim(context, actor, move),)),
                    Commitment("searching", 0, max_loops=h_loops(
                        context, self.waypoint_max_h)),
                    {"progress": abs(move_progress(context, actor, move, goal))},
                    snapshot_id, fingerprint,
                    instance_id=instance.instance_id if instance else None,
                    memory_delta=({"kind": "create", "skill": self.name,
                                   "actors": [actor], "goal": "waypoint",
                                   "phase": "searching",
                                   "max_loops": h_loops(
                                       context, self.waypoint_max_h),
                                   "extra": {"waypoint": goal}}
                                  if instance is None else None),
                    rationale="move to shared search waypoint"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans
