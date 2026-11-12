"""G7 CohesiveSearch: arrival-confirmed clue tracking, bounded coverage (D6).

The tracked waypoint comes from the INSTANCE record (one task, one target,
not re-derived each step). Arrivals/timeouts are confirmed in the OBSERVATION
phase (memory.update_task_facts); propose() only reads. Exploration points
spread symmetrically over the known map area around the search location, so
long-term search is not confined to one direction.
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

    def _exploration_points(self, context: Context) -> List[tuple]:
        """Symmetric coverage around the map search location (no global grid
        assumed): the centre plus eight symmetric offsets."""
        base = context.snapshot.search_location
        radius = max(base[0], base[1]) / 2.0 or 8.0
        points = [base]
        for dx, dy in ((.5, .5), (.5, -.5), (-.5, .5), (-.5, -.5),
                       (1., 0.), (0., 1.), (-1., 0.), (0., -1.)):
            points.append((base[0] + dx * radius, base[1] + dy * radius))
        return points

    def _fresh(self, point, memory, loop) -> bool:
        key = _cell(point)
        if loop - memory.unreachable_search_cells.get(key, -10**9) < \
                self.revisit_cooldown:
            return False            # temporarily unreachable
        if memory.visited_search_cells.get(key, -10**9) < \
                loop - self.revisit_cooldown:
            return True             # unvisited or revisitable
        return False

    def _waypoint(self, context: Context, memory, squad,
                  tracked: Optional[tuple]) -> Optional[tuple]:
        loop = context.snapshot.game_loop
        if tracked is not None:
            return tracked          # keep tracking the instance's own target
        clues = sorted(((loop - seen, position)
                        for enemy_id, (seen, position)
                        in memory.enemy_last_seen.items()),
                       key=lambda item: item[0])
        for age, position in clues:
            if age > self.ttl:
                continue
            if self._fresh(position, memory, loop):
                return position
        for point in self._exploration_points(context):
            if self._fresh(point, memory, loop):
                return point
        return context.snapshot.search_location

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        if context.snapshot.enemies:
            return plans                      # contact info: not a search step
        telemetry.count(self.name, "applicable", len(context.snapshot.allies))
        budget = h_loops(context, self.waypoint_max_h)
        groups = context.engagement_groups(self.cohesion_radius)
        for group in groups:
            squad_positions = [context.snapshot.allies[a].position
                               for a in group if a in context.snapshot.allies
                               and context.snapshot.allies[a].alive]
            if not squad_positions:
                continue
            centre = (sum(p[0] for p in squad_positions) / len(squad_positions),
                      sum(p[1] for p in squad_positions) / len(squad_positions))
            tracked = None
            for actor in group:
                instance = memory.instance_for(self.name, actor)
                if instance is not None and instance.extra.get("waypoint"):
                    tracked = tuple(instance.extra["waypoint"])
                    break
            waypoint = self._waypoint(context, memory, group, tracked)
            if waypoint is None:
                continue
            for actor in sorted(group):
                unit = context.snapshot.allies.get(actor)
                if unit is None or not unit.alive \
                        or context.capabilities[actor].healer:
                    continue
                instance = memory.instance_for(self.name, actor)
                goal = waypoint
                if distance(unit.position, centre) > self.cohesion_radius:
                    goal = centre              # outliers pull toward the squad
                if distance(unit.position, goal) <= self.arrival_tolerance:
                    continue                   # arrival handled in observation
                move = best_move_toward(context, actor, goal)
                if move is None:
                    # stuck searcher: STILL a real bounded task so its budget
                    # is consumed and a timeout writes unreachable/retry (C5)
                    plans.append(make_plan(
                        self.name, {actor: STOP}, "waypoint", {actor: STOP},
                        [EffectClaim("position", actor, None, max(context.H, 1),
                                     HEURISTIC, "search_wait")],
                        ResourceClaims(controls=(actor,)),
                        Commitment("searching", 0, max_loops=budget),
                        {"hold_value": 0.1}, snapshot_id, fingerprint,
                        goal_kind="position",
                        goal_data={"waypoint": list(goal)},
                        instance_id=instance.instance_id if instance else None,
                        memory_delta=({"kind": "create", "skill": self.name,
                                       "actors": [actor], "goal": "waypoint",
                                       "goal_kind": "position",
                                       "phase": "searching", "max_loops": budget,
                                       "extra": {"waypoint": goal,
                                                 "arrival_tolerance":
                                                     self.arrival_tolerance}}
                                      if instance is None else None),
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
                    Commitment("searching", 0, max_loops=budget),
                    {"progress": move_progress(context, actor, move, goal)},
                    snapshot_id, fingerprint,
                    goal_kind="position",
                    goal_data={"waypoint": list(goal),
                               "objective": {"kind": "waypoint",
                                             "position": list(goal)}},
                    instance_id=instance.instance_id if instance else None,
                    memory_delta=({"kind": "create", "skill": self.name,
                                   "actors": [actor], "goal": "waypoint",
                                   "goal_kind": "position",
                                   "phase": "searching", "max_loops": budget,
                                   "extra": {"waypoint": goal,
                                             "arrival_tolerance":
                                                 self.arrival_tolerance}}
                                  if instance is None else None),
                    rationale="move to shared search waypoint"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans


def _cell(position) -> str:
    return f"{int(position[0] // 8)}:{int(position[1] // 8)}"
