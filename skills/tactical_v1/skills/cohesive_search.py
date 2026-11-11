"""G7 CohesiveSearch: cooperative search without contact.

Shared waypoints (map search location plus expired, clearly-labelled
last-seen clues), visited-waypoint suppression, no-progress recovery and
cohesion toward the main group. A stale enemy position is a search CLUE
only — never a legal attack target.
"""
from typing import Dict, List, Optional

from ..context import Context, STOP, distance
from ..contracts import Commitment, EffectClaim, HEURISTIC, ResourceClaims
from .common import best_move_toward, make_plan, move_progress


class CohesiveSearchSkill:
    name = "cohesive_search"

    def __init__(self, config: Dict):
        self.config = config
        params = config.get("cohesive_search", {})
        self.max_steps = params.get("max_search_steps", 120)
        self.cohesion_radius = params.get("cohesion_radius", 10.0)
        self.cell_size = params.get("waypoint_cell_size", 8.0)

    def _waypoint(self, context: Context, memory) -> Optional[tuple]:
        """Shared search waypoint: freshest unvisited last-seen clue, else the
        map search location. Same for every unit this step (cooperative)."""
        loop = context.snapshot.game_loop
        clues = sorted(((loop - seen_loop, position)
                        for enemy_id, (seen_loop, position)
                        in memory.enemy_last_seen.items()),
                       key=lambda item: item[0])
        for age, position in clues:
            if age <= 0:
                continue
            key = _cell(position, self.cell_size)
            if memory.visited_search_cells.get(key, -1) < loop - 160:
                return position
        return context.snapshot.search_location

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        if context.snapshot.enemies:
            return plans  # contact information exists: not a search step
        telemetry.count(self.name, "applicable", len(context.snapshot.allies))
        waypoint = self._waypoint(context, memory)
        if waypoint is None:
            return plans
        cell = _cell(waypoint, self.cell_size)
        visit_delta = {"kind": "visit_cell", "cell": cell}
        allies = [u.position for u in context.snapshot.allies.values() if u.alive]
        centre = (sum(p[0] for p in allies) / len(allies),
                  sum(p[1] for p in allies) / len(allies)) if allies else waypoint
        for actor, unit in sorted(context.snapshot.allies.items()):
            if not unit.alive or context.capabilities[actor].healer:
                continue
            goal = waypoint
            if distance(unit.position, centre) > self.cohesion_radius:
                goal = centre  # keep the squad together: no random walk
            move = best_move_toward(context, actor, goal)
            if move is not None:
                plans.append(make_plan(
                    self.name, {actor: move}, "waypoint", {actor: move},
                    [EffectClaim("position", actor,
                                 move_progress(context, actor, move, goal),
                                 max(context.snapshot.step_mul, 1), HEURISTIC,
                                 "search_waypoint_progress", 0.6)],
                    ResourceClaims(controls=(actor,), space=(f"{actor}@{move}",)),
                    Commitment("searching", 0, self.max_steps,
                               interrupt_on=("contact", "illegal", "no_progress")),
                    {"move_opportunity_cost": 0.0}, snapshot_id, fingerprint,
                    memory_delta=visit_delta,
                    rationale="move to shared search waypoint"))
            else:
                plans.append(make_plan(
                    self.name, {actor: STOP}, "waypoint", {actor: STOP},
                    [EffectClaim("position", actor, 0.0,
                                 max(context.snapshot.step_mul, 1), HEURISTIC,
                                 "search_wait", 0.4)],
                    ResourceClaims(controls=(actor,)),
                    Commitment("searching", 0, self.max_steps,
                               interrupt_on=("contact", "no_progress")),
                    {"raw_bonus": -0.3}, snapshot_id, fingerprint,
                    rationale="no improving search move; hold"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans


def _cell(position, size: float) -> str:
    return f"{int(position[0] // size)}:{int(position[1] // size)}"
