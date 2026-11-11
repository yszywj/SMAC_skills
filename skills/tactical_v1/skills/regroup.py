"""G5 Regroup: local rallying of isolated units.

Only stragglers move; a squad that is fighting effectively keeps firing.
Triggered purely by the observed ally distribution — never by reading
reflect/surround labels. Rally points must be legal moves; the skill has a
wait budget, a no-progress exit and a retry cooldown.
"""
from typing import Dict, List

from ..context import Context, STOP, distance
from ..contracts import Commitment, EffectClaim, HEURISTIC, ResourceClaims
from .common import best_move_toward, make_plan, move_progress


class RegroupSkill:
    name = "regroup"

    def __init__(self, config: Dict):
        self.config = config
        params = config.get("regroup", {})
        self.support_radius = params.get("support_radius", 8.0)
        self.max_wait_steps = params.get("max_wait_steps", 8)

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        allies = {a: u for a, u in sorted(context.snapshot.allies.items()) if u.alive}
        if len(allies) < 2:
            return plans
        engaged = {a for a in allies if context.attackable_enemies(a)}
        for actor, unit in allies.items():
            if actor in engaged:
                continue  # fighting units do not stop firing to rally
            others = [a for a in allies if a != actor]
            nearest = min(others, key=lambda a: (distance(unit.position,
                                                          allies[a].position), a))
            gap = distance(unit.position, allies[nearest].position)
            if gap <= self.support_radius:
                continue
            telemetry.count(self.name, "applicable")
            # Rally toward the nearest ally group member, not the team centroid.
            goal = allies[nearest].position
            move = best_move_toward(context, actor, goal)
            if move is not None:
                progress = move_progress(context, actor, move, goal)
                instance = memory.instance_for(self.name, actor)
                no_progress = (instance is not None
                               and instance.budget_steps >= self.max_wait_steps)
                plans.append(make_plan(
                    self.name, {actor: move}, nearest, {actor: move},
                    [EffectClaim("position", actor, progress,
                                 max(context.snapshot.step_mul, 1), HEURISTIC,
                                 "rally_toward_nearest_ally", 0.7)],
                    ResourceClaims(controls=(actor,), space=(f"{actor}@{move}",)),
                    Commitment("rallying", 0, self.max_wait_steps,
                               interrupt_on=("contact", "illegal", "no_progress")),
                    {"move_opportunity_cost": 0.2,
                     "raw_bonus": 0.6 + min(gap / self.support_radius, 2.0) * 0.3
                     - (1.0 if no_progress else 0.0)},
                    snapshot_id, fingerprint,
                    rationale=f"rally toward {nearest} (gap {gap:.1f})"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans
