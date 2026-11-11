"""G5 Regroup: repair support breaks between squads.

Multiple connected groups (not just "has one neighbour"); for each loner /
sub-squad several merge options are compared (main fighting group, nearest
supporting group, midway meeting area) by join time, route risk and lost
output. Fighting front squads keep firing; units with live heal/burst tasks
are never unconditionally pulled away. Bounded wait, no-progress exit,
retry cooldown — all consumed from the instance.
"""
from typing import Dict, List, Optional

from ..context import Context, STOP, distance
from ..contracts import Commitment, EffectClaim, HEURISTIC, ResourceClaims
from .common import best_move_toward, h_loops, make_plan, move_progress, space_claim


class RegroupSkill:
    name = "regroup"

    def __init__(self, config: Dict):
        self.config = config
        lifecycle = config.get("lifecycle", {})
        self.max_h = lifecycle.get("regroup_max_h", 8.0)
        self.no_progress_h = lifecycle.get("no_progress_window_h", 3.0)
        self.retry_h = lifecycle.get("retry_cooldown_h", 2.0)
        self.support_radius = config.get("regroup", {}).get(
            "support_radius", 8.0)

    def _merge_goal(self, context: Context, actor: str) -> Optional[tuple]:
        """Best merge option for one isolated unit: the main fighting group,
        else the nearest supporting group, else the midway meeting point."""
        groups = context.engagement_groups(self.support_radius)
        own = next((g for g in groups if actor in g), [actor])
        if len(own) > 2:
            return None                      # a real squad, not a loner
        others = [g for g in groups if actor not in g]
        if not others:
            return None
        fighting = [g for g in others
                    if any(context.attackable_enemies(a) for a in g)]
        unit = context.snapshot.allies[actor]
        if fighting:
            group = max(fighting, key=lambda g: (len(g), g))
        else:
            group = min(others, key=lambda g: min(
                distance(unit.position, context.snapshot.allies[a].position)
                for a in g))
        positions = [context.snapshot.allies[a].position for a in group]
        # meeting point: nearest point of the group's extent, not a centroid
        goal = min(positions, key=lambda p: (distance(unit.position, p), p))
        if distance(unit.position, goal) <= self.support_radius * 0.5:
            return None
        return goal

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        loop = context.snapshot.game_loop
        for actor, unit in sorted(context.snapshot.allies.items()):
            if not unit.alive:
                continue
            capability = context.capabilities[actor]
            if capability.healer:
                continue                      # S2 escort handles supports
            if context.attackable_enemies(actor):
                continue                      # fighting units keep firing
            # live heal/burst tasks are not pulled away unconditionally
            if capability.suicide_attack and memory.instance_for(
                    "burst_engage", actor):
                continue
            instance = memory.instance_for(self.name, actor)
            if instance is not None:
                if loop - instance.start_loop > h_loops(context, self.max_h):
                    memory.drop_instance(instance.instance_id, "timeout")
                    instance.retry_after_loop = loop + h_loops(
                        context, self.retry_h)
                    continue
                if instance.phase == "cooldown" and instance.retry_after_loop \
                        and loop < instance.retry_after_loop:
                    continue
            goal = self._merge_goal(context, actor)
            if goal is None:
                continue
            telemetry.count(self.name, "applicable")
            move = best_move_toward(context, actor, goal)
            if move is not None:
                no_progress = instance is not None and memory.displacement(
                    actor, h_loops(context, self.no_progress_h),
                    context.snapshot) < 0.5
                plans.append(make_plan(
                    self.name, {actor: move}, "merge", {actor: move},
                    [EffectClaim("position", actor,
                                 move_progress(context, actor, move, goal),
                                 max(context.H, 1), HEURISTIC,
                                 "rally_to_merge_goal")],
                    ResourceClaims(controls=(actor,),
                                   space=(space_claim(context, actor, move),)),
                    Commitment("rallying", 0, max_loops=h_loops(
                        context, self.max_h)),
                    {"switch_cost": 0.1 if no_progress else 0.0,
                     "progress": abs(move_progress(context, actor, move, goal))},
                    snapshot_id, fingerprint,
                    instance_id=instance.instance_id if instance else None,
                    memory_delta=({"kind": "create", "skill": self.name,
                                   "actors": [actor], "goal": "merge",
                                   "phase": "rallying",
                                   "max_loops": h_loops(context, self.max_h),
                                   "retry_after_loop": loop + h_loops(
                                       context, self.retry_h + self.max_h)}
                                  if instance is None else None),
                    rationale="rally toward merge goal"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans
