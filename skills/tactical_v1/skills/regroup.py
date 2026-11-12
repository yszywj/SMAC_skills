"""G5 Regroup: repair support breaks BETWEEN squads (D4).

Multiple connected groups; for each detached squad several merge options are
compared (join the main ground fighting group, the nearest supporting group,
or the midway meeting area). A tight 5-man squad far from another tight
squad DOES get comparable merge proposals. Fighting front squads keep
firing; flying heal squads are never the primary ground fighting group.
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

    def _merge_options(self, context: Context, actor: str) -> List[tuple]:
        groups = context.engagement_groups(self.support_radius)
        own = next((g for g in groups if actor in g), [actor])
        others = [g for g in groups if actor not in g]
        if not others:
            return []
        ground_others = [g for g in others
                         if not any(context.snapshot.allies[a].is_flying
                                    for a in g)]
        pool = ground_others or others
        fighting = [g for g in pool
                    if any(context.attackable_enemies(a) for a in g)]
        unit = context.snapshot.allies[actor]
        options = []
        if fighting:
            main = max(fighting, key=lambda g: (len(g), g))
            positions = [context.snapshot.allies[a].position for a in main]
            options.append(min(positions,
                               key=lambda p: (distance(unit.position, p), p)))
        else:
            group = min(pool, key=lambda g: min(
                distance(unit.position, context.snapshot.allies[a].position)
                for a in g))
            positions = [context.snapshot.allies[a].position for a in group]
            options.append(min(positions,
                               key=lambda p: (distance(unit.position, p), p)))
        closest = min(pool, key=lambda g: min(
            distance(unit.position, context.snapshot.allies[a].position)
            for a in g))
        own_centre = _centroid(context, own)
        other_centre = _centroid(context, closest)
        options.append(((own_centre[0] + other_centre[0]) / 2,
                        (own_centre[1] + other_centre[1]) / 2))
        return [o for o in options
                if distance(unit.position, o) > self.support_radius * 0.5]

    def _link_pairs(self, context: Context, actor: str):
        """The comparable (own group, other group) pairs for this actor,
        FIXED for the whole search from the current visible state (B3)."""
        groups = context.engagement_groups(self.support_radius)
        own = next((g for g in groups if actor in g), [actor])
        pairs = []
        for other in groups:
            if other == own:
                continue
            pairs.append((own, other))
        return own, pairs

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        loop = context.snapshot.game_loop
        budget = h_loops(context, self.max_h)
        for actor, unit in sorted(context.snapshot.allies.items()):
            if not unit.alive:
                continue
            capability = context.capabilities[actor]
            if capability.healer:
                continue                      # S2 escort owns supports
            if context.attackable_enemies(actor):
                continue                      # fighting units keep firing
            if capability.suicide_attack and memory.instance_for(
                    "burst_engage", actor):
                continue                      # live burst task not pulled away
            if memory.retry_blocked(self.name, [actor], "merge", loop):
                continue
            instance = memory.instance_for(self.name, actor)
            options = self._merge_options(context, actor)
            if not options:
                continue
            telemetry.count(self.name, "applicable")
            no_progress = instance is not None and memory.displacement(
                actor, h_loops(context, self.no_progress_h),
                context.snapshot) < 0.5
            own, pairs = self._link_pairs(context, actor)
            for goal in options[:2]:          # comparable alternatives
                move = best_move_toward(context, actor, goal)
                if move is None:
                    continue
                # the REAL evaluator goal: which support link this closes
                other = min((p[1] for p in pairs),
                            key=lambda g: min(
                                distance(unit.position,
                                         context.snapshot.allies[a].position)
                                for a in g if a in context.snapshot.allies),
                            default=None)
                plans.append(make_plan(
                    self.name, {actor: move}, "merge", {actor: move},
                    [EffectClaim("position", actor,
                                 move_progress(context, actor, move, goal),
                                 max(context.H, 1), HEURISTIC,
                                 "rally_to_merge_goal")],
                    ResourceClaims(controls=(actor,),
                                   space=(space_claim(context, actor, move),)),
                    Commitment("rallying", 0, max_loops=budget),
                    {"switch_cost": 0.1 if no_progress else 0.0},
                    snapshot_id, fingerprint,
                    instance_id=instance.instance_id if instance else None,
                    memory_delta=({"kind": "create", "skill": self.name,
                                   "actors": [actor], "goal": "merge",
                                   "goal_kind": "area", "phase": "rallying",
                                   "max_loops": budget,
                                   "extra": {"waypoint": goal}}
                                  if instance is None else None),
                    goal_kind="group_link",
                    goal_data={
                        "group_a": sorted(own),
                        "group_b": sorted(other) if other else [],
                        "support_radius": self.support_radius,
                        "scale": 4.0, "position": list(goal)},
                    task_key="|".join(("regroup", ",".join(sorted(own)),
                                       ",".join(sorted(other or [])))),
                    rationale="rally toward merge goal"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans


def _centroid(context, group):
    positions = [context.snapshot.allies[a].position for a in group
                 if a in context.snapshot.allies]
    n = len(positions) or 1
    return (sum(p[0] for p in positions) / n, sum(p[1] for p in positions) / n)
