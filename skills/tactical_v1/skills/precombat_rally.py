"""Optional multi-unit rally before a new encounter, selected by Adaptive."""
from ..context import STOP, distance
from ..contracts import Commitment, Dependency, ResourceClaims
from ..encounters import combat_members, rally_readiness, unsafe_to_rally
from .common import h_loops, make_plan, space_claim


class PreCombatRallySkill:
    name = "precombat_rally"

    def __init__(self, config):
        self.config = config
        params = config.get(self.name, {})
        self.min_members = int(params.get("min_members", 3))
        self.ready_radius = float(params.get("ready_radius", 4.0))
        self.max_member_distance = float(params.get("max_member_distance", 20.0))
        self.trigger_enemy_distance = float(params.get("trigger_enemy_distance", 18.0))
        self.ready_fraction = float(params.get("ready_fraction", 0.8))
        self.max_h = float(params.get("max_h", 8.0))
        self.min_commit_h = float(params.get("min_commit_h", 2.0))
        self.no_progress_h = float(params.get("no_progress_h", 3.0))

    def _group(self, context, actors):
        """Choose a safe existing friendly anchor, not a centroid in enemy fire.

        Max-distance membership is frozen at creation, preventing one remote
        straggler from dragging the whole army into an unbounded wait.
        """
        options = []
        for anchor_actor in actors:
            anchor = context.snapshot.allies[anchor_actor].position
            members = tuple(a for a in actors if distance(
                context.snapshot.allies[a].position, anchor) <= self.max_member_distance)
            if len(members) < self.min_members:
                continue
            if any(context.risk_at(a, anchor).threats
                   or context.risk_at(a, anchor).effect_weapon_near for a in members):
                continue
            quality = rally_readiness(context, members, anchor, self.ready_radius)
            # Prefer a useful majority, then the shortest shared regroup path.
            options.append(((-len(members), quality["gap"], anchor_actor),
                            members, anchor, quality))
        if not options:
            return None
        _, members, anchor, quality = min(options, key=lambda option: option[0])
        if quality["fraction"] >= self.ready_fraction:
            return None
        return members, anchor, quality

    def _actions(self, context, members, anchor, radius):
        actions = {}
        for actor in members:
            unit = context.snapshot.allies[actor]
            if distance(unit.position, anchor) <= radius:
                if STOP not in context.legal_actions(actor):
                    return None
                actions[actor] = STOP
                continue
            options = []
            for action, destination in context.move_destinations(actor).items():
                predicted = context.predict_position(actor, action).position
                # Both arrival estimate and endpoint must remain outside
                # currently visible fire. Unknown paths are not simulated.
                if any(context.risk_at(actor, point).threats
                       or context.risk_at(actor, point).effect_weapon_near
                       for point in (predicted, destination)):
                    continue
                gap = distance(predicted, anchor)
                if gap < distance(unit.position, anchor) - 0.05:
                    options.append((gap, action))
            if not options:
                return None  # no all-member legal/progressing group action
            actions[actor] = min(options)[1]
        return actions

    def propose(self, context, memory, telemetry, ledger=None):
        tracker = memory.encounters
        if not tracker.can_rally():
            return []
        instance = next((i for i in memory.instances.values()
                         if i.skill == self.name and not i.termination_reason), None)
        if instance is not None:
            data = dict(instance.extra)
            members = instance.actors
            anchor = tuple(data["anchor"])
            if data["encounter_id"] != tracker.current_id:
                return []
            if unsafe_to_rally(context, members):
                return []
            quality = rally_readiness(context, members, anchor, data["ready_radius"])
            if quality["fraction"] >= data["ready_fraction"]:
                return []  # observation phase owns successful completion
        else:
            members = combat_members(context)
            if len(members) < self.min_members or not context.snapshot.enemies:
                return []
            # Do not pull a fighting frontline out of its attack to assemble.
            if unsafe_to_rally(context, members):
                return []
            if min(distance(context.snapshot.allies[a].position, enemy.position)
                   for a in members for enemy in context.snapshot.enemies.values()
                   ) > self.trigger_enemy_distance:
                return []
            group = self._group(context, members)
            if group is None:
                return []
            members, anchor, quality = group
            data = {"anchor": list(anchor), "waypoint": list(anchor),
                    "members": list(members), "encounter_id": tracker.current_id,
                    "enemy_ids": sorted(context.snapshot.enemies),
                    "ready_radius": self.ready_radius,
                    "ready_fraction": self.ready_fraction,
                    "best_gap": quality["gap"],
                    "rally_last_progress_loop": context.snapshot.game_loop,
                    "no_progress_loops": h_loops(context, self.no_progress_h)}
        goal = f"encounter:{data['encounter_id']}"
        key = memory.task_retry_key(self.name, members, "internal", goal)
        if memory.retry_blocked(*key, context.snapshot.game_loop):
            return []
        actions = self._actions(context, members, anchor, data["ready_radius"])
        if actions is None:
            return []
        telemetry.count(self.name, "applicable")
        objective = {"objective": {"kind": "rally"},
                     "anchor": list(anchor), "members": list(members),
                     "ready_radius": data["ready_radius"],
                     "ready_fraction": data["ready_fraction"],
                     "encounter_id": data["encounter_id"]}
        delta = ({"kind": "continue", "instance_id": instance.instance_id}
                 if instance else {"kind": "create", "skill": self.name,
                                   "actors": list(members), "goal": goal,
                                   "goal_kind": "area", "phase": "rallying",
                                   "max_loops": h_loops(context, self.max_h),
                                   "extra": dict(data, retry_key=key)})
        plan = make_plan(
            self.name, members, goal, actions, [],
            ResourceClaims(controls=members,
                           space=tuple(space_claim(context, a, action)
                                       for a, action in actions.items())),
            Commitment("rallying", h_loops(context, self.min_commit_h),
                       h_loops(context, self.max_h),
                       interrupt_on=("contact", "danger_change", "member_lost")),
            {}, context.snapshot.snapshot_id, self.config.get("_fingerprint", ""),
            instance_id=instance.instance_id if instance else None,
            dependencies=(Dependency("member", members),), memory_delta=delta,
            atomic=True, goal_kind="area", goal_data=objective,
            rationale="assemble a safe support group before this encounter")
        telemetry.count(self.name, "proposed")
        return [plan]
