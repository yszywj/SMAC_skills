"""Atomic in-combat cover and joining: one persistent duty for both squads."""
import math
from ..combat_groups import (connected_groups, forecast, group_status,
                             support_readiness, effective_support, task_interrupt,
                             joiner_progress_errors, viable_rejoin_data, active_rejoin_data)
from ..context import STOP, distance
from ..contracts import Commitment, Dependency, EffectClaim, ResourceClaims
from ..combat_exchange import (imminent_exchange_danger, reserve_fraction, movement_exchange,
                               attack_requires_approach)
from ..targeting import local_front_targets, shared_target_plan
from ..battle_line import actor_line, line_action_rank
from .common import h_loops, make_plan, space_claim


class CombatRejoinSkill:
    name = "combat_rejoin"

    def __init__(self, config):
        self.config = config
        params = config.get(self.name, {})
        self.support_radius = float(params.get("support_radius", 6.))
        self.min_members = int(params.get("min_members", 2))
        self.max_members = int(params.get("max_members", 10))
        self.max_h = float(params.get("max_h", 8.))
        self.min_commit_h = float(params.get("min_commit_h", 2.))
        self.no_progress_h = float(params.get("no_progress_h", 3.))
        self.min_support_ratio = float(params.get("min_support_ratio", 1.2))
        common = config.get("common_combat", {})
        self.shared_front = common.get("shared_front_enabled", True)
        self.local_cover = common.get("local_cover_enabled", True)
        self.arrival_exposure = common.get('arrival_exposure_enabled', True)
        self.joint_assessment = common.get('joint_assessment_enabled', True)

    def _target_data(self, context, memory, members, cover, front=None, previous=None):
        if not self.shared_front:
            return {}
        if front is None and self.joint_assessment:
            # Join one existing firing front, rather than the union of every
            # enemy near every receiver. Cover elsewhere keeps its own shot.
            fronts = {line['front'] for a in cover
                      if (line := actor_line(context, a)) is not None}
            if fronts:
                def fire(frontage):
                    return sum(max((context.fire_estimate(a, e, horizon=3 * context.H).damage_lo or 0.
                                    for e in frontage if context.can_issue_attack(a, e)
                                    and context.contact_status(a, e) == 'in_range'), default=0.)
                               for a in cover)
                front = min(fronts, key=lambda f: (-fire(f),
                    sum(min(context.distance(a, e) for e in f) for a in members), f))
        front = (local_front_targets(context, memory, cover, radius=self.support_radius,
                                     config=self.config) if front is None else
                 tuple(e for e in front if e in context.snapshot.enemies))
        assignments = shared_target_plan(context, memory, self.config).assignments
        allocated = {a: assignments[a].target for a in members if a in assignments
                     and assignments[a].target in front}
        if self.joint_assessment:
            for actor in members:
                line = actor_line(context, actor, front)
                if line is not None and line['target'] in front:
                    allocated[actor] = line['target']
                prior = (previous or {}).get('allocated_targets', {}).get(actor)
                recovery = getattr(memory, 'contact_recovery', lambda *_: None)(context, actor)
                if prior in front and not (recovery and recovery.get('exhausted')):
                    allocated[actor] = prior
                if recovery and recovery.get('exhausted') and recovery.get('target') in front:
                    allocated[actor] = recovery['target']
        targets = {}
        for actor in members:
            allowed = set(front)
            if actor in cover:
                allowed.update(e for e in context.attackable_enemies(actor)
                               if context.contact_status(actor, e) == 'in_range')
            targets[actor] = sorted(allowed, key=lambda e: (
                e != allocated.get(actor), context.distance(actor, e), e))
        return {"front_targets": list(front),
                "targets_by_actor": targets,
                "allocated_targets": allocated}

    @staticmethod
    def _distance(context, left, right):
        return min(distance(context.snapshot.allies[a].position,
                            context.snapshot.allies[b].position) for a in left for b in right)

    def _duties(self, context, memory):
        groups = connected_groups(context, self.support_radius)
        facts = {group: group_status(context, group) for group in groups}
        fighting = [group for group in groups if facts[group]["fighting"]]
        if not fighting:
            return []  # precombat_rally owns preparation before contact
        duties = []
        # Geometric connectivity can hide a long chain of idle backline units.
        # Keep the actors with actual weapon contact as cover and advance only
        # reachable nonparticipants. Cooldown alone does not make a shooter a
        # joiner; targets must be reachable within this duty's unchanged budget.
        for group in fighting:
            cover = tuple(a for a in group if any(
                context.contact_status(a, e) == "in_range"
                for e in context.attackable_enemies(a)))
            if not cover:
                continue
            idle = tuple(a for a in group if a not in cover and not (
                context.snapshot.allies[a].health_fraction <= .25 and context.threats_to(a)))
            if not idle:
                continue
            target_data = self._target_data(context, memory, (*idle, *cover), cover)
            effective = effective_support(context, idle, cover, radius=self.support_radius,
                                          targets_by_actor=target_data.get("targets_by_actor"),
                                          roles={a: "join" for a in idle} if self.local_cover else None)
            joiners = tuple(a for a in idle
                            if (0. if self.joint_assessment else 2.) < effective["eta_by_actor"][a] <= self.max_h
                            and effective["gap_by_actor"][a] > .05)
            if joiners and len(cover) + len(joiners) <= self.max_members:
                duties.append((joiners, cover, "join"))
        # A fighting frontage keeps firing while whole unengaged squads join.
        for cover in fighting:
            joiners = []
            for group in sorted((g for g in groups if not facts[g]["fighting"]),
                                key=lambda g: (self._distance(context, g, cover), g)):
                if len(cover) + len(joiners) + len(group) <= self.max_members:
                    joiners.extend(group)
            if joiners:
                duties.append((tuple(sorted(joiners)), cover, "join"))
        # A pressured pocket gets a coordinated escape-to-support alternative
        # even when it contains multiple shooters; group size alone is no veto.
        for pocket in fighting:
            own = facts[pocket]
            if not own["current_pressure"] or own["support_ratio"] >= self.min_support_ratio:
                continue
            receivers = [g for g in groups if g != pocket and
                         len(g) + len(pocket) <= self.max_members and
                         (facts[g]["available_strength"] >= own["available_strength"] * self.min_support_ratio
                          or (not facts[g]["fighting"] and len(g) >= len(pocket)))]
            if receivers:
                cover = min(receivers, key=lambda g: (
                    -facts[g]["support_ratio"], -facts[g]["available_strength"],
                    self._distance(context, pocket, g), g))
                duties.append((pocket, cover, "escape"))
        result, seen, retained_seen = [], set(), set()
        for joiners, cover, role in duties:
            key = joiners, cover
            if key in seen or len(joiners) + len(cover) < self.min_members:
                continue
            seen.add(key)
            if not self.local_cover and any(context.snapshot.allies[a].health_fraction <= .25
                   and context.threats_to(a) for a in cover):
                continue  # a collapsing receiver cannot promise cover
            data = {"joiners": list(joiners), "supporters": list(cover),
                    "roles": {**{a: role for a in joiners}, **{a: "cover" for a in cover}},
                    "support_radius": self.support_radius,
                    "shared_front_enabled": self.shared_front,
                    "local_cover_enabled": self.local_cover,
                    "arrival_exposure_enabled": self.arrival_exposure,
                    "joint_assessment_enabled": self.joint_assessment,
                    **self._target_data(context, memory, (*joiners, *cover), cover)}
            data = viable_rejoin_data(context, data, self.max_h)
            if data is None:
                continue
            joiners, cover = tuple(data["joiners"]), tuple(data["supporters"])
            if len(joiners) + len(cover) < self.min_members:
                continue
            retained_key = joiners, cover, role
            if retained_key in retained_seen:
                continue
            retained_seen.add(retained_key)
            effective = effective_support(context, joiners, cover, radius=self.support_radius,
                                          targets_by_actor=data.get("targets_by_actor"),
                                          roles=data.get("roles") if self.local_cover else None)
            if not self.joint_assessment and effective["cover_viability"] < .5:
                continue  # receiving frontage cannot plausibly last until arrival
            # Frozen existing friendly position, avoiding enemy-centroid goals.
            anchor_actor = min(cover, key=lambda a: (
                len(forecast(context, a, context.snapshot.allies[a].position).threats),
                sum(distance(context.snapshot.allies[a].position,
                             context.snapshot.allies[j].position) for j in joiners), a))
            anchor = context.snapshot.allies[anchor_actor].position
            ready = support_readiness(context, joiners, cover, self.support_radius)
            result.append({**data,
                "anchor": list(anchor), "waypoint": list(anchor),
                "support_radius": self.support_radius, "readiness_fraction": 1.,
                "best_support_gap": ready["gap"] + sum(
                    0. if self.local_cover and role == "escape" else
                    min(effective["gap_by_actor"][a], 40.) for a in joiners) / len(joiners),
                "last_support_progress_loop": context.snapshot.game_loop,
                "joiner_progress": {a: {"best_error": error,
                    "last_progress_loop": context.snapshot.game_loop, "observed_progress": False}
                    for a, error in joiner_progress_errors(context, joiners, cover,
                        self.support_radius, targets_by_actor=data.get("targets_by_actor"),
                        roles=data.get("roles") if self.local_cover else None).items()},
                "no_progress_loops": h_loops(context, self.no_progress_h),
                "initial_threat_counts": {a: len(context.threats_to(a)) for a in (*joiners, *cover)}})
        # At most three full alternatives get a reserved joint comparison.
        return result[:3]

    def _attack(self, context, actor, allocated, ready_only, data):
        options = []
        for target in context.attackable_enemies(actor):
            if data.get("shared_front_enabled") and target not in data["targets_by_actor"].get(actor, ()) \
                    and not (self.joint_assessment and ready_only):
                continue
            if context.contact_status(actor, target) != "in_range":
                continue
            estimate = context.fire_estimate(actor, target)
            if ready_only and estimate.window_shots < 1:
                continue
            damage = estimate.damage_est or 0.
            remaining = max(context.snapshot.enemies[target].effective_health
                            - allocated.get(target, 0.), 0.)
            preference = (() if not data.get("shared_front_enabled") else (
                not (damage >= remaining > 0.),
                not (remaining > 0. and damage > 0. and target == data["allocated_targets"].get(actor))))
            options.append((*preference, -min(damage, remaining), target not in context.threats_to(actor),
                            context.distance(actor, target), target, damage))
        if not options:
            return None
        target, damage = min(options)[-2:]
        allocated[target] = allocated.get(target, 0.) + damage
        return context.attack_action(actor, target)

    def _join_move(self, context, actor, data, memory=None):
        unit = context.snapshot.allies[actor]
        radius = data["support_radius"]
        supporters = data["supporters"]
        cover_positions = [context.snapshot.allies[a].position for a in supporters]
        # The duty identity/deadline stays frozen. Escape follows the live
        # support boundary; joining additionally has to reach enemy contact.
        before_gap = min(distance(unit.position, point) for point in cover_positions)
        before = forecast(context, actor, unit.position)
        before_count = len(before.threats) + len(before.effect_weapon_near)
        cover_ready = group_status(context, supporters)["ready_shooters"] > 0
        joining = self.joint_assessment and data['roles'].get(actor) == 'join'
        targets = tuple(e for e in data.get('targets_by_actor', {}).get(actor, ())
                        if e in context.snapshot.enemies)
        before_contact = min((context.projected_attack_opportunity(
            actor, e, unit.position).center_gap for e in targets), default=float('inf'))
        recovery = (getattr(memory, 'contact_recovery', lambda *_: None)(context, actor)
                    if memory is not None else None)
        progress = getattr(memory, 'contact_progress', None) if memory is not None else None
        execution = progress.execution(context, actor, memory) if joining and progress is not None else None
        if execution and execution['required'] and execution['reason'] in (
                'synchronize_with_rear_fire', 'advance_rear_to_shared_contact'):
            linked = [a for a in execution['actions'] if context.decode(actor, a).kind == 'move']
            if linked:
                return linked[0]
        preferred = set(execution['actions']) if execution and execution['required'] else (
            set(recovery.get('preferred_actions', ())) if recovery and recovery.get('exhausted') else set())
        options = []
        for action, endpoint in context.move_destinations(actor).items():
            if joining and progress is not None and progress._route_failed(
                    context, actor, recovery['target'] if recovery else None, action):
                continue
            predicted = context.predict_position(actor, action).position
            gap = min(distance(predicted, point) for point in cover_positions)
            line_rank = (line_action_rank(context, actor, action,
                targets=data.get('targets_by_actor', {}).get(actor), memory=memory)
                if self.joint_assessment and data['roles'].get(actor) == 'join' else None)
            after_contact = min((context.projected_attack_opportunity(
                actor, e, predicted).center_gap for e in targets), default=float('inf'))
            if joining:
                # Receiver proximity cannot substitute for reaching an enemy.
                # Keep the same squad boundary, but move toward a firing or
                # interception slot even if this is not toward a teammate.
                if gap > max(before_gap, radius) + .05 or not (
                        after_contact < before_contact - .05 or line_rank is not None):
                    continue
            elif gap >= before_gap - .05:
                continue
            risks = [forecast(context, actor, point) for point in (predicted, endpoint)]
            if any(r.effect_weapon_near for r in risks):
                continue
            if data.get('arrival_exposure_enabled') and data['roles'].get(actor) == 'join':
                exchange = movement_exchange(context, actor, predicted, targets=targets)
                if not data.get('joint_assessment_enabled') and not exchange['admissible']:
                    continue
                if data.get('joint_assessment_enabled') and not math.isfinite(exchange['eta_h']):
                    continue
            after_count = max(len(r.threats) for r in risks)
            in_cover = min(distance(predicted, point) for point in cover_positions) <= radius
            if not data.get('joint_assessment_enabled') and after_count > before_count \
                    and not (before_count == 0 and in_cover and cover_ready):
                continue
            options.append((bool(preferred) and action not in preferred,
                            line_rank is None, line_rank or (), after_contact if joining else gap,
                            after_count, gap, action))
        return min(options)[-1] if options else None

    def _engage_move(self, context, actor, data, memory=None):
        """A healthy receiver closes to usable weapon reach, not a 6-unit ring.

        Retain the existing squad while advancing its idle members. The
        complete evaluator still compares the resulting exposure and fire.
        """
        unit = context.snapshot.allies[actor]
        members = (*data["supporters"], *data["joiners"])
        if data.get("shared_front_enabled"):
            targets = set(data["targets_by_actor"].get(actor, ())) & set(context.snapshot.enemies)
        else:
            targets = {e for a in members for e in context.threats_to(a)}
            targets.update(e for a in members for e in context.attackable_enemies(a)
                           if context.contact_status(a, e) == "in_range")
            if not targets:
                targets = set(context.snapshot.enemies)
        before = forecast(context, actor, unit.position)
        recovery = (getattr(memory, 'contact_recovery', lambda *_: None)(context, actor)
                    if memory is not None else None)
        progress = getattr(memory, 'contact_progress', None) if memory is not None else None
        execution = progress.execution(context, actor, memory) if self.joint_assessment and progress is not None else None
        if execution and execution['required'] and execution['reason'] in (
                'synchronize_with_rear_fire', 'advance_rear_to_shared_contact'):
            linked = [a for a in execution['actions'] if context.decode(actor, a).kind == 'move']
            if linked:
                return linked[0]
        preferred = set(execution['actions']) if execution and execution['required'] else (
            set(recovery.get('preferred_actions', ())) if recovery and recovery.get('exhausted') else set())
        options = []
        for target in sorted(targets):
            current = context.projected_attack_opportunity(actor, target, unit.position)
            for action in context.move_destinations(actor):
                if self.joint_assessment and progress is not None and progress._route_failed(
                        context, actor, target, action):
                    continue
                point = context.predict_position(actor, action).position
                after = context.projected_attack_opportunity(actor, target, point)
                if after.center_gap >= current.center_gap - .05:
                    continue
                risk = forecast(context, actor, point)
                if risk.effect_weapon_near:
                    continue
                if data.get('arrival_exposure_enabled'):
                    exchange = movement_exchange(context, actor, point, targets=(target,))
                    if not data.get('joint_assessment_enabled') and not exchange['admissible']:
                        continue
                    if data.get('joint_assessment_enabled') and not math.isfinite(exchange['eta_h']):
                        continue
                # Full-health units may enter a contested frontage to fight;
                # wounded units cannot buy this progress through extra exposure.
                allowance = 2 if reserve_fraction(unit) > .5 else 0
                if not data.get('joint_assessment_enabled') and len(risk.threats) > len(before.threats) + allowance:
                    continue
                peer_gap = min((distance(point, context.snapshot.allies[b].position)
                                for b in members if b != actor), default=0.)
                old_peer_gap = min((distance(unit.position, context.snapshot.allies[b].position)
                                    for b in members if b != actor), default=0.)
                if peer_gap > max(data["support_radius"], old_peer_gap) + .05:
                    continue
                priority = (() if not data.get("shared_front_enabled") else
                            (target != data["allocated_targets"].get(actor),))
                line_rank = (line_action_rank(context, actor, action,
                    targets=targets, memory=memory) if self.joint_assessment else None)
                options.append((bool(preferred) and action not in preferred,
                                line_rank is None, line_rank or (), *priority,
                                after.center_gap, len(risk.threats), action))
        return min(options)[-1] if options else None

    def _actions(self, context, data, active=False, memory=None):
        actions, allocated = {}, {}
        radius = data["support_radius"]
        for actor in (*data["supporters"], *data["joiners"]):
            unit = context.snapshot.allies[actor]
            emergency = imminent_exchange_danger(context, actor) or (
                not self.joint_assessment and unit.health_fraction <= .25
                and bool(context.threats_to(actor)))
            if data["roles"].get(actor) == "escape":
                move = self._join_move(context, actor, data, memory)
                if move is not None:
                    actions[actor] = move
                    continue
            shot = None if emergency else self._attack(context, actor, allocated, True, data)
            if shot is not None:
                actions[actor] = shot
                continue
            progress = getattr(memory, 'contact_progress', None) if memory is not None else None
            coordination = progress.coordination(context, actor, memory) if self.joint_assessment and progress is not None else None
            if not emergency and coordination and coordination['required']:
                # Share the same pair of arrival obligations as G2/G6;
                # native pursuit must not bypass the rear-fire schedule.
                action = coordination['actions'][0]
                actions[actor] = action
                semantic = context.decode(actor, action)
                if semantic.kind == 'attack':
                    estimate = context.fire_estimate(actor, semantic.target)
                    allocated[semantic.target] = allocated.get(semantic.target, 0.) + (estimate.damage_est or 0.)
                continue
            if self.joint_assessment and not emergency and data['roles'].get(actor) != 'escape' \
                    and progress is not None:
                native = progress.native_approach_actions(context, actor, memory,
                    data.get('targets_by_actor', {}).get(actor) if data.get('shared_front_enabled') else None)
                if native:
                    # Joining can finish through native pursuit. Requiring a
                    # cardinal MOVE here made a near-contact melee participant
                    # keep walking after the command mask had already opened.
                    actions[actor] = native[0]
                    target = context.decode(actor, native[0]).target
                    estimate = context.fire_estimate(actor, target)
                    allocated[target] = allocated.get(target, 0.) + (estimate.damage_est or 0.)
                    continue
            joined = actor in data["supporters"] or min(distance(unit.position,
                context.snapshot.allies[a].position) for a in data["supporters"]) <= radius
            if not joined:
                move = self._join_move(context, actor, data, memory)
                if move is None:
                    if not active:
                        return None
                    # A newly blocked rear member stays in this bounded duty;
                    # do not silently recreate the group with fresh budgets
                    # when the next observation restores its movement menu.
                    hold = self._attack(context, actor, allocated, False, data)
                    risk = context.risk_at(actor, unit.position)
                    if hold is not None:
                        actions[actor] = hold
                    elif STOP in context.legal_actions(actor) and not risk.threats and not risk.effect_weapon_near:
                        actions[actor] = STOP
                    else:
                        return None
                    continue
                actions[actor] = move
                continue
            shot = self._attack(context, actor, allocated, False, data)
            if shot is not None:
                actions[actor] = shot  # preserve an engaged cover order through cooldown
            else:
                move = None if emergency else self._engage_move(context, actor, data, memory)
                if move is not None:
                    actions[actor] = move
                elif STOP in context.legal_actions(actor) and not context.risk_at(actor, unit.position).threats \
                        and not context.risk_at(actor, unit.position).effect_weapon_near:
                    actions[actor] = STOP  # blocked safe unit; idle cost stays explicit
                else:
                    return None
        # A combat joining duty must actually move at least one member.
        if not any(context.decode(a, action).kind == "move" or
                   self.joint_assessment and attack_requires_approach(context, a, action)
                   for a, action in actions.items()) \
                and not (active and any(context.decode(a, action).kind == "attack"
                                        for a, action in actions.items())):
            return None
        return actions, allocated

    def propose(self, context, memory, telemetry, ledger=None):
        instance = next((i for i in memory.instances.values()
                         if i.skill == self.name and not i.termination_reason), None)
        if instance is not None:
            if task_interrupt(context, instance):
                return []
            current = active_rejoin_data(context, instance)
            if current is None:
                return []
            data = dict(current)
            if data.get("shared_front_enabled"):
                data.update(self._target_data(context, memory,
                    (*data["joiners"], *data["supporters"]), data["supporters"],
                    data["front_targets"], previous=data))
            duties = [data]
        else:
            duties = self._duties(context, memory)
        plans = []
        for data in duties:
            members = tuple(sorted((*data["joiners"], *data["supporters"])))
            goal = "+".join(data["joiners"]) + "->" + "+".join(data["supporters"])
            key = memory.task_retry_key(self.name, members, "internal", goal)
            if memory.retry_blocked(*key, context.snapshot.game_loop):
                continue
            result = self._actions(context, data, instance is not None, memory)
            if result is None:
                continue
            actions, allocated = result
            effects = []
            for actor, action in actions.items():
                command = context.decode(actor, action)
                if command.kind == "attack":
                    estimate = context.fire_estimate(actor, command.target)
                    effects.append(EffectClaim("damage", command.target, estimate.damage_est,
                        max(context.H, 1), estimate.evidence, "rejoin_cover_fire",
                        lower=estimate.damage_lo, upper=estimate.damage_hi))
            telemetry.count(self.name, "applicable")
            delta = ({"kind": "continue", "instance_id": instance.instance_id} if instance else
                     {"kind": "create", "skill": self.name, "actors": list(members),
                      "goal": goal, "goal_kind": "area", "phase": "rejoining",
                      "max_loops": h_loops(context, self.max_h), "extra": dict(data, retry_key=key)})
            if instance is not None:
                if members != instance.actors:
                    delta = {"kind": "retain_members", "instance_id": instance.instance_id,
                             "actors": list(members), "extra": data}
                else:
                    changed = {key: value for key, value in data.items()
                               if value != instance.extra.get(key)}
                    if changed:
                        delta["extra"] = changed
            objective = {key: data[key] for key in (
                "joiners", "supporters", "anchor", "roles", "support_radius", "readiness_fraction")}
            objective.update({key: data[key] for key in (
                "front_targets", "targets_by_actor", "allocated_targets",
                "shared_front_enabled", "local_cover_enabled", "arrival_exposure_enabled",
                "joint_assessment_enabled") if key in data})
            objective["objective"] = {"kind": "combat_rejoin"}
            plans.append(make_plan(self.name, members, goal, actions, effects,
                ResourceClaims(controls=members,
                    fire=tuple((target, amount) for target, amount in sorted(allocated.items()) if amount > 0),
                    space=tuple(space_claim(context, a, action) for a, action in actions.items())),
                Commitment("rejoining", h_loops(context, self.min_commit_h), h_loops(context, self.max_h),
                           interrupt_on=("danger_change", "member_lost", "illegal")),
                {}, context.snapshot.snapshot_id, self.config.get("_fingerprint", ""),
                instance_id=instance.instance_id if instance else None,
                dependencies=(Dependency("member", members),), memory_delta=delta,
                atomic=True, goal_kind="area", goal_data=objective,
                rationale="keep cover fire while a fixed squad reaches enemy contact"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans
