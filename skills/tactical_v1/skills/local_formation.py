"""G6: local contact/firing lanes, physical de-crowding and splash spread.

Ordinary units coordinate complete local-front commands and smaller alternatives.
Ready shooters keep firing while missing fire moves into its own effective
range; actual capabilities determine the lane, never a race-specific tank.
S1's protected position and S2's treatment ownership remain separate.
"""
from typing import Dict, List, Optional

from ..context import Context, STOP, distance
from ..contracts import (Commitment, Dependency, EffectClaim, HEURISTIC,
                         ResourceClaims, SpaceClaim)
from .common import best_move_toward, h_loops, make_plan, move_progress, space_claim
from ..battle_line import battle_line, actor_line, line_action_rank, ready_targets
from ..combat_exchange import escape_urgency
from ..joint_combat import action_position


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
        self.joint_assessment = config.get("common_combat", {}).get(
            "joint_assessment_enabled", True)
        self.body_margin = config.get("coordinated_engage", {}).get(
            "contact_slot_margin", .05)

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
        plans: List = ([*self._front_plans(context, memory), *self._line_plans(context, memory)]
                      if self.joint_assessment else [])
        if plans:
            telemetry.count(self.name, "applicable")
        splash_threats = [e for e, cap in sorted(context.enemy_capabilities.items())
                          if cap.splash_attack]
        predicted = {a: u.position for a, u in
                     context.snapshot.allies.items() if u.alive}
        atomic_budget = 2
        for actor, unit in sorted(context.snapshot.allies.items()):
            if not unit.alive or context.capabilities[actor].healer:
                continue
            if self.joint_assessment and context.capabilities[actor].suicide_attack:
                continue  # S3 owns productive contact, approach and burst avoidance
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
            if self.joint_assessment and ready_targets(context, actor) \
                    and not under_splash and not escape_urgency(context, actor)['immediate']:
                # The lane plan can pair this actor's ATTACK with a teammate's
                # move. Ordinary spacing must not throw that shot away.
                continue
            telemetry.count(self.name, "applicable")
            attackable = context.attackable_enemies(actor)
            best_action, best_gain = None, 0.0
            for action in sorted(destinations):
                if self.joint_assessment and context.risk_at(actor,
                        context.predict_position(actor, action).position).effect_weapon_near \
                        and not context.risk_at(actor, unit.position).effect_weapon_near:
                    continue  # spacing cannot authorize entry into a known effect
                if not under_splash and memory.movement_conflict(context, actor, action) > 0:
                    continue  # do not undo a just-committed contact/support move
                prediction = context.predict_position(actor, action)
                after = self._crowding_at(context, actor, prediction.position,
                                          predicted)
                new_splash = any(distance(prediction.position,
                                          context.snapshot.enemies[e].position)
                                 <= self.spread_radius
                                 for e in splash_threats)
                gain = float(crowded - after)
                gain -= 2. * memory.movement_blockage(context, actor, action)
                if under_splash and not new_splash:
                    gain += 1.0
                if role_goal is not None:
                    gain += min(distance(unit.position, role_goal)
                                - distance(prediction.position, role_goal), 2.0)
                if gain > best_gain:
                    best_action, best_gain = action, gain
            guards = [a for a, u in sorted(context.snapshot.allies.items())
                      if a != actor and u.alive and not u.is_flying
                      and distance(u.position, unit.position) <= 6.0]
            if best_action is not None:
                instance = memory.instance_for(self.name, actor)
                group_instance = bool(instance and instance.extra.get('battle_line'))
                is_core = (
                    context.capabilities[actor].specialization
                    == "splash_fire" and guards)
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
                    instance_id=instance.instance_id if instance and not group_instance else None,
                    goal_kind="role" if is_core else "internal",
                    goal_data=(
                        {"role": "colossus_protected", "actor": actor,
                         "guards": guards,
                         "objective": {"kind": "protected_position"}}
                        if is_core else
                        {"objective": {"kind": "spacing"}}),
                    memory_delta=(None if group_instance else {"kind": "create", "skill": self.name,
                                   "actors": [actor], "goal": "formation",
                                   "goal_kind": "internal",
                                   "phase": "adjusting",
                                   "max_loops": budget}
                                  if instance is None else None),
                    rationale=("spread from splash danger" if under_splash
                               else "role/quality position adjustment")))
            # two-unit atomic adjustment: the crowded pair steps apart TOGETHER
            if crowded >= 1 and atomic_budget > 0:
                partner = next((other for other, position in predicted.items()
                                if other != actor
                                and (self.joint_assessment or not
                                     context.snapshot.allies[other].is_flying)
                                and not (self.joint_assessment and
                                         (context.capabilities[other].suicide_attack
                                          or context.capabilities[other].healer))
                                and self._pair_crowded(context, actor, unit.position,
                                                       other, position)), None)
                if partner is not None:
                    atomic_budget -= 1
                    a_move = best_action if best_action is not None else STOP
                    b_move = self._opposite_move(context, partner, actor, memory)
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

    def _front_plans(self, context, memory):
        """Keep a complete local execution group atomic through search.

        No create delta: per-unit execution memories own the deadlines, so
        rebuilding a one-window combination cannot renew a rally task.
        """
        from ..front_execution import front_commands
        proposals = []
        for row in front_commands(context, memory):
            actions, members = row['actions'], row['members']
            positions = {a: action_position(context, a, action)[0] for a, action in actions.items()}
            effects, fire = [], []
            for actor, action in actions.items():
                semantic = context.decode(actor, action)
                if semantic.kind == 'attack':
                    estimate = context.fire_estimate(actor, semantic.target)
                    effects.append(EffectClaim('damage', semantic.target, estimate.damage_est,
                        context.H, estimate.evidence, 'front_execution_fire',
                        lower=estimate.damage_lo, upper=estimate.damage_hi))
                    if estimate.window_shots and estimate.damage_lo is not None:
                        fire.append((semantic.target, estimate.damage_lo))
            proposals.append(make_plan(
                self.name, members, 'front_execution', actions, effects,
                ResourceClaims(controls=members, fire=tuple(fire),
                    space=tuple(SpaceClaim(actor=a, position=positions[a], window_loops=context.H,
                        collision_class='air' if context.snapshot.allies[a].is_flying else 'ground',
                        radius=context.snapshot.allies[a].radius) for a in members)),
                Commitment('adjusting', 0, max_loops=context.H), {},
                context.snapshot.snapshot_id, self.config.get('_fingerprint', ''),
                dependencies=(Dependency('member', members),), atomic=True,
                goal_kind='internal', goal_data={'objective': {'kind': 'battle_line'},
                    'front': list(row['front']), 'roles': row['roles'],
                    'arrival_coordination': row['arrival_coordination'],
                    'execution_duties': row['duties'], 'one_window_composition': True},
                rationale='complete local front: keep fire, fill contact, withdraw endangered members'))
        return proposals

    def _line_plans(self, context, memory):
        """At most two bounded local combinations, not a whole-army reshuffle."""
        facts = battle_line(context)
        units = context.snapshot.allies
        choices = {}
        for actor, lane in facts['actors'].items():
            # Do not turn a specialist's current shot into a generic G6 shot.
            if lane is None or context.capabilities[actor].specialization:
                continue
            options = []
            for action in sorted(context.legal_actions(actor)):
                rank = line_action_rank(context, actor, action, lane['front'], memory)
                if rank is not None:
                    options.append((rank, action))
            if options:
                choices[actor] = min(options)[1]
        moving = [a for a, action in choices.items() if distance(
            units[a].position, action_position(context, a, action)[0]) > .01]
        # First help exposed units or bring overdue nearby fire into range.
        moving.sort(key=lambda a: (
            not bool(context.risk_at(a, units[a].position).threats),
            -facts['actors'][a]['eta_h'], a))
        proposals, used = [], set()
        for actor in moving:
            if actor in used or len(proposals) >= 2:
                continue
            lane = facts['actors'][actor]
            existing = memory.instance_for(self.name, actor)
            if existing is not None and existing.extra.get('battle_line'):
                members = tuple(existing.actors)
                # A changed target is allowed within the same deadline; a
                # missing member cannot silently turn continuation into create.
                if any(a not in choices or not set(facts['actors'][a]['front']) & set(lane['front'])
                       for a in members):
                    continue
                instance = existing
            else:
                peers = [a for a in lane['peers'] if a in choices and a not in used
                         and set(facts['actors'][a]['front']) & set(lane['front'])]
                peers.sort(key=lambda a: (
                    not bool(ready_targets(context, a, lane['front'])),
                    distance(units[a].position, units[actor].position), a))
                if not peers:
                    continue
                members = tuple(sorted((actor, *peers[:2])))
                # Replacing an active formation task must not reset its age.
                active = [memory.instance_for(self.name, a) for a in members]
                if any(item is not None for item in active):
                    continue
                instance = None
            actions = {a: choices[a] for a in members}
            positions = {a: action_position(context, a, action)[0] for a, action in actions.items()}
            # Respect actual same-layer bodies before asking the coordinator
            # to compare the group with attacks, escapes and baseline.
            if any(self._pair_crowded(context, a, positions[a], b, positions[b])
                   for i, a in enumerate(members) for b in members[i + 1:]
                   if distance(positions[a], positions[b]) < distance(units[a].position, units[b].position)):
                continue
            key = memory.task_retry_key(self.name, members, 'internal', 'battle_line')
            if memory.retry_blocked(*key, context.snapshot.game_loop):
                continue
            fire, effects = [], []
            for a, action in actions.items():
                semantic = context.decode(a, action)
                if semantic.kind == 'attack':
                    estimate = context.fire_estimate(a, semantic.target)
                    # Declare each member's actual firing window and target
                    # for mixed-plan validation. Final-command physics remains
                    # the single source of scored damage.
                    effects.append(EffectClaim('damage', semantic.target, estimate.damage_est,
                        max(context.H, 1), estimate.evidence, 'battle_line_fire',
                        lower=estimate.damage_lo, upper=estimate.damage_hi))
                    if estimate.window_shots and estimate.damage_lo is not None:
                        fire.append((semantic.target, estimate.damage_lo))
            delta = ({'kind': 'continue', 'instance_id': instance.instance_id}
                     if instance is not None else {
                         'kind': 'create', 'skill': self.name, 'actors': list(members),
                         'goal': 'battle_line', 'goal_kind': 'internal', 'phase': 'adjusting',
                         'max_loops': h_loops(context, self.max_h),
                         'extra': {'battle_line': True, 'retry_key': key,
                                   'no_progress_loops': h_loops(context, self.max_h)}})
            proposals.append(make_plan(
                self.name, members, 'battle_line', actions, effects,
                ResourceClaims(controls=members, fire=tuple(fire),
                    space=tuple(SpaceClaim(actor=a, position=positions[a],
                        window_loops=context.H,
                        collision_class='air' if units[a].is_flying else 'ground',
                        radius=units[a].radius) for a in members)),
                Commitment('adjusting', 0, max_loops=h_loops(context, self.max_h)),
                {}, context.snapshot.snapshot_id, self.config.get('_fingerprint', ''),
                instance_id=instance.instance_id if instance else None,
                dependencies=(Dependency('member', members),), memory_delta=delta,
                atomic=True, goal_kind='internal',
                goal_data={'objective': {'kind': 'battle_line'},
                           'front': list(lane['front']),
                           'roles': {a: facts['actors'][a]['role'] for a in members},
                           'targets': {a: facts['actors'][a]['target'] for a in members}},
                rationale='join a shared firing front while ready teammates keep attacking'))
            used.update(members)
        return proposals

    def _opposite_move(self, context: Context, mover: str, away_from: str, memory=None):
        unit = context.snapshot.allies[mover]
        if self.joint_assessment and ready_targets(context, mover) \
                and not escape_urgency(context, mover)['immediate'] \
                and not any(cap.splash_attack and distance(unit.position,
                    context.snapshot.enemies[e].position) <= self.spread_radius
                    for e, cap in context.enemy_capabilities.items()):
            return None
        anchor = context.snapshot.allies[away_from].position
        if self.joint_assessment and not self._pair_crowded(
                context, mover, unit.position, away_from, anchor):
            return None
        destinations = context.move_destinations(mover)
        if memory is not None:
            destinations = {a: p for a, p in destinations.items()
                            if not memory.movement_conflict(context, mover, a)
                            and memory.movement_blockage(context, mover, a) < .5}
        if self.joint_assessment and not context.risk_at(
                mover, context.snapshot.allies[mover].position).effect_weapon_near:
            destinations = {a: p for a, p in destinations.items()
                            if not context.risk_at(mover,
                                context.predict_position(mover, a).position).effect_weapon_near}
        if self.joint_assessment:
            # Use the same reachable positions as singleton decongestion. A
            # commanded endpoint beyond this window is not physical clearance.
            destinations = {a: context.predict_position(mover, a).position
                            for a in destinations}
            destinations = {a: p for a, p in destinations.items()
                            if distance(p, anchor) > distance(unit.position, anchor)}
        if not destinations:
            return None
        return max(sorted(destinations),
                   key=lambda a: (distance(destinations[a], anchor), a))

    def _pair_crowded(self, context: Context, actor: str, position,
                      other: str, other_position) -> bool:
        if actor == other:
            return False
        if not self.joint_assessment:
            return distance(position, other_position) < self.min_pair_distance
        unit, neighbor = (context.snapshot.allies[actor],
                          context.snapshot.allies[other])
        if not unit.alive or not neighbor.alive or unit.is_flying != neighbor.is_flying:
            return False
        # Physical congestion is distinct from splash spacing: air above a
        # ground unit does not block it, and a small body needs less clearance.
        return distance(position, other_position) < (
            unit.radius + neighbor.radius + self.body_margin)

    def _crowding(self, context: Context, actor: str, positions) -> int:
        return self._crowding_at(context, actor,
                                 context.snapshot.allies[actor].position, positions)

    def _crowding_at(self, context: Context, actor: str, destination,
                     positions) -> int:
        return sum(1 for other, position in positions.items()
                   if self._pair_crowded(context, actor, destination, other, position))


def math_hypot(x, y):
    import math
    return math.hypot(x, y)
