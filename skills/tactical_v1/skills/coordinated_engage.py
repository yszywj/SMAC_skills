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
from ..targeting import local_front_targets, shared_target_plan
from ..battle_line import actor_line, line_action_rank
from ..combat_exchange import danger_evidence
from ..joint_combat import action_position
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
        self.slot_margin = config.get("coordinated_engage", {}).get(
            "contact_slot_margin", .05)
        self.max_candidates = config.get("search", {}).get(
            "max_candidates_per_unit", 6)
        common = config.get("common_combat", {})
        self.shared_front = common.get("shared_front_enabled", True)
        self.legal_reach = common.get("next_fire_budget_enabled", True)
        self.arrival_exposure = common.get("arrival_exposure_enabled", True)
        self.reentry_window = common.get("reentry_window_enabled", True)
        self.joint_assessment = common.get("joint_assessment_enabled", True)

    def _squad(self, context: Context, actor: str) -> List[str]:
        for group in context.engagement_groups():
            if actor in group:
                return group
        return [actor]

    def _squad_target(self, context: Context, squad: List[str],
                      targets=None) -> Optional[str]:
        targets = context.snapshot.enemies if targets is None else targets
        if not targets:
            return None

        def cost(enemy_id):
            return sum(context.distance(a, enemy_id) for a in squad)

        return min(sorted(targets), key=lambda e: (cost(e), e))

    def _task_delta(self, context, instance, actor, target_id, goal, phase,
                    wait_remaining=None):
        """Continue one enemy task; phase changes never renew its total budget."""
        extra = {"waypoint": goal,
                 "phase_max_loops": (h_loops(context, self.wait_max_h)
                     if wait_remaining is None else wait_remaining)
                 if phase == "waiting" else None}
        if instance is None:
            return {"kind": "create", "skill": self.name,
                    "actors": [actor], "goal": target_id,
                    "goal_kind": "enemy", "phase": phase,
                    "max_loops": h_loops(context, self.target_max_h),
                    "extra": extra}
        if instance.goal != target_id:
            # A different opening on this local front is still the same
            # approach duty. Retargeting never grants a fresh total budget.
            return {"kind": "retarget", "instance_id": instance.instance_id,
                    "goal": target_id, "goal_kind": "enemy", "phase": phase,
                    "extra": extra}
        if instance.phase == phase:
            return {"kind": "continue", "instance_id": instance.instance_id,
                    "extra": {"waypoint": goal}}
        return {"kind": "transition",
                "instance_id": instance.instance_id, "phase": phase,
                "extra": extra}

    def _blocked(self, context: Context, memory, actor: str,
                 target: Optional[str]) -> bool:
        """Real blockage: the same target has been commanded for the whole
        window with no distance improvement — from OBSERVATION history."""
        if target is None:
            return False
        instance = memory.instance_for(self.name, actor)
        if instance is None or instance.goal_kind != "enemy" \
                or instance.goal != target or instance.phase != "approaching":
            return False
        window = h_loops(context, self.no_progress_h)
        threshold = context.snapshot.game_loop - window
        if max(instance.start_loop, instance.phase_start_loop) > threshold:
            return False           # earlier duties are not evidence of blockage
        history = instance.extra.get("approach_history", [])
        enemy = context.snapshot.enemies.get(target)
        if enemy is None:
            return False
        enemy_tag = context.snapshot.raw_state.get("visible_enemies", {}).get(target, {}).get("tag")
        enemy_history = {loop: position for loop, enemy_id, tag, position
                         in instance.extra.get("approach_target_history", [])
                         if enemy_id == target and tag == enemy_tag}
        past = [(positions[actor], enemy_history[loop])
                for loop, positions in history
                if instance.phase_start_loop <= loop <= threshold
                and actor in positions and loop in enemy_history]
        if not past:
            return False  # no synchronized relative-distance evidence yet
        closed = distance(*past[-1]) - distance(
            context.snapshot.allies[actor].position, enemy.position)
        if closed > 0.5:
            return False           # actually closing in: not blocked
        return context.contact_status(actor, target) != "in_range"

    def _slot(self, context: Context, actor: str, target_id: str, index: int):
        target = context.snapshot.enemies[target_id]
        unit = context.snapshot.allies[actor]
        relation = context.snapshot.relations.get(actor, {}).get(target_id)
        reach = relation.native_weapon_range if relation is not None else None
        if reach is None or reach < 0:
            return None, None        # no invented melee contact envelope
        # Radial reach is weapon range plus both radii, not slot spacing.
        # Keep a small margin inside contact, without overlapping the target.
        offset = target.radius + unit.radius + max(reach - min(.05, reach / 2), 0.)
        if self.legal_reach:
            legal = context.effective_attack_reach(actor, target_id)
            bodies = target.radius + unit.radius
            if legal is None or legal < bodies:
                return None, None  # no non-overlapping, legally attackable slot
            offset = max(bodies, legal - min(.05, max(legal - bodies, 0.) / 2))
        occupied = [u for a, u in context.snapshot.allies.items()
                    if a != actor and u.alive and not u.is_flying
                    and distance(u.position, target.position)
                    <= offset + unit.radius + u.radius]
        # Ground bodies determine how many units can actually share the
        # contact ring. A fixed 1.4-unit gap unnecessarily halves a small
        # melee unit's usable frontage. Keep the former layout in legacy mode.
        chord = (2 * unit.radius + self.slot_margin if self.joint_assessment
                 else max(self.slot_spacing, 2 * unit.radius))
        slots = max(1, int(math.pi / math.asin(min(chord / (2 * offset), 1.))))
        available = []
        for number in range(slots):
            angle = number * 2 * math.pi / slots
            position = (target.position[0] + offset * math.cos(angle),
                        target.position[1] + offset * math.sin(angle))
            if all(distance(position, o.position) >= unit.radius + o.radius
                   for o in occupied):
                available.append((position, number))
        if available:
            if self.joint_assessment:
                # The index ranks alternatives for THIS actor, not actors in
                # roster order. Stable ring ids still let the joint allocator
                # exclude two units promising the same future contact slot.
                available.sort(key=lambda item: (
                    self._route_distance(context, actor, unit.position, item[0]),
                    distance(unit.position, item[0]), item[1]))
                position, number = available[index % len(available)]
            else:
                position, number = min(available,
                    key=lambda item: (item[1] - index) % slots)
            return position, f"slot:{target_id}:{slots}:{number}"
        return None, None    # ring full: another target or lateral bypass wins

    @staticmethod
    def _route_distance(context, actor, start, goal):
        """Short geometric route proxy; bodies obstruct, they are not walls.

        A straight segment crossing a visible ground body gets a bounded
        clearance detour. This only ranks/preserves candidates: it does not
        assert pathfinding, freeze teammates in place, or grant safety.
        """
        unit = context.snapshot.allies[actor]
        dx, dy = goal[0] - start[0], goal[1] - start[1]
        length2 = dx * dx + dy * dy
        if length2 <= 1e-12:
            return 0.
        penalty = 0.
        bodies = [(a, u) for a, u in context.snapshot.allies.items() if a != actor]
        bodies += list(context.snapshot.enemies.items())
        for _, body in bodies:
            if not getattr(body, "alive", True) or body.is_flying:
                continue
            along = ((body.position[0] - start[0]) * dx +
                     (body.position[1] - start[1]) * dy) / length2
            if not 0. < along < 1.:
                continue
            nearest = (start[0] + along * dx, start[1] + along * dy)
            overlap = max(unit.radius + body.radius - distance(nearest, body.position), 0.)
            penalty += 2. * overlap
        return math.sqrt(length2) + penalty

    def _useful_melee_route(self, context, memory, actor, target, goal,
                            prediction, contact_after):
        """Reject aimless safe-side shuffling, not contested forward moves.

        A detour can reduce visible body obstruction despite temporarily
        closing little radial distance. Reversing into a recently observed
        position without relative contact progress is not another detour.
        No task clock or observed progress is changed by this query.
        """
        if contact_after:
            return True
        unit = context.snapshot.allies[actor]
        enemy = context.snapshot.enemies[target]
        travel = distance(unit.position, prediction)
        if travel <= 1e-6:
            return False
        meaningful = .1 * travel
        instance = memory.instance_for(self.name, actor)
        if instance is not None and instance.goal_kind == "enemy" and instance.goal == target:
            tag = context.snapshot.raw_state.get("visible_enemies", {}).get(target, {}).get("tag")
            target_history = {loop: position for loop, enemy_id, old_tag, position
                in instance.extra.get("approach_target_history", [])
                if enemy_id == target and old_tag == tag}
            for loop, positions in instance.extra.get("approach_history", []):
                if not context.snapshot.game_loop - 3 * context.H <= loop < context.snapshot.game_loop \
                        or actor not in positions or loop not in target_history:
                    continue
                old = positions[actor]
                if distance(old, prediction) <= min(.2, meaningful) and \
                        distance(old, target_history[loop]) - distance(prediction, enemy.position) < meaningful:
                    return False
        radial_gain = distance(unit.position, enemy.position) - distance(prediction, enemy.position)
        slot_gain = distance(unit.position, goal) - distance(prediction, goal)
        if radial_gain >= meaningful and slot_gain >= -.25 * travel:
            return True
        before_route = self._route_distance(context, actor, unit.position, goal)
        after_route = self._route_distance(context, actor, prediction, goal)
        before_direct = distance(unit.position, goal)
        # Only an actual visible obstruction can justify weak radial closure.
        # Merely facing a distant slot or being marked blocked is insufficient.
        return (before_route - before_direct >= meaningful
                and before_route - after_route >= meaningful
                and slot_gain >= -.25 * travel)

    def _can_wait(self, context, actor, target_id, squad):
        """Only a short precontact cooldown synchronization can justify STOP.

        A teammate already taking damage or firing needs participation now.
        Proximity to that teammate alone is never a reason for the rear to
        wait. Both the waiting actor and a partner must be one legal move
        from the same firing frontage, without assuming future enemy motion.
        """
        unit = context.snapshot.allies[actor]
        if not context.H < unit.weapon_cooldown_loops <= 2 * context.H:
            return False
        fighters = [a for a in squad if context.snapshot.allies[a].alive
                    and not context.capabilities[a].healer
                    and not context.capabilities[a].suicide_attack]
        for member in fighters:
            other = context.snapshot.allies[member]
            risk = context.risk_at(member, other.position)
            if risk.threats or risk.effect_weapon_near or other.health_lost + other.shield_lost > 0:
                return False
            if any(context.contact_status(member, e) == "in_range"
                   for e in context.attackable_enemies(member)):
                return False

        def next_contact(member):
            return any(context.projected_attack_opportunity(member, target_id,
                context.predict_position(member, move).position).available
                for move in context.move_destinations(member))

        return next_contact(actor) and any(
            other != actor and context.snapshot.allies[other].weapon_cooldown_loops <= context.H
            and next_contact(other) for other in fighters)

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        if not context.snapshot.enemies:
            return plans
        target_plan = shared_target_plan(context, memory, self.config) if self.shared_front else None
        melee_index = 0
        for actor, unit in sorted(context.snapshot.allies.items()):
            if not unit.alive:
                continue
            capability = context.capabilities[actor]
            if capability.healer or capability.suicide_attack:
                continue
            # Cooldowns belong to an actor/target task, not the whole actor:
            # another enemy remains eligible while a failed target cools down.
            loop = context.snapshot.game_loop
            targets = [enemy_id for enemy_id in context.snapshot.enemies
                       if not memory.retry_blocked(
                           *memory.task_retry_key(self.name, [actor], "enemy",
                                                  enemy_id), loop)]
            if not targets:
                continue
            instance = memory.instance_for(self.name, actor)
            attackable = context.attackable_enemies(actor)
            if any(context.contact_status(actor, e) == "in_range" for e in attackable):
                continue             # current contact wins over an obsolete goal
            exhausted = getattr(memory, "retreat_exhausted", lambda *_: False)(context, actor)
            if memory.cycle_holds(context, actor, "escape") and not exhausted \
                    and danger_evidence(context, actor, horizon_h=2.)['immediate']:
                continue             # bounded escape survives a skill replacement
            squad = self._squad(context, actor)
            line = actor_line(context, actor) if self.joint_assessment else None
            recovery = getattr(memory, 'contact_recovery', lambda *_: None)(context, actor)
            stalled = bool(recovery and recovery.get('exhausted'))
            target_id = self._squad_target(context, squad, targets)
            battle_target = getattr(memory, "battle_target", lambda *_: None)(context, actor)
            if battle_target in targets:
                target_id = battle_target
            cycle = memory.combat_cycle(context, actor)
            if cycle and memory.cycle_holds(context, actor, "approach") \
                    and cycle["target"] in targets:
                target_id = cycle["target"]
            original = instance.goal if instance is not None and \
                instance.goal_kind == "enemy" and \
                instance.goal in context.snapshot.enemies else None
            blocked = self._blocked(context, memory, actor, original)
            front_targets = targets
            if self.shared_front:
                radius = self.config.get("combat_rejoin", {}).get("support_radius", 6.)
                front = (line['front'] if line is not None else
                    local_front_targets(context, memory, squad,
                        seed_targets=(target_id,), radius=radius, config=self.config))
                front_targets = [e for e in front if e in targets]
                if not front_targets:
                    # A local retry cooldown cannot redirect an actor to a
                    # different distant battle. Other skills may still fire,
                    # cover, move safely or wait for the local retry to expire.
                    continue
                if target_id not in front_targets:
                    target_id = self._squad_target(context, squad, front_targets)
                assignment = target_plan.assignments.get(actor)
                if assignment is not None and assignment.target in front_targets:
                    target_id = assignment.target
                if line is not None and line['target'] in front_targets:
                    target_id = line['target']
                # Keep a live, progressing approach/wait rather than letting
                # a fresh allocation reset its total task deadline each tick.
                if original in front_targets and not blocked and not stalled:
                    target_id = original
                if cycle and memory.cycle_holds(context, actor, "approach") \
                        and cycle["target"] in front_targets and not blocked and not stalled:
                    target_id = cycle["target"]
                if blocked and target_id == original:
                    alternatives = [e for e in front_targets if e != original]
                    if alternatives:
                        target_id = self._squad_target(context, squad, alternatives)
            if stalled and recovery.get('target') in front_targets:
                # The observed participation deadline spans G2/G6/rejoin and
                # escape. Switching providers cannot grant another long trip.
                target_id = recovery['target']
            progress = getattr(memory, 'contact_progress', None)
            execution = progress.execution(context, actor, memory) if progress is not None and self.joint_assessment else None
            if execution and execution.get('target') in front_targets:
                target_id = execution['target']
            telemetry.count(self.name, "applicable")
            if execution and execution['required'] and execution['reason'] in (
                    'synchronize_with_rear_fire', 'advance_rear_to_shared_contact'):
                # Actual linked arrival actions, including safe convergence
                # toward the rear. A contact-slot progress test must not
                # discard these and restore the same solitary charge.
                for action in execution['actions']:
                    semantic = context.decode(actor, action)
                    target = semantic.target if semantic.kind == 'attack' else execution['target']
                    point, _ = action_position(context, actor, action)
                    effects = [EffectClaim('position', actor, None, context.H,
                        HEURISTIC, 'shared_arrival_position_not_contact')]
                    if semantic.kind == 'attack':
                        effects.append(EffectClaim('coverage', target, None, context.H,
                            HEURISTIC, 'native_approach_order_not_confirmed_fire'))
                    plans.append(make_plan(self.name, [actor], target, {actor: action}, effects,
                        ResourceClaims(controls=(actor,), space=(SpaceClaim(actor=actor,
                            position=point, window_loops=context.H,
                            collision_class='air' if unit.is_flying else 'ground', radius=unit.radius),)),
                        Commitment('synchronizing', 0, max_loops=context.H), {}, snapshot_id, fingerprint,
                        dependencies=(Dependency('target_valid', (target,)),), goal_kind='enemy',
                        goal_data={'target_id': target, 'waypoint': list(point),
                            'objective': {'kind': 'shared_arrival'}, 'arrival_coordination': execution},
                        rationale=execution['reason']))
                # No create/retarget: cross-provider records retain the one
                # waiting allowance and the original contact deadline.
                continue
            if self.joint_assessment and capability.combat_class == 'melee' and progress is not None:
                native = progress.native_approach_actions(context, actor, memory, front_targets)
                if native:
                    # A legal native ATTACK owns the final approach. Do not
                    # walk around an ideal ring when SC2 can close and fire.
                    for action in native:
                        target = context.decode(actor, action).target
                        point, _ = action_position(context, actor, action)
                        active_instance = instance if instance is not None and instance.goal_kind == 'enemy' else None
                        plans.append(make_plan(self.name, [actor], target, {actor: action},
                            [EffectClaim('position', actor, None, context.H, HEURISTIC,
                                         'native_attack_closes_contact_not_confirmed_fire')],
                            ResourceClaims(controls=(actor,), space=(SpaceClaim(actor=actor,
                                position=point, window_loops=context.H,
                                collision_class='air' if unit.is_flying else 'ground', radius=unit.radius),)),
                            Commitment('approaching', 0, max_loops=h_loops(context, self.target_max_h)),
                            {}, snapshot_id, fingerprint, instance_id=active_instance.instance_id if active_instance else None,
                            dependencies=(Dependency('target_valid', (target,)),), goal_kind='enemy',
                            goal_data={'target_id': target, 'waypoint': list(point),
                                'objective': {'kind': 'range_band', 'target': target,
                                    'r_min': 0., 'r_max': context.effective_attack_reach(actor, target)},
                                'contact_execution': 'native_contact_commit'},
                            memory_delta=self._task_delta(context, active_instance, actor, target, point, 'approaching'),
                            rationale='native attack completes a reserved near-term melee approach'))
                    continue
            enemy = context.snapshot.enemies[target_id]
            goal, slot_id = enemy.position, None
            if capability.combat_class == "melee":
                goal, slot_id = self._slot(context, actor, target_id,
                                           0 if self.joint_assessment else melee_index)
                if goal is None and self.shared_front:
                    # A full contact ring is not a reason to leave this
                    # actor without an engage candidate. Try another target
                    # on the same local frontage before giving up.
                    alternatives = sorted((e for e in front_targets if e != target_id),
                        key=lambda e: (sum(context.distance(a, e) for a in squad), e))
                    for other in alternatives:
                        other_goal, other_slot = self._slot(context, actor, other,
                            0 if self.joint_assessment else melee_index)
                        if other_goal is not None:
                            target_id, goal, slot_id = other, other_goal, other_slot
                            enemy = context.snapshot.enemies[target_id]
                            break
                if goal is None:
                    continue
                melee_index += 1
                goals = [(goal, slot_id)]
                if self.joint_assessment:
                    for rank in (1, 2):
                        alternative = self._slot(context, actor, target_id, rank)
                        if alternative[0] is not None and alternative not in goals:
                            goals.append(alternative)
            else:
                # ranged: approach the effective range BAND edge, not the
                # enemy centre
                band = self._range_band(context, actor, target_id)
                if band is None and self.legal_reach:
                    continue
                if band is not None and distance(unit.position,
                                                 enemy.position) > band:
                    direction = (unit.position[0] - enemy.position[0],
                                 unit.position[1] - enemy.position[1])
                    norm = math.hypot(*direction) or 1.0
                    goal = (enemy.position[0] + direction[0] / norm * band,
                            enemy.position[1] + direction[1] / norm * band)
                goals = [(goal, None)]
                if self.joint_assessment:
                    firing_line = actor_line(context, actor, (target_id,))
                    if firing_line is not None and firing_line['goal'] != goal:
                        # The same range-aware fire band is used by G6 and
                        # joining duties. Its lateral position is a candidate,
                        # not permission to abandon a currently available shot.
                        goals.insert(0, (firing_line['goal'], None))
            if instance is not None and (instance.goal_kind != "enemy" or
                    (not self.joint_assessment and instance.goal != target_id)):
                instance = None
            routes = []
            primary_goal = goals[0][0]
            for route_goal, route_slot in goals:
                moves = [m for m in [best_move_toward(context, actor, route_goal)]
                         if m is not None]
                for lateral in sorted(context.move_destinations(actor)):
                    if lateral not in moves and len(moves) < self.max_candidates:
                        moves.append(lateral)
                routes.extend((route_goal, route_slot, move) for move in moves)
            no_progress = blocked
            budget = h_loops(context, self.target_max_h)
            actor_plans = []
            for goal, slot_id, move in routes:
                if self.joint_assessment and progress is not None and progress._route_failed(
                        context, actor, target_id, move):
                    continue
                prediction = context.predict_position(actor, move).position
                if self.reentry_window and not self.joint_assessment and not memory.reentry_allowed(
                        context, actor, prediction, targets=front_targets):
                    continue
                exchange = None
                if self.arrival_exposure:
                    from ..combat_exchange import movement_exchange
                    exchange = movement_exchange(context, actor, prediction, targets=(target_id,))
                    if not self.joint_assessment and not exchange["admissible"]:
                        continue
                    # Preserve contested routes until the complete team's
                    # actual cover/actions can be evaluated on one standard.
                    if self.joint_assessment and (not math.isfinite(exchange['eta_h'])
                            or exchange['after']['effect_danger']):
                        continue
                gain = distance(unit.position, goal) - distance(prediction, goal)
                travel = distance(unit.position, prediction)
                contact_after = context.projected_attack_opportunity(
                    actor, target_id, prediction, horizon=context.H).available
                if self.joint_assessment and capability.combat_class == "melee" \
                        and not self._useful_melee_route(context, memory, actor,
                            target_id, goal, prediction, contact_after):
                    continue
                # A fallback direction is not automatically an approach.
                # Small lateral detours are allowed for observed blockage;
                # walking straight back belongs to an escape duty instead.
                if not contact_after and gain < (-.25 * travel if blocked else -.05):
                    continue
                claims_space = (SpaceClaim(
                    actor=actor,
                    position=prediction,
                    window_loops=context.H,
                    collision_class="air" if unit.is_flying else "ground",
                    radius=unit.radius, slot_id=slot_id),)
                plan = make_plan(
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
                    goal_data={"target_id": target_id, "waypoint": goal,
                               **({"movement_contract": {"kind": "approach",
                                   "targets": list(front_targets),
                                   "eta_h": exchange['eta_h'],
                                   "reserve_ratio": exchange['after']['reserve_ratio']}}
                                  if self.joint_assessment and exchange is not None else {}),
                               "objective": _objective(context, actor,
                                                       target_id, goal, self.legal_reach)},
                    memory_delta=self._task_delta(
                        context, instance, actor, target_id, goal, "approaching"),
                    rationale=f"approach {target_id}"
                              + (f" {slot_id}" if slot_id else ""))
                if self.joint_assessment and slot_id:
                    plan.proposal_id += "-" + slot_id
                actor_plans.append(plan)
            if self.joint_assessment:
                ranked = [(line_action_rank(context, actor, p.current_actions[actor],
                    targets=front_targets, memory=memory), p) for p in actor_plans]
                # Keep legacy route alternatives for full-plan comparison;
                # preferred shared-line actions are retained before truncation.
                preferred = set(execution['actions']) if execution and execution['required'] else (
                    set(recovery.get('preferred_actions', ())) if stalled else set())
                ranked.sort(key=lambda item: (
                    bool(preferred) and item[1].current_actions[actor] not in preferred,
                    item[0] is None, item[0] or ()))
                actor_plans = [plan for _, plan in ranked]
            # Keep action diversity and a few alternative ring reservations.
            # Proposal order never reserves a slot for an earlier actor.
            chosen, deferred, actions_seen = [], [], set()
            for plan in actor_plans:
                action = plan.current_actions[actor]
                if action not in actions_seen:
                    chosen.append(plan)
                    actions_seen.add(action)
                else:
                    deferred.append(plan)
            plans.extend((chosen + deferred)[:self.max_candidates])
            local_wait, wait_remaining = None, None
            if self.joint_assessment:
                from ..local_entry import wait_opportunity, wait_remaining_loops
                wait_remaining = wait_remaining_loops(context, memory, actor,
                    instance, self.wait_max_h)
                local_wait = wait_opportunity(context, memory, actor, target_id,
                    instance, self.wait_max_h)
            can_wait = local_wait is not None or self._can_wait(context, actor, target_id, squad)
            if self.joint_assessment:
                can_wait = can_wait and wait_remaining >= context.H and not stalled \
                    and not (execution and execution['required'])
            if can_wait:
                plans.append(make_plan(
                    self.name, {actor: STOP}, target_id, {actor: STOP},
                    [EffectClaim("position", actor, None, max(context.H, 1),
                                 HEURISTIC, "bounded_wait")],
                    ResourceClaims(controls=(actor,)),
                    Commitment("waiting", 0, max_loops=h_loops(
                        context, self.wait_max_h)),
                    {}, snapshot_id, fingerprint,
                    instance_id=instance.instance_id if instance else None,
                    dependencies=(Dependency("target_valid", (target_id,)),),
                    goal_kind="enemy",
                    goal_data={"target_id": target_id, "waypoint": primary_goal,
                               **({"local_entry_wait": local_wait} if local_wait else {})},
                    memory_delta=self._task_delta(
                        context, instance, actor, target_id, primary_goal, "waiting",
                        wait_remaining=wait_remaining),
                    rationale=("bounded local fire arrival synchronization" if local_wait
                               else "bounded precontact cooldown synchronization")))
        telemetry.count(self.name, "proposed", len(plans))
        return plans

    def _range_band(self, context: Context, actor: str, target_id: str):
        if self.legal_reach:
            reach = context.effective_attack_reach(actor, target_id)
            if reach is None or reach < 0:
                return None
            bodies = (context.snapshot.allies[actor].radius
                      + context.snapshot.enemies[target_id].radius)
            if reach < bodies:
                return None
            return max(bodies, reach - min(.5, (reach - bodies) / 2))
        relation = context.snapshot.relations[actor].get(target_id)
        if relation is not None and relation.native_weapon_range:
            return max(relation.native_weapon_range - 0.5, 0.5)
        return None


def _objective(context, actor, target_id, goal, legal_reach=False):
    """2.2: melee evaluate the CONTACT SLOT; ranged evaluate the RANGE BAND
    around the target — never the enemy centre for everyone."""
    capability = context.capabilities[actor]
    if capability.combat_class == "melee":
        return {"kind": "contact_point", "position": list(goal)}
    relation = context.snapshot.relations[actor].get(target_id)
    r_max = (context.effective_attack_reach(actor, target_id) if legal_reach else
             (relation.native_weapon_range if relation is not None else None) or 5.0)
    return {"kind": "range_band", "target": target_id,
            "r_min": max(r_max - 1.5, 0.0), "r_max": r_max}
