"""Observation memory, task instances and the real commit path (C1–C3).

One decision cycle: observe -> compare with the last committed record ->
update confirmed facts (progress, arrivals, invalidations) -> candidates
(read-only) -> joint decision -> verify final actions -> commit task
changes and command records. propose() may never mutate any of this.
"""
import copy
import math
from typing import Any, Dict, List, Optional, Tuple

from .context import distance
from .contracts import CandidatePlan, SkillInstance, Snapshot
from .encounters import EncounterTracker
from .battle_intent import BattleIntent
from .combat_progress import CombatProgress
from .contact_progress import ContactProgress
from .burst_progress import BurstProgress
from .recovery import RecoveryMemory
from .combat_exchange import PressureMemory, ReentryWindowMemory, safe_reentry_window

GOAL_KINDS = ("enemy", "ally", "position", "area", "internal")


class TacticalMemory:
    def __init__(self, observation_ttl_loops: int = 240,
                 search_revisit_cooldown_loops: int = 160,
                 retry_ttl_loops: int = 96, retry_cooldown_h: float = 2.0,
                 no_progress_window_h: float = 3.0,
                 encounter_config: Optional[Dict] = None):
        self.observation_ttl_loops = observation_ttl_loops
        self.search_revisit_cooldown_loops = search_revisit_cooldown_loops
        self.retry_ttl_loops = retry_ttl_loops
        self.retry_cooldown_h = retry_cooldown_h
        self.no_progress_window_h = no_progress_window_h
        self.encounter_config = copy.deepcopy(encounter_config or {})
        self.encounters = EncounterTracker(self.encounter_config)
        self.battle = BattleIntent(self.encounter_config)
        self.combat_progress = CombatProgress(self.encounter_config)
        self.contact_progress = ContactProgress(self.encounter_config)
        self.burst_progress = BurstProgress(self.encounter_config)
        self.recovery = RecoveryMemory(self.encounter_config)
        self.pressure_memory = PressureMemory(self.encounter_config)
        self.reentry_window = ReentryWindowMemory(self.encounter_config)
        self.lifecycle_counts: Dict[str, Dict[str, int]] = {}
        self.last_committed_loop = None
        self.episode = None
        self.last_game_loop = None
        self.last_snapshot_id = None
        self.observed_key = None             # (snapshot_id, game_loop) guard
        self.instances: Dict[str, SkillInstance] = {}
        self.last_committed_actions: Dict[str, int] = {}
        self.committed: bool = False         # commit idempotence (C2)
        self.committed_positions: Dict[str, Tuple[int, Tuple[float, float]]] = {}
        self.committed_risk: Dict[str, int] = {}
        self.move_history: Dict[str, List[Tuple[int, Tuple[float, float]]]] = {}
        self.enemy_last_seen: Dict[str, Tuple[int, Tuple[float, float]]] = {}
        self.visited_search_cells: Dict[str, int] = {}     # arrival-confirmed
        self.search_targets: Dict[str, Tuple[int, Tuple[float, float]]] = {}
        self.unreachable_search_cells: Dict[str, int] = {}  # temporary, TTL
        # retry cooldowns: (skill, actors-id, goal) -> loop until which a new
        # instance of the SAME failed task may not be re-created (C3)
        self.retry_records: Dict[Tuple[str, str, Any], int] = {}
        # Cross-skill combat intent survives replacement of a G2/G3/G4 task.
        # Deadlines are frozen on real direction changes, never on each step.
        self.combat_cycles: Dict[str, Dict[str, Any]] = {}
        self.attack_history: Dict[str, Dict[str, Any]] = {}
        self.movement_intents: Dict[str, Dict[str, Any]] = {}
        self.pending_moves: Dict[str, Dict[str, Any]] = {}
        self.blocked_directions: Dict[Tuple[str, int], Dict[str, Any]] = {}

    # -- lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        self.episode = None
        self.last_game_loop = None
        self.last_snapshot_id = None
        self.observed_key = None
        self.instances.clear()
        self.last_committed_actions.clear()
        self.committed = False
        self.committed_positions.clear()
        self.committed_risk.clear()
        self.move_history.clear()
        self.enemy_last_seen.clear()
        self.visited_search_cells.clear()
        self.search_targets.clear()
        self.unreachable_search_cells.clear()
        self.retry_records.clear()
        self.combat_cycles.clear()
        self.attack_history.clear()
        self.movement_intents.clear()
        self.pending_moves.clear()
        self.blocked_directions.clear()
        self.encounters = EncounterTracker(self.encounter_config)
        self.battle = BattleIntent(self.encounter_config)
        self.combat_progress = CombatProgress(self.encounter_config)
        self.contact_progress = ContactProgress(self.encounter_config)
        self.burst_progress = BurstProgress(self.encounter_config)
        self.recovery = RecoveryMemory(self.encounter_config)
        self.pressure_memory = PressureMemory(self.encounter_config)
        self.reentry_window = ReentryWindowMemory(self.encounter_config)
        self.lifecycle_counts.clear()
        self.last_committed_loop = None

    def begin_step(self, snapshot: Snapshot) -> bool:
        """False => identical snapshot already committed: serve the cache."""
        if self.episode != snapshot.episode:
            self.reset()
            self.episode = snapshot.episode
        elif (self.last_game_loop is not None
              and snapshot.game_loop < self.last_game_loop):
            self.reset()
            self.episode = snapshot.episode
        self.last_game_loop = snapshot.game_loop
        if self.last_snapshot_id == snapshot.snapshot_id:
            return False
        self.last_snapshot_id = snapshot.snapshot_id
        self.committed = False          # a fresh snapshot may be committed
        return True

    @staticmethod
    def _actors_key(actors) -> str:
        # task_retry_key already returns a joined string. Do not sort its
        # characters when that key is passed back into a query or record.
        return actors if isinstance(actors, str) else "+".join(sorted(actors))

    def retry_blocked(self, skill: str, actors, goal,
                      game_loop: int) -> bool:
        """PURE query (C3): never deletes or mutates retry records."""
        until = self.retry_records.get(
            (skill, self._actors_key(actors), goal))
        return until is not None and game_loop < until

    def cleanup_retry_records(self, game_loop: int) -> None:
        """TTL housekeeping — observation phase only (C3)."""
        for other in list(self.retry_records):
            if self.retry_records[other] < game_loop - self.retry_ttl_loops:
                del self.retry_records[other]

    @staticmethod
    def task_retry_key(skill: str, actors, goal_kind: str, goal: str,
                       goal_data=None) -> tuple:
        """3.3: the single normalized retry key. goal_kind decides the goal
        normalization (enemy id / waypoint CELL / undirected group pair /
        internal purpose) so propose-side checks and finish-time records can
        never drift apart."""
        participants = TacticalMemory._actors_key(actors)
        data = goal_data or {}
        if goal_kind == "enemy":
            return (skill, participants, f"enemy:{goal}")
        if goal_kind == "position":
            waypoint = data.get("waypoint") or data.get("position") or goal
            cell = f"{int(waypoint[0] // 8)}:{int(waypoint[1] // 8)}" \
                if isinstance(waypoint, (tuple, list)) and len(waypoint) == 2 \
                else str(waypoint)
            return (skill, participants, f"waypoint:{cell}")
        if goal_kind == "group_link":
            a = tuple(sorted(data.get("group_a", [])))
            b = tuple(sorted(data.get("group_b", [])))
            left, right = sorted((a, b))
            return (skill, participants, ("link", left, right))
        return (skill, participants, str(goal))

    def record_retry(self, skill: str, actors, goal: str, game_loop: int,
                     cooldown_loops: int) -> None:
        self.retry_records[(skill, self._actors_key(actors), goal)] = \
            game_loop + cooldown_loops

    # -- observation phase (idempotent; replaces same-clock records) --------
    def displacement(self, actor: str, loops: int, snapshot: Snapshot) -> float:
        """Distance travelled over the recorded window (loop-based)."""
        history = self.move_history.get(actor, [])
        threshold = snapshot.game_loop - loops
        past = [position for loop, position in history if loop <= threshold]
        if not past:
            return 0.0
        return distance(past[-1], snapshot.allies[actor].position)

    def observe(self, snapshot: Snapshot, risk_by_actor=None) -> None:
        key = (snapshot.snapshot_id, snapshot.game_loop)
        if self.observed_key == key:
            return
        # move history: one record per game_loop — a corrected same-clock
        # observation REPLACES the previous record instead of appending
        for actor, unit in snapshot.allies.items():
            history = self.move_history.setdefault(actor, [])
            if history and history[-1][0] == snapshot.game_loop:
                history[-1] = (snapshot.game_loop, unit.position)
            else:
                history.append((snapshot.game_loop, unit.position))
            if len(history) > 16:
                del history[:len(history) - 16]
        for enemy_id, enemy in snapshot.enemies.items():
            self.enemy_last_seen[enemy_id] = (snapshot.game_loop, enemy.position)
        stale = [e for e, (loop, _) in self.enemy_last_seen.items()
                 if snapshot.game_loop - loop > self.observation_ttl_loops]
        for enemy_id in stale:
            del self.enemy_last_seen[enemy_id]
        # C1: the risk BASELINE stays untouched here; commit() compares
        # progress against the PREVIOUS step's records first and only then
        # installs this observation as the new baseline.
        if risk_by_actor is not None:
            self.observed_risk = dict(risk_by_actor)
        self.observed_key = key

    def update_task_facts(self, snapshot: Snapshot, context) -> List[str]:
        """3.1: the SINGLE ordered observation phase per real game_loop.

        identity/protocol invalid -> observed progress -> completion ->
        deadline -> no-progress. Completion is ALWAYS judged before the
        deadline: an instance that arrives on its deadline frame is
        recorded as ARRIVED, never as a failed timeout."""
        events: List[str] = []
        if context is not None:
            context.deployment_memory = self
            self._observe_movement(snapshot, context)
            self.pressure_memory.observe(context)
            self.recovery.observe(context)
            self.combat_progress.observe(context)
            self.contact_progress.observe(context)
            self.burst_progress.observe(context)
            self.reentry_window.observe(context)
            self._observe_combat_memory(snapshot, context)
            self.encounters.observe(context)
            self.battle.observe(context)
        self.cleanup_retry_records(snapshot.game_loop)
        events.extend(self.retire_invalid(snapshot))   # identity/protocol
        # deadlines for every remaining instance, AFTER completion checks
        for instance_id in list(self.instances):
            instance = self.instances.get(instance_id)
            if instance is None:
                continue
            if context is not None and self.task_progress(snapshot, instance, context):
                instance.last_progress_loop = snapshot.game_loop
            # Observed completion must win over a same-frame deadline.
            self._check_search_completion(instance, snapshot, events)
            if instance.skill == "precombat_rally" and context is not None:
                reason = self.encounters.task_outcome(context, instance)
                if reason:
                    self.finish_task(instance_id, reason, snapshot)
                    events.append(f"{reason}:{instance_id}")
            if instance.skill == "combat_rejoin" and context is not None:
                from .combat_groups import task_outcome
                reason = task_outcome(context, instance)
                if reason:
                    self.finish_task(instance_id, reason, snapshot)
                    events.append(f"{reason}:{instance_id}")
            if instance.skill == "coordinated_engage" and context is not None \
                    and instance.goal_kind == "enemy" and all(
                        any(context.contact_status(actor, enemy) == "in_range"
                            for enemy in context.attackable_enemies(actor))
                        for actor in instance.actors):
                self.finish_task(instance_id, "contact_established", snapshot)
                events.append(f"contact_established:{instance_id}")
            self._check_support_completion(instance, snapshot, context, events)
            if instance.skill == "disengage_reenter" and \
                    instance.phase == "ESCAPE" and context is not None and \
                    self._deadline_reached(instance, snapshot.game_loop):
                safe = all(not context.risk_at(a, snapshot.allies[a].position).threats
                           and not context.risk_at(a, snapshot.allies[a].position).effect_weapon_near
                           for a in instance.actors)
                if self.encounter_config.get('common_combat', {}).get('reentry_window_enabled', True):
                    safe = all(safe_reentry_window(context, a,
                        pressure_memory_enabled=self.encounter_config.get('common_combat', {}).get(
                            'pressure_memory_enabled', True)) for a in instance.actors)
                if safe:
                    self.finish_task(instance_id, "safe", snapshot)
                    events.append(f"safe:{instance_id}")
            if self.instances.get(instance_id) is None:
                continue                       # completed this observation
            if self._deadline_reached(instance, snapshot.game_loop):
                self.finish_task(instance_id, "timeout", snapshot)
                events.append(f"task_timeout:{instance_id}")
                continue
            # Waiting/holding/healing are not failed movement. Only duties
            # promising approach progress use this shorter stall deadline.
            moving_duty = ((instance.skill == "coordinated_engage"
                            and instance.phase == "approaching")
                           or (instance.skill == 'burst_engage'
                               and instance.phase in ('approaching', 'bursting'))
                           or instance.skill == "regroup")
            stall_limit = instance.extra.get("no_progress_loops")
            progress_start = max(instance.last_progress_loop,
                                 instance.phase_start_loop)
            if moving_duty and stall_limit is not None and (
                    snapshot.game_loop - progress_start >= stall_limit):
                self.finish_task(instance_id, "no_progress", snapshot)
                events.append(f"task_no_progress:{instance_id}")
        # Observation may complete/shrink tasks or update their anchor. The
        # coordinator must build facts from that resulting state, not from a
        # squad cache populated while judging an earlier instance.
        if context is not None:
            for name in ('_deployment_squads', '_deployment_phases', '_deployment_anchors'):
                context.__dict__.pop(name, None)
        return events

    def _combat_ttl(self, snapshot):
        lifecycle = self.encounter_config.get("lifecycle", {})
        return max(snapshot.step_mul, int(round(snapshot.step_mul *
                   lifecycle.get("combat_cycle_memory_h", 6.0))))

    def _observe_movement(self, snapshot, context):
        """Compare a real submitted move with its next observed displacement.

        Blockage is local and short-lived, never an inferred impassable map.
        A skipped observation is not evidence that one command failed.
        """
        for actor, record in list(self.movement_intents.items()):
            unit = snapshot.allies.get(actor)
            if unit is None or not unit.alive or unit.tag != record["tag"] \
                    or (record.get("enemy") and record["enemy"] not in snapshot.enemies) \
                    or snapshot.game_loop - record["last_loop"] > self._combat_ttl(snapshot):
                del self.movement_intents[actor]
        for key, record in list(self.blocked_directions.items()):
            unit = snapshot.allies.get(key[0])
            if unit is None or not unit.alive or unit.tag != record["tag"] \
                    or snapshot.game_loop >= record["until_loop"] \
                    or distance(unit.position, record["position"]) > record["local_radius"]:
                del self.blocked_directions[key]
        for actor, record in list(self.pending_moves.items()):
            if snapshot.game_loop <= record["loop"]:
                continue
            del self.pending_moves[actor]
            unit = snapshot.allies.get(actor)
            if unit is None or not unit.alive or unit.tag != record["tag"] \
                    or snapshot.game_loop - record["loop"] != record["window"]:
                continue
            direction, expected = record["direction"], record["expected"]
            actual = sum((unit.position[i] - record["position"][i]) * direction[i]
                         for i in (0, 1))
            key = (actor, record["action"])
            if expected > .1 and actual < .35 * expected:
                route_end = tuple(record["position"][i] + direction[i] * expected
                                  for i in (0, 1))
                # These are possible body blockers, not a claim that terrain
                # is passable. Keep the evidence with this observation/TTL.
                friends, enemies = self._route_bodies(
                    context, actor, record["position"], route_end) if context is not None else ((), ())
                self.blocked_directions[key] = {
                    "tag": unit.tag, "position": unit.position,
                    "local_radius": max(1., 2 * expected),
                    "until_loop": snapshot.game_loop + 3 * record["window"],
                    "severity": min(1., max(0., 1 - actual / expected)),
                    "friendly_blockers": {a: snapshot.allies[a].tag for a in friends},
                    "enemy_blocker_observed": bool(enemies)}
            else:
                self.blocked_directions.pop(key, None)

    @staticmethod
    def _segment_distance(point, start, end):
        delta = tuple(end[i] - start[i] for i in (0, 1))
        length2 = sum(x * x for x in delta)
        t = max(0., min(1., sum((point[i] - start[i]) * delta[i]
                               for i in (0, 1)) / length2)) if length2 else 0.
        return distance(point, tuple(start[i] + t * delta[i] for i in (0, 1)))

    def _route_bodies(self, context, actor, start, end):
        unit = context.snapshot.allies[actor]
        if unit.is_flying:
            return (), ()  # air stacking is not a ground-body obstruction
        def intersects(other):
            return (not other.is_flying and self._segment_distance(
                other.position, start, end) < unit.radius + other.radius - .02)
        friends = tuple(a for a, other in context.snapshot.allies.items()
                        if a != actor and other.alive and intersects(other))
        enemies = tuple(e for e, other in context.snapshot.enemies.items()
                        if intersects(other))
        return friends, enemies

    def _joint_route_relief(self, context, actor, action, record,
                            joint_actions, joint_positions):
        """Bounded discount when observed friendly obstruction is vacated.

        Both endpoints and synchronized straight-line paths must clear. The
        remaining half-cost preserves unmodeled terrain/collision uncertainty;
        this query never erases observed failure or extends its lifetime.
        """
        witnesses = record.get("friendly_blockers", {})
        if not witnesses or record.get("enemy_blocker_observed"):
            return 0.
        units = context.snapshot.allies
        if any(a not in units or not units[a].alive or units[a].tag != tag
               for a, tag in witnesses.items()):
            return 0.
        unit = units[actor]
        start = unit.position
        end = joint_positions.get(actor, context.predict_position(actor, action).position)
        friends, enemies = self._route_bodies(context, actor, start, end)
        if enemies:
            return 0.
        # All current friendly blockers must move clear, including bodies
        # newly entering this path since the original failed command.
        for other_actor in set(friends) | set(witnesses):
            other = units[other_actor]
            other_action = joint_actions.get(other_actor)
            if other_actor in friends and (other_action is None
                    or other_action not in context.legal_actions(other_actor)
                    or context.decode(other_actor, other_action).kind != "move"
                    or context.speed(other_actor) is None):
                return 0.
            target = joint_positions.get(other_actor, other.position)
            radius = unit.radius + other.radius
            if self._segment_distance(target, start, end) < radius + .02:
                return 0.
            # Crossing paths can collide despite clear final destinations.
            relative_start = tuple(start[i] - other.position[i] for i in (0, 1))
            relative_end = tuple(end[i] - target[i] for i in (0, 1))
            minimum = self._segment_distance((0., 0.), relative_start, relative_end)
            if minimum < min(radius, distance(start, other.position)) - .02:
                return 0.
        # A different moving body must not enter the freshly vacated route.
        for other_actor, other in units.items():
            if other_actor == actor or not other.alive or other.is_flying:
                continue
            target = joint_positions.get(other_actor, other.position)
            radius = unit.radius + other.radius
            relative_start = tuple(start[i] - other.position[i] for i in (0, 1))
            relative_end = tuple(end[i] - target[i] for i in (0, 1))
            if self._segment_distance((0., 0.), relative_start, relative_end) < \
                    min(radius, distance(start, other.position)) - .02:
                return 0.
        return .5

    def movement_blockage(self, context, actor, action, *, joint_actions=None,
                          joint_positions=None):
        """Pure [0,1] cost; joint plans may discount verified friendly vacating."""
        record = self.blocked_directions.get((actor, action))
        unit = context.snapshot.allies[actor]
        capability = context.capabilities[actor]
        if not unit.alive or capability.healer or capability.suicide_attack \
                or context.decode(actor, action).kind != "move":
            return 0.
        if record and record["tag"] == unit.tag \
                and context.snapshot.game_loop < record["until_loop"] \
                and distance(unit.position, record["position"]) <= record["local_radius"]:
            relief = self._joint_route_relief(context, actor, action, record,
                        joint_actions, joint_positions) if joint_actions is not None \
                        and joint_positions is not None else 0.
            return record["severity"] * (1. - relief)
        return 0.

    def movement_conflict(self, context, actor, action):
        """Intent reversal, barren-loop and observed round-trip costs share one bound.

        The round-trip term is the cross-skill ping-pong guard: a move that
        returns to a position abandoned within the last few windows, after a
        real walk and observed damage, costs 1.0 unless the same move is
        genuine relief. It reaches candidate filtering, joint scoring,
        repair and baseline through this single channel.
        """
        from .deployment import roundtrip_cost
        return max(self._intent_movement_conflict(context, actor, action),
                   self.movement_cycle_cost(context, actor, action),
                   roundtrip_cost(context, actor, action, self))

    def movement_cycle_cost(self, context, actor, action):
        return self.combat_progress.cycle_cost(context, actor, action, self)

    def advance_credit_scale(self, context, actor, action):
        """Do not pay again for retracing a recent unproductive approach."""
        return 1. - self.movement_cycle_cost(context, actor, action)

    def contact_recovery(self, context, actor):
        return self.contact_progress.recovery(context, actor, self)

    def return_fire_pending(self, context, actor):
        return self.combat_progress.return_pending(context, actor, self)

    def return_fire_cost(self, context, actor, action):
        return self.combat_progress.return_cost(context, actor, action, self)

    def next_fire_cost(self, context, actor, action, *, eligible_targets=None):
        return self.combat_progress.next_fire_cost(
            context, actor, action, self, eligible_targets=eligible_targets)

    def reentry_allowed(self, context, actor, position, targets=None):
        """Pure, short-lived cross-skill return-to-danger guard."""
        return self.reentry_window.allowed(context, actor, position, targets)

    def _intent_movement_conflict(self, context, actor, action):
        """Pure reversal cost for ALL final actions, independent of skill label.

        The short deadline is fixed at a real intent change. Emergency moves,
        a blocked old direction, and a move leaving a physical danger band
        remain free to override it; sideways movement is not reversal.
        """
        record = self.movement_intents.get(actor)
        unit = context.snapshot.allies[actor]
        capability = context.capabilities[actor]
        if not unit.alive or capability.healer or capability.suicide_attack:
            return 0.
        if not record or record["tag"] != unit.tag \
                or context.snapshot.game_loop >= record["until_loop"] \
                or context.decode(actor, action).kind != "move" \
                or self.combat_emergency(context, actor) \
                or self.movement_blockage(context, actor, record["action"]) > .5:
            return 0.
        point = context.predict_position(actor, action).position
        vector = (point[0] - unit.position[0], point[1] - unit.position[1])
        length = math.hypot(*vector)
        if length <= .05:
            return 0.
        current = context.risk_at(actor, unit.position)
        after = context.risk_at(actor, point)
        # A measurable escape from current danger outranks locomotion intent.
        if len(after.threats) + len(after.effect_weapon_near) < \
                len(current.threats) + len(current.effect_weapon_near):
            return 0.
        cosine = sum(vector[i] * record["direction"][i] for i in (0, 1)) / length
        return max(0., min(1., -cosine)) if cosine < -.25 else 0.

    def _commit_movement(self, snapshot, context, actions, plans):
        if context is None:
            return
        duration = max(0, int(round(snapshot.step_mul * self.encounter_config.get(
            "lifecycle", {}).get("combat_cycle_commit_h", 2.))))
        by_actor = {a: p for p in plans for a in p.controlled_actors}
        purposeful = {"coordinated_engage", "combat_reposition", "disengage_reenter",
                      "combat_rejoin", "regroup", "precombat_rally"}
        for actor, action in actions.items():
            unit = snapshot.allies.get(actor)
            if unit is None or not unit.alive:
                continue
            semantic = context.decode(actor, action)
            if semantic.kind != "move":
                continue
            point = context.predict_position(actor, action).position
            vector = (point[0] - unit.position[0], point[1] - unit.position[1])
            length = math.hypot(*vector)
            if length <= .05:
                continue
            direction = tuple(x / length for x in vector)
            self.pending_moves[actor] = {"tag": unit.tag, "loop": snapshot.game_loop,
                "window": snapshot.step_mul, "position": unit.position, "direction": direction,
                "expected": length, "action": action}
            previous = self.movement_intents.get(actor)
            opposing = previous and sum(direction[i] * previous["direction"][i]
                                         for i in (0, 1)) < -.25
            if previous and opposing and snapshot.game_loop >= previous["until_loop"]:
                self.movement_intents.pop(actor, None)
                previous = None
            plan = by_actor.get(actor)
            if plan is None or plan.primary_skill not in purposeful:
                continue  # spacing/baseline cannot lock out a deliberate approach
            purpose = (plan.primary_skill, plan.goal, plan.commitment.phase)
            if previous is None or (previous["purpose"] != purpose
                                    and snapshot.game_loop >= previous["until_loop"]):
                previous = {"tag": unit.tag, "purpose": purpose, "direction": direction,
                            "action": action, "until_loop": snapshot.game_loop + duration,
                            "enemy": plan.goal if plan.goal_kind == "enemy"
                                and plan.goal in snapshot.enemies else None}
                self.movement_intents[actor] = previous
            previous["last_loop"] = snapshot.game_loop

    def _observe_combat_memory(self, snapshot, context):
        """Expire intent by observed identity/target loss, not skill labels."""
        for actor in self.combat_progress.fired:
            self.movement_intents.pop(actor, None)
            self.combat_cycles.pop(actor, None)
        ttl = self._combat_ttl(snapshot)
        for records in (self.combat_cycles, self.attack_history):
            for actor, record in list(records.items()):
                unit = snapshot.allies.get(actor)
                if unit is None or not unit.alive or unit.tag != record["tag"] \
                        or record["target"] not in snapshot.enemies \
                        or snapshot.game_loop - record["last_loop"] > ttl:
                    del records[actor]
        for actor, unit in snapshot.allies.items():
            if not unit.alive:
                continue
            for order in snapshot.raw_state.get("allies", {}).get(actor, {}).get("orders", []):
                target = order.get("target_unit")
                if order.get("kind") == "attack" and target in snapshot.enemies:
                    self.attack_history[actor] = {
                        "target": target, "tag": unit.tag,
                        "last_loop": snapshot.game_loop}
                    break
            cycle = self.combat_cycles.get(actor)
            if cycle and cycle["mode"] == "approach" \
                    and not cycle.get("edge_started") \
                    and self.one_step_attack_target(context, actor, cycle["target"]):
                # A long approach only starts its short boundary-crossing
                # allowance once the next step can unlock a real attack.
                duration = max(0, int(round(snapshot.step_mul *
                    self.encounter_config.get("lifecycle", {}).get("combat_cycle_commit_h", 2.0))))
                cycle["until_loop"] = snapshot.game_loop + duration
                cycle["edge_started"] = True

    def battle_state(self, context):
        """Observed team objective; a query never resets its commitment."""
        return {**self.battle.summary(), **self.combat_progress.summary()}

    def battle_target(self, context, actor):
        return self.battle.target_for(context, actor)

    def retreat_exhausted(self, context, actor):
        return self.battle.exhausted(context, actor)

    def recent_attack_target(self, context, actor, candidates):
        """A real firing intention, independent of the current skill owner."""
        candidates = set(candidates)
        instance = self.instance_for("focus_fire", actor)
        if instance is not None and instance.goal in candidates:
            return instance.goal
        unit = context.snapshot.allies[actor]
        for order in context.snapshot.raw_state.get("allies", {}).get(actor, {}).get("orders", []):
            if order.get("kind") == "attack" and order.get("target_unit") in candidates:
                return order["target_unit"]
        record = self.attack_history.get(actor)
        if record and record["tag"] == unit.tag and record["target"] in candidates \
                and context.snapshot.game_loop - record["last_loop"] <= self._combat_ttl(context.snapshot):
            return record["target"]
        return None

    def combat_cycle(self, context, actor):
        """Pure lookup of a bounded intent; proposal generation never mutates it."""
        record = self.combat_cycles.get(actor)
        unit = context.snapshot.allies[actor]
        if record and record["tag"] == unit.tag and record["target"] in context.snapshot.enemies \
                and context.snapshot.game_loop - record["last_loop"] <= self._combat_ttl(context.snapshot):
            return record
        return None

    def combat_emergency(self, context, actor):
        """Expected contact and small chip damage do not undo approach intent."""
        unit = context.snapshot.allies[actor]
        if self.pressure_memory.enabled and self.pressure_memory.evidence(context).get(actor, 0.):
            from .combat_exchange import imminent_exchange_danger
            if imminent_exchange_danger(context, actor):
                return True  # same short observed danger interrupts every skill's intent
        risk = context.risk_at(actor, unit.position)
        critical = self.encounter_config.get("disengage_reenter", {}).get(
            "critical_health_fraction", .35)
        if self.encounter_config.get('common_combat', {}).get('joint_assessment_enabled', True):
            from .combat_exchange import exposure_window, next_fire_horizon, observed_pressure
            facts = exposure_window(context, actor, horizon_h=next_fire_horizon(context, actor),
                                    pressure_memory_enabled=self.pressure_memory.enabled)
            severe = observed_pressure(context, actor, enabled=self.pressure_memory.enabled) > 0 \
                and facts['amount'] >= unit.effective_health
        else:
            severe = unit.health_fraction <= critical and (
                unit.health_lost + unit.shield_lost > 0 or unit.health_fraction <= critical / 2)
        if severe or risk.effect_weapon_near or unit.health_lost >= .15 * max(unit.max_health, 1.):
            return True
        record = self.combat_cycle(context, actor)
        return bool(record and len(set(risk.threats) - set(record.get("expected_threats", ()))) >= 2)

    def cycle_holds(self, context, actor, mode):
        record = self.combat_cycle(context, actor)
        return bool(record and record["mode"] == mode
                    and context.snapshot.game_loop < record["until_loop"])

    @staticmethod
    def one_step_attack_target(context, actor, preferred=None):
        """A legal MOVE may unlock a following attack; never return an attack action."""
        targets = [preferred] if preferred in context.snapshot.enemies \
            else list(context.snapshot.enemies)
        for enemy in targets:
            for action in context.move_destinations(actor):
                position = context.predict_position(actor, action).position
                opportunity = context.projected_attack_opportunity(
                    actor, enemy, position, horizon=2 * context.H)
                if opportunity.available and opportunity.ready:
                    return enemy
        return None

    def _commit_combat_memory(self, snapshot, context, actions, plans):
        if context is None:
            return
        lifecycle = self.encounter_config.get("lifecycle", {})
        duration = max(0, int(round(snapshot.step_mul *
                       lifecycle.get("combat_cycle_commit_h", 2.0))))
        by_actor = {a: p for p in plans for a in p.controlled_actors}
        for actor, action in actions.items():
            unit = snapshot.allies.get(actor)
            if unit is None or not unit.alive:
                continue
            semantic = context.decode(actor, action)
            if semantic.kind == "attack" and semantic.target in snapshot.enemies:
                self.attack_history[actor] = {
                    "target": semantic.target, "tag": unit.tag,
                    "last_loop": snapshot.game_loop}
                # Only an observed weapon cycle releases locomotion/return
                # memory; a contact attack command is not proof of firing.
                continue
            plan = by_actor.get(actor)
            if plan is None or semantic.kind != "move":
                continue
            mode = ("approach" if plan.primary_skill == "coordinated_engage"
                    or (plan.primary_skill == "disengage_reenter"
                        and plan.commitment.phase == "REENTER") else
                    "escape" if plan.primary_skill == "disengage_reenter"
                    and plan.commitment.phase == "ESCAPE" else None)
            if mode is None:
                continue
            old = self.combat_cycle(context, actor)
            target = plan.goal if plan.goal in snapshot.enemies else \
                old["target"] if old else context.nearest_enemy(actor)
            if target is None:
                continue
            same_encounter = old is not None and (old["target"] == target or distance(
                snapshot.enemies[old["target"]].position,
                snapshot.enemies[target].position) <= 8.0)
            if not same_encounter or old["mode"] != mode:
                old = {"target": target, "tag": unit.tag, "mode": mode,
                       "start_loop": snapshot.game_loop,
                       "until_loop": snapshot.game_loop + duration}
                self.combat_cycles[actor] = old
            old["last_loop"] = snapshot.game_loop
            # Entering the already-predicted threat band is not a surprise.
            position = context.predict_position(actor, action).position
            old["expected_threats"] = tuple(context.risk_at(actor, position).threats)

    @staticmethod
    def _deadline_reached(instance, game_loop: int) -> bool:
        phase_max = instance.extra.get("phase_max_loops")
        return (instance.max_loops is not None
                and game_loop - instance.start_loop >= instance.max_loops) or (
                    phase_max is not None
                    and game_loop - instance.phase_start_loop >= phase_max)

    def _check_support_completion(self, instance, snapshot, context, events):
        """Observed support outcomes win over a deadline on the same frame."""
        if instance.skill != "heal_support":
            return
        reason = None
        if instance.phase in ("meet_patient", "sustained_heal"):
            patient = snapshot.allies.get(instance.goal)
            if patient is not None and patient.health >= patient.max_health:
                reason = "patient_recovered"
            elif instance.phase == "meet_patient" and context is not None:
                healer = instance.extra.get("healer")
                if healer and context.heal_action(healer, instance.goal) is not None:
                    reason = "patient_reached"
        elif instance.phase == "escape" and context is not None:
            actor = instance.actors[0]
            risk = context.risk_at(actor, snapshot.allies[actor].position)
            if not risk.threats and not risk.effect_weapon_near:
                reason = "safe"
        elif instance.phase == "escort" and self._deadline_reached(
                instance, snapshot.game_loop):
            # Holding a useful escort band is success, not a stuck approach.
            waypoint = instance.extra.get("waypoint")
            members = [snapshot.allies[a].position
                       for a in instance.extra.get("escort_members", ())
                       if a in snapshot.allies and snapshot.allies[a].alive]
            if members:
                waypoint = (sum(p[0] for p in members) / len(members),
                            sum(p[1] for p in members) / len(members))
            if waypoint is not None and distance(
                    snapshot.allies[instance.actors[0]].position, waypoint
            ) <= instance.extra.get("follow_distance", 5.0):
                reason = "escort_position_reached"
        if reason:
            self.finish_task(instance.instance_id, reason, snapshot)
            events.append(f"{reason}:{instance.instance_id}")

    def _check_search_completion(self, instance, snapshot, events) -> None:
        """Arrival confirmation for one search instance (3.1 step 2)."""
        if instance.skill != "cohesive_search" \
                or instance.goal_kind != "position":
            return
        waypoint = instance.extra.get("waypoint")
        if waypoint is None:
            return
        for actor in instance.actors:
            if actor not in snapshot.allies or \
                    not snapshot.allies[actor].alive:
                continue
            if distance(snapshot.allies[actor].position, waypoint) \
                    <= instance.extra.get("arrival_tolerance", 2.0):
                cell = _cell(waypoint)
                self.finish_task(instance.instance_id, "waypoint_reached",
                                 snapshot)
                events.append(f"search_arrived:{cell}")
                return

    # -- instances ----------------------------------------------------------
    def instance_for(self, skill: str, actor: str) -> Optional[SkillInstance]:
        for instance in self.instances.values():
            if instance.skill == skill and actor in instance.actors:
                return instance
        return None

    def instance_by_id(self, instance_id: str) -> Optional[SkillInstance]:
        return self.instances.get(instance_id)

    def create_instance(self, skill: str, actors, goal: str, phase: str,
                        loop: int, max_loops: Optional[int],
                        ally_tags=(), instance_id: str = "",
                        goal_kind: str = "internal",
                        extra: Optional[Dict[str, Any]] = None) -> SkillInstance:
        extra = dict(extra or {})
        extra.setdefault(
            "retry_key", self.task_retry_key(skill, actors, goal_kind, goal,
                                             extra))
        instance_id = instance_id or f"{skill}:{'+'.join(sorted(actors))}:{goal}:{loop}"
        instance = SkillInstance(
            instance_id=instance_id, skill=skill, actors=tuple(sorted(actors)),
            ally_tags=tuple(ally_tags), goal=goal, goal_kind=goal_kind,
            phase=phase, start_loop=loop, phase_start_loop=loop,
            last_progress_loop=loop, budget_loops=0, executed_steps=0,
            max_loops=max_loops, extra=extra)
        self.instances[instance_id] = instance
        return instance

    def drop_instance(self, instance_id: str, reason: str = "") -> None:
        instance = self.instances.get(instance_id)
        if instance is not None:
            instance.termination_reason = reason or "dropped"
            self._count_lifecycle(instance.skill, "exited")
        self.instances.pop(instance_id, None)

    def finish_task(self, instance_id: str, reason: str,
                    snapshot: Snapshot) -> None:
        """C2: the ONE termination entry point. Writes the observed result
        facts (visited / unreachable / retry) BEFORE releasing the instance —
        never delete first and lose the arrival fact. Idempotent per
        instance+reason."""
        instance = self.instances.get(instance_id)
        if instance is None:
            return
        if instance.extra.get("finished_reason") == reason:
            return
        instance.extra["finished_reason"] = reason
        instance.extra["finished_loop"] = snapshot.game_loop
        if instance.skill == 'burst_engage' and reason in ('timeout', 'no_progress', 'target_lost'):
            for actor in instance.actors:
                self.burst_progress.release_goal(actor, instance.goal)
        if instance.skill == "precombat_rally":
            if reason not in ('rally_ready', 'member_lost', 'switched_or_baseline'):
                self.encounters.close(instance.extra.get("encounter_id"), reason)
        if reason == "waypoint_reached":
            waypoint = instance.extra.get("waypoint")
            if waypoint is not None:
                self.visited_search_cells[_cell(waypoint)] = snapshot.game_loop
        elif reason in ("waypoint_timeout", "escape_failed", "timeout",
                        "no_progress", "cover_emergency", "danger_escalated",
                        "effect_danger"):
            # classify by the TASK, not by the caller's string (3.2): a
            # generic timeout on a search instance is a search timeout and
            # writes unreachable — never visited
            search_timeout = (
                instance.skill == "cohesive_search"
                and reason in ("waypoint_timeout", "timeout", "no_progress")
                and instance.extra.get("waypoint") is not None)
            if search_timeout:
                waypoint = instance.extra.get("waypoint")
                self.unreachable_search_cells[_cell(waypoint)] = \
                    snapshot.game_loop
            # retry cooldown under the SAME normalized key propose() checks
            keys = [instance.extra.get('retry_key') or self.task_retry_key(
                instance.skill, instance.actors, instance.goal_kind,
                instance.goal, instance.extra)]
            if instance.skill == 'combat_rejoin' and instance.extra.get('local_cover_enabled'):
                keys.extend(instance.extra.get('retained_retry_keys', ()))
            until = snapshot.game_loop + instance.extra.get(
                'retry_cooldown_loops', int(round(snapshot.step_mul * self.retry_cooldown_h)))
            for key in keys:
                self.retry_records[key] = until
        elif reason == "switched_or_baseline":
            pass    # normal switch to a better task: no failure penalty
        self.drop_instance(instance_id, reason)

    def retire_invalid(self, snapshot: Snapshot) -> List[str]:
        """Death, tag change, typed target loss, loop-budget exhaustion."""
        retired = []
        for instance_id in list(self.instances):
            instance = self.instances[instance_id]
            actors_alive = [a for a in instance.actors
                            if a in snapshot.allies and snapshot.allies[a].alive]
            tags_by_actor = dict(zip(instance.actors, instance.ally_tags))
            retained = {a for a in actors_alive if a not in tags_by_actor
                        or snapshot.allies[a].tag == tags_by_actor[a]}
            if instance.skill == 'precombat_rally' and retained:
                # Observation-owned shrink preserves the original deadline.
                instance.actors = tuple(a for a in instance.actors if a in retained)
                instance.ally_tags = tuple(snapshot.allies[a].tag for a in instance.actors)
                instance.extra['members'] = list(instance.actors)
                actors_alive = list(instance.actors)
            can_shrink = instance.skill == 'combat_rejoin' \
                and instance.extra.get('local_cover_enabled', False) \
                and bool(retained.intersection(instance.extra.get('joiners', ()))) \
                and bool(retained.intersection(instance.extra.get('supporters', ())))
            if (not actors_alive or len(actors_alive) != len(instance.actors)) and not can_shrink:
                self.finish_task(instance_id, "member_lost", snapshot)
                retired.append(instance_id)
                continue
            if instance.ally_tags and len(retained) != len(instance.actors) and not can_shrink:
                self.finish_task(instance_id, "tag_changed", snapshot)
                retired.append(instance_id)
                continue
            if instance.goal_kind == "enemy" and instance.goal \
                    and instance.goal not in snapshot.enemies:
                self.finish_task(instance_id, "target_lost", snapshot)
                retired.append(instance_id)
                continue
            if instance.skill == 'burst_engage' and instance.goal_kind == 'enemy':
                prior = [row for row in instance.extra.get('approach_target_history', ())
                         if row[1] == instance.goal]
                tag = snapshot.raw_state.get('visible_enemies', {}).get(instance.goal, {}).get('tag')
                if prior and prior[-1][2] != tag:
                    self.finish_task(instance_id, 'target_lost', snapshot)
                    retired.append(instance_id)
                    continue
            if instance.goal_kind == "ally" and (
                    instance.goal not in snapshot.allies
                    or not snapshot.allies[instance.goal].alive):
                self.finish_task(instance_id, "target_lost", snapshot)
                retired.append(instance_id)
                continue
            # deadlines are judged in update_task_facts AFTER completion
            # checks — never deleted here first (3.1)
        return retired

    # -- progress evidence (C1: task-relevant, compared with last record) ---
    def task_progress(self, snapshot: Snapshot, instance: SkillInstance,
                      context, previous_positions=None,
                      previous_risk=None) -> bool:
        previous = (previous_positions if previous_positions is not None
                    else self.committed_positions)
        if instance.skill == 'burst_engage':
            target = snapshot.enemies.get(instance.goal)
            if target is None:
                return False
            tag = snapshot.raw_state.get('visible_enemies', {}).get(instance.goal, {}).get('tag')
            history = instance.extra.get('approach_target_history', ())
            for actor in instance.actors:
                record = previous.get(actor)
                if record is None or actor not in snapshot.allies:
                    continue
                enemy_positions = [row[3] for row in history if tuple(row[:3]) ==
                                   (record[0], instance.goal, tag)]
                if enemy_positions and distance(record[1], enemy_positions[-1]) - distance(
                        snapshot.allies[actor].position, target.position) > .25:
                    return True
            return False
        progressed = False
        for actor in instance.actors:
            if actor not in snapshot.allies:
                continue
            record = previous.get(actor)
            if record is not None and distance(
                    record[1], snapshot.allies[actor].position) > 0.25:
                # displacement exists; task-relevant when it closes distance
                # to a typed goal (enemy/ally/position), or risk dropped
                if instance.goal_kind == "enemy" and instance.goal in snapshot.enemies:
                    before = distance(record[1],
                                      snapshot.enemies[instance.goal].position)
                    now = distance(snapshot.allies[actor].position,
                                   snapshot.enemies[instance.goal].position)
                    if now < before - 0.25:
                        progressed = True
                elif instance.goal_kind in ("position", "area") and \
                        "waypoint" in instance.extra:
                    waypoint = instance.extra.get("waypoint", (0., 0.))
                    if distance(snapshot.allies[actor].position, waypoint) \
                            < distance(record[1], waypoint) - 0.25:
                        progressed = True
                else:
                    progressed = True
            risk = len(context.threats_to(actor)) if context else 0
            risk_previous = (previous_risk if previous_risk is not None
                             else self.committed_risk)
            if instance.skill == "disengage_reenter" and \
                    risk_previous.get(actor, 0) > risk:
                progressed = True
        return progressed

    # -- commit -------------------------------------------------------------
    def commit(self, snapshot: Snapshot, actions: Dict[str, int],
               accepted_plans: List[CandidatePlan],
               risk_by_actor: Optional[Dict[str, int]] = None,
               context=None) -> None:
        """Commit ONLY plans whose final actions survived untouched (C1/C3).
        Idempotent for a repeated submission of the same actions."""
        if self.committed and self.last_committed_actions == actions:
            return
        if self.committed and self.last_committed_actions != actions:
            raise RuntimeError(
                "conflicting commit at the same decision step: "
                f"{self.last_committed_actions} vs {actions}")
        deployment_facts = {}
        if context is not None:
            context.deployment_memory = self
            from .action_duties import actor_duty
            # Capture the original prioritized duty BEFORE applying deltas
            # or retiring tasks. A coincidentally equal rescue/fire command
            # must not start an unrelated deployment task.
            for actor, action in actions.items():
                duty = actor_duty(context, actor, self)
                if duty.get('required') and duty.get('reason', '').startswith('deployment_') \
                        and action in duty['actions']:
                    deployment_facts[actor] = dict(duty)
        self.committed = True
        self.last_committed_actions = dict(actions)
        for actor, action in actions.items():
            if actor in snapshot.allies:
                self.committed_positions[actor] = (
                    snapshot.game_loop, snapshot.allies[actor].position)
        committed_plans = []
        for plan in accepted_plans:
            unchanged = all(actions.get(a) == plan.current_actions[a]
                            for a in plan.controlled_actors)
            if not unchanged:
                continue    # externally altered: nothing commits for it
            committed_plans.append(plan)
            was_live = plan.instance_id in self.instances if plan.instance_id else False
            self.apply_delta(snapshot, plan.proposed_memory_delta, plan, context)
            instance = self.instances.get(plan.instance_id) \
                if plan.instance_id else None
            if instance is not None:
                self._count_lifecycle(instance.skill,
                                      "continued" if was_live else "created")
                instance.executed_steps += 1
                instance.budget_loops = max(
                    snapshot.game_loop - instance.start_loop,
                    instance.budget_loops)
                if (instance.skill == 'coordinated_engage' and instance.phase == 'approaching') \
                        or instance.skill == 'burst_engage':
                    history = instance.extra.setdefault("approach_history", [])
                    history.append((snapshot.game_loop, {
                        a: snapshot.allies[a].position for a in instance.actors}))
                    del history[:-32]
                    target = snapshot.enemies.get(instance.goal)
                    if target is not None:
                        targets = instance.extra.setdefault('approach_target_history', [])
                        target_tag = snapshot.raw_state.get('visible_enemies', {}).get(
                            instance.goal, {}).get('tag')
                        targets.append((snapshot.game_loop, instance.goal, target_tag, target.position))
                        del targets[:-32]
        self._commit_combat_memory(snapshot, context, actions, committed_plans)
        if context is not None:
            self.battle.commit(context, actions, committed_plans)
            self.combat_progress.commit(context, actions)
            self.contact_progress.commit(context, actions, self)
            self.burst_progress.commit(context, actions)
            self.reentry_window.commit(context, actions)
        self._commit_movement(snapshot, context, actions, committed_plans)
        self.committed_risk = dict(getattr(self, "observed_risk", {})
                                   or self.committed_risk)
        self._retire_switched(snapshot, committed_plans, actions, context, deployment_facts)
        if context is not None:
            self._commit_deployment(snapshot, context, actions, deployment_facts)
        self.last_committed_loop = snapshot.game_loop

    def _commit_deployment(self, snapshot, context, actions, deployment_facts):
        """Persist the deployment commitment actually EXECUTED this step.

        Deployment duties frequently win through repair/baseline execution
        rather than a selected rally plan; without this hook the next step
        re-groups and re-picks an anchor every observation (the observed
        "anchor changes every step" regression). Only COMMITTED actions can
        create the task; unselected candidates never touch state. The task
        reuses the ordinary instance machinery (retry key, deadlines,
        observation-phase completion) — no parallel state system.
        """
        from .encounters import rally_readiness
        created = set()
        for actor, action in sorted(actions.items()):
            unit = snapshot.allies.get(actor)
            if unit is None or not unit.alive:
                continue
            active = next((t for t in self.instances.values()
                           if t.skill == 'precombat_rally' and actor in t.actors
                           and not t.termination_reason), None)
            if active is not None:
                continue        # a selected rally plan already owns the task
            duty = deployment_facts.get(actor)
            if duty is None or action not in duty['actions']:
                continue
            if duty['squad'] in created:
                continue
            members = list(duty['members'])
            goal = f"encounter:{self.encounters.current_id}"
            key = self.task_retry_key('precombat_rally', members, 'internal', goal)
            if self.retry_blocked(*key, snapshot.game_loop):
                continue
            params = self.encounter_config.get('precombat_rally', {})
            step = max(snapshot.step_mul, 1)
            remaining = int(round(step * float(params.get('max_h', 8.)))) \
                - (snapshot.game_loop - self.encounters.started_loop)
            if remaining <= 0:
                continue
            instance = self.create_instance(
                'precombat_rally', members, goal, 'rallying', snapshot.game_loop,
                remaining,
                tuple(snapshot.allies[a].tag for a in sorted(members)
                      if a in snapshot.allies),
                goal_kind='area',
                extra={'anchor': list(duty['anchor']), 'waypoint': list(duty['anchor']),
                       'members': members, 'encounter_id': self.encounters.current_id,
                       'enemy_ids': sorted(snapshot.enemies),
                       'ready_radius': float(params.get('ready_radius', 4.)),
                       'ready_fraction': float(params.get('ready_fraction', .8)),
                       'best_gap': rally_readiness(context, duty['pending_members'], duty['anchor'],
                           float(params.get('ready_radius', 4.)))['gap'],
                       'squad_id': duty['squad'],
                       'front_actor': duty.get('front_actor'),
                       'deployment_phase': duty['deployment_phase'],
                       'rally_last_progress_loop': snapshot.game_loop,
                       'no_progress_loops': int(round(
                           step * float(params.get('no_progress_h', 3.)))),
                       'retry_key': key,
                       'created_by': 'deployment_duty_commit',
                       'duty_benefit': duty.get('benefit', '')})
            instance.executed_steps = 1
            self._count_lifecycle(instance.skill, 'created')
            created.add(duty['squad'])

    def _count_lifecycle(self, skill, event):
        counts = self.lifecycle_counts.setdefault(
            skill, {"created": 0, "continued": 0, "exited": 0})
        counts[event] += 1

    def _retire_switched(self, snapshot, accepted_plans, actions, context=None,
                         deployment_facts=None) -> None:
        selected_ids = {plan.instance_id for plan in accepted_plans
                        if plan.instance_id}
        for instance_id in list(self.instances):
            if instance_id in selected_ids:
                continue    # still executing its own task
            instance = self.instances[instance_id]
            if context is not None and instance.skill == 'burst_engage' and len(instance.actors) == 1:
                actor = instance.actors[0]
                replaced = any(p.primary_skill == 'burst_engage' and actor in p.controlled_actors
                               for p in accepted_plans)
                within_budget = (instance.max_loops is None or
                    snapshot.game_loop - instance.start_loop < instance.max_loops)
                if not replaced and within_budget and actor in actions and instance.goal in snapshot.enemies:
                    from .burst import continuation_state
                    fact = continuation_state(context, actor, actions[actor], instance.goal)
                    if fact is not None:
                        if instance.phase != fact['phase']:
                            instance.phase = fact['phase']
                            instance.phase_start_loop = snapshot.game_loop
                        instance.extra['waypoint'] = fact['center']
                        history = instance.extra.setdefault('approach_history', [])
                        history.append((snapshot.game_loop, {actor: snapshot.allies[actor].position}))
                        del history[:-32]
                        targets = instance.extra.setdefault('approach_target_history', [])
                        target = snapshot.enemies[instance.goal]
                        tag = snapshot.raw_state.get('visible_enemies', {}).get(instance.goal, {}).get('tag')
                        targets.append((snapshot.game_loop, instance.goal, tag, target.position))
                        del targets[:-32]
                        instance.executed_steps += 1
                        self._count_lifecycle(instance.skill, 'continued')
                        continue  # no new deadline or claimed observed progress
            # A duty-executed deployment task continues while its actors'
            # COMMITTED actions still satisfy the same duty; the ordinary
            # retirement rule would otherwise kill it every step it wins
            # through repair/baseline instead of a selected rally plan.
            if context is not None and instance.skill == 'precombat_rally':
                executing = any(a in (deployment_facts or {}) for a in instance.actors)
                if executing:
                    instance.executed_steps += 1
                    self._count_lifecycle(instance.skill, 'continued')
                # A fire/rescue interruption pauses movement, not the task's
                # identity. Observation alone completes/fails it; original
                # deadline and stall timer keep running, never renewed.
                if not self._deadline_reached(instance, snapshot.game_loop):
                    continue
            # A different duty of the SAME skill also replaces the old task.
            # New instances are bound to their accepted plan in apply_delta.
            self.finish_task(instance_id, "switched_or_baseline", snapshot)

    # -- unified task deltas (C3: create/continue/retarget/transition/...) --
    def apply_delta(self, snapshot: Snapshot, delta: Dict[str, Any],
                    plan: Optional[CandidatePlan], context=None) -> None:
        if not delta:
            return
        kind = delta.get("kind")
        if kind == "create":
            extra = dict(delta.get("extra") or {})
            extra.setdefault("retry_cooldown_loops", int(round(
                snapshot.step_mul * self.retry_cooldown_h)))
            extra.setdefault("no_progress_loops", int(round(
                snapshot.step_mul * self.no_progress_window_h)))
            if plan is not None:
                extra.setdefault("min_loops", plan.commitment.min_loops)
                extra.setdefault("interrupt_on", list(plan.commitment.interrupt_on))
                if context is not None:
                    extra.setdefault("initial_threats", {
                        a: sorted(context.threats_to(a)) for a in delta["actors"]})
            key = extra.get("retry_key") or self.task_retry_key(
                delta["skill"], delta["actors"],
                delta.get("goal_kind", "internal"), delta.get("goal", ""),
                extra)
            if self.retry_blocked(*key, snapshot.game_loop):
                return      # same failed task may not restart with a fresh budget
            tags = tuple(snapshot.allies[a].tag for a in sorted(delta["actors"])
                         if a in snapshot.allies)
            instance = self.create_instance(
                delta["skill"], delta["actors"], delta.get("goal", ""),
                delta.get("phase", "active"), snapshot.game_loop,
                delta.get("max_loops"), tags,
                goal_kind=delta.get("goal_kind", "internal"),
                extra=extra)
            if plan is not None:
                plan.instance_id = instance.instance_id
        elif kind == 'retain_members':
            instance = self.instances.get(delta.get('instance_id'))
            actors = tuple(sorted(delta.get('actors', ())))
            if instance is None or instance.skill != 'combat_rejoin' \
                    or not instance.extra.get('local_cover_enabled', False) \
                    or not actors or len(set(actors)) != len(actors) \
                    or not set(actors) < set(instance.actors):
                return
            tags = dict(zip(instance.actors, instance.ally_tags))
            if any(a not in snapshot.allies or not snapshot.allies[a].alive
                   or (a in tags and snapshot.allies[a].tag != tags[a]) for a in actors):
                return
            if plan is not None and set(plan.controlled_actors) != set(actors):
                return
            extra = copy.deepcopy(delta.get('extra') or {})
            joiners, cover = set(extra.get('joiners', ())), set(extra.get('supporters', ()))
            if not joiners or not cover or joiners & cover or joiners | cover != set(actors) \
                    or not joiners <= set(instance.extra.get('joiners', ())) \
                    or not cover <= set(instance.extra.get('supporters', ())):
                return
            # A smaller feasible duty inherits its original budget and all
            # progress/failure baselines. Member removal is not progress.
            for key in ('retry_key', 'anchor', 'best_support_gap',
                        'last_support_progress_loop', 'no_progress_loops',
                        'min_loops', 'retry_cooldown_loops', 'interrupt_on',
                        'arrival_exposure_enabled'):
                if key in instance.extra:
                    extra[key] = copy.deepcopy(instance.extra[key])
            for key in ('roles', 'joiner_progress', 'initial_threat_counts', 'initial_threats'):
                if key in instance.extra:
                    extra[key] = {a: copy.deepcopy(value)
                                  for a, value in instance.extra[key].items() if a in actors}
            for key in ('targets_by_actor', 'allocated_targets'):
                if key in extra:
                    extra[key] = {a: value for a, value in extra[key].items() if a in actors}
            retry_keys = list(instance.extra.get('retained_retry_keys', ()))
            retained_goal = '+'.join(sorted(joiners)) + '->' + '+'.join(sorted(cover))
            retained_key = self.task_retry_key(instance.skill, actors, 'internal', retained_goal)
            if retained_key not in retry_keys:
                retry_keys.append(retained_key)
            extra['retained_retry_keys'] = retry_keys
            instance.actors = actors
            instance.ally_tags = tuple(snapshot.allies[a].tag for a in actors)
            instance.extra.update(extra)
            if plan is not None:
                plan.instance_id = instance.instance_id
        elif kind in ("continue", "retarget", "transition"):
            instance = self.instances.get(delta["instance_id"])
            if instance is None:
                return
            if delta.get("phase") and instance.phase != delta["phase"]:
                instance.phase = delta["phase"]
                instance.phase_start_loop = snapshot.game_loop   # set ONCE
            if delta.get("goal") is not None:
                instance.goal = delta["goal"]
                instance.goal_kind = delta.get("goal_kind") or instance.goal_kind
            if "max_loops" in delta:
                instance.max_loops = delta["max_loops"]
            instance.extra.update(delta.get("extra") or {})
            if kind == "retarget":
                instance.extra["retry_key"] = self.task_retry_key(
                    instance.skill, instance.actors, instance.goal_kind,
                    instance.goal, instance.extra)
            if plan is not None:
                plan.instance_id = instance.instance_id
        elif kind == "drop":
            reason = delta.get("reason", "dropped")
            self.finish_task(delta["instance_id"], reason, snapshot)
        elif kind == "retry":
            key = self.task_retry_key(
                delta["skill"], delta["actors"],
                delta.get("goal_kind", "internal"), delta.get("goal", ""),
                delta.get("extra"))
            self.record_retry(*key, snapshot.game_loop,
                              delta.get("cooldown_loops", 16))
        elif kind == "visit_cell":
            self.visited_search_cells[delta["cell"]] = snapshot.game_loop
        elif kind == "unreachable_cell":
            self.unreachable_search_cells[delta["cell"]] = snapshot.game_loop
        elif kind == "search_target":
            self.search_targets[delta["actor"]] = (
                snapshot.game_loop, tuple(delta["target"]))
        else:
            raise ValueError(f"unknown memory delta kind: {kind}")


def _cell(position) -> str:
    return f"{int(position[0] // 8)}:{int(position[1] // 8)}"
