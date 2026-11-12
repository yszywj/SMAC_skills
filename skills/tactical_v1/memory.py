"""Observation memory, task instances and the real commit path (C1–C3).

One decision cycle: observe -> compare with the last committed record ->
update confirmed facts (progress, arrivals, invalidations) -> candidates
(read-only) -> joint decision -> verify final actions -> commit task
changes and command records. propose() may never mutate any of this.
"""
from typing import Any, Dict, List, Optional, Tuple

from .context import distance
from .contracts import CandidatePlan, SkillInstance, Snapshot

GOAL_KINDS = ("enemy", "ally", "position", "area", "internal")


class TacticalMemory:
    def __init__(self, observation_ttl_loops: int = 240,
                 search_revisit_cooldown_loops: int = 160,
                 retry_ttl_loops: int = 96):
        self.observation_ttl_loops = observation_ttl_loops
        self.search_revisit_cooldown_loops = search_revisit_cooldown_loops
        self.retry_ttl_loops = retry_ttl_loops
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
        self.retry_records: Dict[Tuple[str, str, str], int] = {}

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

    def retry_blocked(self, skill: str, actors, goal: str,
                      game_loop: int) -> bool:
        key = (skill, "+".join(sorted(actors)), goal)
        until = self.retry_records.get(key)
        if until is not None and game_loop < until:
            return True
        if until is not None:
            del self.retry_records[key]
        # TTL housekeeping
        for other in list(self.retry_records):
            if self.retry_records[other] < game_loop - self.retry_ttl_loops:
                del self.retry_records[other]
        return False

    def record_retry(self, skill: str, actors, goal: str, game_loop: int,
                     cooldown_loops: int) -> None:
        self.retry_records[(skill, "+".join(sorted(actors)), goal)] = \
            game_loop + cooldown_loops

    # -- observation phase (idempotent; replaces same-clock records) --------
    def displacement(self, actor: str, loops: int, snapshot: Snapshot) -> float:
        """Distance travelled over the recorded window (loop-based)."""
        history = self.move_history.get(actor, [])
        threshold = snapshot.game_loop - loops
        past = [position for loop, position in history if loop <= threshold]
        if not past:
            return 0.0
        return distance(past[0], snapshot.allies[actor].position)

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
        if risk_by_actor is not None:
            self.committed_risk = dict(risk_by_actor)
        self.observed_key = key

    def update_task_facts(self, snapshot: Snapshot, context) -> List[str]:
        """Observation-phase fact updates ONLY (C2): arrivals, timeouts,
        invalidations. Candidates never do this later."""
        events: List[str] = []
        events.extend(self.retire_invalid(snapshot))
        # search arrival confirmation (G7): a search instance whose actor is
        # within tolerance of its tracked waypoint HAS arrived — write the
        # visit fact now; a timeout marks the cell temporarily unreachable.
        for instance_id in list(self.instances):
            instance = self.instances[instance_id]
            if instance.skill != "cohesive_search" \
                    or instance.goal_kind != "position":
                continue
            waypoint = instance.extra.get("waypoint")
            if waypoint is None:
                continue
            for actor in instance.actors:
                if actor not in snapshot.allies or \
                        not snapshot.allies[actor].alive:
                    continue
                if distance(snapshot.allies[actor].position, waypoint) \
                        <= instance.extra.get("arrival_tolerance", 2.0):
                    cell = _cell(waypoint)
                    self.visited_search_cells[cell] = snapshot.game_loop
                    self.drop_instance(instance_id, "waypoint_reached")
                    events.append(f"search_arrived:{cell}")
                    break
                budget = instance.max_loops or (12 * max(snapshot.step_mul, 1))
                if snapshot.game_loop - instance.start_loop >= budget:
                    cell = _cell(waypoint)
                    self.unreachable_search_cells[cell] = snapshot.game_loop
                    self.record_retry("cohesive_search", instance.actors,
                                      instance.goal, snapshot.game_loop,
                                      self.search_revisit_cooldown_loops)
                    self.drop_instance(instance_id, "waypoint_timeout")
                    events.append(f"search_unreachable:{cell}")
                    break
        return events

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
        instance_id = instance_id or f"{skill}:{'+'.join(sorted(actors))}:{goal}:{loop}"
        instance = SkillInstance(
            instance_id=instance_id, skill=skill, actors=tuple(sorted(actors)),
            ally_tags=tuple(ally_tags), goal=goal, goal_kind=goal_kind,
            phase=phase, start_loop=loop, phase_start_loop=loop,
            last_progress_loop=loop, budget_loops=0, executed_steps=0,
            max_loops=max_loops, extra=dict(extra or {}))
        self.instances[instance_id] = instance
        return instance

    def drop_instance(self, instance_id: str, reason: str = "") -> None:
        instance = self.instances.get(instance_id)
        if instance is not None:
            instance.termination_reason = reason or "dropped"
        self.instances.pop(instance_id, None)

    def retire_invalid(self, snapshot: Snapshot) -> List[str]:
        """Death, tag change, typed target loss, loop-budget exhaustion."""
        retired = []
        for instance_id in list(self.instances):
            instance = self.instances[instance_id]
            actors_alive = [a for a in instance.actors
                            if a in snapshot.allies and snapshot.allies[a].alive]
            if not actors_alive or len(actors_alive) != len(instance.actors):
                self.drop_instance(instance_id, "member_lost")
                retired.append(instance_id)
                continue
            tags = tuple(snapshot.allies[a].tag for a in instance.actors)
            if instance.ally_tags and tags != instance.ally_tags:
                self.drop_instance(instance_id, "tag_changed")
                retired.append(instance_id)
                continue
            if instance.goal_kind == "enemy" and instance.goal \
                    and instance.goal not in snapshot.enemies:
                self.drop_instance(instance_id, "target_lost")
                retired.append(instance_id)
                continue
            if instance.max_loops is not None \
                    and snapshot.game_loop - instance.start_loop >= instance.max_loops:
                self.drop_instance(instance_id, "timeout")
                retired.append(instance_id)
        return retired

    # -- progress evidence (C1: task-relevant, compared with last record) ---
    def task_progress(self, snapshot: Snapshot, instance: SkillInstance,
                      context, previous_positions=None) -> bool:
        previous = previous_positions if previous_positions is not None             else self.committed_positions
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
                elif instance.goal_kind == "position":
                    waypoint = instance.extra.get("waypoint", (0., 0.))
                    if distance(snapshot.allies[actor].position, waypoint) \
                            < distance(record[1], waypoint) - 0.25:
                        progressed = True
                else:
                    progressed = True
            risk = len(context.threats_to(actor)) if context else 0
            if self.committed_risk.get(actor, 0) > risk:
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
        previous_positions = dict(self.committed_positions)   # C1: compare FIRST
        self.committed = True
        self.last_committed_actions = dict(actions)
        for actor, action in actions.items():
            if actor in snapshot.allies:
                self.committed_positions[actor] = (
                    snapshot.game_loop, snapshot.allies[actor].position)
        for plan in accepted_plans:
            unchanged = all(actions.get(a) == plan.current_actions[a]
                            for a in plan.controlled_actors)
            if not unchanged:
                continue    # externally altered: nothing commits for it
            self.apply_delta(snapshot, plan.proposed_memory_delta, plan)
            instance = self.instances.get(plan.instance_id) \
                if plan.instance_id else None
            if instance is not None:
                instance.executed_steps += 1
                instance.budget_loops = max(
                    snapshot.game_loop - instance.start_loop,
                    instance.budget_loops)
                if context is not None and self.task_progress(
                        snapshot, instance, context, previous_positions):
                    instance.last_progress_loop = snapshot.game_loop
        self._retire_switched(snapshot, accepted_plans, actions)

    def _retire_switched(self, snapshot, accepted_plans, actions) -> None:
        selected = {}
        for plan in accepted_plans:
            for actor in plan.controlled_actors:
                selected[actor] = plan
        for instance_id in list(self.instances):
            instance = self.instances[instance_id]
            if instance_id in {p.instance_id for p in accepted_plans
                               if p.instance_id}:
                continue    # still executing its own task
            for actor in instance.actors:
                plan = selected.get(actor)
                if plan is None or plan.primary_skill != instance.skill:
                    # unit fell to baseline or another task: end the old one
                    self.drop_instance(instance_id, "switched_or_baseline")
                    break

    # -- unified task deltas (C3: create/continue/retarget/transition/...) --
    def apply_delta(self, snapshot: Snapshot, delta: Dict[str, Any],
                    plan: Optional[CandidatePlan]) -> None:
        if not delta:
            return
        kind = delta.get("kind")
        if kind == "create":
            if self.retry_blocked(delta["skill"], delta["actors"],
                                  delta.get("goal", ""), snapshot.game_loop):
                return      # same failed task may not restart with a fresh budget
            tags = tuple(snapshot.allies[a].tag for a in delta["actors"]
                         if a in snapshot.allies)
            self.create_instance(
                delta["skill"], delta["actors"], delta.get("goal", ""),
                delta.get("phase", "active"), snapshot.game_loop,
                delta.get("max_loops"), tags,
                goal_kind=delta.get("goal_kind", "internal"),
                extra=delta.get("extra"))
        elif kind == "retarget":
            instance = self.instances.get(delta["instance_id"])
            if instance is not None:
                instance.goal = delta["goal"]
                instance.goal_kind = delta.get("goal_kind", instance.goal_kind)
                instance.last_progress_loop = snapshot.game_loop
        elif kind == "transition":
            instance = self.instances.get(delta["instance_id"])
            if instance is not None and instance.phase != delta["phase"]:
                instance.phase = delta["phase"]
                instance.phase_start_loop = snapshot.game_loop   # set ONCE
            if delta.get("goal"):
                instance.goal = delta["goal"]
                instance.goal_kind = delta.get("goal_kind") or instance.goal_kind
        elif kind == "drop":
            self.drop_instance(delta["instance_id"],
                               delta.get("reason", "dropped"))
        elif kind == "retry":
            self.record_retry(delta["skill"], delta["actors"],
                              delta.get("goal", ""), snapshot.game_loop,
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
