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
        """PURE query (C3): never deletes or mutates retry records."""
        until = self.retry_records.get(
            (skill, "+".join(sorted(actors)), goal))
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
        participants = "+".join(sorted(actors))
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
        self.cleanup_retry_records(snapshot.game_loop)
        events.extend(self.retire_invalid(snapshot))   # identity/protocol
        # deadlines for every remaining instance, AFTER completion checks
        for instance_id in list(self.instances):
            instance = self.instances.get(instance_id)
            if instance is None:
                continue
            # completion first (arrival handled in the block below for
            # search; G4/G1 completions flow through switched/target_lost)
            self._check_search_completion(instance, snapshot, events)
            if self.instances.get(instance_id) is None:
                continue                       # completed this observation
            if instance.max_loops is not None \
                    and snapshot.game_loop - instance.start_loop \
                    >= instance.max_loops:
                self.finish_task(instance_id, "timeout", snapshot)
                events.append(f"task_timeout:{instance_id}")
        return events

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
        if reason == "waypoint_reached":
            waypoint = instance.extra.get("waypoint")
            if waypoint is not None:
                self.visited_search_cells[_cell(waypoint)] = snapshot.game_loop
        elif reason in ("waypoint_timeout", "escape_failed", "timeout",
                        "no_progress"):
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
            self.retry_records[instance.extra.get(
                "retry_key") or self.task_retry_key(
                    instance.skill, instance.actors, instance.goal_kind,
                    instance.goal, instance.extra)] = \
                snapshot.game_loop + self.retry_ttl_loops
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
            if not actors_alive or len(actors_alive) != len(instance.actors):
                self.finish_task(instance_id, "member_lost", snapshot)
                retired.append(instance_id)
                continue
            tags = tuple(snapshot.allies[a].tag for a in instance.actors)
            if instance.ally_tags and tags != instance.ally_tags:
                self.finish_task(instance_id, "tag_changed", snapshot)
                retired.append(instance_id)
                continue
            if instance.goal_kind == "enemy" and instance.goal \
                    and instance.goal not in snapshot.enemies:
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
            risk_previous = previous_risk if previous_risk is not None                 else self.committed_risk
            if risk_previous.get(actor, 0) > risk:
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
        previous_risk = dict(self.committed_risk)
        previous_risk = dict(self.committed_risk)
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
                        snapshot, instance, context, previous_positions,
                        previous_risk):
                    instance.last_progress_loop = snapshot.game_loop
        self.committed_risk = dict(getattr(self, "observed_risk", {})
                                   or self.committed_risk)
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
                    self.finish_task(instance_id, "switched_or_baseline",
                                     snapshot)
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
            reason = delta.get("reason", "dropped")
            if reason in ("escape_failed", "timeout", "no_progress"):
                self.finish_task(delta["instance_id"], reason, snapshot)
            else:
                self.drop_instance(delta["instance_id"], reason)
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
