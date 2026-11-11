"""Per-episode observation memory and committed skill instances (R4).

- Instances are created ONCE when a task is first selected; later candidates
  carry the same instance_id. No per-step re-creation, no budget resets by
  renaming.
- Duration = game_loop deltas; a repeated call at the same snapshot never
  adds time or step counts.
- Progress is marked only with observed evidence (displacement, contact
  improvement, threat reduction), never unconditionally.
- commit_final_actions verifies every plan's final actions; a multi-unit
  plan whose members were altered commits NOTHING for that plan.
"""
from typing import Any, Dict, List, Optional, Tuple

from .context import distance
from .contracts import CandidatePlan, SkillInstance, Snapshot


class TacticalMemory:
    def __init__(self, observation_ttl_loops: int = 240,
                 search_revisit_cooldown_loops: int = 160):
        self.observation_ttl_loops = observation_ttl_loops
        self.search_revisit_cooldown_loops = search_revisit_cooldown_loops
        self.episode = None
        self.last_game_loop = None
        self.last_snapshot_id = None
        self.observed_snapshot_id = None      # idempotence guard for observe()
        self.instances: Dict[str, SkillInstance] = {}
        self.last_committed_actions: Dict[str, int] = {}
        self.committed_positions: Dict[str, Tuple[int, Tuple[float, float]]] = {}
        self.committed_risk: Dict[str, int] = {}
        self.move_history: Dict[str, List[Tuple[int, Tuple[float, float]]]] = {}
        self.enemy_last_seen: Dict[str, Tuple[int, Tuple[float, float]]] = {}
        self.visited_search_cells: Dict[str, int] = {}     # arrival-confirmed only
        self.search_targets: Dict[str, Tuple[int, Tuple[float, float]]] = {}

    # -- lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        self.episode = None
        self.last_game_loop = None
        self.last_snapshot_id = None
        self.observed_snapshot_id = None
        self.instances.clear()
        self.last_committed_actions.clear()
        self.committed_positions.clear()
        self.committed_risk.clear()
        self.move_history.clear()
        self.enemy_last_seen.clear()
        self.visited_search_cells.clear()
        self.search_targets.clear()

    def begin_step(self, snapshot: Snapshot) -> bool:
        """Validate episode/clock. False => same snapshot already committed;
        the caller must serve the cached result instead of advancing."""
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
        return True

    # -- observation memory (idempotent per snapshot) ------------------------
    def observe(self, snapshot: Snapshot, risk_by_actor=None) -> None:
        if self.observed_snapshot_id == snapshot.snapshot_id:
            return                      # never append history twice (R4)
        self.observed_snapshot_id = snapshot.snapshot_id
        for actor, unit in snapshot.allies.items():
            history = self.move_history.setdefault(actor, [])
            history.append((snapshot.game_loop, unit.position))
            if len(history) > 16:
                del history[:len(history) - 16]
        for enemy_id, enemy in snapshot.enemies.items():
            self.enemy_last_seen[enemy_id] = (snapshot.game_loop, enemy.position)
        stale = [e for e, (loop, _) in self.enemy_last_seen.items()
                 if snapshot.game_loop - loop > self.observation_ttl_loops]
        for enemy_id in stale:
            del self.enemy_last_seen[enemy_id]
        if risk_by_actor:
            self.committed_risk = dict(risk_by_actor)

    def recent_move(self, actor: str) -> Optional[Tuple[float, float]]:
        history = self.move_history.get(actor, [])
        return history[-1][1] if history else None

    def displacement(self, actor: str, loops: int, snapshot: Snapshot) -> float:
        history = self.move_history.get(actor, [])
        threshold = snapshot.game_loop - loops
        past = [position for loop, position in history if loop <= threshold]
        if not past:
            return 0.0
        return distance(past[0], snapshot.allies[actor].position)

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
                        retry_after_loop: Optional[int] = None,
                        extra: Optional[Dict[str, Any]] = None) -> SkillInstance:
        instance_id = instance_id or f"{skill}:{'+'.join(sorted(actors))}:{goal}:{loop}"
        instance = SkillInstance(
            instance_id=instance_id, skill=skill, actors=tuple(sorted(actors)),
            ally_tags=tuple(ally_tags), goal=goal, phase=phase,
            start_loop=loop, phase_start_loop=loop, last_progress_loop=loop,
            budget_loops=0, executed_steps=0, max_loops=max_loops,
            retry_after_loop=retry_after_loop, extra=dict(extra or {}))
        self.instances[instance_id] = instance
        return instance

    def drop_instance(self, instance_id: str, reason: str = "") -> None:
        instance = self.instances.get(instance_id)
        if instance is not None:
            instance.termination_reason = reason or "dropped"
        self.instances.pop(instance_id, None)

    def retire_invalid(self, snapshot: Snapshot) -> List[str]:
        """Death, tag change, target loss, episode bounds; atomic multi-member
        tasks lose any required member -> whole instance invalid."""
        retired = []
        for instance_id in list(self.instances):
            instance = self.instances[instance_id]
            actors_alive = [a for a in instance.actors
                            if a in snapshot.allies and snapshot.allies[a].alive]
            if not actors_alive or len(actors_alive) != len(instance.actors):
                retired.append(instance_id)
                self.drop_instance(instance_id, "member_lost")
                continue
            # tag change: identity no longer matches the task's units
            tags = tuple(snapshot.allies[a].tag for a in instance.actors)
            if instance.ally_tags and tags != instance.ally_tags:
                retired.append(instance_id)
                self.drop_instance(instance_id, "tag_changed")
                continue
            if instance.goal and instance.goal.startswith("e") \
                    and instance.goal not in snapshot.enemies:
                retired.append(instance_id)
                self.drop_instance(instance_id, "target_lost")
                continue
            if instance.max_loops is not None and instance.budget_loops >= instance.max_loops:
                retired.append(instance_id)
                self.drop_instance(instance_id, "timeout")
        return retired

    # -- commit -------------------------------------------------------------
    def progress_evidence(self, snapshot: Snapshot, actor: str,
                          risk_now: int) -> bool:
        """Observed evidence only: real displacement since the last commit,
        risk reduction, or contact is the coordinator's to add."""
        record = self.committed_positions.get(actor)
        moved = False
        if record is not None and actor in snapshot.allies:
            moved = distance(record[1], snapshot.allies[actor].position) > 0.25
        risk_down = self.committed_risk.get(actor, 0) > risk_now
        return moved or risk_down

    def commit(self, snapshot: Snapshot, actions: Dict[str, int],
               accepted_plans: List[CandidatePlan],
               risk_by_actor: Optional[Dict[str, int]] = None) -> None:
        """Commit only plans whose FINAL actions survived untouched (R4)."""
        self.last_committed_actions = dict(actions)
        for actor, action in actions.items():
            if actor in snapshot.allies:
                self.committed_positions[actor] = (
                    snapshot.game_loop, snapshot.allies[actor].position)
        for plan in accepted_plans:
            unchanged = all(actions.get(a) == plan.current_actions[a]
                            for a in plan.controlled_actors)
            if not unchanged:
                # Multi-unit plan altered externally: nothing is committed for
                # it — no phase advance, no claims, no memory delta.
                continue
            self.apply_delta(snapshot, plan.proposed_memory_delta, plan)
            instance = self.instances.get(plan.instance_id) \
                if plan.instance_id else None
            if instance is not None:
                instance.executed_steps += 1
                instance.budget_loops = max(
                    snapshot.game_loop - instance.start_loop, instance.budget_loops)
                progressed = any(
                    self.progress_evidence(snapshot, a,
                                           (risk_by_actor or {}).get(a, 0))
                    for a in instance.actors)
                if progressed:
                    instance.last_progress_loop = snapshot.game_loop
        # actors whose final action no longer matches any held instance task
        self._retire_switched(snapshot, accepted_plans, actions)

    def _retire_switched(self, snapshot: Snapshot,
                         accepted_plans: List[CandidatePlan],
                         actions: Dict[str, int]) -> None:
        selected = {}
        for plan in accepted_plans:
            for actor in plan.controlled_actors:
                selected[actor] = plan
        for instance_id in list(self.instances):
            instance = self.instances[instance_id]
            for actor in instance.actors:
                plan = selected.get(actor)
                if plan is not None and plan.instance_id != instance_id \
                        and plan.primary_skill != instance.skill:
                    self.drop_instance(instance_id, "switched_skill")
                    break

    def apply_delta(self, snapshot: Snapshot, delta: Dict[str, Any],
                    plan: CandidatePlan) -> None:
        kind = delta.get("kind")
        if kind == "create":
            self.create_instance(
                delta["skill"], delta["actors"], delta.get("goal", ""),
                delta.get("phase", "active"), snapshot.game_loop,
                delta.get("max_loops"), delta.get("ally_tags", ()),
                delta.get("instance_id", ""),
                delta.get("retry_after_loop"), delta.get("extra"))
        elif kind == "phase":
            instance = self.instances.get(delta["instance_id"])
            if instance is not None:
                instance.phase = delta["phase"]
                instance.phase_start_loop = snapshot.game_loop
                if delta.get("goal") is not None:
                    instance.goal = delta["goal"]
        elif kind == "drop":
            self.drop_instance(delta["instance_id"],
                               delta.get("reason", "dropped"))
        elif kind == "visit_cell":
            # written ONLY on arrival confirmation (G7), never on move issue
            self.visited_search_cells[delta["cell"]] = snapshot.game_loop
        elif kind == "search_target":
            self.search_targets[delta["actor"]] = (
                snapshot.game_loop, tuple(delta["target"]))
