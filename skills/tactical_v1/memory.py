"""Per-episode observation memory and committed skill instances.

Rules (P1.2): candidate trial evaluation never mutates memory; only plans
selected by the coordinator commit their proposed_memory_delta. Reset, clock
rollback or a new episode clears everything; a repeated call at the same
game loop is idempotent.
"""
from typing import Any, Dict, List, Optional, Tuple

from .contracts import SkillInstance, Snapshot


class TacticalMemory:
    def __init__(self, observation_ttl_loops: int = 240):
        self.observation_ttl_loops = observation_ttl_loops
        self.episode = None
        self.last_game_loop = None
        self.last_snapshot_id = None
        self.instances: Dict[str, SkillInstance] = {}
        # Committed per-step records (only what was actually submitted).
        self.last_committed_actions: Dict[str, int] = {}
        self.move_history: Dict[str, List[Tuple[int, Tuple[float, float]]]] = {}
        # Enemy last-seen memory (for search only; never an attack target).
        self.enemy_last_seen: Dict[str, Tuple[int, Tuple[float, float]]] = {}
        self.actor_skill_history: Dict[str, List[str]] = {}
        # Search waypoint suppression: visited cells by coarse grid key.
        self.visited_search_cells: Dict[str, int] = {}

    # -- lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        self.episode = None
        self.last_game_loop = None
        self.last_snapshot_id = None
        self.instances.clear()
        self.last_committed_actions.clear()
        self.move_history.clear()
        self.enemy_last_seen.clear()
        self.actor_skill_history.clear()
        self.visited_search_cells.clear()

    def begin_step(self, snapshot: Snapshot) -> bool:
        """Validate episode/clock. Returns False when the step is a repeat
        (same snapshot already committed) and must be served from cache."""
        if self.episode != snapshot.episode:
            # New episode (or first): full reset, never leak old instances.
            self.reset()
            self.episode = snapshot.episode
        elif (self.last_game_loop is not None
              and snapshot.game_loop < self.last_game_loop):
            # Clock rollback inside an episode: treat as a new episode.
            self.reset()
            self.episode = snapshot.episode
        self.last_game_loop = snapshot.game_loop
        if self.last_snapshot_id == snapshot.snapshot_id:
            return False
        self.last_snapshot_id = snapshot.snapshot_id
        return True

    # -- observation memory -------------------------------------------------
    def observe(self, snapshot: Snapshot) -> None:
        """Update read-only observation memory from the current snapshot."""
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

    def recent_move(self, actor: str) -> Optional[Tuple[float, float]]:
        history = self.move_history.get(actor, [])
        return history[-1][1] if history else None

    def displacement(self, actor: str, loops: int, snapshot: Snapshot) -> float:
        """Distance travelled over the recorded window (approach progress)."""
        history = self.move_history.get(actor, [])
        threshold = snapshot.game_loop - loops
        past = [position for loop, position in history if loop <= threshold]
        if not past:
            return 0.0
        from .context import distance
        return distance(past[0], snapshot.allies[actor].position)

    # -- instances ----------------------------------------------------------
    def instance_for(self, skill: str, actor: str) -> Optional[SkillInstance]:
        for instance in self.instances.values():
            if instance.skill == skill and actor in instance.actors:
                return instance
        return None

    def create_instance(self, skill: str, actors, goal: str, phase: str,
                        loop: int, max_steps: Optional[int],
                        min_steps: int, extra: Optional[Dict[str, Any]] = None,
                        instance_id: str = "") -> SkillInstance:
        instance_id = instance_id or f"{skill}:{'+'.join(actors)}:{loop}"
        instance = SkillInstance(instance_id=instance_id, skill=skill,
                                 actors=tuple(sorted(actors)), goal=goal,
                                 phase=phase, start_loop=loop,
                                 last_progress_loop=loop,
                                 committed_until_step=min_steps,
                                 budget_steps=0, max_steps=max_steps,
                                 extra=dict(extra or {}))
        self.instances[instance_id] = instance
        return instance

    def drop_instance(self, instance_id: str) -> None:
        self.instances.pop(instance_id, None)

    def retire_invalid(self, snapshot: Snapshot) -> List[str]:
        """Drop instances whose actors died or whose goal vanished."""
        retired = []
        for instance_id in list(self.instances):
            instance = self.instances[instance_id]
            actors = [a for a in instance.actors
                      if a in snapshot.allies and snapshot.allies[a].alive]
            if not actors:
                retired.append(instance_id)
                self.drop_instance(instance_id)
                continue
            if instance.goal and instance.goal.startswith("e") \
                    and instance.goal not in snapshot.enemies:
                retired.append(instance_id)
                self.drop_instance(instance_id)
                continue
            if instance.max_steps is not None and instance.budget_steps >= instance.max_steps:
                retired.append(instance_id)
                self.drop_instance(instance_id)
        return retired

    # -- commit -------------------------------------------------------------
    def commit(self, snapshot: Snapshot, actions: Dict[str, int],
               deltas: List[Dict[str, Any]]) -> None:
        """Commit the final joint plan. Called once per environment step.

        Memory deltas are plain command dicts so candidates can carry them
        without capturing mutable state: {"kind": "create"|"advance"|"drop",
        ...}. Applying anything here happens only after the coordinator chose
        the plan and the runner accepted its actions.
        """
        self.last_committed_actions = dict(actions)
        for delta in deltas:
            self.apply_delta(snapshot, delta)

    def apply_delta(self, snapshot: Snapshot, delta: Dict[str, Any]) -> None:
        kind = delta.get("kind")
        if kind == "create":
            self.create_instance(delta["skill"], delta["actors"], delta.get("goal", ""),
                                 delta.get("phase", "active"), snapshot.game_loop,
                                 delta.get("max_steps"), delta.get("min_steps", 0),
                                 delta.get("extra"), delta.get("instance_id", ""))
        elif kind == "advance":
            instance = self.instances.get(delta["instance_id"])
            if instance is not None:
                instance.budget_steps += 1
                if delta.get("phase"):
                    instance.phase = delta["phase"]
                if delta.get("goal") is not None:
                    instance.goal = delta["goal"]
                if delta.get("progress"):
                    instance.mark_progress(snapshot.game_loop)
        elif kind == "drop":
            self.drop_instance(delta["instance_id"])
        elif kind == "visit_cell":
            self.visited_search_cells[delta["cell"]] = snapshot.game_loop
