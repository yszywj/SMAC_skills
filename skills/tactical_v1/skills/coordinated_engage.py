"""G2 CoordinatedEngage: establish effective shared contact.

Local squads from observed ally positions (flying supports do not chain
distant ground groups). Melee get real contact slots around the target —
slot width from body sizes, ALREADY-OCCUPIED positions subtracted — instead
of an average ring over the whole melee count. Waiting is bounded (2H) and
proposed only when it clearly helps. Progress/no-progress guards consume
the instance budgets.
"""
import math
from typing import Dict, List, Optional

from ..context import Context, STOP, distance
from ..contracts import Commitment, Dependency, EffectClaim, HEURISTIC, ResourceClaims
from .common import best_move_toward, h_loops, make_plan, move_progress, space_claim


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
        self.max_candidates = config.get("search", {}).get(
            "max_candidates_per_unit", 6)

    def _squad(self, context: Context, actor: str) -> List[str]:
        for group in context.engagement_groups():
            if actor in group:
                return group
        return [actor]

    def _squad_target(self, context: Context, squad: List[str]) -> Optional[str]:
        """One approachable LOCAL enemy cluster for the squad (not per-unit
        nearest): fewest total approach distance, deterministic."""
        if not context.snapshot.enemies:
            return None
        def cost(enemy_id):
            return sum(min(context.distance(a, enemy_id) for enemy_id in
                           [enemy_id]) for a in squad)
        return min(sorted(context.snapshot.enemies), key=lambda e: (cost(e), e))

    def _occupied_slots(self, context: Context, target_id: str) -> List[tuple]:
        """Positions currently crowding the target (allies in contact)."""
        target = context.snapshot.enemies[target_id]
        return [u.position for u in context.snapshot.allies.values()
                if u.alive and distance(u.position, target.position)
                <= target.radius + u.radius + 0.6]

    def _contact_slot(self, context: Context, actor: str, target_id: str,
                      index: int, occupied: List[tuple]) -> tuple:
        """A deterministic free contact slot: ring positions around the
        target sized by both bodies, skipping already-occupied ones."""
        unit = context.snapshot.allies[actor]
        target = context.snapshot.enemies[target_id]
        offset = target.radius + unit.radius + self.slot_spacing * 0.5
        candidates = []
        for step in range(8):
            angle = (index + step) * math.pi / 4.0
            position = (target.position[0] + offset * math.cos(angle),
                        target.position[1] + offset * math.sin(angle))
            if all(distance(position, o) > unit.radius + 0.2 for o in occupied):
                candidates.append(position)
        return candidates[0] if candidates else target.position

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        if not context.snapshot.enemies:
            return plans
        melee_index = 0
        for actor, unit in sorted(context.snapshot.allies.items()):
            if not unit.alive:
                continue
            capability = context.capabilities[actor]
            if capability.healer or capability.suicide_attack:
                continue
            # Blocked-attack detection: an issuable attack that has produced
            # no progress for the whole no-progress window also engages.
            attackable = context.attackable_enemies(actor)
            instance = memory.instance_for(self.name, actor)
            blocked = False
            if attackable:
                if instance is not None and instance.phase == "blocked_check":
                    blocked = memory.displacement(
                        actor, h_loops(context, self.no_progress_h),
                        context.snapshot) < 0.5
                if not blocked:
                    continue       # effective contact: G1/G3 own this unit
            telemetry.count(self.name, "applicable")
            if instance is not None and instance.retry_after_loop \
                    and context.snapshot.game_loop < instance.retry_after_loop \
                    and instance.phase == "cooldown":
                continue
            squad = self._squad(context, actor)
            target_id = self._squad_target(context, squad)
            if target_id is None:
                continue
            enemy = context.snapshot.enemies[target_id]
            goal = enemy.position
            slot = None
            if capability.combat_class == "melee":
                occupied = self._occupied_slots(context, target_id)
                slot = self._contact_slot(context, actor, target_id,
                                          melee_index, occupied)
                goal = slot
                melee_index += 1
            # candidate moves: direct, and lateral alternatives
            moves = []
            direct = best_move_toward(context, actor, goal)
            if direct is not None and direct not in moves:
                moves.append(direct)
            for lateral in sorted(context.move_destinations(actor)):
                if lateral not in moves:
                    moves.append(lateral)
            no_progress = instance is not None and memory.displacement(
                actor, h_loops(context, self.no_progress_h),
                context.snapshot) < 0.5
            budget = h_loops(context, self.target_max_h)
            for move in moves[:self.max_candidates]:
                plans.append(make_plan(
                    self.name, {actor: move}, target_id, {actor: move},
                    [EffectClaim("position", actor,
                                 move_progress(context, actor, move, goal),
                                 max(context.H, 1), HEURISTIC,
                                 "approach_contact_position")],
                    ResourceClaims(controls=(actor,),
                                   space=(space_claim(context, actor, move),)),
                    Commitment("approaching", 0, max_loops=budget,
                               interrupt_on=("contact", "no_progress",
                                             "danger_change")),
                    {"switch_cost": 0.1 if no_progress else 0.0,
                     "progress": abs(move_progress(context, actor, move, goal))},
                    snapshot_id, fingerprint,
                    instance_id=instance.instance_id if instance else None,
                    dependencies=(Dependency("target_valid", (target_id,)),),
                    memory_delta=({"kind": "create", "skill": self.name,
                                   "actors": [actor], "goal": target_id,
                                   "phase": "approaching",
                                   "max_loops": budget}
                                  if instance is None else None),
                    rationale=f"approach {target_id}"
                              + (f" slot {slot}" if slot else "")))
            # bounded wait: only when squadmates are close to arriving
            mates_close = any(
                distance(context.snapshot.allies[m].position, enemy.position)
                <= 2 * (unit.radius + enemy.radius) + 3.0
                for m in squad if m != actor)
            if mates_close:
                plans.append(make_plan(
                    self.name, {actor: STOP}, target_id, {actor: STOP},
                    [EffectClaim("position", actor, None, max(context.H, 1),
                                 HEURISTIC, "bounded_wait")],
                    ResourceClaims(controls=(actor,)),
                    Commitment("waiting", 0, max_loops=h_loops(
                        context, self.wait_max_h)),
                    {"hold_value": 0.3}, snapshot_id, fingerprint,
                    rationale="bounded wait for squad contact"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans
