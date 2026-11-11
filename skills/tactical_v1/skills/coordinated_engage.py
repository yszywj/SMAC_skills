"""G2 CoordinatedEngage: obtain an effective fighting position.

Melee units get deterministic contact-slot allocation around the target
(body size + already-approaching allies considered); ranged units pick a
legal move that best closes on their chosen enemy without the whole squad
mirroring one unit. Front squads may fight while rear units catch up: the
skill proposes per-unit, never one team-wide phase.
"""
import math
from typing import Dict, List

from ..context import Context, STOP, distance
from ..contracts import Commitment, EffectClaim, HEURISTIC, ResourceClaims
from .common import best_move_toward, make_plan, move_progress


class CoordinatedEngageSkill:
    name = "coordinated_engage"

    def __init__(self, config: Dict):
        self.config = config
        params = config.get("coordinated_engage", {})
        self.slot_spacing = params.get("slot_spacing", 1.4)
        self.wait_max_steps = params.get("wait_max_steps", 3)
        self.max_approach_steps = params.get("max_approach_steps", 60)

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        if not context.snapshot.enemies:
            return plans
        melee_slots = self._melee_slots(context)
        for actor, unit in sorted(context.snapshot.allies.items()):
            if not unit.alive:
                continue
            capability = context.capabilities[actor]
            if capability.healer or capability.suicide_attack:
                continue
            attackable = context.attackable_enemies(actor)
            if attackable:
                continue  # already in effective contact: G1/G3 own this unit
            telemetry.count(self.name, "applicable")
            target_id = context.nearest_enemy(actor)
            if target_id is None:
                continue
            enemy = context.snapshot.enemies[target_id]
            if capability.combat_class == "melee":
                goal = melee_slots.get(actor, enemy.position)
            else:
                goal = enemy.position
            move = best_move_toward(context, actor, goal)
            if move is not None:
                progress = move_progress(context, actor, move, goal)
                plans.append(make_plan(
                    self.name, {actor: move}, target_id, {actor: move},
                    [EffectClaim("position", actor, progress,
                                 max(context.snapshot.step_mul, 1), HEURISTIC,
                                 "approach_engagement_position", 0.7)],
                    ResourceClaims(controls=(actor,),
                                   space=(f"{actor}@{move}",)),
                    Commitment("approaching", 0, self.max_approach_steps,
                               interrupt_on=("contact", "illegal", "danger_change",
                                             "no_progress")),
                    {"move_opportunity_cost": 0.0}, snapshot_id, fingerprint,
                    rationale=f"approach {target_id} slot"))
            else:
                # No legal move improves the approach: bounded wait.
                plans.append(make_plan(
                    self.name, {actor: STOP}, target_id, {actor: STOP},
                    [EffectClaim("position", actor, 0.0,
                                 max(context.snapshot.step_mul, 1), HEURISTIC,
                                 "bounded_wait", 0.5)],
                    ResourceClaims(controls=(actor,)),
                    Commitment("waiting", 0, self.wait_max_steps,
                               interrupt_on=("contact", "danger_change")),
                    {"raw_bonus": -0.2}, snapshot_id, fingerprint,
                    rationale="no improving move; bounded wait"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans

    def _melee_slots(self, context: Context) -> Dict[str, tuple]:
        """Deterministic ring slots around each melee unit's nearest enemy so
        melee attackers spread over the contact face instead of stacking."""
        slots: Dict[str, tuple] = {}
        assigned: Dict[str, list] = {}
        melee = [a for a, cap in sorted(context.capabilities.items())
                 if cap.combat_class == "melee"
                 and context.snapshot.allies[a].alive
                 and not context.attackable_enemies(a)]
        for actor in melee:
            target_id = context.nearest_enemy(actor)
            if target_id is None:
                continue
            enemy = context.snapshot.enemies[target_id]
            users = assigned.setdefault(target_id, [])
            angle = (len(users) * 2 * math.pi / max(len(melee), 1)) % (2 * math.pi)
            offset = enemy.radius + context.snapshot.allies[actor].radius + 0.1
            slots[actor] = (enemy.position[0] + offset * math.cos(angle),
                            enemy.position[1] + offset * math.sin(angle))
            users.append(actor)
        return slots
