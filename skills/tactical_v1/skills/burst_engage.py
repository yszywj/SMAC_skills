"""S3 BurstEngage: consumption-burst approach, timing and split investment (D9).

Unknown contact/explosion damage never makes the legal attack path vanish:
`contact_status == unknown` still yields an attack ATTEMPT candidate (no
claimed detonation or window benefit). Proximity is computed from the
ACTOR's position. A single remaining enemy still gets a meaningful exchange
attempt. Self-consumption is a planned FUTURE cost, charged only by the
effects model for in-contact detonations.
"""
from typing import Dict, List, Optional

from ..context import Context, STOP, distance
from ..contracts import Commitment, EffectClaim, HEURISTIC, ResourceClaims
from .common import best_move_toward, h_loops, make_plan, move_progress, space_claim

CONTACT_UNKNOWN = "unknown"


class BurstEngageSkill:
    name = "burst_engage"

    def __init__(self, config: Dict):
        self.config = config
        params = config.get("burst_engage", {})
        self.proxy_radius = params.get("proxy_radius", 2.5)
        self.value_per_enemy = params.get("value_per_enemy", 2.0)
        self.min_exchange_value = params.get("min_exchange_value", 2.0)
        self.self_cost = params.get("self_cost", 1.0)
        self.lonely_exception = params.get("lonely_enemy_exception", True)

    def cluster_value(self, context: Context, center) -> float:
        value = 0.0
        for enemy in context.snapshot.enemies.values():
            if distance(enemy.position, center) <= self.proxy_radius + enemy.radius:
                weight = 0.4 if enemy.type_name == "baneling" else 1.0
                value += self.value_per_enemy * weight
        return value

    def _regions(self, context: Context, actor: str,
                 current_goal: Optional[str]) -> List:
        """Diverse regions: current target, high coverage, near-approachable —
        proximity measured FROM THE ACTOR, never from (0, 0)."""
        unit = context.snapshot.allies[actor]
        scored = [(enemy_id, enemy.position,
                   self.cluster_value(context, enemy.position),
                   distance(unit.position, enemy.position))
                  for enemy_id, enemy in sorted(context.snapshot.enemies.items())]
        if not scored:
            return []
        ordered = sorted(scored, key=lambda s: (s[0] != current_goal, -s[2], s[3]))
        result, seen = [], set()
        for enemy_id, position, value, _ in ordered:
            if enemy_id not in seen:
                result.append((enemy_id, position, value))
                seen.add(enemy_id)
            if len(result) >= 3:
                break
        return result

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        banelings = [a for a, cap in sorted(context.capabilities.items())
                     if cap.suicide_attack and context.snapshot.allies[a].alive]
        if not banelings:
            return plans
        telemetry.count(self.name, "applicable", len(banelings))
        lonely = len(context.snapshot.enemies) == 1 and self.lonely_exception
        for actor in banelings:
            instance = memory.instance_for(self.name, actor)
            current_goal = instance.goal if instance is not None and \
                instance.goal_kind == "enemy" else None
            # 1. detonation ATTEMPT: issuable attack; in-range contact is not
            #    required (unknown contact keeps the attempt, no fake credit)
            attempts = []
            for enemy_id in sorted(context.snapshot.enemies):
                if not context.can_issue_attack(actor, enemy_id):
                    continue
                if memory.retry_blocked(
                        *memory.task_retry_key(self.name, [actor], "enemy",
                                               enemy_id),
                        context.snapshot.game_loop):
                    continue
                contact = context.contact_status(actor, enemy_id)
                if contact == "out_of_range":
                    continue
                value = self.cluster_value(
                    context, context.snapshot.enemies[enemy_id].position)
                attempts.append((value, enemy_id, contact))
            attempts.sort(reverse=True)
            threshold = 0.0 if lonely else self.min_exchange_value
            viable = [a for a in attempts if a[0] >= threshold][:2]
            if viable:
              for value, enemy_id, contact in viable:
                action = context.attack_action(actor, enemy_id)
                effects = [EffectClaim(
                    "coverage", enemy_id, None, max(context.H, 1), HEURISTIC,
                    "burst_attempt_contact_" + contact)]
                if contact == "in_range":
                    # only a genuine in-contact detonation plans the cost;
                    # the effects model charges it once, never here again
                    effects.append(EffectClaim(
                        "loss", actor, self.self_cost, max(context.H, 1),
                        HEURISTIC, "planned_self_consumption_once"))
                plans.append(make_plan(
                    self.name, {actor: action}, enemy_id, {actor: action},
                    effects,
                    ResourceClaims(controls=(actor,)),
                    Commitment("bursting", 0, max_loops=2 * context.H),
                    {"urgency": 0.5}, snapshot_id, fingerprint,
                    provider="burst_engage",
                    instance_id=instance.instance_id if instance else None,
                    memory_delta=({"kind": "create", "skill": self.name,
                                   "actors": [actor], "goal": enemy_id,
                                   "goal_kind": "enemy", "phase": "bursting",
                                   "max_loops": 2 * context.H}
                                  if instance is None else
                                  {"kind": "retarget",
                                   "instance_id": instance.instance_id,
                                   "goal": enemy_id, "goal_kind": "enemy"}
                                  if instance.goal != enemy_id else None),
                    rationale=f"burst attempt at {enemy_id} "
                              f"(contact={contact}, proxy value {value:.1f})"))
              continue
            # 2. approach candidates: diverse regions, signed progress
            regions = self._regions(context, actor, current_goal)
            worthwhile = [r for r in regions if r[2] >= threshold or lonely]
            if worthwhile:
                for enemy_id, position, value in worthwhile[:2]:
                    move = best_move_toward(context, actor, position)
                    if move is None:
                        continue
                    plans.append(make_plan(
                        self.name, {actor: move}, enemy_id, {actor: move},
                        [EffectClaim("position", actor,
                                     move_progress(context, actor, move,
                                                   position),
                                     max(context.H, 1), HEURISTIC,
                                     "approach_burst_region")],
                        ResourceClaims(
                            controls=(actor,),
                            space=(space_claim(context, actor, move),)),
                        Commitment("approaching", 0,
                                   max_loops=40 * context.H,
                                   interrupt_on=("no_progress", "danger_change")),
                        {"progress": move_progress(context, actor, move,
                                                   position)},
                        snapshot_id, fingerprint, provider="burst_engage",
                        instance_id=instance.instance_id if instance else None,
                        memory_delta=({"kind": "create", "skill": self.name,
                                       "actors": [actor], "goal": enemy_id,
                                       "goal_kind": "enemy",
                                       "phase": "approaching",
                                       "max_loops": 40 * context.H,
                                       "extra": {"waypoint": position}}
                                      if instance is None else None),
                        goal_kind="enemy",
                        goal_data={"target_id": enemy_id,
                                   "waypoint": list(position)},
                        rationale=f"approach burst region {enemy_id}"))
                continue
            # 3. nothing worth the trade: hold (declared, real Stop)
            plans.append(make_plan(
                self.name, {actor: STOP}, "hold", {actor: STOP},
                [EffectClaim("position", actor, None, max(context.H, 1),
                             HEURISTIC, "declined_low_value_contact")],
                ResourceClaims(controls=(actor,)),
                Commitment("declined", 0, max_loops=4 * context.H,
                           interrupt_on=("danger_change",)),
                {"hold_value": 0.2}, snapshot_id, fingerprint,
                provider="burst_engage",
                rationale="low-value contact declined; wait for a cluster"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans
