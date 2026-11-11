"""S3 BurstEngage: consumption-burst approach, timing and split investment.

Replaces ordinary sustained fire AND the low-HP survival value; keeps the
shared movement/congestion/search tools. "Attack command issuable" is NOT
"already detonated": out-of-contact candidates are approach/tracking with
unknown burst timing. All magnitudes are capped coverage OPPORTUNITY
proxies; the planned self-consumption is counted ONCE (here, not again as
ordinary damage-taken).
"""
from typing import Dict, List, Optional

from ..context import Context, STOP, distance
from ..contracts import Commitment, EffectClaim, HEURISTIC, ResourceClaims
from .common import best_move_toward, make_plan, move_progress, space_claim


class BurstEngageSkill:
    name = "burst_engage"

    def __init__(self, config: Dict):
        self.config = config
        params = config.get("burst_engage", {})
        self.proxy_radius = params.get("proxy_radius", 2.5)   # proxy, not SC2 truth
        self.value_per_enemy = params.get("value_per_enemy", 2.0)
        self.min_exchange_value = params.get("min_exchange_value", 2.0)
        self.self_cost = params.get("self_cost", 1.0)         # fraction scale
        self.lonely_exception = params.get("lonely_enemy_exception", True)

    def cluster_value(self, context: Context, center) -> float:
        value = 0.0
        for enemy in context.snapshot.enemies.values():
            if distance(enemy.position, center) <= self.proxy_radius + enemy.radius:
                weight = 0.4 if enemy.type_name == "baneling" else 1.0
                value += self.value_per_enemy * weight
        return value

    def _best_clusters(self, context: Context, current_goal=None) -> List:
        """Diverse retention: current target, high-coverage, near-approachable."""
        scored = []
        for enemy_id, enemy in sorted(context.snapshot.enemies.items()):
            scored.append((enemy_id, enemy.position,
                           self.cluster_value(context, enemy.position),
                           distance((0, 0), enemy.position)))
        if not scored:
            return []
        by_value = sorted(scored, key=lambda s: (-s[2], s[0]))
        result, seen = [], set()
        for enemy_id, position, value, _ in sorted(
                scored, key=lambda s: (s[0] != current_goal, -s[2], s[0])):
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
            current_goal = instance.goal if instance and \
                instance.goal.startswith("e") else None
            # 1. detonation opportunity: issuable AND in physical contact
            best = None
            for enemy_id in sorted(context.snapshot.enemies):
                if not context.can_issue_attack(actor, enemy_id):
                    continue
                if context.contact_status(actor, enemy_id) != "in_range":
                    continue   # command issuable but not yet in burst contact
                value = self.cluster_value(
                    context, context.snapshot.enemies[enemy_id].position)
                if best is None or value > best[0]:
                    best = (value, enemy_id)
            threshold = 0.0 if lonely else self.min_exchange_value
            if best is not None and best[0] >= threshold:
                value, enemy_id = best
                action = context.attack_action(actor, enemy_id)
                plans.append(make_plan(
                    self.name, {actor: action}, enemy_id, {actor: action},
                    [EffectClaim("coverage", enemy_id, None, max(context.H, 1),
                                 HEURISTIC,
                                 "in_contact_burst_opportunity_timing_unknown"),
                     EffectClaim("loss", actor, self.self_cost, max(context.H, 1),
                                 HEURISTIC, "planned_self_consumption_once")],
                    ResourceClaims(controls=(actor,)),
                    Commitment("bursting", 0, max_loops=2 * max(context.H, 1)),
                    {"urgency": 0.5}, snapshot_id, fingerprint,
                    provider="burst_engage",
                    instance_id=instance.instance_id if instance else None,
                    rationale=f"burst opportunity at {enemy_id} "
                              f"(proxy value {value:.1f})"))
                continue
            # 2. approach candidates: diverse cluster regions
            clusters = self._best_clusters(context, current_goal)
            worthwhile = [c for c in clusters
                          if c[2] >= threshold or lonely]
            if worthwhile:
                for enemy_id, position, value in worthwhile[:2]:
                    move = best_move_toward(context, actor, position)
                    if move is None:
                        continue
                    plans.append(make_plan(
                        self.name, {actor: move}, enemy_id, {actor: move},
                        [EffectClaim("position", actor,
                                     move_progress(context, actor, move, position),
                                     max(context.H, 1), HEURISTIC,
                                     "approach_burst_region")],
                        ResourceClaims(
                            controls=(actor,),
                            space=(space_claim(context, actor, move),)),
                        Commitment("approaching", 0,
                                   max_loops=40 * max(context.H, 1),
                                   interrupt_on=("no_progress", "danger_change")),
                        {"progress": abs(move_progress(
                            context, actor, move, position))},
                        snapshot_id, fingerprint, provider="burst_engage",
                        instance_id=instance.instance_id if instance else None,
                        memory_delta=({"kind": "create", "skill": self.name,
                                       "actors": [actor], "goal": enemy_id,
                                       "phase": "approaching"}
                                      if instance is None else None),
                        rationale=f"approach burst region {enemy_id}"))
                continue
            # 3. nothing worth the trade: hold (declared, not a fake no-op)
            plans.append(make_plan(
                self.name, {actor: STOP}, "hold", {actor: STOP},
                [EffectClaim("position", actor, None, max(context.H, 1),
                             HEURISTIC, "declined_low_value_contact")],
                ResourceClaims(controls=(actor,)),
                Commitment("declined", 0, max_loops=4 * max(context.H, 1),
                           interrupt_on=("danger_change",)),
                {"hold_value": 0.2}, snapshot_id, fingerprint,
                provider="burst_engage",
                rationale="low-value contact declined; wait for a cluster"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans
