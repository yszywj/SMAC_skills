"""S3 BurstEngage: Baneling suicide-burst engagement.

The explosion radius/damage is NOT calibrated here, so the exchange value is
an explicitly-labelled geometric proxy (cluster size weighted by value near
the detonation point). The planned self-consumption enters the evaluation as
a cost, not as an invalid normal-unit loss; a Baneling disappearing is never
by itself recorded as a successful burst. No manual detonation or burrow is
added: the attack action against a legal target IS the environment's burst.
"""
from typing import Dict, List

from ..context import Context, STOP, distance
from ..contracts import Commitment, EffectClaim, HEURISTIC
from .common import best_move_toward, make_plan


class BurstEngageSkill:
    name = "burst_engage"

    def __init__(self, config: Dict):
        self.config = config
        params = config.get("burst_engage", {})
        self.proxy_radius = params.get("proxy_radius", 2.5)   # proxy, not SC2 truth
        self.value_per_enemy = params.get("value_per_enemy", 2.0)
        self.min_exchange_value = params.get("min_exchange_value", 2.0)
        self.self_cost = params.get("self_cost", 3.0)
        self.max_approach_steps = params.get("max_approach_steps", 40)

    def cluster_value(self, context: Context, center) -> float:
        value = 0.0
        for enemy in context.snapshot.enemies.values():
            if distance(enemy.position, center) <= self.proxy_radius + enemy.radius:
                weight = 1.0
                if enemy.type_name in ("baneling",):
                    weight = 0.4   # trading into banelings is worth less
                value += self.value_per_enemy * weight
        return value

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        banelings = [a for a, cap in sorted(context.capabilities.items())
                     if cap.suicide_attack and context.snapshot.allies[a].alive]
        if not banelings:
            return plans
        telemetry.count(self.name, "applicable", len(banelings))
        for actor in banelings:
            # Detonation opportunity: a legal attack on an adjacent target.
            best = None
            for enemy_id in sorted(context.snapshot.enemies):
                action = context.attack_action(actor, enemy_id)
                if action is None:
                    continue
                enemy = context.snapshot.enemies[enemy_id]
                value = self.cluster_value(context, enemy.position)
                if best is None or value > best[0] or (value == best[0]
                                                       and enemy_id < best[2]):
                    best = (value, action, enemy_id)
            if best is not None and best[0] >= self.min_exchange_value:
                value, action, enemy_id = best
                plans.append(make_plan(
                    self.name, {actor: action}, enemy_id, {actor: action},
                    [EffectClaim("damage", enemy_id, 0.0,
                                 max(context.snapshot.step_mul, 1), HEURISTIC,
                                 "uncalibrated_burst_proxy; excluded from kill budget",
                                 0.3),
                     EffectClaim("loss", actor, self.self_cost,
                                 max(context.snapshot.step_mul, 1), HEURISTIC,
                                 "planned_self_consumption", 0.9)],
                    _claims(actor, enemy_id), Commitment("bursting", 0, 2),
                    {"raw_bonus": value}, snapshot_id, fingerprint,
                    provider="burst_engage",
                    rationale=f"burst into cluster at {enemy_id} "
                              f"(proxy value {value:.1f})"))
                continue
            # Approach the best expected burst point (cluster centre proxy).
            centre = _best_cluster_centre(context)
            if centre is not None and self.cluster_value(context, centre) \
                    >= self.min_exchange_value:
                move = best_move_toward(context, actor, centre)
                if move is not None:
                    from .common import move_progress
                    plans.append(make_plan(
                        self.name, {actor: move}, "cluster", {actor: move},
                        [EffectClaim("position", actor,
                                     move_progress(context, actor, move, centre),
                                     max(context.snapshot.step_mul, 1), HEURISTIC,
                                     "approach_cluster", 0.7),
                         EffectClaim("loss", actor, 0.0,
                                     max(context.snapshot.step_mul, 1), HEURISTIC,
                                     "exposure_during_approach", 0.5)],
                        _claims(actor, "cluster"),
                        Commitment("approaching", 0, self.max_approach_steps,
                                   interrupt_on=("illegal", "danger_change",
                                                 "no_progress")),
                        {"raw_bonus": 0.2}, snapshot_id, fingerprint,
                        provider="burst_engage",
                        rationale="approach burst cluster"))
                    continue
            # Low-value contact: do not force the nearest enemy. Hold/regroup.
            plans.append(make_plan(
                self.name, {actor: STOP}, "hold", {actor: STOP},
                [EffectClaim("position", actor, 0.0,
                             max(context.snapshot.step_mul, 1), HEURISTIC,
                             "declined_low_value_contact", 0.6)],
                _claims(actor, "hold"),
                Commitment("declined", 0, 4, interrupt_on=("danger_change",)),
                {"raw_bonus": 0.0}, snapshot_id, fingerprint,
                provider="burst_engage",
                rationale="low-value contact declined; wait for a better cluster"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans


def _claims(actor, goal):
    from ..contracts import ResourceClaims
    return ResourceClaims(controls=(actor,), space=(f"{actor}->{goal}",))


def _best_cluster_centre(context: Context):
    """Proxy: the visible enemy position with the highest cluster value."""
    best, best_value = None, -1.0
    for enemy_id, enemy in sorted(context.snapshot.enemies.items()):
        value = BurstEngageSkill({"burst_engage": {}}).cluster_value(
            context, enemy.position)
        if value > best_value:
            best, best_value = enemy.position, value
    return best
