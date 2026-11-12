"""G1 FocusFire: ordinary fire allocation and joint interception (D1).

Candidate classes (original target / kill-completable / high real threat /
nearby effective) are INTERLEAVED so no class exhausts the quota by id
order. Colossus candidates come from the enabled S1 provider's target
ranking. Joint interception members are picked by window opportunity and
effect timing, with one alternative combination so important members may
keep healing or escaping instead.
"""
from typing import Dict, List, Optional

from ..context import Context, IN_RANGE
from ..contracts import (Commitment, Dependency, EffectClaim, HEURISTIC,
                         ResourceClaims)
from .common import h_loops, make_plan, space_claim


class FocusFireSkill:
    name = "focus_fire"

    def __init__(self, config: Dict, providers: Optional[Dict] = None):
        self.config = config
        params = config.get("focus_fire", {})
        self.kill_window_h = params.get("kill_window_h", 3.0)
        self.joint_kill_max_members = params.get("joint_kill_max_members", 5)
        self.urgency_value = params.get("urgency_value", 0.5)
        self.max_candidates = config.get("search", {}).get(
            "max_candidates_per_unit", 6)
        self.urgent_types = ["baneling"]
        self.providers = providers or {}

    # -- impact model -------------------------------------------------------
    def _impact(self, context: Context, actor: str, enemy_id: str
                ) -> List[EffectClaim]:
        provider = self.providers.get("splash_fire")
        capability = context.capabilities[actor]
        if capability.specialization == "splash_fire" and provider is not None:
            return provider.impact_claims(context, actor, enemy_id)
        estimate = context.fire_estimate(actor, enemy_id)
        window = max(context.H, 1)
        if estimate.window_shots >= 1 and estimate.damage_est is not None:
            return [EffectClaim("damage", enemy_id, estimate.damage_est, window,
                                estimate.evidence,
                                "shared_damage_model_window_shot",
                                lower=estimate.damage_lo, upper=estimate.damage_hi)]
        return [EffectClaim("damage", enemy_id, None, window, HEURISTIC,
                            "attack_command_without_window_damage")]

    def _window_damage(self, context: Context, actor: str, enemy_id: str) -> float:
        for effect in self._impact(context, actor, enemy_id):
            if effect.kind == "damage" and effect.target == enemy_id \
                    and effect.amount is not None:
                return effect.amount
        return 0.0

    # -- candidate ordering -------------------------------------------------
    def _ordered_targets(self, context: Context, actor: str, issuable,
                         original) -> List[str]:
        provider = self.providers.get("splash_fire")
        capability = context.capabilities[actor]
        if capability.specialization == "splash_fire" and provider is not None:
            ranked = provider.target_candidates(context, actor)
            # interleave the provider ranking with killable/threat/nearby
            classes = [ranked]
        else:
            classes = []
        killable = [e for e in issuable
                    if self._window_damage(context, actor, e)
                    >= context.snapshot.enemies[e].effective_health]
        high_threat = [e for e in issuable
                       if any(e in context.threats_to(a)
                              for a in context.snapshot.allies)]
        nearby = sorted(issuable, key=lambda e: (
            context.snapshot.relations[actor][e].distance, e))
        classes = [killable, [original] if original else [], high_threat,
                   nearby, *classes]
        # round-robin interleave: one from each class per round
        ordered, seen = [], set()
        for round_index in range(max(len(c) for c in classes if c)):
            for group in classes:
                if not group:
                    continue
                if round_index < len(group) and group[round_index] not in seen:
                    seen.add(group[round_index])
                    ordered.append(group[round_index])
        return ordered

    def propose(self, context: Context, memory, telemetry,
                ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        attackers: Dict[str, List[str]] = {}
        for actor, unit in sorted(context.snapshot.allies.items()):
            if not unit.alive:
                continue
            capability = context.capabilities[actor]
            if capability.healer or capability.suicide_attack:
                continue
            issuable = [e for e in sorted(context.snapshot.enemies)
                        if context.can_issue_attack(actor, e)]
            if issuable:
                attackers[actor] = issuable
                telemetry.count(self.name, "applicable")
        for actor, issuable in sorted(attackers.items()):
            instance = memory.instance_for(self.name, actor)
            original = None
            if instance is not None and instance.goal_kind == "enemy" \
                    and instance.goal in context.snapshot.enemies \
                    and context.can_issue_attack(actor, instance.goal):
                original = instance.goal
            for enemy_id in self._ordered_targets(
                    context, actor, issuable, original)[:self.max_candidates]:
                action = context.attack_action(actor, enemy_id)
                if action is None:
                    continue
                amount = self._window_damage(context, actor, enemy_id)
                effects = self._impact(context, actor, enemy_id)
                capability = context.capabilities[actor]
                provider = ("splash_fire"
                            if capability.specialization == "splash_fire"
                            and self.providers.get("splash_fire") else None)
                delta = None
                if instance is None:
                    delta = {"kind": "create", "skill": self.name,
                             "actors": [actor], "goal": enemy_id,
                             "goal_kind": "enemy", "phase": "attacking"}
                elif instance.goal != enemy_id:
                    delta = {"kind": "retarget", "instance_id":
                             instance.instance_id, "goal": enemy_id,
                             "goal_kind": "enemy"}   # keep budget, update goal
                switch = 0.2 if instance is not None and instance.goal != enemy_id \
                    else 0.0
                plans.append(make_plan(
                    self.name, {actor: action}, enemy_id, {actor: action},
                    effects,
                    ResourceClaims(controls=(actor,),
                                   fire=((enemy_id, amount),) if amount else ()),
                    Commitment("attacking", min_loops=0),
                    {"switch_cost": switch, "urgency": (
                        self.urgency_value if any(
                            enemy_id in context.threats_to(a)
                            for a in context.snapshot.allies) else 0.0)},
                    snapshot_id, fingerprint, provider=provider,
                    instance_id=instance.instance_id if instance else None,
                    memory_delta=delta,
                    goal_kind="enemy",
                    goal_data={"target_id": enemy_id},
                    rationale=f"attack {enemy_id}"))
        plans.extend(self._joint_interceptions(
            context, memory, telemetry, attackers, fingerprint, snapshot_id))
        telemetry.count(self.name, "proposed", len(plans))
        return plans

    def _joint_interceptions(self, context, memory, telemetry, attackers,
                             fingerprint, snapshot_id) -> List:
        plans: List = []
        max_atomic = self.config.get("search", {}).get(
            "max_atomic_candidates", 12)
        produced = 0
        for enemy_id, enemy in sorted(context.snapshot.enemies.items()):
            if enemy.type_name not in self.urgent_types:
                continue
            # member pool ranked by opportunity: window damage, effect delay,
            # then the member's own foregone output (opportunity cost)
            pool = []
            for actor in sorted(attackers):
                if not context.can_issue_attack(actor, enemy_id):
                    continue
                estimate = context.fire_estimate(actor, enemy_id)
                if estimate.window_shots < 1 or not estimate.damage_est:
                    continue
                foregone = max(
                    (context.fire_estimate(actor, other).damage_est or 0.0)
                    for other in context.attackable_enemies(actor)
                    if other != enemy_id) if len(
                    context.attackable_enemies(actor)) > 1 else 0.0
                pool.append((estimate.effective_after_loops or 0,
                             -estimate.damage_est, foregone, actor,
                             estimate.damage_est))
            pool.sort()
            # primary minimal-sufficient set + one alternative combination
            # (the alternative leaves the best-timing member free to keep
            # healing / escaping / other output)
            for skip_first_member in (False, True):
                if produced >= max_atomic:
                    break
                cumulative, members, effects, fire = 0.0, {}, [], []
                member_pool = pool[1:] if (skip_first_member and pool) else pool
                for delay, neg_damage, foregone, actor, damage in member_pool:
                    if len(members) >= self.joint_kill_max_members:
                        break
                    action = context.attack_action(actor, enemy_id)
                    members[actor] = action
                    effects.extend(self._impact(context, actor, enemy_id))
                    fire.append((enemy_id, damage))
                    cumulative += damage
                    if cumulative >= enemy.effective_health and len(members) >= 2:
                        produced += 1
                        plans.append(make_plan(
                            self.name, dict(members), enemy_id, dict(members),
                            effects,
                            ResourceClaims(controls=tuple(sorted(members)),
                                           fire=tuple(fire)),
                            Commitment("intercepting", 0,
                                       max_loops=h_loops(context,
                                                         self.kill_window_h)),
                            {"urgency": self.urgency_value},
                            snapshot_id, fingerprint, atomic=True,
                            goal_kind="enemy",
                            goal_data={"target_id": enemy_id},
                            dependencies=(Dependency("member",
                                                     tuple(sorted(members))),
                                         Dependency("target_valid", (enemy_id,)),
                                         Dependency("fire_window", (enemy_id,))),
                            rationale=f"attempted joint interception of "
                                      f"{enemy_id} with {len(members)} attackers"))
                        break
        return plans
