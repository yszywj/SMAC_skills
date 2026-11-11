"""G1 FocusFire: ordinary fire allocation and joint interception.

Reuses the shared fire/contact queries; the S1 provider (registry-injected,
respecting its real enabled state) replaces only the impact model for
splash units. Candidate retention keeps: the still-valid original target,
kill-completable targets, high-threat targets and nearby effective targets
— deduped, capped per unit, never "first three by lowest HP".
"""
from typing import Dict, List, Optional

from ..context import Context, IN_RANGE
from ..contracts import (CALIBRATED, Commitment, Dependency, EffectClaim,
                         HEURISTIC, ResourceClaims)
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
        # Registry-injected providers (R5): a disabled S1 is simply absent.
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
        # Command is legal but no window-qualified damage (cooldown/range):
        # an approach/keep opportunity only, never a fabricated amount.
        return [EffectClaim("damage", enemy_id, None, window, HEURISTIC,
                            "attack_command_without_window_damage")]

    def _window_damage(self, context: Context, actor: str, enemy_id: str) -> float:
        for effect in self._impact(context, actor, enemy_id):
            if effect.kind == "damage" and effect.target == enemy_id \
                    and effect.amount is not None:
                return effect.amount
        return 0.0

    # -- target ordering ----------------------------------------------------
    def _target_value(self, context: Context, actor: str, enemy_id: str) -> float:
        """Lower is better: contact feasibility, real threat, kill cost."""
        enemy = context.snapshot.enemies[enemy_id]
        contact = context.contact_status(actor, enemy_id)
        feasibility = {"in_range": 0.0, "unknown": 1.0,
                       "out_of_range": 2.0}[contact]
        threats_us = sum(1 for a in context.snapshot.allies
                         if enemy_id in context.threats_to(a))
        damage = self._window_damage(context, actor, enemy_id)
        kill_cost = enemy.effective_health / max(damage, 1.0)
        return feasibility - 0.5 * min(threats_us, 3) + 0.1 * min(kill_cost, 10)

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
                continue                      # S2 / S3 own these units
            issuable = [e for e in sorted(context.snapshot.enemies)
                        if context.can_issue_attack(actor, e)]
            if issuable:
                attackers[actor] = issuable
                telemetry.count(self.name, "applicable")
        # ---- per-attacker candidate set (diverse retention, dedup, cap) ----
        for actor, issuable in sorted(attackers.items()):
            original = self._original_target(context, memory, actor)
            killable = [e for e in issuable
                        if self._window_damage(context, actor, e)
                        >= context.snapshot.enemies[e].effective_health]
            high_threat = [e for e in issuable
                           if any(e in context.threats_to(a)
                                  for a in context.snapshot.allies)]
            nearby = sorted(issuable, key=lambda e: (
                context.snapshot.relations[actor][e].distance, e))
            ordered: List[str] = []
            for group in (killable, original and [original], high_threat, nearby):
                for enemy_id in group or []:
                    if enemy_id and enemy_id not in ordered:
                        ordered.append(enemy_id)
            for enemy_id in ordered[:self.max_candidates]:
                action = context.attack_action(actor, enemy_id)
                if action is None:
                    continue
                amount = self._window_damage(context, actor, enemy_id)
                effects = self._impact(context, actor, enemy_id)
                capability = context.capabilities[actor]
                provider = ("splash_fire"
                            if capability.specialization == "splash_fire"
                            and self.providers.get("splash_fire") else None)
                instance = memory.instance_for(self.name, actor)
                switch = 0.0
                if instance and instance.goal and instance.goal != enemy_id:
                    switch = 0.3     # bounded target-switch cost, not distance faking
                plans.append(make_plan(
                    self.name, {actor: action}, enemy_id, {actor: action},
                    effects,
                    ResourceClaims(controls=(actor,),
                                   fire=((enemy_id, amount),) if amount else ()),
                    Commitment("attacking", min_loops=0),
                    {"switch_cost": switch, "urgency": (
                        self.urgency_value if enemy_id in high_threat else 0.0)},
                    snapshot_id, fingerprint, provider=provider,
                    instance_id=instance.instance_id if instance else None,
                    memory_delta=({"kind": "create", "skill": self.name,
                                   "actors": [actor], "goal": enemy_id,
                                   "phase": "attacking"}
                                  if instance is None else None),
                    rationale=f"attack {enemy_id}"))
        # ---- atomic joint interception on urgent targets (P4.2) ----------
        plans.extend(self._joint_interceptions(
            context, memory, telemetry, attackers, fingerprint, snapshot_id))
        telemetry.count(self.name, "proposed", len(plans))
        return plans

    def _original_target(self, context: Context, memory, actor: str) -> Optional[str]:
        instance = memory.instance_for(self.name, actor)
        if instance and instance.goal.startswith("e") \
                and instance.goal in context.snapshot.enemies \
                and context.can_issue_attack(actor, instance.goal):
            return instance.goal
        return None

    def _joint_interceptions(self, context, memory, telemetry, attackers,
                             fingerprint, snapshot_id) -> List:
        plans: List = []
        budget_h = self.kill_window_h
        max_atomic = self.config.get("search", {}).get(
            "max_atomic_candidates", 12)
        produced = 0
        for enemy_id, enemy in sorted(context.snapshot.enemies.items()):
            if enemy.type_name not in self.urgent_types:
                continue
            # member pool ranked by window-qualified damage then fire delay
            pool = []
            for actor in sorted(attackers):
                if not context.can_issue_attack(actor, enemy_id):
                    continue
                estimate = context.fire_estimate(actor, enemy_id)
                if estimate.window_shots < 1:
                    continue
                damage = (context.fire_estimate(actor, enemy_id).damage_est
                          or self._window_damage(context, actor, enemy_id))
                if not damage:
                    continue
                delay = estimate.effective_after_loops or 0
                pool.append((-damage, delay, actor, damage))
            pool.sort()
            # minimal sufficient member sets, few distinct combinations
            cumulative, members, effects, fire = 0.0, {}, [], []
            for neg_damage, delay, actor, damage in pool:
                if len(members) >= self.joint_kill_max_members:
                    break
                action = context.attack_action(actor, enemy_id)
                members[actor] = action
                effects.extend(self._impact(context, actor, enemy_id))
                fire.append((enemy_id, damage))
                cumulative += damage
                # emit this prefix as one atomic candidate when the
                # CONSERVATIVE total covers the target's effective health
                # inside the window; feasibility is "attempted", never
                # guaranteed interception (arrival times unverified).
                if cumulative >= enemy.effective_health \
                        and len(members) >= 2 and produced < max_atomic:
                    produced += 1
                    plans.append(make_plan(
                        self.name, dict(members), enemy_id, dict(members),
                        effects,
                        ResourceClaims(controls=tuple(sorted(members)),
                                       fire=tuple(fire)),
                        Commitment("intercepting", min_loops=0,
                                   max_loops=h_loops(context, budget_h),
                                   interrupt_on=("target_lost", "illegal",
                                                 "member_withdrawn",
                                                 "danger_change")),
                        {"urgency": self.urgency_value},
                        snapshot_id, fingerprint, atomic=True,
                        dependencies=(Dependency("member", tuple(sorted(members))),
                                     Dependency("target_valid", (enemy_id,)),
                                     Dependency("fire_window", (enemy_id,))),
                        rationale=f"attempted joint interception of {enemy_id} "
                                  f"with {len(members)} attackers"))
                    break   # minimal sufficient set: stop adding members
        return plans
