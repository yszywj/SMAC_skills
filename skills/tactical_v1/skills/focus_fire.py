"""G1 FocusFire: joint fire allocation.

Reuses the shared calibrated damage model through Context.expected_damage.
Supports grouped multi-target allocation and an atomic joint-kill candidate
(P4.2) for urgent high-risk targets (e.g. an approaching enemy Baneling).
For splash units the S1 provider replaces the target value/impact model.
"""
from typing import Dict, List, Optional

from ..context import Context
from ..contracts import CALIBRATED, CandidatePlan, Commitment, EffectClaim, HEURISTIC, ResourceClaims
from ..ledger import Ledger
from . import splash_fire as splash_provider


class FocusFireSkill:
    name = "focus_fire"

    def __init__(self, config: Dict):
        self.config = config
        params = config.get("focus_fire", {})
        self.kill_window_loops = params.get("kill_window_loops", 24)
        self.joint_kill_max_attackers = params.get("joint_kill_max_attackers", 5)
        self.urgent_targets = params.get("urgent_enemy_types", ["baneling"])
        # The S1 mechanism provider replaces this skill's target-value and
        # impact model for splash units (coverage matrix, P3).
        self.splash = splash_provider.SplashFireSkill(config)
        self.risk_critical_fraction = params.get(
            "risk_critical_fraction",
            config.get("disengage_reenter", {}).get(
                "critical_health_fraction", 0.35))
        self.risk_min_cooldown = params.get(
            "risk_min_cooldown",
            config.get("combat_reposition", {}).get("min_window_loops", 8))
        self.risk_loss_per_threat = params.get("risk_loss_per_threat", 0.45)

    def _target_value(self, context: Context, actor: str, enemy_id: str) -> float:
        enemy = context.snapshot.enemies[enemy_id]
        if context.capabilities.get(actor) is not None and \
                context.capabilities[actor].specialization == "splash_fire":
            return self.splash.target_value(context, actor, enemy_id)
        # Lower effective health is better; distance breaks ties toward focus.
        return max(enemy.effective_health, 1.0)

    def _impact(self, context: Context, actor: str, enemy_id: str
                ) -> List[EffectClaim]:
        enemy = context.snapshot.enemies[enemy_id]
        capability = context.capabilities[actor]
        if capability.specialization == "splash_fire":
            return self.splash.impact_claims(context, actor, enemy_id)
        amount, evidence, confidence = context.expected_damage(actor, enemy_id)
        window = max(int(context.snapshot.step_mul), 1)
        if amount <= 0:
            # Unknown damage: no fabricated number; tiny positional credit only.
            return [EffectClaim("damage", enemy_id, 0.0, window, HEURISTIC,
                                "damage_model_unknown", 0.0)]
        return [EffectClaim("damage", enemy_id, amount, window, evidence,
                            "shared_damage_model_single_attack", confidence)]

    def propose(self, context: Context, memory, telemetry,
                ledger: Optional[Ledger] = None) -> List[CandidatePlan]:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List[CandidatePlan] = []
        attackers: Dict[str, List[str]] = {}
        for actor, unit in sorted(context.snapshot.allies.items()):
            if not unit.alive:
                continue
            capability = context.capabilities[actor]
            if capability.healer or capability.suicide_attack:
                continue  # S2/S3 own these units
            targets = context.attackable_enemies(actor)
            if targets:
                attackers[actor] = targets
            telemetry.count(self.name, "applicable")
        # -- atomic joint-kill candidate on an urgent target (P4.2) -------
        for enemy_id, enemy in sorted(context.snapshot.enemies.items()):
            if enemy.type_name not in self.urgent_targets:
                continue
            members: Dict[str, int] = {}
            effects: List[EffectClaim] = []
            fire: List[tuple] = []
            already = (ledger.fire_on(enemy_id) if ledger else 0.0)
            for actor in sorted(attackers):
                if len(members) >= self.joint_kill_max_attackers:
                    break
                action = context.attack_action(actor, enemy_id)
                if action is None:
                    continue
                members[actor] = action
                effects.extend(self._impact(context, actor, enemy_id))
                amount, _, _ = context.expected_damage(actor, enemy_id)
                if amount <= 0:  # window estimate for unknown damage
                    amount = 0.0
                fire.append((enemy_id, amount))
            if len(members) >= 2 and sum(a for _, a in fire) + already \
                    >= enemy.effective_health * 0.9:
                plans.append(_joint_plan(context, fingerprint, self.name,
                                         members, enemy_id, effects, fire))
        # -- per-attacker single-shot candidates ---------------------------
        for actor, targets in sorted(attackers.items()):
            provider = ("splash_fire"
                        if context.capabilities[actor].specialization == "splash_fire"
                        else None)
            scored = []
            for enemy_id in targets:
                value = self._target_value(context, actor, enemy_id)
                effects = self._impact(context, actor, enemy_id)
                scored.append((value, -context.distance(actor, enemy_id),
                               enemy_id, effects))
            scored.sort()
            for value, neg_distance, enemy_id, effects in scored[:3]:
                action = context.attack_action(actor, enemy_id)
                amount = next((e.amount for e in effects
                               if e.kind == "damage" and e.target == enemy_id), 0.0)
                effects = list(effects)
                # Continued-exposure risk: an attack plan for a critical unit
                # that is taking fire carries an expected-loss estimate. This
                # is a heuristic probability proxy, never a certain outcome,
                # and it COMPETES with damage instead of overriding it.
                unit = context.snapshot.allies[actor]
                threats = context.threats_to(actor)
                if (unit.health_fraction <= self.risk_critical_fraction
                        and threats
                        and unit.weapon_cooldown_loops >= self.risk_min_cooldown):
                    effects.append(EffectClaim(
                        "loss", actor,
                        self.risk_loss_per_threat * len(threats) * unit.max_health,
                        max(context.snapshot.step_mul, 1), HEURISTIC,
                        "continued_exposure_while_critical", 0.7))
                features = {"raw_bonus": 0.0}
                # A verified-legal attack with no calibrated damage estimate
                # still carries intent value (never a fabricated amount).
                features["attack_intent"] = 1.0 if amount <= 0 else 0.0
                if provider:
                    features["raw_bonus"] = self.splash.coverage_bonus(
                        context, actor, enemy_id)
                plans.append(_attack_plan(context, fingerprint, self.name,
                                          actor, action, enemy_id, effects, amount,
                                          provider=provider, features=features))
        telemetry.count(self.name, "proposed", len(plans))
        return plans


def _attack_plan(context, fingerprint, skill, actor, action, enemy_id, effects,
                 amount, provider=None, features=None) -> CandidatePlan:
    from .common import make_plan
    plan_features = {"switch_cost": 0.0, "raw_bonus": 0.0}
    plan_features.update(features or {})
    return make_plan(
        skill, {actor: action}, enemy_id, {actor: action}, effects,
        ResourceClaims(controls=(actor,), fire=((enemy_id, amount),)),
        Commitment("attacking", min_steps=0, max_steps=None),
        plan_features,
        context.snapshot.snapshot_id, fingerprint, provider=provider,
        rationale=f"attack {enemy_id}")


def _joint_plan(context, fingerprint, skill, members, enemy_id, effects,
                fire) -> CandidatePlan:
    from .common import make_plan
    return make_plan(
        skill, dict(members), enemy_id, dict(members), effects,
        ResourceClaims(controls=tuple(sorted(members)), fire=tuple(fire)),
        Commitment("joint_kill", min_steps=1, max_steps=4,
                   interrupt_on=("target_lost", "illegal", "member_withdrawn",
                                 "danger_change")),
        {"raw_bonus": 1.0, "attack_intent": 1.0},  # urgency + verified intent
        context.snapshot.snapshot_id, fingerprint,
        dependencies=tuple(f"member_ready:{a}" for a in sorted(members)),
        rationale=f"joint kill {enemy_id} with {len(members)} attackers")
