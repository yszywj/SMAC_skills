"""G4 DisengageReenter: phased, bounded local escape (C4).

ESCAPE -> STABILIZE -> REENTER with transitions that only fire on OBSERVED
conditions and are COMMITTED through unified transition deltas (phase_start
set once by memory). No propose-side mutation of instances; total budget
never resets, exhausted escapes terminate with a retry record.
"""
from typing import Dict, List, Optional

from ..context import Context, STOP, distance
from ..contracts import Commitment, EffectClaim, HEURISTIC, ResourceClaims
from .common import best_move_away, best_move_toward, h_loops, make_plan, move_progress, space_claim

ESCAPE, STABILIZE, REENTER = "ESCAPE", "STABILIZE", "REENTER"


class DisengageReenterSkill:
    name = "disengage_reenter"

    def __init__(self, config: Dict):
        self.config = config
        lifecycle = config.get("lifecycle", {})
        params = config.get("disengage_reenter", {})
        self.escape_max_h = lifecycle.get("escape_max_h", 3.0)
        self.stabilize_max_h = lifecycle.get("stabilize_max_h", 2.0)
        self.reenter_max_h = lifecycle.get("reenter_max_h", 6.0)
        self.total_max_h = lifecycle.get("disengage_total_max_h", 12.0)
        self.critical = params.get("critical_health_fraction", 0.35)
        self.reenter_frac = params.get("reenter_health_fraction", 0.6)
        self.risk_threshold = params.get("escape_risk_threshold", 2)

    def _entry(self, context: Context, actor: str) -> bool:
        unit = context.snapshot.allies[actor]
        capability = context.capabilities[actor]
        if capability.suicide_attack or capability.healer:
            return False            # S3 / S2 replace survival value
        risk = context.risk_at(actor, unit.position)
        pincer = len(risk.threats) >= self.risk_threshold
        severe = (unit.health_fraction <= self.critical
                  and (unit.health_lost + unit.shield_lost > 0
                       or unit.health_fraction <= self.critical / 2))
        shielded = unit.shield > 0 and unit.shield_lost > 0 \
            and unit.health_fraction > self.critical
        return (severe or pincer or (risk.effect_weapon_near
                                     and unit.health_fraction < 0.7)) \
            and not shielded

    def _phase(self, context: Context, actor: str, instance) -> Optional[str]:
        """Observed-condition phase machine: never an unconditional step to
        STABILIZE merely because an escape move was selected before."""
        loop = context.snapshot.game_loop
        if loop - instance.start_loop >= h_loops(context, self.total_max_h):
            return None             # total timeout: terminate, do not restart
        unit = context.snapshot.allies[actor]
        risk = context.risk_at(actor, unit.position)
        healers = any(cap.healer and context.snapshot.allies[a].alive
                      for a, cap in context.capabilities.items())
        if instance.phase == ESCAPE:
            in_escape_long = loop - instance.phase_start_loop >= h_loops(
                context, self.escape_max_h)
            if not risk.threats:
                # risk acceptable: stabilize only when waiting really pays
                # (a healer exists and health is still low); else REENTER now
                if healers and unit.health_fraction < self.reenter_frac:
                    return STABILIZE
                return REENTER
            if in_escape_long:
                return "STILL_DANGEROUS"   # compare kill / terminate paths
            return ESCAPE
        if instance.phase == STABILIZE:
            if risk.threats:
                return ESCAPE        # danger returned: flee again, no reset
            waited = loop - instance.phase_start_loop
            if waited >= h_loops(context, self.stabilize_max_h) \
                    or not healers \
                    or unit.health_fraction >= self.reenter_frac:
                return REENTER
            return STABILIZE
        return REENTER

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        loop = context.snapshot.game_loop
        total_budget = h_loops(context, self.total_max_h)
        for actor, unit in sorted(context.snapshot.allies.items()):
            if not unit.alive or context.capabilities[actor].healer \
                    or context.capabilities[actor].suicide_attack:
                continue
            if memory.retry_blocked(self.name, [actor], "escape", loop):
                continue
            instance = memory.instance_for(self.name, actor)
            active = instance is not None and instance.termination_reason is None
            if not active:
                if not self._entry(context, actor):
                    continue
                telemetry.count(self.name, "applicable")
                phase = ESCAPE
            else:
                phase = self._phase(context, actor, instance)
                if phase is None:
                    continue        # total timeout handled by memory facts
            risk = context.risk_at(actor, unit.position)
            killable = next((e for e in context.attackable_enemies(actor)
                             if (context.fire_estimate(actor, e).damage_est or 0)
                             >= context.snapshot.enemies[e].effective_health),
                            None)
            if phase in (ESCAPE, "STILL_DANGEROUS"):
                move = best_move_away(context, actor, list(risk.threats))
                improvement = any(
                    len(context.risk_at(actor, context.predict_position(
                        actor, a).position).threats) < len(risk.threats)
                    for a in context.move_destinations(actor))
                if move is not None and (improvement or killable is None):
                    plans.append(make_plan(
                        self.name, {actor: move}, "safe", {actor: move},
                        [EffectClaim("position", actor,
                                     move_progress(context, actor, move, (0., 0.)),
                                     max(context.H, 1), HEURISTIC,
                                     "escape_risk_improvement")],
                        ResourceClaims(controls=(actor,),
                                       space=(space_claim(context, actor, move),)),
                        Commitment(ESCAPE, 0, max_loops=total_budget),
                        {}, snapshot_id, fingerprint,
                        instance_id=instance.instance_id if instance else None,
                        memory_delta=self._delta(instance, ESCAPE, actor),
                        rationale="escape: risk improvement available"))
                elif killable is not None:
                    action = context.attack_action(actor, killable)
                    plans.append(make_plan(
                        self.name, {actor: action}, killable, {actor: action},
                        [EffectClaim("damage", killable, None, max(context.H, 1),
                                     HEURISTIC, "stand_and_finish_kill")],
                        ResourceClaims(controls=(actor,)),
                        Commitment(ESCAPE, 0, max_loops=total_budget),
                        {}, snapshot_id, fingerprint,
                        instance_id=instance.instance_id if instance else None,
                        memory_delta=self._delta(instance, ESCAPE, actor),
                        rationale="no safe improvement; finish the kill"))
                else:
                    # escape budget spent and still dangerous: terminate with
                    # a retry cooldown instead of a fake stabilize (C4)
                    plans.append(make_plan(
                        self.name, {actor: STOP}, "safe", {actor: STOP},
                        [EffectClaim("position", actor, None, max(context.H, 1),
                                     HEURISTIC, "escape_failed_hold")],
                        ResourceClaims(controls=(actor,)),
                        Commitment(ESCAPE, 0, 1),
                        {"hold_value": 0.1}, snapshot_id, fingerprint,
                        instance_id=instance.instance_id if instance else None,
                        memory_delta=({"kind": "drop",
                                       "instance_id": instance.instance_id,
                                       "reason": "escape_failed"}
                                      if instance else None),
                        rationale="escape exhausted; hold, retry later"))
                continue
            if phase == STABILIZE:
                plans.append(make_plan(
                    self.name, {actor: STOP}, "stabilize", {actor: STOP},
                    [EffectClaim("position", actor, None, max(context.H, 1),
                                 HEURISTIC,
                                 "stabilize_wait_real_stop_command")],
                    ResourceClaims(controls=(actor,)),
                    Commitment(STABILIZE, 0, max_loops=h_loops(
                        context, self.stabilize_max_h)),
                    {"hold_value": 0.3}, snapshot_id, fingerprint,
                    instance_id=instance.instance_id if instance else None,
                    memory_delta=self._delta(instance, STABILIZE, actor),
                    rationale="stabilize (bounded, conditions real)"))
                continue
            target = context.nearest_enemy(actor)
            if target is None:
                continue
            move = best_move_toward(
                context, actor, context.snapshot.enemies[target].position)
            if move is not None:
                prediction = context.predict_position(actor, move)
                if len(context.risk_at(actor, prediction.position).threats) \
                        > len(risk.threats) + 1:
                    continue        # no serious risk jump on re-entry
                plans.append(make_plan(
                    self.name, {actor: move}, target, {actor: move},
                    [EffectClaim("position", actor,
                                 move_progress(context, actor, move,
                                               context.snapshot.enemies[target].position),
                                 max(context.H, 1), HEURISTIC, "reenter")],
                    ResourceClaims(controls=(actor,),
                                   space=(space_claim(context, actor, move),)),
                    Commitment(REENTER, 0, max_loops=h_loops(
                        context, self.reenter_max_h)),
                    {"progress": move_progress(
                        context, actor, move,
                        context.snapshot.enemies[target].position)},
                    snapshot_id, fingerprint,
                    instance_id=instance.instance_id if instance else None,
                    memory_delta=self._delta(instance, REENTER, actor,
                                             goal=target),
                    rationale=f"re-enter toward {target}"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans

    def _delta(self, instance, phase, actor, goal=None):
        total_budget = int(self.total_max_h * 8)
        if instance is None:
            return {"kind": "create", "skill": self.name, "actors": [actor],
                    "goal": goal or "safe", "goal_kind": "internal",
                    "phase": phase, "max_loops": total_budget,
                    "extra": {"phase_reason": "observed_conditions"}}
        return {"kind": "transition", "instance_id": instance.instance_id,
                "phase": phase, "goal": goal,
                "goal_kind": "enemy" if goal else None}
