"""G4 DisengageReenter: phased, bounded local escape (R-fixed).

Phases ESCAPE -> STABILIZE -> REENTER with real loop budgets consumed from
the instance; transitions on observed evidence (risk drop, contact return,
target loss, timeout). Entry considers risk vs. the whole position set —
including "keep attacking" when no safe improvement exists and the target
is nearly dead. Shields are read separately from health. The skill
controls ONLY the escaping unit; covering fire stays with G1/G3.
"""
from typing import Dict, List

from ..context import Context, STOP, distance
from ..contracts import Commitment, EffectClaim, HEURISTIC, ResourceClaims
from .common import best_move_away, best_move_toward, h_loops, make_plan, move_progress, space_claim


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
        """Risk of the CURRENT behavior clearly exceeds a bounded escape:
        severe damage, approaching area danger, or local pincers. Not a
        bare health_fraction check."""
        unit = context.snapshot.allies[actor]
        capability = context.capabilities[actor]
        if capability.suicide_attack:
            return False                       # S3 replaces survival value
        if capability.healer:
            return False                       # S2 routes its own escape
        risk = context.risk_at(actor, unit.position)
        approaching_area_danger = bool(risk.effect_weapon_near)
        pincer = len(risk.threats) >= self.risk_threshold
        severe = (unit.health_fraction <= self.critical
                  and (unit.health_lost + unit.shield_lost > 0
                       or unit.health_fraction <= self.critical / 2))
        shields_left = unit.shield > 0 and unit.shield_lost > 0 \
            and unit.health_fraction > self.critical
        return severe or pincer or (approaching_area_danger and
                                    unit.health_fraction < 0.7) \
            and not shields_left

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        loop = context.snapshot.game_loop
        for actor, unit in sorted(context.snapshot.allies.items()):
            if not unit.alive or context.capabilities[actor].healer \
                    or context.capabilities[actor].suicide_attack:
                continue
            instance = memory.instance_for(self.name, actor)
            in_phase = instance is not None and instance.termination_reason is None
            if not in_phase and not self._entry(context, actor):
                continue
            if not in_phase and self._entry(context, actor):
                telemetry.count(self.name, "applicable")
                phase = "ESCAPE"
            else:
                phase = instance.phase
                total = loop - instance.start_loop
                if total > h_loops(context, self.total_max_h):
                    memory.drop_instance(instance.instance_id, "total_timeout")
                    continue
            risk = context.risk_at(actor, unit.position)
            if phase == "ESCAPE":
                budget = h_loops(context, self.escape_max_h)
                if instance is not None:
                    budget = max(budget, instance.max_loops or budget)
                move = best_move_away(context, actor, list(risk.threats))
                # compare: does ANY legal position actually improve risk?
                improvement = False
                for action in context.move_destinations(actor):
                    prediction = context.predict_position(actor, action)
                    if len(context.risk_at(actor, prediction.position).threats) \
                            < len(risk.threats):
                        improvement = True
                        break
                killable = next((e for e in context.attackable_enemies(actor)
                                 if (context.fire_estimate(actor, e).damage_est or 0)
                                 >= context.snapshot.enemies[e].effective_health),
                                None)
                if move is not None and (improvement or killable is None):
                    plans.append(make_plan(
                        self.name, {actor: move}, "safe", {actor: move},
                        [EffectClaim("position", actor,
                                     move_progress(context, actor, move, (0., 0.)),
                                     max(context.H, 1), HEURISTIC,
                                     "escape_risk_improvement")],
                        ResourceClaims(controls=(actor,),
                                       space=(space_claim(context, actor, move),)),
                        Commitment("ESCAPE", 0, max_loops=h_loops(
                            context, self.escape_max_h)),
                        {}, snapshot_id, fingerprint,
                        instance_id=instance.instance_id if instance else None,
                        memory_delta=({"kind": "create", "skill": self.name,
                                       "actors": [actor], "goal": "safe",
                                       "phase": "ESCAPE",
                                       "max_loops": h_loops(
                                           context, self.total_max_h)}
                                      if instance is None else
                                      {"kind": "phase",
                                       "instance_id": instance.instance_id,
                                       "phase": "STABILIZE"}),
                        rationale="escape: risk improvement available"))
                elif killable is not None:
                    # no safe improvement AND a completable kill: attacking
                    # is a legitimate winner, not "move for movement's sake"
                    plans.append(make_plan(
                        self.name,
                        {actor: context.attack_action(actor, killable)},
                        killable, {actor: context.attack_action(actor, killable)},
                        [EffectClaim("damage", killable, None, max(context.H, 1),
                                     HEURISTIC, "stand_and_finish_kill")],
                        ResourceClaims(controls=(actor,)),
                        Commitment("ESCAPE", 0, max_loops=h_loops(
                            context, self.escape_max_h)),
                        {}, snapshot_id, fingerprint,
                        rationale="no safe improvement; finish the kill"))
                else:
                    plans.append(make_plan(
                        self.name, {actor: STOP}, "safe", {actor: STOP},
                        [EffectClaim("position", actor, None, max(context.H, 1),
                                     HEURISTIC, "escape_blocked_hold")],
                        ResourceClaims(controls=(actor,)),
                        Commitment("ESCAPE", 0, max_loops=h_loops(
                            context, self.escape_max_h)),
                        {"hold_value": 0.2}, snapshot_id, fingerprint,
                        rationale="no improving escape direction; hold"))
                continue
            if phase == "STABILIZE":
                healers = any(cap.healer and
                              context.snapshot.allies[a].alive
                              for a, cap in context.capabilities.items())
                risk_ok = not risk.threats
                waited = loop - (instance.phase_start_loop if instance else loop)
                if risk_ok and (not healers or unit.health_fraction >= self.reenter_frac
                                or waited > h_loops(context, self.stabilize_max_h)):
                    phase = "REENTER"      # fall through to re-entry below
                else:
                    plans.append(make_plan(
                        self.name, {actor: STOP}, "stabilize", {actor: STOP},
                        [EffectClaim("position", actor, None, max(context.H, 1),
                                     HEURISTIC,
                                     "stabilize_wait_real_stop_command")],
                        ResourceClaims(controls=(actor,)),
                        Commitment("STABILIZE", 0, max_loops=h_loops(
                            context, self.stabilize_max_h)),
                        {"hold_value": 0.1 if not risk_ok else 0.3},
                        snapshot_id, fingerprint,
                        instance_id=instance.instance_id if instance else None,
                        rationale="stabilize (bounded)"))
                    continue
            if phase == "REENTER":
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
                        Commitment("REENTER", 0, max_loops=h_loops(
                            context, self.reenter_max_h)),
                        {"progress": abs(move_progress(
                            context, actor, move,
                            context.snapshot.enemies[target].position))},
                        snapshot_id, fingerprint,
                        instance_id=instance.instance_id if instance else None,
                        rationale=f"re-enter toward {target}"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans
