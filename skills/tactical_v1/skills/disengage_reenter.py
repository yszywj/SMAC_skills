"""G4 DisengageReenter: phased, bounded local escape (C4).

ESCAPE -> STABILIZE -> REENTER with transitions that only fire on OBSERVED
conditions and are COMMITTED through unified transition deltas (phase_start
set once by memory). No propose-side mutation of instances; total budget
never resets, exhausted escapes terminate with a retry record.
"""
import math
from typing import Dict, List, Optional

from ..context import Context, STOP, distance
from ..contracts import Commitment, EffectClaim, HEURISTIC, ResourceClaims
from ..combat_exchange import (escape_urgency, exposure_window, imminent_exchange_danger,
                               next_fire_horizon, movement_exchange, safe_reentry_window,
                               danger_evidence)
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
        self.pressure_memory_enabled = config.get('common_combat', {}).get(
            'pressure_memory_enabled', True)
        self.reentry_window_enabled = config.get('common_combat', {}).get(
            'reentry_window_enabled', True)
        self.joint_enabled = config.get('common_combat', {}).get('joint_assessment_enabled', True)
        self.recovery_enabled = config.get('common_combat', {}).get('mechanism_recovery_enabled', True)

    def _recovery_reason(self, context, actor, remaining=None):
        from ..healing import potential_healing_services
        from ..recovery import recovery_opportunity
        services = potential_healing_services(context)
        if services.get(actor):
            return 'allocated_healing_required'
        if self.joint_enabled:
            # Passive regeneration also occurs while moving/firing and is
            # valued by the final evaluator. It is not a reason to STOP.
            return None
        if self.recovery_enabled and recovery_opportunity(
                context, actor, horizon_loops=remaining)['gain'] > 0.:
            return 'observed_passive_recovery'
        return None

    def _safe_window(self, context, actor):
        """Use the same visible next-output window as escape/re-entry routes."""
        return safe_reentry_window(context, actor,
            pressure_memory_enabled=self.pressure_memory_enabled)

    def _forecast_pressure(self, context, actor, position, certain=True):
        """Same per-enemy exposure and volley normalization as Evaluator.

        log/health factors are constant monotone factors for one actor, so
        pressure preserves its ordering without another full-plan scoring.
        """
        params = self.config.get("forecasting", {})
        discount = params.get("forecast_reach_discount", .5)
        floor = params.get("uncertain_position_floor", .35)
        risk = context.forecast_risk(actor, position)
        current = set(risk.current_threats)
        exposure = {e: 1. if e in current else discount for e in risk.reachable_threats}
        if not certain:
            origin = context.forecast_risk(actor, context.snapshot.allies[actor].position)
            for enemy in origin.reachable_threats:
                amount = floor * (1. if enemy in origin.current_threats else discount)
                exposure[enemy] = max(exposure.get(enemy, 0.), amount)
        total = 0.
        for enemy, fraction in exposure.items():
            damage = context.enemy_attack_damage(enemy, actor)
            ability = max(.35, min(4., damage / 14.)) if damage is not None else 1.
            total += fraction * ability
        window = exposure_window(context, actor, position,
            horizon_h=next_fire_horizon(context, actor, position), certain=certain,
            forecast_config=params, pressure_memory_enabled=self.pressure_memory_enabled)
        return max(total, window["amount"] / 14.)

    def _entry(self, context: Context, actor: str) -> bool:
        unit = context.snapshot.allies[actor]
        capability = context.capabilities[actor]
        if capability.suicide_attack or capability.healer:
            return False            # S3 / S2 replace survival value
        if self.joint_enabled:
            return self._escape_evidence(context, actor)['urgent'] \
                and not self._protect_shot(context, actor)
        risk = context.risk_at(actor, unit.position)
        pincer = len(risk.threats) >= self.risk_threshold
        severe = self._severe(context, actor)
        if self._protect_shot(context, actor):
            return False
        shielded = unit.shield > 0 and unit.shield_lost > 0 \
            and unit.health_fraction > self.critical
        return severe or ((pincer or (risk.effect_weapon_near
                                      and unit.health_fraction < 0.7))
                          and not shielded)

    def _reentry_objective(self, context, actor, target):
        """Center-distance progress to both native contact and command reach.

        The evaluator's band is measured center-to-center. In particular a
        melee unit has not rejoined combat merely by reaching distance five.
        Unknown native/command reach is not replaced by an invented range.
        """
        relation = context.snapshot.relations.get(actor, {}).get(target)
        if relation is None or relation.native_weapon_range is None \
                or relation.weapon_range_status == "no_compatible_weapon":
            return None
        unit, enemy = context.snapshot.allies[actor], context.snapshot.enemies[target]
        raw = context.snapshot.raw_state
        limit = raw.get("relations", {}).get(actor, {}).get(target, {}).get(
            "smac_attack_distance_limit")
        if limit is None:
            limit = raw.get("action_contract", {}).get(
                "candidate_center_distance_by_type", {}).get(unit.type_name)
        if not isinstance(limit, (int, float)) or not math.isfinite(limit):
            if not context.can_issue_attack(actor, target):
                return None
            limit = distance(unit.position, enemy.position)
        reach = min(float(limit), relation.native_weapon_range + unit.radius + enemy.radius)
        if not math.isfinite(reach) or reach < 0:
            return None
        return {"kind": "range_band", "target": target, "r_min": 0., "r_max": reach}

    def _escape_evidence(self, context, actor):
        return escape_urgency(context, actor, critical_fraction=self.critical,
            min_threats=self.risk_threshold,
            pressure_memory_enabled=self.pressure_memory_enabled)

    def _severe(self, context: Context, actor: str) -> bool:
        unit = context.snapshot.allies[actor]
        if self.joint_enabled:
            return self._escape_evidence(context, actor)['immediate']
        return imminent_exchange_danger(context, actor,
                pressure_memory_enabled=self.pressure_memory_enabled) or (unit.health_fraction <= self.critical
                and (unit.health_lost + unit.shield_lost > 0
                     or unit.health_fraction <= self.critical / 2))

    def _protect_shot(self, context: Context, actor: str) -> bool:
        """Multiple nearby enemies alone must not cancel an engaged volley."""
        unit = context.snapshot.allies[actor]
        if self._severe(context, actor) or context.risk_at(
                actor, unit.position).effect_weapon_near:
            return False
        return any(context.fire_estimate(actor, target).window_shots >= 1
                   for target in context.attackable_enemies(actor))

    def _phase(self, context: Context, actor: str, instance) -> Optional[str]:
        """Observed-condition phase machine: never an unconditional step to
        STABILIZE merely because an escape move was selected before."""
        loop = context.snapshot.game_loop
        total_max = instance.max_loops if instance.max_loops is not None \
            else h_loops(context, self.total_max_h)
        if loop - instance.start_loop >= total_max:
            return None             # total timeout: terminate, do not restart
        unit = context.snapshot.allies[actor]
        risk = context.risk_at(actor, unit.position)
        healers = any(cap.healer and context.snapshot.allies[a].alive
                      for a, cap in context.capabilities.items())
        if instance.phase == ESCAPE:
            phase_max = instance.extra.get("phase_max_loops")
            if phase_max is None:
                phase_max = h_loops(context, self.escape_max_h)
            in_escape_long = loop - instance.phase_start_loop >= phase_max
            if in_escape_long:
                return None        # observation records failure/safe completion
            safe = (self._safe_window(context, actor) if self.reentry_window_enabled
                    else not risk.threats and not risk.effect_weapon_near)
            if self.joint_enabled and not self._escape_evidence(context, actor)['immediate']:
                # Remaining inside somebody's possible range is not evidence
                # that a necessary escape is still in progress. Stop spending
                # the entire phase on avoidance after observed pressure eased.
                return STABILIZE if safe and self._recovery_reason(context, actor) else REENTER
            if safe:
                # risk acceptable: stabilize only when waiting really pays
                # (a NEARBY SERVICEABLE healer, not merely "a medivac exists")
                # and health is still low; else REENTER now (C4)
                if self._recovery_reason(context, actor) is not None:
                    return STABILIZE
                return REENTER
            return ESCAPE
        if instance.phase == STABILIZE:
            if (not self._safe_window(context, actor) if self.reentry_window_enabled
                    else risk.threats or risk.effect_weapon_near):
                if self.joint_enabled and not self._escape_evidence(context, actor)['immediate']:
                    return REENTER
                return ESCAPE        # observed danger returned, same deadline
            waited = loop - instance.phase_start_loop
            if waited >= h_loops(context, self.stabilize_max_h) \
                    or self._recovery_reason(context, actor, max(0,
                        h_loops(context, self.stabilize_max_h) - waited)) is None:
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
            if self.joint_enabled:
                from ..action_duties import rescue_actions
                rescue = rescue_actions(context, actor, memory)
                if rescue:
                    # Current death evidence must not wait for a failed
                    # task's cooldown or an approach/return-fire commitment.
                    telemetry.count(self.name, 'applicable')
                    plans.extend(self._rescue_plans(context, actor, memory, rescue))
                    continue
            if getattr(memory, "return_fire_pending", lambda *_: False)(context, actor) \
                    and not self._severe(context, actor) \
                    and not memory.combat_emergency(context, actor):
                continue
            if self._protect_shot(context, actor):
                continue            # G1 keeps the volley, including after escape
            cycle = memory.combat_cycle(context, actor)
            if memory.cycle_holds(context, actor, "approach") \
                    and not memory.combat_emergency(context, actor) \
                    and not self._severe(context, actor) \
                    and memory.one_step_attack_target(context, actor, cycle["target"]):
                continue            # cross the command-mask edge once; no ping-pong
            if memory.retry_blocked(
                    *memory.task_retry_key(self.name, [actor], "internal",
                                           "safe"), loop):
                continue
            instance = memory.instance_for(self.name, actor)
            active = instance is not None and instance.termination_reason is None
            evidence = self._escape_evidence(context, actor) if self.joint_enabled else None
            emergency = self._severe(context, actor) or bool(context.risk_at(
                actor, unit.position).effect_weapon_near)
            recovery = getattr(memory, 'contact_recovery', lambda *_: None)(context, actor)
            contact_overdue = bool(recovery and recovery['exhausted'])
            exhausted = getattr(memory, "retreat_exhausted", lambda *_: False)(context, actor) \
                and not emergency
            exhausted = exhausted or (contact_overdue and not emergency)
            if not active:
                if not exhausted and not self._entry(context, actor):
                    continue
                telemetry.count(self.name, "applicable")
                phase = REENTER if exhausted and evidence is not None and not evidence['urgent'] else ESCAPE
            else:
                phase = self._phase(context, actor, instance)
                if phase is None:
                    continue        # total timeout handled by memory facts
            if exhausted:
                # Overdue contact must not manufacture a fresh escape duty.
                # Keep an escape alternative only while actual danger evidence
                # supports it; the final joint score still prices return risk.
                phase = ESCAPE if evidence is None or evidence['urgent'] else REENTER
            risk = context.risk_at(actor, unit.position)
            killable = next((e for e in context.attackable_enemies(actor)
                             if (context.fire_estimate(actor, e).damage_est or 0)
                             >= context.snapshot.enemies[e].effective_health),
                            None)
            if phase == ESCAPE:
                preferred = best_move_away(context, actor, list(risk.threats))
                options = []
                # Joint-mode survivable rescues were already emitted above.
                # Do not reintroduce the old "less damage but still lethal"
                # moves through the ordinary ESCAPE candidate path.
                allowed_rescues = set(rescue_actions(context, actor, memory)) if self.joint_enabled else None
                current_pressure = self._forecast_pressure(context, actor, unit.position)
                current_count = len(risk.threats) + len(risk.effect_weapon_near)
                for action in context.move_destinations(actor):
                    if allowed_rescues is not None and action not in allowed_rescues:
                        continue
                    prediction = context.predict_position(actor, action)
                    after = context.risk_at(actor, prediction.position)
                    pressure = self._forecast_pressure(context, actor, prediction.position,
                                                       prediction.certain)
                    count = len(after.threats) + len(after.effect_weapon_near)
                    if contact_overdue and not emergency:
                        # A new G4 instance cannot buy another long departure.
                        # Keep genuine relief, with an observable nearby return
                        # frontage, rather than an arbitrarily tiny risk drop.
                        if pressure >= current_pressure - max(.1, .15 * current_pressure):
                            continue
                        return_window = movement_exchange(context, actor, prediction.position,
                            targets=(recovery['target'],),
                            pressure_memory_enabled=self.pressure_memory_enabled)
                        if return_window['eta_h'] > 3.:
                            continue
                    # Keep several improving directions; static cardinal
                    # preselection must not hide a better pursuit-safe route.
                    if pressure < current_pressure - .01 or \
                            (count < current_count and pressure <= current_pressure + .01):
                        blockage = memory.movement_blockage(context, actor, action)
                        options.append((pressure + 2. * blockage,
                                        blockage, count, action != preferred, action))
                options.sort()
                for _, _, _, _, move in options[:3]:
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
                        memory_delta=self._delta(instance, ESCAPE, actor, context=context),
                        goal_kind="internal",
                        goal_data={"objective": {"kind": "escape"}},
                        rationale="escape: forecast improvement with observed route reliability"))
                if not options and killable is not None:
                    action = context.attack_action(actor, killable)
                    plans.append(make_plan(
                        self.name, {actor: action}, killable, {actor: action},
                        [EffectClaim("damage", killable, None, max(context.H, 1),
                                     HEURISTIC, "stand_and_finish_kill")],
                        ResourceClaims(controls=(actor,)),
                        Commitment(ESCAPE, 0, max_loops=total_budget),
                        {}, snapshot_id, fingerprint,
                        instance_id=instance.instance_id if instance else None,
                        memory_delta=self._delta(instance, ESCAPE, actor, context=context),
                        rationale="no safe improvement; finish the kill"))
                # No improving escape: let ordinary combat compete, rather
                # than adding arbitrary movement or a rewarded stop action.
                if not exhausted:
                    continue
                phase = REENTER
            if phase == STABILIZE:
                plans.append(make_plan(
                    self.name, {actor: STOP}, "stabilize", {actor: STOP},
                    [EffectClaim("position", actor, None, max(context.H, 1),
                                 HEURISTIC,
                                 "stabilize_wait_real_stop_command")],
                    ResourceClaims(controls=(actor,)),
                    Commitment(STABILIZE, 0, max_loops=h_loops(
                        context, self.stabilize_max_h)),
                    {}, snapshot_id, fingerprint,
                    instance_id=instance.instance_id if instance else None,
                    memory_delta=self._delta(instance, STABILIZE, actor, context=context),
                    goal_kind="internal",
                    goal_data={"recovery_reason": self._recovery_reason(context, actor)},
                    rationale="stabilize (bounded, conditions real)"))
                continue
            # Contact already restored: retain ordinary fire through cooldown.
            # Re-entry should never drive an engaged ranged unit into the enemy.
            if any(context.contact_status(actor, e) == "in_range"
                   for e in context.attackable_enemies(actor)):
                continue
            battle_target = (recovery['target'] if recovery else
                             getattr(memory, "battle_target", lambda *_: None)(context, actor))
            choices = ((target, self._reentry_objective(context, actor, target))
                       for target in sorted(context.snapshot.enemies,
                           key=lambda e: (e != battle_target, distance(unit.position,
                               context.snapshot.enemies[e].position), e)))
            target, objective = next(((target, objective) for target, objective in choices
                                      if objective is not None), (None, None))
            if target is None:
                continue
            move = best_move_toward(context, actor, context.snapshot.enemies[target].position)
            moves = [move] if move is not None else []
            if self.reentry_window_enabled:
                # A side route may reach the same firing frontage without
                # undoing the escape. Do not let one cardinal preselection
                # hide that route; compare only progressing, viable moves.
                moves = []
                preferred_return = set(recovery.get('preferred_actions', ())) if recovery else set()
                for action in context.move_destinations(actor):
                    point = context.predict_position(actor, action).position
                    facts = movement_exchange(context, actor, point, targets=(target,),
                        pressure_memory_enabled=self.pressure_memory_enabled)
                    if not facts['closing'] or (not self.joint_enabled and not facts['admissible']):
                        continue
                    if self.joint_enabled:
                        danger = danger_evidence(context, actor, point, horizon_h=facts['eta_h'],
                            pressure_memory_enabled=self.pressure_memory_enabled)
                        if danger['effect_danger'] or danger['immediate'] and not facts['relief']:
                            continue
                    if not self.joint_enabled and not getattr(memory, 'reentry_allowed', lambda *_: True)(
                            context, actor, point, (target,)):
                        continue
                    if memory.movement_blockage(context, actor, action) >= .5:
                        continue
                    moves.append((bool(preferred_return) and action not in preferred_return,
                                  facts['eta_h'], memory.movement_conflict(context, actor, action),
                                  facts['after']['reserve_ratio'], action))
                moves = [row[-1] for row in sorted(moves)[:2]]
            for move in moves:
                prediction = context.predict_position(actor, move)
                if not self.reentry_window_enabled and len(context.risk_at(actor, prediction.position).threats) \
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
                    {}, snapshot_id, fingerprint,
                    instance_id=instance.instance_id if instance else None,
                    memory_delta=self._delta(instance, REENTER, actor, goal=target,
                                             context=context),
                    goal_kind="enemy",
                    goal_data={"target_id": target,
                               "objective": objective},
                    rationale=f"re-enter toward {target}"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans

    def _rescue_plans(self, context, actor, memory, moves):
        """Evidence-backed relief survives ordinary task locks and retries."""
        instance = memory.instance_for(self.name, actor)
        if instance is not None:
            total = instance.max_loops if instance.max_loops is not None else h_loops(context, self.total_max_h)
            if instance.termination_reason is not None or context.snapshot.game_loop - instance.start_loop >= total:
                instance = None
        result = []
        for move in moves:
            # With no live instance this is a one-window safety action,
            # not a fresh twelve-window escape that renews a failed duty.
            result.append(make_plan(
                self.name, (actor,), 'safe', {actor: move}, [],
                ResourceClaims(controls=(actor,), space=(space_claim(context, actor, move),)),
                Commitment(ESCAPE, 0, max_loops=context.H), {},
                context.snapshot.snapshot_id, self.config.get('_fingerprint', ''),
                instance_id=instance.instance_id if instance else None,
                memory_delta=self._delta(instance, ESCAPE, actor, context=context) if instance else None,
                goal_kind='internal', goal_data={'objective': {'kind': 'escape'},
                    'execution_reason': 'lethal_pressure_with_survivable_route',
                    'one_window_rescue': True},
                rationale='execute observed emergency relief before ordinary firing commitments'))
        return result

    def _delta(self, instance, phase, actor, goal=None, context=None):
        # H0-based budget: converted from the control window at task creation
        # (C4) — never a hardcoded loop constant.
        total_budget = int((context.H if context else 8) * self.total_max_h)
        phase_h = {ESCAPE: self.escape_max_h,
                   REENTER: self.reenter_max_h}.get(phase)
        extra = {"phase_reason": "observed_conditions",
                 "phase_max_loops": int((context.H if context else 8) * phase_h)
                 if phase_h is not None else None}
        if instance is None:
            return {"kind": "create", "skill": self.name, "actors": [actor],
                    "goal": goal or "safe", "goal_kind": "enemy" if goal else "internal",
                    "phase": phase, "max_loops": total_budget,
                    "extra": extra}
        return {"kind": "continue" if instance.phase == phase else "transition",
                "instance_id": instance.instance_id,
                "phase": phase, "goal": goal,
                "goal_kind": "enemy" if goal else None,
                "extra": {} if instance.phase == phase else extra}
