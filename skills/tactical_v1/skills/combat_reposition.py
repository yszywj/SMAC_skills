"""G3 CombatReposition: retreat, sidestep, pursuit and hold IN the cycle (D3).

Candidates are classified by GEOMETRY against the target/threat directions
(pursuit = closing on the target, retreat = growing min-threat distance,
sidestep = neither), so each class is genuinely non-empty. The lost window
shot is already priced by the physical scoring (attack action earns its
damage, the move does not); it is NOT charged a second time as switch cost.
"""
from typing import Dict, List, Optional

from ..context import Context, distance, GAME_LOOPS_PER_SPEED_SECOND
from ..contracts import Commitment, EffectClaim, HEURISTIC, ResourceClaims
from ..targeting import next_shot_target, planning_enabled, shared_target_plan
from .common import h_loops, make_plan, move_progress, space_claim


class CombatRepositionSkill:
    name = "combat_reposition"

    def __init__(self, config: Dict, providers: Optional[Dict] = None):
        self.config = config
        self.providers = providers or {}
        self.max_candidates = config.get("search", {}).get(
            "max_candidates_per_unit", 6)
        self.joint_assessment = config.get("common_combat", {}).get(
            "joint_assessment_enabled", True)

    def _classify(self, context: Context, actor: str, action: int,
                  target_id: str, threats) -> str:
        unit = context.snapshot.allies[actor]
        enemy = context.snapshot.enemies[target_id]
        destination = context.move_destinations(actor).get(action)
        if destination is None:
            return "hold"
        to_target = distance(unit.position, enemy.position) - \
            distance(destination, enemy.position)
        if to_target > 0.15:
            return "pursuit"
        threat_gains = [distance(context.snapshot.enemies[e].position,
                                 destination) -
                        distance(context.snapshot.enemies[e].position,
                                 unit.position) for e in threats]
        if threat_gains and min(threat_gains) > 0.15:
            return "retreat"
        return "sidestep"

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        if not context.snapshot.enemies:
            return plans
        target_plan = shared_target_plan(context, memory, self.config)
        for actor, unit in sorted(context.snapshot.allies.items()):
            if not unit.alive:
                continue
            capability = context.capabilities[actor]
            if capability.healer or capability.suicide_attack:
                continue
            attackable = [e for e in context.attackable_enemies(actor)
                          if not memory.retry_blocked(*memory.task_retry_key(
                              self.name, [actor], "enemy", e),
                              context.snapshot.game_loop)]
            if not attackable:
                continue                      # G2 owns out-of-contact units
            telemetry.count(self.name, "applicable")
            original = (next_shot_target(context, memory, actor, attackable)
                        if planning_enabled(self.config) else
                        memory.recent_attack_target(context, actor, attackable))
            assignment = target_plan.assignments.get(actor)
            assigned = assignment.target if assignment is not None else None
            if assigned and memory.retry_blocked(*memory.task_retry_key(
                    self.name, [actor], "enemy", assigned), context.snapshot.game_loop):
                assigned = None
            target_id = original or assigned or attackable[0]
            shot_available = any(context.fire_estimate(actor, target).window_shots >= 1
                                 for target in attackable)
            risk_now = context.risk_at(actor, unit.position)
            under_pressure = (unit.health_lost + unit.shield_lost > 0
                              and risk_now.threats) or risk_now.effect_weapon_near
            cooldown_window = unit.weapon_cooldown_loops > context.H
            melee = capability.combat_class == "melee"
            # Full-health ranged may still reposition inside a valuable
            # cooldown window; melee only when contact can be kept. Injury
            # alone never opens another reposition window for a zergling.
            worth_moving = cooldown_window or not shot_available or \
                (under_pressure and not melee) or \
                (under_pressure and melee and capability.key != 'zergling'
                 and unit.health_fraction < 0.5)
            if not worth_moving:
                continue
            # one candidate per class, classified by geometry (D3)
            by_class: Dict[str, int] = {}
            from ..deployment import unsupported_cooldown_advance
            def cooldown_blocked(action):
                return cooldown_window and unsupported_cooldown_advance(
                    context, actor,
                    context.predict_position(actor, action).position)["blocked"]
            for action in sorted(context.move_destinations(actor)):
                label = self._classify(context, actor, action, target_id,
                                       risk_now.threats)
                # During cooldown, a pursuit step that adds ordinary enemy
                # weapon coverage without teammate window-fire support is
                # not offered at all — the joint scorer's fallback must not
                # be able to re-select the same lone advance either.
                if label == "pursuit" and cooldown_blocked(action):
                    continue
                if label not in by_class:
                    by_class[label] = action
            # keep one candidate per class, but never fewer than the plain
            # four directions for the unified evaluator to compare (2.3)
            seen_actions = set()
            ordered = sorted(by_class.items())
            for label, move in ordered:
                seen_actions.add(move)
            for action in sorted(context.move_destinations(actor)):
                if action not in seen_actions and len(ordered) < \
                        self.max_candidates:
                    if cooldown_blocked(action):
                        continue
                    ordered.append((self._classify(context, actor, action,
                                                   target_id, risk_now.threats), action))
                    seen_actions.add(action)
            candidates = [(target_id, label, move)
                          for label, move in ordered[:self.max_candidates]]
            # Preserve the original four directions. Only spare quota goes
            # to one allocated alternative, and only where a next shot is
            # geometrically possible within the same short firing horizon.
            if assigned and assigned != target_id and len(candidates) < self.max_candidates:
                alternatives = []
                for move in sorted(context.move_destinations(actor)):
                    point = context.predict_position(actor, move).position
                    opportunity = context.projected_attack_opportunity(
                        actor, assigned, point, horizon=target_plan.horizon_loops)
                    if opportunity.available and opportunity.ready:
                        alternatives.append((
                            len(context.forecast_risk(actor, point).reachable_threats),
                            distance(point, context.snapshot.enemies[assigned].position), move))
                for _, _, move in sorted(alternatives)[:self.max_candidates - len(candidates)]:
                    candidates.append((assigned, self._classify(
                        context, actor, move, assigned, risk_now.threats), move))
            provider = self.providers.get("splash_fire")
            budget = h_loops(context, 8.0)
            for target_id, label, move in candidates:
                enemy = context.snapshot.enemies[target_id]
                relation = context.snapshot.relations[actor][target_id]
                r_max = relation.native_weapon_range or 5.0
                ordinary_reach = self.joint_assessment and capability.specialization is None
                if ordinary_reach:
                    r_max = context.effective_attack_reach(actor, target_id)
                    if r_max is None or r_max < 0:
                        continue  # unknown reach does not invent a firing band
                speed = context.speed(actor)
                band_width = (min(1.5, speed * context.H / GAME_LOOPS_PER_SPEED_SECOND)
                              if ordinary_reach and speed is not None else 1.5)
                prediction = context.predict_position(actor, move)
                if melee and label != "pursuit":
                    after = distance(prediction.position, enemy.position)
                    relation = context.snapshot.relations[actor][target_id]
                    reach = r_max if ordinary_reach else \
                        (relation.native_weapon_range or 0.0) + unit.radius + enemy.radius
                    if after > reach:
                        continue      # leaving the contact face is a G4 matter
                instance = memory.instance_for(self.name, actor)
                extra = {"target_id": target_id}
                if planning_enabled(self.config):
                    # Queries never start/renew this timer. The selected G3
                    # plan carries its preference to G1 through existing
                    # task memory, with a fixed two-window expiry.
                    prior = (instance.extra if instance is not None
                             and instance.goal == target_id else {})
                    started = prior.get("next_shot_started_loop", context.snapshot.game_loop)
                    extra.update(next_shot_target=target_id,
                                 next_shot_started_loop=started,
                                 next_shot_until_loop=prior.get(
                                     "next_shot_until_loop", started + target_plan.horizon_loops))
                delta = None
                if instance is None:
                    delta = {"kind": "create", "skill": self.name,
                             "actors": [actor], "goal": target_id, "goal_kind": "enemy",
                             "phase": "repositioning", "max_loops": budget, "extra": extra}
                elif instance.goal != target_id or any(
                        instance.extra.get(k) != v for k, v in extra.items()):
                    delta = {"kind": "retarget", "instance_id": instance.instance_id,
                             "goal": target_id, "goal_kind": "enemy", "extra": extra}
                plans.append(make_plan(
                    self.name, {actor: move}, target_id, {actor: move},
                    [EffectClaim("position", actor,
                                 move_progress(context, actor, move,
                                               enemy.position),
                                 max(context.H, 1), HEURISTIC,
                                 f"reposition_{label}")],
                    ResourceClaims(controls=(actor,),
                                   space=(space_claim(context, actor, move),)),
                    Commitment("repositioning", 0, max_loops=budget,
                               interrupt_on=("weapon_ready", "danger_change")),
                    # the lost window shot is priced by physics only
                    {"switch_cost": 0.0},
                    snapshot_id, fingerprint,
                    provider="splash_fire"
                    if capability.specialization == "splash_fire" and provider
                    else None,
                    instance_id=instance.instance_id if instance else None,
                    memory_delta=delta,
                    goal_kind="enemy",
                    goal_data={"target_id": target_id,
                               "allocated_target": assigned,
                               "objective": {
                                   "kind": "range_band", "target": target_id,
                                   "r_min": max(r_max - band_width, 0.0)
                                   if not melee else 0.0,
                                   "r_max": r_max if not melee or ordinary_reach else
                                        r_max + unit.radius + enemy.radius,
                                    "next_attack_target": target_id,
                                    "evaluate_splash": capability.specialization == "splash_fire"
                                        and provider is not None,
                                    "horizon_loops": min(2 * context.H,
                                        max(context.H, int(unit.weapon_cooldown_loops) + 1))}},
                    rationale=f"{label} during weapon cycle"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans
