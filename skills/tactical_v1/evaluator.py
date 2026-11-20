"""Unified full-plan evaluation (A3/B1-B6).

score_joint reads the FINAL action list, rebuilds physical effects, and
scores them on shared normalized scales. Per-source damage aggregation: a
cooling or unknown source never cancels another source's known
contribution. Tactical opportunity (progress toward a declared goal,
support links, meetings, role positions) is recomputed here from final
predicted positions — never added up from self-reported features.
"""
import math
from typing import Dict, List, Optional, Tuple

from .context import Context, distance
from .targeting import DAMAGE_SCALE, avoided_output
from .contracts import CandidatePlan
from .effects import EffectsReport, physical_effects
from .ledger import Ledger

DEFAULT_WEIGHTS = {
    "useful_damage": 3.0,   # per target-fraction of effective health removed
    "kill": 2.0,            # once per target, conservative feasibility only
    "heal": 2.0,            # per bounded heal OPPORTUNITY (uncalibrated proxy)
    "coverage": 1.0,        # saturated opportunity proxy per enemy
    "risk": 0.8,            # threats able to reach the FINAL position
    "congestion": 0.2,      # predicted overlaps between same-class movers
    "switch": 0.3,          # abandoning a live instance task
    "reentry_cost": 0.2,
    "planned_loss": 1.5,    # planned in-contact self-consumption, once
    "hold_value": 0.3,      # bounded wait/hold intent value
    # bounded TACTICAL opportunity terms (B1-B5; engineering starts, not
    # calibrated game facts)
    "tactic_progress": 0.6,  # signed approach-to-goal progress per task
    "link": 0.8,             # support-link closure (G5), once per pair key
    "meeting": 0.8,          # FUTURE heal opportunity from closing a pair
    "role": 0.6,             # role-position quality change (G6)
    "support_break": 0.6,   # lost fraction of existing local support, per unit
    "attack_continuity": 0.4,  # interruption of a ready, established attack
    "rally_wait_cost": 0.05,  # unresolved group time, per decision window
    "future_attack": 0.35,   # prospective attack opportunity, never damage/kill credit
    "target_threat": 0.25,   # bounded reduction of an enemy's objective threat
    "combat_idle": 0.35,     # healthy ready combatant waiting outside effective contact
    "movement_continuity": 0.8,
    "blocked_move": 0.6,
    "unsupported_frontline": 0.5,
    "sustained_contact": 0.8,  # bounded reduction in time to effective participation
    "ranged_fire_formation": 2.0,  # change in local next-volley fire balance
}

RISK_NORM = 3.0


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _clip(value: float) -> float:
    return max(min(float(value), 1.0), -1.0)


def _cap(value: Optional[float], limit: float = 1.0) -> float:
    if value is None:
        return 0.0
    return max(min(float(value), limit), -limit)


def reserve_risk_penalty(ratio: float) -> float:
    """Keep the existing ordinary-risk scale, without a flat lethal plateau.

    Up to twice the remaining reserve this is the original squared penalty.
    Beyond that point a smooth, increasing tail stays bounded below eight.
    Exposure is uncertain: this does not turn a lethal proxy into a command
    to flee, remove attacks, or claim that the least exposed route survives.
    """
    ratio = max(0., ratio)
    if ratio <= 2.:
        return ratio * ratio
    return 8. - 4. / (ratio - 1.)


class Evaluator:
    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 mechanisms: Optional[Dict[str, bool]] = None,
                 splash_radius: float = 3.0, splash_weight: float = 0.5,
                 burst_radius: float = 2.5, burst_weight: float = 0.5,
                 burst_self_cost: float = 1.0,
                 forecast_config: Optional[Dict[str, float]] = None,
                 urgency_value: float = 0.5,
                 target_planning_enabled: bool = True,
                 splash_reliability: float = 0.5,
                 common_combat: Optional[Dict[str, bool]] = None):
        self.weights = dict(DEFAULT_WEIGHTS)
        self.common_combat = dict(common_combat or {})
        if weights:
            self.weights.update(weights)
        self.mechanisms = mechanisms or {"splash": True, "burst": True}
        forecast = forecast_config or {}
        self.forecast_reach_discount = max(0., min(1., float(
            forecast.get("forecast_reach_discount", .5))))
        self.uncertain_position_floor = max(0., min(1., float(
            forecast.get("uncertain_position_floor", .35))))
        self.urgency_value = max(0., float(urgency_value))
        self.target_config = {"focus_fire": {"target_planning_enabled": target_planning_enabled}}
        self.splash_reliability = max(0., min(1., float(splash_reliability)))
        self.effect_params = dict(splash_radius=splash_radius,
                                  splash_weight=splash_weight,
                                  splash_reliability=self.splash_reliability,
                                  burst_radius=burst_radius,
                                  burst_weight=burst_weight,
                                  burst_self_cost=burst_self_cost)

    def effects(self, context: Context,
                actions: Dict[str, int]) -> EffectsReport:
        report = physical_effects(context, actions, self.mechanisms, **self.effect_params)
        if self.common_combat.get('joint_assessment_enabled', True):
            facts = self.joint_facts(context, actions)
            for actor, row in facts['actors'].items():
                if row['ordinary']:
                    report.positions[actor] = (row['position'], row['certain'])
        return report

    def joint_facts(self, context, actions):
        from .joint_combat import assess_joint
        return assess_joint(context, actions, pressure_memory_enabled=
                            self.common_combat.get('pressure_memory_enabled', True), forecast_config={
                                'forecast_reach_discount': self.forecast_reach_discount,
                                'uncertain_position_floor': self.uncertain_position_floor})

    # ---------------------------------------------------- B1-B5 tactical terms
    def _move_scale(self, context: Context, actor: str) -> float:
        """L: plausible distance inside one window; speed-derived when known,
        else a flagged conservative constant (no path simulation)."""
        speed = context.speed(actor)
        from .context import GAME_LOOPS_PER_SPEED_SECOND
        return max(speed * context.H / GAME_LOOPS_PER_SPEED_SECOND, 0.5) if speed else 1.0

    def _available_fire(self, context, actor):
        """A legal native-range shot in this window, independent of provider.

        Geometric progress must not buy postponement of this opportunity.
        Damage, survival and actual next-shot timing still decide the action;
        this is not a hard requirement to stand and fire through danger.
        """
        if not self.common_combat.get('joint_assessment_enabled', True):
            return False
        cache = getattr(context, '_geometric_ready_fire', None)
        if cache is None:
            cache = context._geometric_ready_fire = {}
        if actor not in cache:
            unit, cap = context.snapshot.allies.get(actor), context.capabilities.get(actor)
            cache[actor] = bool(unit and unit.alive and cap and not cap.healer
                and not cap.suicide_attack and any(
                    context.fire_estimate(actor, enemy).window_shots >= 1
                    for enemy in context.attackable_enemies(actor)))
        return cache[actor]

    def _geometry_credit(self, context, actor, gain):
        return min(gain, 0.) if self._available_fire(context, actor) else gain

    def task_quality(self, context: Context, actions: Dict[str, int],
                     plans: List[CandidatePlan],
                     report: EffectsReport) -> Tuple[float, Dict[str, float]]:
        """Bounded tactical opportunity recomputed from FINAL predicted
        positions and each task's declared OBJECTIVE (2.1). Unknown or
        incomplete objectives score a conservative zero with a stated reason
        — never a fallback to the enemy centre or the origin. Relations are
        deduplicated by an UNDIRECTED quality key (2.5)."""
        w = self.weights
        breakdown = {"progress": 0.0, "link": 0.0, "meeting": 0.0,
                     "role": 0.0, "rally_wait": 0.0}
        seen_tasks = set()
        link_relations = set()
        for plan in plans:
            if plan.task_key and plan.task_key in seen_tasks:
                continue
            seen_tasks.add(plan.task_key)
            objective = (plan.goal_data or {}).get("objective") or {}
            kind = objective.get("kind")
            if kind is None:
                kind = {"enemy": "target_object", "ally": "target_object",
                        "position": "waypoint", "group_link": "support_link",
                        "meeting": "meeting", "role": "role"}.get(
                            plan.goal_kind, "none")
            if kind in ("none", "target_object", "burst_region", "attempt"):
                if kind == "target_object":
                    goal = self._goal_position(context, plan)
                    if goal is not None:
                        breakdown["progress"] += \
                            w["tactic_progress"] * self._toward(
                                context, report, plan, goal)
                continue
            if kind == "contact_point":
                position = objective.get("position")
                if position:
                    breakdown["progress"] += w["tactic_progress"] * \
                        self._toward(context, report, plan, tuple(position))
                continue
            if kind == "waypoint":
                position = objective.get("position") or \
                    (plan.goal_data or {}).get("waypoint")
                if position:
                    breakdown["progress"] += w["tactic_progress"] * \
                        self._toward(context, report, plan, tuple(position))
                continue
            if kind == "range_band":
                breakdown["progress"] += w["tactic_progress"] * \
                    self._range_band_gain(context, report, plan, objective)
                continue
            if kind == "shared_fire":
                breakdown["progress"] += w["tactic_progress"] * \
                    self._shared_fire_gain(context, report, plan, objective)
                continue
            if kind == "counter_fire_support":
                breakdown["progress"] += w["tactic_progress"] * \
                    self._counter_fire_gain(context, actions, report, plan, objective)
                continue
            if kind == "escape":
                # The final-position risk term already prices this exact
                # reduction. Do not reward the same threat drop twice.
                continue
            if kind == "rally":
                gain, unresolved = self._rally_gain(context, report, plan)
                breakdown["link"] += w["link"] * gain
                breakdown["rally_wait"] -= w["rally_wait_cost"] * unresolved
                continue
            if kind == "combat_rejoin":
                gain, unresolved = self._combat_rejoin_gain(context, report, plan, actions)
                breakdown["link"] += w["link"] * gain
                breakdown["rally_wait"] -= w["rally_wait_cost"] * unresolved
                continue
            if kind == "spacing":
                continue    # congestion is priced once by the physics term
            if kind in ("battle_line", "ranged_fire_formation"):
                continue    # final-action shared firing facts, never a label bonus
            if kind == "support_link":
                key = self._link_key(plan)
                if key not in link_relations:
                    link_relations.add(key)
                    breakdown["link"] += w["link"] * self._link_gain(
                        context, actions, plan, report)
                continue
            if kind == "meeting":
                breakdown["meeting"] += w["meeting"] * self._meeting_gain(
                    context, actions, plan, report)
                continue
            if kind == "escort_band":
                breakdown["progress"] += w["tactic_progress"] * \
                    self._escort_gain(context, report, plan, objective)
                continue
            if kind in ("protected_position", "role"):
                breakdown["role"] += w["role"] * self._role_gain(
                    context, actions, plan, report)
                continue
            breakdown["unknown_objectives"] = \
                breakdown.get("unknown_objectives", 0.0) + 1.0
        total = (breakdown["progress"] + breakdown["link"]
                 + breakdown["meeting"] + breakdown["role"]
                 + breakdown["rally_wait"])
        return total, breakdown

    def _rally_gain(self, context, report, plan):
        """Reduction in windows until the required group fraction is ready.

        Already-ready members get no reward for repeatedly issuing Stop.
        The unresolved member fraction prices time until group readiness.
        """
        data = plan.goal_data or {}
        anchor = data.get("anchor")
        radius = float(data.get("ready_radius", 4.0))
        members = [a for a in data.get("members", plan.controlled_actors)
                   if a in context.snapshot.allies
                   and context.snapshot.allies[a].alive]
        if anchor is None or not members:
            return 0.0, 0.0
        before_windows, after_windows = [], []
        unresolved = 0
        for actor in members:
            x = context.snapshot.allies[actor].position
            z = report.positions.get(actor, (x, False))[0]
            before = max(distance(x, anchor) - radius, 0.0)
            after = max(distance(z, anchor) - radius, 0.0)
            if self._available_fire(context, actor):
                after = max(after, before)
            scale = self._move_scale(context, actor)
            before_windows.append(before / scale)
            after_windows.append(after / scale)
            unresolved += after > 0.0
        required = max(1, min(len(members), math.ceil(
            float(data.get("ready_fraction", 0.8)) * len(members))))
        critical_before = sorted(before_windows)[required - 1]
        critical_after = sorted(after_windows)[required - 1]
        # Shared readiness serves the whole participating group. A member
        # already in band cannot collect repeated holding/progress rewards.
        gain = len(members) * _clip(critical_before - critical_after)
        return gain, unresolved / len(members)

    def _combat_rejoin_gain(self, context, report, plan, actions=None):
        """Marginal progress of actual joiners, never multiplied by cover size.

        Geometry and time until effective support share one bounded credit
        per joiner. A receiver unlikely to survive until arrival cannot
        promise the whole benefit. Current damage remains in its own ledger.
        """
        data = plan.goal_data or {}
        units = context.snapshot.allies
        controlled = set(plan.controlled_actors)
        joiners = [a for a in data.get("joiners", ()) if a in controlled
                   and a in units and units[a].alive]
        supporters = [a for a in data.get("supporters", ()) if a in controlled
                      and a in units and units[a].alive and a not in joiners]
        if not joiners or not supporters:
            return 0., 0.
        radius = float(data.get("support_radius", 6.))
        from .combat_groups import effective_support
        positions = {a: p[0] for a, p in report.positions.items()}
        ready_before = effective_support(context, joiners, supporters, radius=radius,
                                         targets_by_actor=data.get('targets_by_actor'),
                                         roles=data.get('roles') if data.get('local_cover_enabled') else None)
        ready_after = effective_support(context, joiners, supporters, positions, radius=radius,
                                        targets_by_actor=data.get('targets_by_actor'),
                                        roles=data.get('roles') if data.get('local_cover_enabled') else None)
        joint = self.joint_facts(context, actions or plan.current_actions) if self.common_combat.get(
            'joint_assessment_enabled', True) else None
        gain = 0.
        for actor in joiners:
            now = min(distance(units[actor].position, units[b].position)
                      for b in supporters)
            point = report.positions.get(actor, (units[actor].position, False))[0]
            future = min(distance(point, report.positions.get(
                b, (units[b].position, False))[0]) for b in supporters)
            scale = self._move_scale(context, actor)
            geometry = _clip((max(now - radius, 0.) - max(future - radius, 0.)) / scale)
            eta_before = ready_before["eta_by_actor"].get(actor, float("inf"))
            eta_after = ready_after["eta_by_actor"].get(actor, float("inf"))
            arrival = (_clip(eta_before - eta_after)
                       if math.isfinite(eta_before) and math.isfinite(eta_after) else 0.)
            if data.get('local_cover_enabled') and data.get('roles', {}).get(actor) == 'escape':
                # An escape joins safe support, not the enemy's weapon band.
                # The two halves share this one bounded geometric credit.
                arrival = geometry
            if geometry <= 0. and future > radius:
                # Both squads advancing in parallel without closing their
                # gap is not joining progress; future fire has its own term.
                arrival = min(arrival, 0.)
            contribution = .5 * geometry + .5 * arrival
            contribution = self._geometry_credit(context, actor, contribution)
            if joint is not None and contribution > 0.:
                relevant = set(data.get('targets_by_actor', {}).get(actor, ()))
                relevant.update(context.forecast_risk(actor, point).reachable_threats)
                relevant.update(context.risk_at(actor, units[actor].position).threats)
                if not relevant:
                    compatible = [e for e in context.snapshot.enemies
                        if context.effective_attack_reach(actor, e) is not None
                        and context.effective_attack_reach(actor, e) >= 0.]
                    if compatible:
                        relevant.add(min(compatible, key=lambda e: distance(
                            point, context.snapshot.enemies[e].position)))
                timely = []
                for peer in supporters:
                    row = joint['actors'].get(peer)
                    if row is None or not row['ordinary']:
                        continue
                    if any(e in relevant and eta <= min(3., max(1., eta_after))
                           for e, eta in row['target_etas'].items()):
                        timely.append(row.get('team_exposure_after', row['exposure'])['viability'])
                # Per-joiner reserve: a healthy receiver fighting elsewhere
                # cannot lend its HP to the actual exposed receiver.
                contribution *= max(timely, default=0.) / max(
                    1., eta_after / 3.) if math.isfinite(eta_after) else 0.
            gain += contribution
        viability = min(ready_before["cover_viability"], ready_after["cover_viability"])
        if joint is not None:
            viability = 1.    # already priced each joiner's actual receiver
        gain = max(gain, 0.) * viability + min(gain, 0.)
        return gain, 1. - ready_after["fraction"]

    def _toward(self, context, report, plan, goal) -> float:
        """Signed closeness gain per controlled actor toward a POINT."""
        total = 0.0
        for actor in plan.controlled_actors:
            if actor not in context.snapshot.allies:
                continue
            x = context.snapshot.allies[actor].position
            z = report.positions.get(actor, (x, True))[0]
            gain = _clip((_dist(x, goal) - _dist(z, goal))
                         / self._move_scale(context, actor))
            total += self._geometry_credit(context, actor, gain)
        return _clip(total)

    def _shared_fire_gain(self, context, report, plan, objective):
        """Progress until two of the group's shooters can reach the same enemy."""
        enemy_id = objective.get("target")
        enemy = context.snapshot.enemies.get(enemy_id)
        if enemy is None:
            return 0.
        before, after = [], []
        for actor in set(objective.get("shooters", ())).intersection(plan.controlled_actors):
            unit = context.snapshot.allies[actor]
            reach = context.effective_attack_reach(actor, enemy_id)
            if not unit.alive or reach is None or reach < 0.:
                continue
            endpoint = report.positions.get(actor, (unit.position, False))[0]
            stride = self._move_scale(context, actor)
            start = max(0., distance(unit.position, enemy.position) - reach) / stride
            end = max(0., distance(endpoint, enemy.position) - reach) / stride
            before.append(start)
            after.append(max(start, end) if self._available_fire(context, actor) else end)
        if len(before) < 2:
            return 0.
        return _clip(sorted(before)[1] - sorted(after)[1])

    def _counter_fire_gain(self, context, actions, report, plan, objective):
        """Credit progress of the actors actually seeking a firing position."""
        pair = plan.goal_data.get("support_pair", ())
        enemy = objective.get("target")
        if not pair or enemy not in context.snapshot.enemies:
            return 0.
        actors = list(pair[1:])
        if not plan.goal_data.get("withdrawing"):
            actors.append(pair[0])
        total = 0.
        target = context.snapshot.enemies[enemy].position
        for actor in actors:
            if actor not in actions or context.decode(actor, actions[actor]).kind != "move":
                continue
            reach = context.effective_attack_reach(actor, enemy)
            if reach is None or reach < 0 or self._available_fire(context, actor):
                continue
            unit = context.snapshot.allies[actor]
            endpoint = report.positions.get(actor, (unit.position, False))[0]
            before = max(0., distance(unit.position, target) - reach)
            after = max(0., distance(endpoint, target) - reach)
            total += (before - after) / self._move_scale(context, actor)
        return _clip(total)

    def _range_band_gain(self, context, report, plan, objective) -> float:
        """2.3: band-error progress toward [r_min, r_max] around the target;
        inside the band neither closing nor retreating gains or loses."""
        target = context.snapshot.enemies.get(objective.get("target"))
        if target is None:
            return 0.0
        r_min = float(objective.get("r_min", 0.0))
        r_max = float(objective.get("r_max", r_min))
        total = 0.0
        for actor in plan.controlled_actors:
            if actor not in context.snapshot.allies:
                continue
            x = context.snapshot.allies[actor].position
            z = report.positions.get(actor, (x, True))[0]
            d_now = _dist(x, target.position)
            d_after = _dist(z, target.position)
            gain = _clip((self._band_error(d_now, r_min, r_max)
                          - self._band_error(d_after, r_min, r_max))
                         / self._move_scale(context, actor))
            total += self._geometry_credit(context, actor, gain)
        return _clip(total)

    @staticmethod
    def _band_error(d: float, r_min: float, r_max: float) -> float:
        if d < r_min:
            return r_min - d
        if d > r_max:
            return d - r_max
        return 0.0

    def _escape_gain(self, context, report, plan) -> float:
        """2.1 escape: risk reduction at the final position — never measured
        against the map origin."""
        total = 0.0
        for actor in plan.controlled_actors:
            unit = context.snapshot.allies.get(actor)
            if unit is None or not unit.alive:
                continue
            z = report.positions.get(actor, (unit.position, True))[0]
            risk_now = context.risk_at(actor, unit.position)
            risk_after = context.risk_at(actor, z)
            drop = (len(risk_now.threats) + len(risk_now.effect_weapon_near)
                    - len(risk_after.threats)
                    - len(risk_after.effect_weapon_near))
            total += _clip(drop / RISK_NORM)
        return _clip(total)

    def _escort_gain(self, context, report, plan, objective) -> float:
        """escort_band: distance band to the served squad reference."""
        position = objective.get("position")
        if not position:
            return 0.0
        r_min = float(objective.get("r_min", 1.0))
        r_max = float(objective.get("r_max", 5.0))
        total = 0.0
        for actor in plan.controlled_actors:
            unit = context.snapshot.allies.get(actor)
            if unit is None:
                continue
            z = report.positions.get(actor, (unit.position, True))[0]
            d_now = _dist(unit.position, tuple(position))
            d_after = _dist(z, tuple(position))
            gain = _clip((self._band_error(d_now, r_min, r_max)
                          - self._band_error(d_after, r_min, r_max))
                         / self._move_scale(context, actor))
            total += self._geometry_credit(context, actor, gain)
        return _clip(total)

    @staticmethod
    def _link_key(plan) -> tuple:
        """2.5: UNDIRECTED pair identity — A->B and B->A are one relation."""
        data = plan.goal_data or {}
        a = tuple(sorted(data.get("group_a", [])))
        b = tuple(sorted(data.get("group_b", [])))
        left, right = sorted((a, b))
        return ("support_link", left, right)

    def _goal_position(self, context: Context, plan: CandidatePlan):
        data = plan.goal_data or {}
        if plan.goal_kind == "enemy":
            enemy = context.snapshot.enemies.get(plan.goal)
            return enemy.position if enemy else None
        if plan.goal_kind == "ally":
            ally = context.snapshot.allies.get(plan.goal)
            return ally.position if ally else None
        position = data.get("position") or data.get("waypoint")
        return tuple(position) if position else None

    def _link_gain(self, context: Context, actions: Dict[str, int],
                   plan: CandidatePlan, report: EffectsReport) -> float:
        """2.5: support-gap potential with FIXED group membership; gain =
        current gap minus the plan's final gap (both directions averaged)."""
        data = plan.goal_data or {}
        group_a, group_b = data.get("group_a", []), data.get("group_b", [])
        radius = data.get("support_radius", 8.0)
        scale = data.get("scale", 4.0)
        units = context.snapshot.allies
        if not group_a or not group_b:
            return 0.0

        def gap_one_way(pa, pb):
            values = [max(0.0, min(_dist(a, b) for b in pb) - radius)
                      for a in pa]
            return sum(values) / len(values) if values else 0.0

        def phi(pa, pb):
            return 0.5 * (gap_one_way(pa, pb) + gap_one_way(pb, pa))

        current_a = [units[a].position for a in group_a if a in units]
        current_b = [units[b].position for b in group_b if b in units]
        # Existing shooters cannot supply a positive regroup reward merely
        # by leaving their firing position. Other members can still close.
        final_a = [units[a].position if self._available_fire(context, a)
                   else report.positions.get(a, (units[a].position, True))[0]
                   for a in group_a if a in units]
        final_b = [units[b].position if self._available_fire(context, b)
                   else report.positions.get(b, (units[b].position, True))[0]
                   for b in group_b if b in units]
        if not current_a or not current_b:
            return 0.0
        return _clip((phi(current_a, current_b) - phi(final_a, final_b))
                     / scale)

    def _meeting_gain(self, context: Context, actions: Dict[str, int],
                      plan: CandidatePlan, report: EffectsReport) -> float:
        """Bounded, energy-qualified rescue progress from final joint positions."""
        from .healing import healing_approach_gain
        return healing_approach_gain(context, actions, plan, report)

    def _role_gain(self, context: Context, actions: Dict[str, int],
                   plan: CandidatePlan, report: EffectsReport) -> float:
        """2.6: protected-position quality. Uses the core unit's OWN reliable
        range (own weapon data — never a hardcoded 6), the guards' FINAL
        predicted positions, and a front/back PROJECTION along the local
        enemy direction. The single front/back term is dropped when the
        direction is unreliable (multi-direction pressure)."""
        data = plan.goal_data or {}
        actor = data.get("actor")
        units = context.snapshot.allies
        if not actor or actor not in units:
            return 0.0
        unit = units[actor]
        z = report.positions.get(actor, (unit.position, True))[0]
        enemies = list(context.snapshot.enemies.values())
        guards = [g for g in (data.get("guards") or []) if g in units]
        if not enemies or not guards:
            return 0.0
        # local enemy direction + reliability via direction-vote coherence
        votes = []
        for enemy in enemies:
            dx = enemy.position[0] - z[0]
            dy = enemy.position[1] - z[1]
            length = math.hypot(dx, dy) or 1.0
            votes.append((dx / length, dy / length))
        sx = sum(v[0] for v in votes)
        sy = sum(v[1] for v in votes)
        resultant = math.hypot(sx, sy)
        reliable = bool(votes) and resultant / len(votes) >= 0.4
        n = (sx / resultant if resultant else 0.0,
             sy / resultant if resultant else 0.0)
        own_range = None
        for relation in context.snapshot.relations.get(actor, {}).values():
            if relation.native_weapon_range:
                own_range = max(own_range or 0.0,
                                relation.native_weapon_range)
        r_max = own_range or float(data.get("range", 6.0))
        guard_positions = [report.positions.get(
            g, (units[g].position, True))[0] for g in guards]

        def quality(position):
            range_error = min(abs(_dist(position, e.position) - r_max)
                              for e in enemies)
            support = min(_dist(position, g) for g in guard_positions)
            value = -(range_error + 0.3 * max(support - 4.0, 0.0))
            if reliable:
                # back_gap: projection of (guard - core) along the enemy
                # direction; positive means the core is behind its front
                guard_ref = min(guard_positions,
                                key=lambda g: _dist(g, position))
                back_gap = ((position[0] - guard_ref[0]) * -n[0]
                            + (position[1] - guard_ref[1]) * -n[1])
                value -= abs(max(back_gap, 0.0)
                             - float(data.get("back_gap_ideal", 1.5)))
            return value

        gain = _clip((quality(z) - quality(unit.position))
                     / max(float(data.get("scale", 6.0)), 1.0))
        return self._geometry_credit(context, actor, gain)

    def _support_reference(self, context):
        """Local directed support links, cached per immutable observation.

        Ranged units can support farther away than melee; healers only serve
        wounded ground allies. Combat links need a common compatible visible
        enemy, so an unrelated group across the map creates no cohesion duty.
        """
        cached = getattr(self, "_support_cache", None)
        if cached is not None and cached[0] is context:
            return cached[1]
        units = {a: u for a, u in context.snapshot.allies.items() if u.alive}
        reachable = {}
        ranges = {}
        for actor, unit in units.items():
            targets = set()
            own_ranges = []
            for enemy_id, enemy in context.snapshot.enemies.items():
                relation = context.snapshot.relations.get(actor, {}).get(enemy_id)
                if relation is None or relation.weapon_range_status == "no_compatible_weapon":
                    continue
                if relation.native_weapon_range is not None:
                    own_ranges.append(relation.native_weapon_range)
                if distance(unit.position, enemy.position) <= 14.0:
                    targets.add(enemy_id)
            reachable[actor] = targets
            ranges[actor] = max(own_ranges, default=2.0)
        links = {}
        for actor, unit in units.items():
            peers = []
            for peer, other in units.items():
                if actor == peer:
                    continue
                cap = context.capabilities[peer]
                if cap.suicide_attack:
                    continue
                if cap.healer:
                    if unit.is_flying or unit.health >= unit.max_health:
                        continue
                    radius = 4.0
                else:
                    if not (reachable[actor] & reachable[peer]):
                        continue
                    radius = max(3.0, min(8.0, ranges[peer] + other.radius))
                strength = max(0.0, min(1.0,
                    (radius + 1.0 - distance(unit.position, other.position)) / 2.0))
                if strength > 0:
                    peers.append((peer, radius, strength))
            links[actor] = peers
        self._support_cache = (context, links)
        return links

    def _final_next_fire_cost(self, context, actor, action, actions, report, memory):
        """One final-branch target set shared by continuity and progress terms."""
        from .combat_exchange import (next_fire_timing, escape_urgency,
                                      attack_requires_approach, attack_approach_relief)
        from .fire_opportunities import final_action_budget
        contacts = any(context.contact_status(actor, e) == 'in_range'
                       for e in context.attackable_enemies(actor))
        duty = getattr(getattr(memory, 'combat_progress', None),
                       'output_duty', lambda *_: None)(context, actor)
        if not contacts and not duty:
            return 0.
        timing = next_fire_timing(context, actor, action, allow_outside=True)
        before, after = timing['before_eta_by_target'], timing['after_eta_by_target']
        earliest = {e: min(before.get(e, float('inf')), after.get(e, float('inf')))
                    for e in set(before) | set(after)}
        eligible = tuple(final_action_budget(context, actions).future_targets(actor, earliest)['primary'])
        if memory is not None and hasattr(memory, 'next_fire_cost'):
            return memory.next_fire_cost(context, actor, action, eligible_targets=eligible)
        pressure_enabled = self.common_combat.get('pressure_memory_enabled', True)
        if escape_urgency(context, actor, pressure_memory_enabled=pressure_enabled)['immediate']:
            if not attack_requires_approach(context, actor, action) or attack_approach_relief(
                    context, actor, action, pressure_memory_enabled=pressure_enabled):
                return 0.
        unit = context.snapshot.allies[actor]
        return next_fire_timing(context, actor, action,
            position=report.positions.get(actor, (unit.position, False))[0],
            eligible_targets=eligible, allow_outside=True)['cost']

    def _coordination_costs(self, context, actions, report, memory, plans):
        """Costs reconstructed from final actions, independent of skill labels."""
        units = context.snapshot.allies
        support_loss = continuity = reentry = commitment = 0.0
        idle = reversal = blockage = return_fire = fire_delay = 0.
        joint_positions = {a: p[0] for a, p in report.positions.items()}
        links = {} if self.common_combat.get('joint_assessment_enabled', True) \
            else self._support_reference(context)
        for actor, peers in links.items():
            x = report.positions.get(actor, (units[actor].position, False))[0]
            # No cost for leaving an empty region together. Retaining an
            # exposed member while its local cover withdraws is the concern.
            if not context.risk_at(actor, x).threats:
                continue
            before = sum(strength for _, _, strength in peers)
            after = sum(max(0.0, min(1.0, (radius + 1.0 - distance(
                x, report.positions.get(peer, (units[peer].position, False))[0])) / 2.0))
                        for peer, radius, _ in peers)
            support_loss += max(before - after, 0.0) / max(before, 1.0)
        if self.common_combat.get('joint_assessment_enabled', True):
            from .local_entry import assess_local_entry
            joint = self.joint_facts(context, actions)
            support_loss = sum(row.get('support_loss', 0.) for row in joint['actors'].values())
            entry = assess_local_entry(context, actions, joint, plans, memory)
            # Waiting is judged by final commands, even for baseline STOP.
            # Charge invalid waits once per actor; averaging them by frontage
            # would make a whole idle squad cheaper than independent idleness.
            idle += len(entry['invalid_wait_actors'])
        else:
            entry = {'cost': 0., 'waits': {}}
        previous = getattr(memory, "last_committed_actions", {}) if memory else {}
        for actor, action in actions.items():
            unit = units.get(actor)
            if unit is None or not unit.alive:
                continue
            capability = context.capabilities[actor]
            if capability.healer or capability.suicide_attack:
                continue
            if memory is not None:
                reversal += memory.movement_conflict(context, actor, action)
                blockage += memory.movement_blockage(
                    context, actor, action, joint_actions=actions,
                    joint_positions=joint_positions)
            # Native STOP may auto-fire. Once the team is fighting, a healthy
            # rear unit can also be idle several moves away from useful range.
            if actor in entry['waits']:
                continue  # valid short wait exempt; invalid wait priced above
            if context.decode(actor, action).kind != "stop" \
                    or unit.health_fraction < .5 or unit.weapon_cooldown_loops > context.H:
                continue
            if any(context.fire_estimate(actor, e).window_shots >= 1
                   for e in context.attackable_enemies(actor)):
                continue
            cache = getattr(context, "_idle_opportunity_cache", None)
            if cache is None:
                cache = context._idle_opportunity_cache = {}
            battle = self._engagement_state(context, memory)
            idle_key = (actor, bool(battle.get("engaged")), battle.get("phase"),
                        self._participation_targets(context, actor, battle, memory))
            if idle_key not in cache:
                immediate = any(context.projected_attack_opportunity(
                    actor, target, context.predict_position(actor, move).position,
                    horizon=2 * context.H).available
                    for move in context.move_destinations(actor)
                    for target in context.snapshot.enemies)
                productive = self._productive_advance(context, actor, memory)
                cache[idle_key] = (immediate or productive, productive)
            idle_now, productive = cache[idle_key]
            # Observed damage stagnation adds bounded urgency, not a new
            # estimate of safety. Only an actually feasible advance qualifies.
            idle += bool(idle_now) * (1.25 if battle.get("phase") == "press"
                                     and productive else 1.)
        for actor, action in actions.items():
            unit = units.get(actor)
            if unit is None or not unit.alive:
                continue
            from .combat_exchange import attack_requires_approach
            joint_budget = self.common_combat.get('joint_assessment_enabled', True)
            approach = (joint_budget and self.common_combat.get('next_fire_budget_enabled', True)
                        and attack_requires_approach(context, actor, action))
            if context.decode(actor, action).kind != "move" and not approach:
                continue
            if context.capabilities[actor].healer or context.capabilities[actor].suicide_attack:
                continue
            from .combat_exchange import imminent_exchange_danger, escape_urgency
            if joint_budget:
                # Low HP plus a tiny chip is not a separate path around the
                # shared next-shot obligation. Use the same survival evidence
                # as the cross-skill output memory and G4.
                critical = getattr(getattr(memory, 'combat_progress', None), 'critical_fraction', .35)
                emergency = escape_urgency(context, actor, critical_fraction=critical,
                    pressure_memory_enabled=self.common_combat.get('pressure_memory_enabled', True))['immediate']
            else:
                emergency = imminent_exchange_danger(context, actor,
                    pressure_memory_enabled=self.common_combat.get('pressure_memory_enabled', True)) or (
                    unit.health_fraction <= 0.25 and unit.health_lost > 0)
            if emergency and not approach:
                continue
            contacts = [e for e in context.attackable_enemies(actor)
                        if context.contact_status(actor, e) == "in_range"]
            if not contacts and not (self.common_combat.get('joint_assessment_enabled', True)
                    and memory is not None and getattr(getattr(memory, 'combat_progress', None),
                        'output_duty', lambda *_: None)(context, actor)):
                continue
            orders = context.snapshot.raw_state.get("allies", {}).get(actor, {}).get("orders", [])
            established = {o.get("target_unit") for o in orders if o.get("kind") == "attack"}
            if actor in previous:
                old = context.decode(actor, previous[actor])
                if old.kind == "attack":
                    established.add(old.target)
            ready = [e for e in contacts if context.fire_estimate(actor, e).window_shots >= 1]
            if self.common_combat.get('next_fire_budget_enabled', True):
                if joint_budget:
                    # A presently available shot remains eligible even when a
                    # detour would miss it. Other sources' earlier reserved
                    # kills cannot provide a fictitious later return target.
                    cost = self._final_next_fire_cost(context, actor, action, actions, report, memory)
                elif memory is not None and hasattr(memory, 'next_fire_cost'):
                    cost = memory.next_fire_cost(context, actor, action)
                else:
                    from .combat_progress import exposure_at
                    from .combat_exchange import next_fire_delay
                    z = report.positions.get(actor, (unit.position, False))[0]
                    before = exposure_at(context, actor, unit.position)
                    after = exposure_at(context, actor, z)
                    relief = before > 0. and before - after >= max(.1, .15 * before)
                    cost = 0. if relief else next_fire_delay(
                        context, actor, action, position=z, use_native_period=False)
                if any(e in established for e in ready):
                    continuity += cost
                else:
                    fire_delay += cost
                continue  # exactly one next-shot delay, from first departure onward
            if any(e in established for e in ready):
                continuity += 1.0
                continue  # no second reentry charge for the same ready shot
            if memory is not None:
                return_fire += getattr(memory, "return_fire_cost", lambda *_: 0.)(
                    context, actor, action)
            if not ready:
                z = report.positions.get(actor, (unit.position, False))[0]
                errors = []
                for enemy_id in contacts:
                    enemy = context.snapshot.enemies[enemy_id]
                    reach = context.snapshot.relations[actor][enemy_id].native_weapon_range
                    if reach is not None:
                        errors.append(max(distance(z, enemy.position) - reach
                                          - unit.radius - enemy.radius, 0.0))
                if errors:
                    reentry += min(min(errors) / self._move_scale(context, actor), 1.0)
        selected = {p.instance_id for p in plans if p.instance_id}
        for instance in getattr(memory, "instances", {}).values() if memory else ():
            minimum = instance.extra.get("min_loops", 0)
            elapsed = context.snapshot.game_loop - instance.start_loop
            if not minimum or elapsed >= minimum or instance.instance_id in selected:
                continue
            if self._commitment_interrupted(context, instance, memory):
                continue
            members = instance.actors
            if instance.skill == 'combat_rejoin' and instance.extra.get('local_cover_enabled'):
                from .combat_groups import active_rejoin_data
                data = active_rejoin_data(context, instance)
                if data is None:
                    continue
                members = (*data['joiners'], *data['supporters'])
            commitment += len(members) * (minimum - elapsed) / minimum
        return {"support_break": -self.weights["support_break"] * support_loss,
                "local_entry": -self.weights["support_break"] * entry['cost'],
                "attack_continuity": -self.weights["attack_continuity"] * continuity,
                "return_fire_continuity": -self.weights["attack_continuity"] * return_fire,
                "contact_reentry": -self.weights["reentry_cost"] * reentry,
                "next_fire_delay": -self.weights["attack_continuity"] * fire_delay,
                "commitment_switch": -self.weights["switch"] * commitment,
                "combat_idle": -self.weights["combat_idle"] * idle,
                "movement_continuity": -self.weights["movement_continuity"] * reversal,
                "blocked_move": -self.weights["blocked_move"] * blockage,
                "rejoin_abandonment": -self.weights["switch"] * self._rejoin_abandonment(
                    context, actions, joint_positions, memory, selected)}

    def _rejoin_abandonment(self, context, actions, positions, memory, selected):
        """Bounded cost of stopping feasible, observed joining progress.

        Action effects, not task labels, decide whether the duty continues.
        This replaces no damage/risk term and never renews a task deadline.
        Initial minimum commitment is already priced separately above.
        """
        from .combat_groups import (effective_support, rejoin_continuation_opportunity,
                                    active_rejoin_data)
        total = 0.
        for instance in getattr(memory, "instances", {}).values() if memory else ():
            if instance.skill != "combat_rejoin" or instance.termination_reason \
                    or instance.instance_id in selected:
                continue
            if context.snapshot.game_loop - instance.start_loop < instance.extra.get("min_loops", 0):
                continue
            opportunity = rejoin_continuation_opportunity(context, instance)
            if not opportunity:
                continue
            data = active_rejoin_data(context, instance)
            if data is None:
                continue
            joiners, supporters = data["joiners"], data["supporters"]
            radius = data["support_radius"]
            before = effective_support(context, joiners, supporters, radius=radius,
                                       targets_by_actor=data.get('targets_by_actor'),
                                       roles=data.get('roles') if data.get('local_cover_enabled') else None)
            after = effective_support(context, joiners, supporters, positions, radius=radius,
                                      targets_by_actor=data.get('targets_by_actor'),
                                      roles=data.get('roles') if data.get('local_cover_enabled') else None)
            stalled = 0.
            for actor, weight in opportunity.items():
                unit = context.snapshot.allies[actor]
                command = context.decode(actor, actions[actor])
                if command.kind == "attack" and context.contact_status(actor, command.target) == "in_range":
                    continue
                if memory.combat_emergency(context, actor):
                    continue
                point = positions.get(actor, unit.position)
                if len(context.risk_at(actor, point).threats) < len(context.risk_at(actor, unit.position).threats):
                    continue
                old_gap = min(distance(unit.position, context.snapshot.allies[b].position) for b in supporters)
                new_gap = min(distance(point, positions.get(b, context.snapshot.allies[b].position)) for b in supporters)
                closing = max(old_gap - radius, 0.) - max(new_gap - radius, 0.) > .05
                engaging = after["gap_by_actor"][actor] < before["gap_by_actor"][actor] - .05
                if closing or (new_gap <= radius + .05 and engaging):
                    continue
                stalled += weight
            total += stalled / len(opportunity)
        return min(total, 1.)

    @staticmethod
    def _commitment_interrupted(context, instance, memory=None):
        units = context.snapshot.allies
        if any(a not in units or not units[a].alive for a in instance.actors):
            return True
        if instance.goal_kind == "enemy" and instance.goal not in context.snapshot.enemies:
            return True
        if instance.goal_kind == "ally" and (instance.goal not in units
                                             or not units[instance.goal].alive):
            return True
        interrupts = instance.extra.get("interrupt_on", ())
        initial = instance.extra.get("initial_threats", {})
        previous = getattr(memory, "last_committed_actions", {}) if memory else {}
        if instance.skill == "combat_rejoin":
            if "illegal" in interrupts and any(
                    a in previous and previous[a] not in context.legal_actions(a)
                    for a in instance.actors):
                return True
            from .combat_groups import task_interrupt
            return task_interrupt(context, instance) is not None
        for actor in instance.actors:
            if "illegal" in interrupts and actor in previous \
                    and previous[actor] not in context.legal_actions(actor):
                return True
            if "weapon_ready" in interrupts and units[actor].weapon_ready:
                return True
            if "contact" in interrupts and any(
                    context.contact_status(actor, e) == "in_range"
                    for e in context.attackable_enemies(actor)):
                return True
            if "danger_change" in interrupts:
                unit = units[actor]
                if unit.health_lost > 0 or unit.shield_lost > 0:
                    return True
                now = set(context.threats_to(actor))
                if now - set(initial.get(actor, ())):
                    return True
        return False

    def _forecast_exposure(self, context, actor, position, certain):
        """Per-enemy exposure weights; no assumed enemy cooldown or shot."""
        cached = getattr(self, "_exposure_cache", None)
        if cached is None or cached[0] is not context:
            cached = (context, {})
            self._exposure_cache = cached
        key = (actor, tuple(position), certain)
        if key in cached[1]:
            return cached[1][key]
        unit = context.snapshot.allies[actor]
        forecast = context.forecast_risk(actor, position)
        current = set(forecast.current_threats)
        exposure = {e: (1. if e in current else self.forecast_reach_discount)
                    for e in forecast.reachable_threats}
        if not certain:
            origin = context.forecast_risk(actor, unit.position)
            original_current = set(origin.current_threats)
            for enemy in origin.reachable_threats:
                floor = self.uncertain_position_floor * (
                    1. if enemy in original_current else self.forecast_reach_discount)
                exposure[enemy] = max(exposure.get(enemy, 0.), floor)
        cached[1][key] = exposure
        return exposure

    @staticmethod
    def _threat_ability(context, actor, enemy):
        damage = context.enemy_attack_damage(enemy, actor)
        # One ordinary volley is a normalization unit, not a claimed damage
        # amount for unknown weapons. Catalog damage remains None upstream.
        return max(.35, min(4., damage / 14.)) if damage is not None else 1.

    def _forecast_cost(self, context, report):
        """Continuous pressure after three enemies, adjusted for EHP reserve."""
        costs, unknown = self._forecast_costs(context, report)
        return sum(costs.values()), unknown

    def _forecast_costs(self, context, report):
        """Per-actor costs allow correlated exposure to be priced only once."""
        costs, unknown = {}, set()
        for actor, (position, certain) in report.positions.items():
            unit = context.snapshot.allies[actor]
            if not unit.alive:
                continue
            exposure = self._forecast_exposure(context, actor, position, certain)
            pressure = 0.
            for enemy, fraction in exposure.items():
                pressure += fraction * self._threat_ability(context, actor, enemy)
                if context.enemy_attack_damage(enemy, actor) is None:
                    unknown.add((actor, enemy))
            reserve = max(unit.max_health + unit.max_shield, unit.effective_health, 1.)
            vulnerability = min(3., math.sqrt(reserve / max(unit.effective_health, 1.)))
            # log1p has diminishing returns but NO flat cap: six equal
            # enemies always cost more than three equal enemies.
            costs[actor] = (self.weights["risk"] * math.log1p(pressure)
                            / math.log(4.) * vulnerability)
        return costs, len(unknown)

    @staticmethod
    def _engagement_state(context, memory):
        if memory is not None and hasattr(memory, "battle_state"):
            return memory.battle_state(context)
        engaged = any(unit.alive and (
            unit.health_lost + unit.shield_lost > 0 or any(
                context.contact_status(actor, enemy) == "in_range"
                for enemy in context.attackable_enemies(actor)))
            for actor, unit in context.snapshot.allies.items())
        return {"engaged": engaged, "phase": "engage" if engaged else "search",
                "targets": tuple(context.snapshot.enemies)}

    def _participation_eta(self, context, actor, position, targets):
        """Static arrival estimate in control windows, not future damage.

        Both native weapon range and the command mask must be satisfied.
        Unknown speed/range gives no optimistic progress credit. Cooldown
        permits harmless movement that still preserves the next firing window.
        """
        speed = context.speed(actor)
        if speed is None:
            return float("inf")
        from .context import GAME_LOOPS_PER_SPEED_SECOND
        scale = speed * context.H / GAME_LOOPS_PER_SPEED_SECOND
        unit = context.snapshot.allies[actor]
        raw = context.snapshot.raw_state
        gaps = []
        for enemy in targets:
            limit = raw.get("relations", {}).get(actor, {}).get(enemy, {}).get(
                "smac_attack_distance_limit")
            if limit is None:
                limit = raw.get("action_contract", {}).get(
                    "candidate_center_distance_by_type", {}).get(unit.type_name)
            if (not isinstance(limit, (int, float)) or not math.isfinite(limit)) \
                    and not context.can_issue_attack(actor, enemy):
                continue  # unknown command reach is not a measurable arrival
            gaps.append(context.projected_attack_opportunity(actor, enemy, position).center_gap)
        gap = min(gaps, default=float("inf"))
        return max(gap / max(scale, 1e-6), unit.weapon_cooldown_loops / max(context.H, 1))

    def _participation_viability(self, context, actor, position, certain=False):
        """Conservative reserve discount; firing does not suppress threats."""
        unit = context.snapshot.allies[actor]
        if context.risk_at(actor, position).effect_weapon_near:
            return 0.
        if not self.common_combat.get('joint_assessment_enabled', True) and (
                unit.effective_health / max(unit.max_health + unit.max_shield, 1.) <= .25):
            return 0.             # explicit legacy ablation only
        incoming, unknown = 0., False
        for enemy, fraction in self._forecast_exposure(context, actor, position, certain).items():
            damage = context.enemy_attack_damage(enemy, actor)
            unknown |= damage is None
            incoming += fraction * max(damage or 0., 0.)
        from .combat_exchange import exposure_window, next_fire_horizon
        window = exposure_window(context, actor, position,
            horizon_h=next_fire_horizon(context, actor, position), certain=certain,
            forecast_config={"forecast_reach_discount": self.forecast_reach_discount,
                             "uncertain_position_floor": self.uncertain_position_floor})
        incoming = max(incoming, window["amount"])
        unknown |= window["unknown"]
        viable = max(0., 1. - incoming / max(unit.effective_health, 1.))
        return min(viable, .5 if unknown else 1.)

    def _participation_targets(self, context, actor, state, memory=None):
        # A persistent group objective must never force an already engaged
        # unit to abandon another valid local target.
        local = [e for e in context.attackable_enemies(actor)
                 if context.contact_status(actor, e) == "in_range"]
        if memory is not None and hasattr(memory, "battle_target"):
            target = memory.battle_target(context, actor)
            if target in context.snapshot.enemies:
                local.append(target)
        return tuple(sorted(set(local) | {e for e in state.get("targets", ())
                                         if e in context.snapshot.enemies}))

    def _productive_advance(self, context, actor, memory):
        state = self._engagement_state(context, memory)
        if not state.get("engaged") or state.get("phase") not in ("engage", "press"):
            return False
        unit = context.snapshot.allies[actor]
        targets = self._participation_targets(context, actor, state, memory)
        before = self._participation_eta(context, actor, unit.position, targets)
        if not math.isfinite(before):
            return False
        for action in context.move_destinations(actor):
            prediction = context.predict_position(actor, action)
            if self.common_combat.get('joint_assessment_enabled', True):
                from .combat_exchange import danger_evidence
                if danger_evidence(context, actor, prediction.position, horizon_h=2.,
                        pressure_memory_enabled=self.common_combat.get(
                            'pressure_memory_enabled', True))['immediate']:
                    continue
            elif self._participation_viability(context, actor, prediction.position,
                                               prediction.certain) <= 0.:
                continue
            if memory is not None and memory.movement_blockage(context, actor, action) >= .5:
                continue
            if self._participation_eta(context, actor, prediction.position, targets) < before - .05:
                return True
        return False

    def _counter_fire_relief(self, context, actions, report, memory):
        """Bounded threat-relief for counter-fire at an ally's ACTUAL threat.

        For each final window-qualified attack (actor, target): every OTHER
        alive ally that this target currently covers (relation flag or a
        current forecast threat) gains a bounded relief fraction
        min(1, known window damage / target effective health), weighted by
        the existing target_threat scale. (actor, target) pairs already
        credited by allocation_progress are excluded so the same volley is
        never priced twice; each (shooter, target) contributes once per
        step. Pure scoring read: threat evidence from the observation, no
        damage-attribution claims.
        """
        if not self.common_combat.get('joint_assessment_enabled', True):
            return 0.
        from .battle_pressure import enemy_pressure
        from .targeting import shared_target_plan
        assignments = {}
        plan = shared_target_plan(context, memory, self.target_config)
        for shooter, row in plan.assignments.items():
            assignments[(shooter, row.target)] = True
        relief = 0.
        seen = set()
        for actor, action in actions.items():
            unit = context.snapshot.allies.get(actor)
            if unit is None or not unit.alive:
                continue
            semantic = context.decode(actor, action)
            if semantic.kind != "attack"                     or semantic.target not in context.snapshot.enemies:
                continue
            target = semantic.target
            if (actor, target) in assignments or (actor, target) in seen:
                continue
            estimate = context.fire_estimate(actor, target)
            if estimate.window_shots < 1 or estimate.damage_lo is None:
                continue            # only executable known fire prices relief
            seen.add((actor, target))
            fraction = min(1., estimate.damage_lo / max(
                context.snapshot.enemies[target].effective_health, 1.))
            for peer, mate in context.snapshot.allies.items():
                if peer == actor or not mate.alive:
                    continue
                relation = context.snapshot.relations.get(peer, {}).get(target)
                covered = (relation is not None
                           and relation.in_enemy_weapon_range is True)                     or target in context.forecast_risk(
                        peer, mate.position).current_threats
                if not covered:
                    continue        # cover must match the actual threat
                relief += fraction * min(1., enemy_pressure(context, target))
                break               # one covered teammate per shot suffices
        return self.weights["target_threat"] * self.urgency_value             * min(1., relief)

    def _sustained_contact_value(self, context, actions, report, memory):
        """Bounded progress toward a sustained, shared firing front.

        This evaluates final physical positions, independent of skill labels.
        It is an opportunity potential, never a shot/kill prediction. Joint
        mode can value a shared entry before first contact; legacy progress
        still starts after observed combat. Neither rewards postponing a
        current shot or charges real escape relief as lost participation.
        """
        line = None
        if self.common_combat.get('joint_assessment_enabled', True):
            from .battle_line import assess_battle_line, battle_line
            line = battle_line(context)
            participation = assess_battle_line(
                context, actions, self.joint_facts(context, actions), memory)
            # Replace the old independent approach reward. A nearby body or
            # a plan label cannot promise a joint firing window on its own.
            total = sum(value for actor, value in participation['actor_gains'].items()
                        if not self._available_fire(context, actor))
        else:
            total = 0.
        state = self._engagement_state(context, memory)
        if not state.get("engaged") or state.get("phase") not in ("engage", "press"):
            return self.weights["sustained_contact"] * total
        for actor, action in actions.items():
            unit = context.snapshot.allies.get(actor)
            if unit is None or not unit.alive:
                continue
            capability = context.capabilities[actor]
            if capability.healer or capability.suicide_attack:
                continue
            if context.decode(actor, action).kind != "move":
                continue
            if any(context.fire_estimate(actor, e).window_shots >= 1
                   for e in context.attackable_enemies(actor)):
                continue  # immediate fire/attack continuity already values this
            lane = line['actors'].get(actor) if line is not None else None
            targets = (lane['front'] if lane is not None else
                       self._participation_targets(context, actor, state, memory))
            before = self._participation_eta(context, actor, unit.position, targets)
            point, certain = report.positions[actor]
            after = self._participation_eta(context, actor, point, targets)
            if not math.isfinite(before) or not math.isfinite(after):
                continue
            gain = _clip(before - after)
            if line is not None and gain >= 0.:
                continue  # common-front participation already supplies this credit
            if gain > 0. and memory is not None:
                gain *= getattr(memory, "advance_credit_scale", lambda *_: 1.)(
                    context, actor, action)
            if line is not None:
                row = self.joint_facts(context, actions)['actors'][actor]
                viability = row['team_exposure_after']['viability']
            else:
                viability = self._participation_viability(context, actor, point, certain)
            if gain < 0.:
                # Genuine escape retains its full safety benefit. Persistent
                # ineffective withdrawal is handled from observations by memory.
                origin = self._forecast_exposure(context, actor, unit.position, True)
                final = self._forecast_exposure(context, actor, point, certain)
                old = sum(v * self._threat_ability(context, actor, e) for e, v in origin.items())
                new = sum(v * self._threat_ability(context, actor, e) for e, v in final.items())
                if old - new >= max(.1, .15 * old) or context.risk_at(actor, unit.position).effect_weapon_near:
                    continue
                if self.common_combat.get('joint_assessment_enabled', True) and memory is not None \
                        and self.common_combat.get('next_fire_budget_enabled', True) \
                        and self._final_next_fire_cost(context, actor, action, actions, report, memory) > 0.:
                    continue      # the same delayed shot is already priced
            # Distant movement earns less credit than an imminent firing
            # window. A single source cannot scale its reward by target count.
            urgency = (1.25 if state.get("phase") == "press" and gain > 0.
                       and unit.health_fraction >= .5 else 1.)
            total += urgency * gain * viability / (1. + .25 * min(before, after))
        return self.weights["sustained_contact"] * total

    def _future_attack_value(self, context, actions, report, plans):
        """Bounded prospective opportunity; never enters damage/kill ledgers.

        A source with an available current-window shot cannot earn a reward
        for postponing it. Cooling sources and sources still outside the
        current attack mask can value next-window contact after a legal move.
        Credits saturate once per target, regardless of plan packaging.
        """
        objectives = {actor: (p.goal_data or {}).get("objective", {})
                      for p in plans for actor in p.controlled_actors}
        by_target = {}
        ordinary_by_target, ordinary_caps = {}, {}
        from .fire_opportunities import final_action_budget
        budget = (final_action_budget(context, actions)
                  if self.common_combat.get('joint_assessment_enabled', True) else None)
        for actor, action in actions.items():
            unit = context.snapshot.allies.get(actor)
            if unit is None or not unit.alive or context.capabilities[actor].healer:
                continue
            semantic = context.decode(actor, action)
            if semantic.kind not in ("move", "attack"):
                continue
            if any(context.fire_estimate(actor, e).window_shots >= 1
                   for e in context.attackable_enemies(actor)):
                continue  # immediate fire is already valued, not deferred twice
            objective = objectives.get(actor, {})
            horizon = int(objective.get("horizon_loops", 2 * context.H))
            declared = objective.get("next_attack_target")
            targets = ([semantic.target] if semantic.kind == "attack" else
                       [declared] if declared in context.snapshot.enemies else
                       sorted(context.snapshot.enemies))
            position = report.positions[actor][0]
            capability = context.capabilities[actor]
            ordinary = (budget is not None and not capability.suicide_attack
                        and capability.specialization != "splash_fire")
            eta = max(unit.weapon_cooldown_loops / max(context.H, 1),
                      1. if semantic.kind == "move" or distance(unit.position, position) > .01 else 0.)
            # A move does not commit a future attack target. If its declared
            # next target is already budgeted, consider another compatible
            # target at the SAME final position, not imaginary extra movement.
            if ordinary and semantic.kind == "move" and declared in context.snapshot.enemies \
                    and budget.remaining_fraction(actor, declared, eta) <= 1e-9:
                targets = sorted(context.snapshot.enemies)
            best = (0., "")
            best_primary = False
            best_credit = 1.
            for enemy_id in targets:
                opportunity = context.projected_attack_opportunity(
                    actor, enemy_id, position, horizon=horizon)
                if not opportunity.available or not opportunity.ready:
                    continue
                enemy = context.snapshot.enemies[enemy_id]
                if opportunity.damage_est is None:
                    value = .15
                elif ordinary:
                    # Price the first shot opened by a move in the same
                    # units as a current volley, including its overkill cap.
                    useful = min(opportunity.damage_est, enemy.effective_health)
                    value = min(1., 3. * useful / DAMAGE_SCALE)
                else:
                    value = min(1., 3. * opportunity.damage_est / max(enemy.effective_health, 1.))
                if context.capabilities[actor].specialization == "splash_fire" \
                        and self.mechanisms.get("splash", True) \
                        and objective.get("evaluate_splash", True):
                    neighbours = sum(not other.is_flying and e != enemy_id
                        and distance(enemy.position, other.position) <= self.effect_params["splash_radius"]
                        for e, other in context.snapshot.enemies.items())
                    value += self.effect_params["splash_weight"] * self.splash_reliability * min(neighbours, 2)
                if ordinary:
                    credit = budget.credit(actor, enemy_id, eta)
                    primary = budget.remaining_fraction(actor, enemy_id, eta) > 1e-9
                    candidate = (min(value, 1.) * credit, enemy_id)
                    if (primary, candidate) > (best_primary, best):
                        best, best_primary, best_credit = candidate, primary, credit
                else:
                    best = max(best, (min(value, 1.), enemy_id))
            if best[0]:
                if ordinary:
                    ordinary_by_target[best[1]] = ordinary_by_target.get(best[1], 0.) + best[0]
                    pool = context.snapshot.enemies[best[1]].effective_health
                    cap = best_credit * min(1., 3. * pool / DAMAGE_SCALE)
                    ordinary_caps[best[1]] = max(ordinary_caps.get(best[1], 0.), cap)
                else:
                    by_target[best[1]] = min(1., by_target.get(best[1], 0.) + best[0])
        for target, value in ordinary_by_target.items():
            by_target[target] = min(1., by_target.get(target, 0.) + min(value, ordinary_caps[target]))
        return self.weights["future_attack"] * sum(by_target.values())

    # ------------------------------------------------------------ main scoring
    def score_joint(self, context: Context, plans: List[CandidatePlan],
                    actions: Dict[str, int],
                    memory=None) -> Tuple[float, Dict[str, float]]:
        w = self.weights
        snapshot = context.snapshot
        report = self.effects(context, actions)
        breakdown: Dict[str, float] = {}

        # -- useful damage & kills: aggregate KNOWN sources per target ----
        damage_score = kill_score = 0.0
        for enemy_id, total in report.damage_known.items():
            enemy = snapshot.enemies.get(enemy_id)
            if enemy is None:
                continue
            # Fixed reference pool (same scale as candidate allocation and
            # the mixed gun/burst settlement): base value tracks damage
            # applied, capped by remaining effective health, without the
            # old bonus for small health pools.
            useful = min(total, enemy.effective_health)
            damage_score += w["useful_damage"] * useful / DAMAGE_SCALE
            if total >= enemy.effective_health:
                kill_score += w["kill"]      # EXPECTED kill, never a real one
        breakdown["useful_damage"] = damage_score
        breakdown["expected_kills"] = kill_score
        breakdown["unknown_damage_sources"] = float(sum(
            len(v) for v in report.unknown_sources.values()))

        # Only actual ordinary firing groups earn forecast output avoided.
        # Unknown projectile timing never changes the current damage ledger.
        future_relief = sum(avoided_output(context, enemy, shooters)
                            for enemy, shooters in report.window_sources.items())
        breakdown["early_kill_avoidance"] = w["target_threat"] * min(2., future_relief)

        # -- heal: bounded opportunity proxy (need x sustain x urgency) -----
        heal_score = 0.0
        for opportunity in report.heal.values():
            heal_score += w["heal"] * min(opportunity.opportunity, 1.0)
        breakdown["heal"] = heal_score

        # -- coverage opportunity (window-qualified, saturating) ------------
        breakdown["coverage"] = w["coverage"] * sum(report.coverage.values())
        breakdown["coverage_raw"] = w["coverage"] * sum(report.coverage_raw.values())
        breakdown["splash_reliability"] = self.splash_reliability
        breakdown["coverage_sources"] = float(sum(
            len(v) for v in report.coverage_sources.values()))

        # One-use explosions and ordinary fire share target HP in arrival
        # buckets. Unknown/same-window timing never gives guns automatic
        # priority. Kill evidence above remains exclusively ordinary damage.
        burst_value = 0.
        if report.burst_outcomes:
            from .burst_allocation import allocate_exchange
            ordinary, explosion = allocate_exchange(context, actions, report)
            damage_score = w['useful_damage'] * ordinary
            burst_value = w['useful_damage'] * explosion
            breakdown['useful_damage'] = damage_score
        breakdown["burst_opportunity"] = burst_value
        breakdown["burst_sources"] = float(sum(
            outcome.opportunity > 0. for outcome in report.burst_outcomes.values()))
        breakdown["burst_unknown_sources"] = float(sum(
            len(sources) for sources in report.burst_unknown_sources.values()))
        from .burst import approach_value
        burst_approach = w["tactic_progress"] * approach_value(context, actions, report) \
            if self.mechanisms["burst"] else 0.
        breakdown["burst_approach"] = burst_approach
        # Conditional death splash is a risk floor, not a second damage sum.
        breakdown["enemy_death_burst_potential"] = sum(report.enemy_death_burst.values())

        # -- risk at FINAL positions ----------------------------------------
        risk_costs, risk_unknown = self._forecast_costs(context, report)
        conservative_risk = sum(risk_costs.values())
        joint = self.joint_facts(context, actions) if self.common_combat.get(
            'joint_assessment_enabled', True) else None
        if joint is not None:
            for actor, row in joint['actors'].items():
                if row['ordinary']:
                    # Same per-enemy team budget for ATTACK/MOVE/STOP. Taking
                    # max with the old all-enemies-per-ally upper bound here
                    # would silently reinstate the veto this model replaces.
                    ratio = row.get('team_exposure_after', row['exposure'])['reserve_ratio']
                    risk_costs[actor] = w['risk'] * reserve_risk_penalty(ratio)
        for actor, amount in report.enemy_death_burst.items():
            floor = w['risk'] * reserve_risk_penalty(
                amount / max(snapshot.allies[actor].effective_health, 1.))
            risk_costs[actor] = max(risk_costs.get(actor, 0.), floor)
        risk_score = sum(risk_costs.values())
        breakdown["risk"] = -risk_score
        breakdown['conservative_risk_diagnostic'] = -conservative_risk
        breakdown["risk_unknown_sources"] = float(risk_unknown)

        # -- congestion between predicted same-class positions --------------
        congestion = 0.0
        movers = [a for a in report.positions
                  if a in snapshot.allies and snapshot.allies[a].alive]
        for i, a in enumerate(movers):
            for b in movers[i + 1:]:
                ua, ub = snapshot.allies[a], snapshot.allies[b]
                if ua.is_flying != ub.is_flying:
                    continue
                if distance(report.positions[a][0], report.positions[b][0]) \
                        < (ua.radius + ub.radius):
                    congestion += 1.0
        breakdown["congestion"] = -w["congestion"] * min(congestion, 4.0)

        # -- tactical terms: recomputed from FINAL positions and declared
        # goals; a displacement is counted ONCE here, never again from
        # self-reported features (B2). Packaging-independent (B6).
        tactic, tactic_breakdown = self.task_quality(context, actions, plans,
                                                     report)
        breakdown.update(tactic_breakdown)
        hold = switch = reentry = 0.0
        for plan in plans:
            # A recovery wait is justified by final service/recovery facts,
            # not by a label supplying free holding reward each step.
            if not (plan.primary_skill == 'disengage_reenter'
                    and plan.commitment.phase == 'STABILIZE'):
                hold += w["hold_value"] * _cap(plan.features.get("hold_value", 0.0))
            switch += w["switch"] * _cap(plan.features.get("switch_cost", 0.0))
            reentry += w["reentry_cost"] * _cap(plan.features.get("reentry_cost", 0.0))
        breakdown["hold_value"] = hold
        breakdown["switch"] = -switch
        breakdown["reentry_cost"] = -reentry
        planned_raw = sum(report.planned_loss.values()) * w["planned_loss"]
        planned = sum(max(0., value * w["planned_loss"]
                          - risk_costs.get(actor, 0.))
                      for actor, value in report.planned_loss.items())
        breakdown["planned_loss"] = -planned
        breakdown["planned_loss_raw"] = -planned_raw

        coordination = self._coordination_costs(context, actions, report, memory, plans)
        breakdown.update(coordination)
        future_attack = self._future_attack_value(context, actions, report, plans)
        from .battle_pressure import joint_pressure
        pressure = joint_pressure(context, actions, report)
        target_threat = (self.weights["target_threat"] * self.urgency_value
                         * pressure["pressure_relief"])
        from .targeting import allocation_progress
        allocation_value = self.weights["target_threat"] * allocation_progress(
            context, actions, report, memory, self.target_config)
        # Counter-fire relief against an ally's ACTUAL threat (bounded).
        # A window-qualified shot whose target is an enemy currently
        # covering a teammate prices the threat that shot answers — even
        # when the shared target plan assigned that shooter elsewhere. The
        # correspondence check is per (shooter, threat): "near an ally" or
        # "attacking an unrelated nearby unit" earns nothing here. Shots
        # already credited by allocation_progress for the same (shooter,
        # target) are skipped, so one volley never receives the threat
        # credit twice. Same scale as target_threat; predictions only.
        counter_relief = self._counter_fire_relief(context, actions, report,
                                                   memory)
        # Both terms observe the same enemy volley. Price the larger concern
        # per actor, not their sum; do not let one safe unit cancel another's
        # exposure. Unknown risk remains in risk_costs and lethal relief remains
        # independent. A disabled risk weight still leaves the exposure term.
        exposure_by_actor = dict(pressure['unsupported_actors'])
        if joint is not None:
            for actor, row in joint['actors'].items():
                if row['ordinary']:
                    exposure_by_actor[actor] = min(1., row['team_exposure_after']['reserve_ratio'])
        unsupported = -sum(max(0., self.weights["unsupported_frontline"] * exposure
                               - max(risk_costs.get(actor, 0.),
                                     w["planned_loss"] * report.planned_loss.get(actor, 0.)))
                           for actor, exposure in exposure_by_actor.items())
        sustained = self._sustained_contact_value(context, actions, report, memory)
        breakdown["future_attack"] = future_attack
        breakdown["target_threat"] = target_threat
        breakdown["allocation_progress"] = allocation_value
        breakdown["counter_fire_relief"] = counter_relief
        breakdown["unsupported_frontline"] = unsupported
        breakdown["sustained_contact"] = sustained
        breakdown["pressure_unknown_pairs"] = float(pressure["unknown_enemy_pairs"])

        recovery_value = 0.
        if self.common_combat.get('mechanism_recovery_enabled', True):
            from .recovery import recovery_opportunity
            for actor, action in actions.items():
                unit = snapshot.allies.get(actor)
                if unit is None or not unit.alive:
                    continue
                # Passive regeneration does not require STOP. Equal safe
                # positions get equal opportunity even while issuing ATTACK.
                regain = recovery_opportunity(context, actor,
                    position=report.positions.get(actor, (unit.position, False))[0])
                recovery_value += w['heal'] * regain['gain'] / max(
                    unit.max_health + unit.max_shield, 1.)
        breakdown['passive_recovery'] = recovery_value
        breakdown['joint_exposed_moves'] = float(sum(
            row['ordinary'] and row['moving'] and row['exposure']['reserve_ratio'] >= 1.
            for row in joint['actors'].values())) if joint is not None else 0.

        # Cross-task progress is measured from observations, not the current
        # provider's newly created task. No generic gun-cycle cost for S3.
        contact_debt = burst_motion = team_delay = 0.
        delay_by_actor = joint['team_delay']['actors'] if joint is not None else {}
        if memory is not None and self.common_combat.get('joint_assessment_enabled', True):
            contact_progress = getattr(memory, 'contact_progress', None)
            burst_progress = getattr(memory, 'burst_progress', None)
            for actor, action in actions.items():
                actor_debt = 0.
                if contact_progress is not None:
                    actor_debt = contact_progress.action_cost(context, actor, action, memory)
                    contact_debt += actor_debt
                # Both price participation delay. A receiver losing HP raises
                # its urgency, but never charges this same actor twice.
                team_delay += max(0., delay_by_actor.get(actor, 0.) - actor_debt)
                if burst_progress is not None:
                    burst_motion += burst_progress.action_cost(context, actor, action)
        contact_debt *= w['combat_idle']
        burst_motion *= w['movement_continuity']
        team_delay *= w['combat_idle']
        breakdown['contact_debt'] = -contact_debt
        breakdown['burst_motion'] = -burst_motion
        breakdown['team_delay'] = -team_delay

        total = (damage_score + kill_score + heal_score + breakdown["coverage"]
                 + burst_value + burst_approach
                 + tactic + hold - risk_score + breakdown["congestion"]
                 - switch - reentry - planned + sum(coordination.values())
                 + future_attack + target_threat + allocation_value
                 + counter_relief + breakdown["early_kill_avoidance"]
                 + unsupported + sustained
                 + recovery_value - contact_debt - burst_motion - team_delay)
        fronts = getattr(context, "ranged_fire_reference", ())
        if fronts:
            from .skills.ranged_fire_formation import formation_value
            # A pocket can focus on either target. Credit its best common
            # firing line once, including when the branch switches targets.
            current_best, baseline_best = {}, {}
            for anchor, enemy, baseline in fronts:
                value = formation_value(context, anchor, enemy, actions)
                current_best[anchor] = max(current_best.get(anchor, value), value)
                baseline_best[anchor] = max(baseline_best.get(anchor, baseline), baseline)
            fire_balance = w["ranged_fire_formation"] * sum(
                value - baseline_best[anchor] for anchor, value in current_best.items())
            breakdown["ranged_fire_formation"] = fire_balance
            total += fire_balance
        breakdown["total"] = total
        return total, breakdown
