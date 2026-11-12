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


class Evaluator:
    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 mechanisms: Optional[Dict[str, bool]] = None,
                 splash_radius: float = 3.0, splash_weight: float = 0.5,
                 burst_radius: float = 2.5, burst_weight: float = 0.5,
                 burst_self_cost: float = 1.0):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.mechanisms = mechanisms or {"splash": True, "burst": True}
        self.effect_params = dict(splash_radius=splash_radius,
                                  splash_weight=splash_weight,
                                  burst_radius=burst_radius,
                                  burst_weight=burst_weight,
                                  burst_self_cost=burst_self_cost)

    def effects(self, context: Context,
                actions: Dict[str, int]) -> EffectsReport:
        return physical_effects(context, actions, self.mechanisms,
                                **self.effect_params)

    # ---------------------------------------------------- B1-B5 tactical terms
    def _move_scale(self, context: Context, actor: str) -> float:
        """L: plausible distance inside one window; speed-derived when known,
        else a flagged conservative constant (no path simulation)."""
        speed = context.speed(actor)
        return max(speed * context.H / 22.4, 0.5) if speed else 1.0

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
                     "role": 0.0}
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
            if kind == "escape":
                breakdown["progress"] += w["tactic_progress"] * \
                    self._escape_gain(context, report, plan)
                continue
            if kind == "spacing":
                continue    # congestion is priced once by the physics term
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
                 + breakdown["meeting"] + breakdown["role"])
        return total, breakdown

    def _toward(self, context, report, plan, goal) -> float:
        """Signed closeness gain per controlled actor toward a POINT."""
        total = 0.0
        for actor in plan.controlled_actors:
            if actor not in context.snapshot.allies:
                continue
            x = context.snapshot.allies[actor].position
            z = report.positions.get(actor, (x, True))[0]
            total += _clip((_dist(x, goal) - _dist(z, goal))
                           / self._move_scale(context, actor))
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
            total += _clip((self._band_error(d_now, r_min, r_max)
                            - self._band_error(d_after, r_min, r_max))
                           / self._move_scale(context, actor))
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
            total += _clip((self._band_error(d_now, r_min, r_max)
                            - self._band_error(d_after, r_min, r_max))
                           / self._move_scale(context, actor))
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
        final_a = [report.positions.get(a, (units[a].position, True))[0]
                   for a in group_a if a in units]
        final_b = [report.positions.get(b, (units[b].position, True))[0]
                   for b in group_b if b in units]
        if not current_a or not current_b:
            return 0.0
        return _clip((phi(current_a, current_b) - phi(final_a, final_b))
                     / scale)

    def _meeting_gain(self, context: Context, actions: Dict[str, int],
                      plan: CandidatePlan, report: EffectsReport) -> float:
        """B5: FUTURE heal opportunity from closing a healer-patient pair —
        never a heal amount credited this window."""
        data = plan.goal_data or {}
        healer, patient = data.get("healer"), data.get("patient")
        units = context.snapshot.allies
        if not healer or not patient or healer not in units \
                or patient not in units:
            return 0.0
        gap_now = _dist(units[healer].position, units[patient].position)
        gap_after = _dist(
            report.positions.get(healer, (units[healer].position, True))[0],
            report.positions.get(patient, (units[patient].position, True))[0])
        unit = units[patient]
        need = min(max(unit.max_health - unit.health, 0.0)
                   / max(unit.max_health, 1.0), 1.0)
        return need * _clip((gap_now - gap_after)
                            / (2.0 * self._move_scale(context, healer)))

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

        return _clip((quality(z) - quality(unit.position))
                     / max(float(data.get("scale", 6.0)), 1.0))

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
            useful = min(total, enemy.effective_health)
            damage_score += w["useful_damage"] * useful / max(enemy.effective_health, 1.0)
            if total >= enemy.effective_health:
                kill_score += w["kill"]      # EXPECTED kill, never a real one
        breakdown["useful_damage"] = damage_score
        breakdown["expected_kills"] = kill_score
        breakdown["unknown_damage_sources"] = float(sum(
            len(v) for v in report.unknown_sources.values()))

        # -- heal: bounded opportunity proxy (need x sustain x urgency) -----
        heal_score = 0.0
        for opportunity in report.heal.values():
            heal_score += w["heal"] * min(opportunity.opportunity, 1.0)
        breakdown["heal"] = heal_score

        # -- coverage opportunity (window-qualified, saturating) ------------
        breakdown["coverage"] = w["coverage"] * sum(report.coverage.values())
        breakdown["coverage_sources"] = float(sum(
            len(v) for v in report.coverage_sources.values()))

        # -- risk at FINAL positions ----------------------------------------
        risk_score = 0.0
        for actor, (position, _certain) in report.positions.items():
            if actor not in snapshot.allies or not snapshot.allies[actor].alive:
                continue
            risk = context.risk_at(actor, position)
            normalized = min(len(risk.threats) / RISK_NORM, 1.0)
            if risk.effect_weapon_near:
                normalized = min(normalized + 0.3, 1.0)
            risk_score += w["risk"] * normalized
        breakdown["risk"] = -risk_score

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
            hold += w["hold_value"] * _cap(plan.features.get("hold_value", 0.0))
            switch += w["switch"] * _cap(plan.features.get("switch_cost", 0.0))
            reentry += w["reentry_cost"] * _cap(plan.features.get("reentry_cost", 0.0))
        breakdown["hold_value"] = hold
        breakdown["switch"] = -switch
        breakdown["reentry_cost"] = -reentry
        planned = sum(report.planned_loss.values()) * w["planned_loss"]
        breakdown["planned_loss"] = -planned

        total = (damage_score + kill_score + heal_score + breakdown["coverage"]
                 + tactic + hold - risk_score + breakdown["congestion"]
                 - switch - reentry - planned)
        breakdown["total"] = total
        return total, breakdown
