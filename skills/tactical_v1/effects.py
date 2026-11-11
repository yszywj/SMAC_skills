"""Unified physical effect model (A3).

score_joint rebuilds the physical consequences of a COMPLETE action list
through this module — skills never add self-reported damage. The same
action gets the same physical interpretation regardless of which skill
proposed it or how plans are packaged.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .context import Context, IN_RANGE, distance
from .contracts import CALIBRATED, HEURISTIC

# Heal opportunity proxy: no calibrated HP/second exists for this protocol,
# so one committed heal action in the window is worth this fraction of the
# patient's missing health — an explicitly heuristic, capped proxy (S2.7).
HEAL_WINDOW_FRACTION = 0.5
# Coverage opportunity weight per additional in-contact splash/burst source.
COVERAGE_SOURCE_WEIGHT = 0.5


@dataclass
class EffectsReport:
    # enemy -> (window damage estimate, all_estimates_known: bool)
    damage: Dict[str, float] = field(default_factory=dict)
    damage_known: Dict[str, bool] = field(default_factory=dict)
    # patient -> heal opportunity proxy (already capped by missing health)
    heal: Dict[str, float] = field(default_factory=dict)
    heal_blocked_by_move: Dict[str, bool] = field(default_factory=dict)
    # enemy -> saturated coverage opportunity in [0, 1]
    coverage: Dict[str, float] = field(default_factory=dict)
    # actor -> predicted final position (uncertain flag)
    positions: Dict[str, Tuple[Tuple[float, float], bool]] = field(default_factory=dict)
    attacks_in_window: Dict[str, List[str]] = field(default_factory=dict)  # enemy -> attackers
    # actor -> fire estimate used (for diagnostics)
    planned_loss: Dict[str, float] = field(default_factory=dict)


def physical_effects(context: Context, actions: Dict[str, int],
                     coverage_radius: float = 3.0) -> EffectsReport:
    """Aggregate window effects of the FINAL action list."""
    snapshot = context.snapshot
    report = EffectsReport()
    # 1. positions
    for actor, action in actions.items():
        unit = snapshot.allies.get(actor)
        if unit is None or not unit.alive:
            continue
        prediction = context.predict_position(actor, action)
        report.positions[actor] = (prediction.position, prediction.certain)
    # 2. attacks and heals, by decoded semantics
    for actor, action in actions.items():
        unit = snapshot.allies.get(actor)
        if unit is None or not unit.alive:
            continue
        semantic = context.decode(actor, action)
        if semantic.kind == "attack" and semantic.target in snapshot.enemies:
            enemy_id = semantic.target
            estimate = context.fire_estimate(actor, enemy_id)
            if estimate.window_shots >= 1 and estimate.damage_est:
                report.damage[enemy_id] = report.damage.get(enemy_id, 0.0) \
                    + estimate.damage_est
                report.damage_known[enemy_id] = \
                    report.damage_known.get(enemy_id, True)
            else:
                # command submitted but no window-qualified damage
                report.damage_known.setdefault(enemy_id, True)
                if estimate.damage_est is None and estimate.window_shots == 0:
                    report.damage_known[enemy_id] = False
            report.attacks_in_window.setdefault(enemy_id, []).append(actor)
        elif semantic.kind in ("heal", "keep_heal") and semantic.target:
            patient = semantic.target
            target = snapshot.allies.get(patient)
            if target is not None and 0 < target.health < target.max_health:
                missing = target.max_health - target.health
                # One committed heal source per window: additional medivacs
                # on the SAME patient saturate — spreading heals patients is
                # worth strictly more (S2 occupancy-aware allocation).
                report.heal[patient] = min(
                    report.heal.get(patient, 0.0) + HEAL_WINDOW_FRACTION * missing,
                    HEAL_WINDOW_FRACTION * missing)
                # moving patients interrupt sustained healing: the patient's
                # own move reduces this window's heal credit (S2.8)
                patient_action = actions.get(patient)
                if patient_action is not None:
                    patient_semantic = context.decode(patient, patient_action)
                    report.heal_blocked_by_move[patient] = \
                        patient_semantic.kind == "move"
    # coverage opportunities from in-contact splash / burst attackers
    for enemy_id, attackers in report.attacks_in_window.items():
        weights = []
        for actor in attackers:
            capability = context.capabilities.get(actor)
            if capability is None or not capability.splash_attack:
                continue
            weights.append(COVERAGE_SOURCE_WEIGHT)
            for other_id, other in snapshot.enemies.items():
                if other_id == enemy_id:
                    continue
                if distance(other.position,
                            snapshot.enemies[enemy_id].position) \
                        <= coverage_radius:
                    report.coverage.setdefault(other_id, [])
                    report.coverage[other_id].append(COVERAGE_SOURCE_WEIGHT)
        if weights:
            report.coverage.setdefault(enemy_id, []).extend(weights)
    for enemy_id, weights in report.coverage.items():
        # saturation proxy: min(1, sum of geometric opportunity weights);
        # reduced when ordinary fire already covers the target
        saturated = min(1.0, sum(weights))
        if report.damage.get(enemy_id, 0.0) >= snapshot.enemies[enemy_id].effective_health:
            saturated *= 0.3
        report.coverage[enemy_id] = saturated
    # planned self-consumption of suicide attackers that detonate in contact
    for actor, action in actions.items():
        unit = snapshot.allies.get(actor)
        capability = context.capabilities.get(actor)
        if unit is None or capability is None or not capability.suicide_attack:
            continue
        semantic = context.decode(actor, action)
        if semantic.kind == "attack":
            report.planned_loss[actor] = 1.0   # fraction of the unit, once
    return report
