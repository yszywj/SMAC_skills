"""Unified physical effect model (B1–B4, B6).

score_joint rebuilds physical consequences from the FINAL action list. Per
attack SOURCE: a window-qualified known contribution, or zero this window
(known cooldown), or an UNKNOWN source listed separately — an unknown or
cooling source never cancels another source's known contribution. Coverage
credit requires a window-qualified action from an ENABLED mechanism. Heal
value is an explicitly-uncalibrated opportunity proxy (need x sustain x
urgency), bounded and protocol-capped at one source per patient.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .actions import KEEP_HEAL_ORDER
from .context import Context, IN_RANGE, distance
from .contracts import CALIBRATED, HEURISTIC


@dataclass(frozen=True)
class DamageSource:
    actor: str
    amount: Optional[float]        # None = unknown magnitude
    lower: Optional[float]
    upper: Optional[float]
    evidence: str
    window_qualified: bool         # contact ok AND cooldown inside H


@dataclass(frozen=True)
class HealOpportunity:
    patient: str
    healer: str
    need: float                    # missing HEALTH / max_health (shields excluded)
    sustain: float                 # supported continuation factor
    urgency: float                 # bounded pressure composite
    opportunity: float             # need * sustain * (0.5 + 0.5*urgency)
    patient_moving_away: bool


@dataclass
class EffectsReport:
    # enemy -> known window-qualified damage (known sources only)
    damage_known: Dict[str, float] = field(default_factory=dict)
    damage_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # enemy -> sources with unknown magnitude/timing (never cancel the known)
    unknown_sources: Dict[str, List[str]] = field(default_factory=dict)
    # enemy -> actors that issued an attack command this step (deduped)
    attack_commands: Dict[str, List[str]] = field(default_factory=dict)
    # enemy -> actors whose attack is window-qualified (contact + cooldown)
    window_sources: Dict[str, List[str]] = field(default_factory=dict)
    heal: Dict[str, HealOpportunity] = field(default_factory=dict)
    new_heal_assignments: Dict[str, str] = field(default_factory=dict)  # patient->healer
    heal_capacity_conflicts: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    # enemy -> saturated coverage opportunity in [0, 1]
    coverage: Dict[str, float] = field(default_factory=dict)
    coverage_sources: Dict[str, List[str]] = field(default_factory=dict)
    positions: Dict[str, Tuple[Tuple[float, float], bool]] = field(default_factory=dict)
    planned_loss: Dict[str, float] = field(default_factory=dict)  # in-contact detonation only


def physical_effects(context: Context, actions: Dict[str, int],
                     mechanisms: Optional[Dict[str, bool]] = None,
                     splash_radius: float = 3.0, splash_weight: float = 0.5,
                     burst_radius: float = 2.5, burst_weight: float = 0.5,
                     burst_self_cost: float = 1.0) -> EffectsReport:
    mechanisms = mechanisms or {"splash": True, "burst": True}
    snapshot = context.snapshot
    report = EffectsReport()
    # ---- 1. positions: attack/heal commands may imply approach (B6) ------
    for actor, action in actions.items():
        unit = snapshot.allies.get(actor)
        if unit is None or not unit.alive:
            continue
        prediction = context.predict_position(actor, action)
        report.positions[actor] = (prediction.position, prediction.certain)
    # ---- 2. attack sources, per actor (B2) --------------------------------
    seen_attack = set()
    for actor, action in actions.items():
        unit = snapshot.allies.get(actor)
        if unit is None or not unit.alive:
            continue
        semantic = context.decode(actor, action)
        if semantic.kind != "attack" or semantic.target not in snapshot.enemies:
            continue
        enemy_id = semantic.target
        key = (actor, enemy_id, action)
        if key in seen_attack:      # one credit per actual (actor, action, target)
            continue
        seen_attack.add(key)
        report.attack_commands.setdefault(enemy_id, []).append(actor)
        estimate = context.fire_estimate(actor, enemy_id)
        if estimate.window_shots >= 1:
            report.window_sources.setdefault(enemy_id, []).append(actor)
            if estimate.damage_est is not None:
                report.damage_known[enemy_id] = \
                    report.damage_known.get(enemy_id, 0.0) + estimate.damage_est
                lo, hi = report.damage_bounds.get(enemy_id, (0.0, 0.0))
                report.damage_bounds[enemy_id] = (
                    lo + (estimate.damage_lo if estimate.damage_lo is not None
                          else estimate.damage_est),
                    hi + (estimate.damage_hi if estimate.damage_hi is not None
                          else estimate.damage_est))
            else:
                # window opportunity with unknown magnitude
                report.unknown_sources.setdefault(enemy_id, []).append(actor)
        else:
            # known-not-this-window (cooldown out of range of H) or unknown
            # contact: zero contribution now, no cancellation of others.
            if estimate.damage_est is None:
                report.unknown_sources.setdefault(enemy_id, []).append(actor)
    # ---- 3. heal opportunities (B1/B4) -------------------------------------
    owners = context.heal_occupancy()
    assigned: Dict[str, str] = {}
    for actor, action in actions.items():
        unit = snapshot.allies.get(actor)
        if unit is None or not unit.alive \
                or not context.capabilities.get(actor, object()).healer:
            continue
        semantic = context.decode(actor, action)
        if semantic.kind not in ("heal", KEEP_HEAL_ORDER) or not semantic.target:
            continue
        patient = semantic.target
        target = snapshot.allies.get(patient)
        if target is None or not (0 < target.health < target.max_health):
            continue
        # protocol capacity: observed orders keep ownership until a later
        # observation confirms release; at most ONE active source per patient
        observed_owner = owners.get(patient)
        if observed_owner is not None and observed_owner != actor:
            continue
        if patient in assigned and assigned[patient] != actor:
            # protocol capacity (one active healer per patient): the later
            # source gets NO credit; the conflict is recorded so the
            # validator can reject the plan (D8 true constraint)
            report.heal_capacity_conflicts[patient] = (assigned[patient], actor)
            continue
        assigned[patient] = actor
        # need: missing HEALTH only — 1 missing and 40 missing differ
        need = min((target.max_health - target.health)
                   / max(target.max_health, 1.0), 1.0)
        # sustain: relative predicted position of healer vs patient (B4)
        healer_pos = report.positions.get(actor, (unit.position, True))[0]
        patient_action = actions.get(patient)
        patient_moving_away = False
        if patient_action is not None:
            patient_pred = report.positions.get(patient,
                                                 (target.position, True))[0]
            gap_now = distance(unit.position, target.position)
            gap_after = distance(healer_pos, patient_pred)
            patient_moving_away = gap_after > gap_now + 0.25
        sustain = 0.35 if patient_moving_away else 0.6
        pressure = len(context.threats_to(patient))
        urgency = min(0.6 * need + (0.2 if target.health_lost > 0 else 0.0)
                      + 0.1 * min(pressure, 2), 1.0)
        opportunity = need * sustain * (0.5 + 0.5 * urgency)
        if patient in report.heal:     # keep vs re-issue: same base opportunity
            existing = report.heal[patient]
            report.heal[patient] = HealOpportunity(
                patient, existing.healer, need, sustain, urgency,
                max(existing.opportunity, opportunity), patient_moving_away)
        else:
            report.heal[patient] = HealOpportunity(
                patient, actor, need, sustain, urgency, opportunity,
                patient_moving_away)
    report.new_heal_assignments = assigned
    # ---- 4. coverage opportunities from window-qualified mechanism sources
    credited = set()   # (actor, target): one coverage credit per source/target
    for enemy_id, actors in report.window_sources.items():
        for actor in actors:
            capability = context.capabilities.get(actor)
            if capability is None:
                continue
            if capability.specialization == "splash_fire" and mechanisms["splash"]:
                radius, weight = splash_radius, splash_weight
            elif capability.suicide_attack and mechanisms["burst"]:
                radius, weight = burst_radius, burst_weight
            else:
                continue
            primary = snapshot.enemies[enemy_id]
            for other_id, other in sorted(snapshot.enemies.items()):
                # colossus/baneling mechanisms are GROUND-domain in this
                # roster: flying neighbours never enter coverage (B7)
                if other.is_flying:
                    continue
                if other_id == enemy_id and report.damage_known.get(enemy_id, 0.0) > 0:
                    continue   # known single-target damage is not re-packaged
                if distance(other.position, primary.position) <= radius:
                    key = (actor, other_id)
                    if key in credited:
                        continue
                    credited.add(key)
                    report.coverage_sources.setdefault(other_id, []).append(actor)
                    report.coverage[other_id] = min(
                        1.0, report.coverage.get(other_id, 0.0) + weight)
    for enemy_id in list(report.coverage):
        if report.damage_known.get(enemy_id, 0.0) >= \
                snapshot.enemies[enemy_id].effective_health:
            report.coverage[enemy_id] *= 0.3   # ordinary fire already suffices
    # ---- 5. planned self-consumption: only an in-contact detonation (D9) --
    for actor, action in actions.items():
        unit = snapshot.allies.get(actor)
        capability = context.capabilities.get(actor)
        if unit is None or capability is None or not capability.suicide_attack:
            continue
        semantic = context.decode(actor, action)
        if semantic.kind == "attack" and semantic.target in snapshot.enemies \
                and context.contact_status(actor, semantic.target) == IN_RANGE:
            report.planned_loss[actor] = burst_self_cost   # future planned cost, once
    return report


