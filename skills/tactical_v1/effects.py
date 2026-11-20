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
from .healing import healing_opportunity
from .burst import BurstOutcome, affects, damage as burst_damage


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
    coverage_raw: Dict[str, float] = field(default_factory=dict)
    coverage_sources: Dict[str, List[str]] = field(default_factory=dict)
    positions: Dict[str, Tuple[Tuple[float, float], bool]] = field(default_factory=dict)
    planned_loss: Dict[str, float] = field(default_factory=dict)  # in-contact detonation only
    burst_outcomes: Dict[str, BurstOutcome] = field(default_factory=dict)
    # Potential effect opportunities, never ordinary confirmed fire windows.
    burst_damage_potential: Dict[str, float] = field(default_factory=dict)
    burst_unknown_sources: Dict[str, List[str]] = field(default_factory=dict)
    enemy_death_burst: Dict[str, float] = field(default_factory=dict)
    enemy_death_burst_unknown: Dict[str, List[str]] = field(default_factory=dict)


def physical_effects(context: Context, actions: Dict[str, int],
                     mechanisms: Optional[Dict[str, bool]] = None,
                     splash_radius: float = 3.0, splash_weight: float = 0.5,
                     burst_radius: float = 2.5, burst_weight: float = 0.5,
                     burst_self_cost: float = 1.0,
                     splash_reliability: float = 1.0) -> EffectsReport:
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
        if context.capabilities[actor].suicide_attack:
            from .burst import action_position as burst_position
            report.positions[actor] = (burst_position(context, actor, action), False)
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
        if context.capabilities[actor].suicide_attack:
            # An empty native weapon list is handled by the one-use effect
            # model below. Never also count it as an ordinary volley.
            continue
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
        opportunity_data = healing_opportunity(context, actor, patient, report.positions)
        if opportunity_data is None:
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
        # Same energy, biological eligibility, final-position exposure and
        # capability-retention model as S2. This remains opportunity, not HP.
        need, sustain, urgency, opportunity, patient_moving_away = opportunity_data
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
                raw_weight = weight
                weight *= max(0., min(1., splash_reliability))
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
                    report.coverage_raw[other_id] = min(
                        1.0, report.coverage_raw.get(other_id, 0.0) + raw_weight)
    for enemy_id in list(report.coverage):
        if report.damage_known.get(enemy_id, 0.0) >= \
                snapshot.enemies[enemy_id].effective_health:
            report.coverage[enemy_id] *= 0.3   # ordinary fire already suffices
            report.coverage_raw[enemy_id] *= 0.3
    # ---- 5. one-use effects at actual/projected source centers (D9) ------
    for actor, action in actions.items():
        unit = snapshot.allies.get(actor)
        capability = context.capabilities.get(actor)
        if unit is None or not unit.alive or capability is None \
                or not capability.suicide_attack or not mechanisms["burst"]:
            continue
        outcome = context.burst_outcome(actor, action, report.positions[actor][0])
        report.burst_outcomes[actor] = outcome
        if outcome.opportunity <= 0.:
            continue
        report.planned_loss[actor] = burst_self_cost * outcome.opportunity
        for enemy_id, amount in outcome.damage_potential.items():
            if amount is None:
                report.burst_unknown_sources.setdefault(enemy_id, []).append(actor)
            else:
                report.burst_damage_potential[enemy_id] = (
                    report.burst_damage_potential.get(enemy_id, 0.)
                    + amount * outcome.opportunity)
    # Killing an enemy Baneling does not erase the same local explosion.
    # A potential combined burst kill also carries conditional retaliation.
    # It remains discounted exposure, never ordinary damage or a known kill.
    # (Tasks 4-6) Death sources are built per DAMAGED target — including
    # splash SIDE targets recorded in damage_potential — then accumulated in
    # time order per bomber: a death scenario forms only at the first moment
    # when the sources ALLOWED at or before that moment reach the target's
    # remaining effective health. Ally coverage is judged with BOTH sides at
    # that same moment. Multiple scenarios per bomber are mutually
    # exclusive alternatives: each ally takes the MAX predicted loss across
    # them, never their sum.
    _build_enemy_death_bursts(context, snapshot, actions, report,
                              burst_self_cost, mechanisms)
    return report


def _death_sources_for_bomber(context, snapshot, actions, report):
    """Local per-bomber source table (task 4): predicted, not observed.

    For each enemy bomber EVERY damaging source is collected — burst
    outcomes via their FULL damage_potential (so splash side targets count
    even when the bomber was not the attack's primary target) plus
    ordinary window-qualified shooters from the FINAL actions. Fields:
      source_actor, victim_enemy_id, damage, opportunity,
      evaluation_after_loops, timing_kind ('predicted' | 'unknown'),
      raw_damage (pre-discount potential, kept separate from the
      opportunity-discounted amount — an uncalibrated proxy, NOT a
      calibrated probability). Unknown damage is recorded as unknown, never
      converted to zero. One entry per (source, victim); no benefit ledgers
      are touched here.
    """
    sources = {}
    for actor, result in report.burst_outcomes.items():
        if result.opportunity <= 0.:
            continue
        for enemy_id, amount in result.damage_potential.items():
            if enemy_id not in snapshot.enemies \
                    or not context.enemy_capabilities[enemy_id].suicide_attack:
                continue
            if amount is not None and amount <= 0.:
                continue
            sources.setdefault(enemy_id, []).append({
                'source_actor': actor, 'victim_enemy_id': enemy_id,
                'damage': None if amount is None else amount * result.opportunity,
                'raw_damage': amount,
                'opportunity': result.opportunity,
                'evaluation_after_loops': max(0., result.evaluation_after_loops),
                'timing_kind': 'predicted', 'outcome_actor': actor})
    for enemy_id, shooters in report.window_sources.items():
        if enemy_id not in snapshot.enemies \
                or not context.enemy_capabilities[enemy_id].suicide_attack:
            continue
        rows = sources.setdefault(enemy_id, [])
        seen_sources = {row['source_actor'] for row in rows}
        for actor in shooters:
            if actor in seen_sources:
                continue
            estimate = context.fire_estimate(actor, enemy_id)
            if estimate.window_shots < 1:
                continue
            seen_sources.add(actor)
            # Cooldown gives a readiness lower bound for both timing scenarios.
            rows.append({
                'source_actor': actor, 'victim_enemy_id': enemy_id,
                'damage': estimate.damage_est, 'raw_damage': estimate.damage_est,
                'opportunity': 1.,
                'evaluation_after_loops': max(0., estimate.effective_after_loops or 0.),
                'timing_kind': 'unknown', 'outcome_actor': None})
    return sources


def _bomber_death_scenarios(context, snapshot, sources, enemy_id):
    """Conditional death scenarios for ONE bomber (task 5).

    Returns a list of (moment, contributing_rows). Sources with a predicted
    moment are grouped in time order; a scenario forms at the FIRST moment
    when the cumulative allowed damage reaches the target's remaining
    effective health — later-arriving sources can NEVER contribute to an
    earlier moment. 'unknown' ordinary shots (no impact upper bound) run as
    two clearly-marked alternative scenarios (earliest-lower-bound vs.
    window-end), each counting each shot once; both are conditional model
    assumptions, not confirmed kills and not a basis for strong safety
    claims. No data at all -> [].
    """
    remaining = snapshot.enemies[enemy_id].effective_health
    if remaining <= 0.:
        return []
    known = [row for row in sources.get(enemy_id, ()) if row['damage'] is not None]
    has_shots = any(row['timing_kind'] == 'unknown' for row in known)
    scenarios = []
    for shot_time_floor in ((0., context.H) if has_shots else (0.,)):
        by_moment = {}
        for row in known:
            moment = row['evaluation_after_loops']
            if row['timing_kind'] == 'unknown':
                moment = max(moment, shot_time_floor)
            by_moment.setdefault(moment, []).append(row)

        cumulative = 0.
        contributing = []
        for moment in sorted(by_moment):
            hits = by_moment[moment]
            contributing.extend(hits)
            cumulative += sum(max(0., row['damage']) for row in hits)
            if cumulative >= remaining:
                scenario = (moment, contributing)
                if scenario not in scenarios:
                    scenarios.append(scenario)
                break
    return scenarios


def _build_enemy_death_bursts(context, snapshot, actions, report,
                              burst_self_cost, mechanisms):
    """One final risk account per enemy bomber (tasks 4-6).

    For each bomber's FIRST forming scenario (and the two marked
    alternatives when ordinary-shot timing is unknown), both sides are
    placed at THAT scenario's moment: enemies via a moment-matched position
    query (the claiming outcome's own map when it shares the moment,
    otherwise a fresh query), allies via burst.action_position_at on the
    FINAL actions. Per bomber and ally the predicted losses across
    alternative scenarios are merged by MAX — never summed across moments.
    report.positions keeps its original whole-window semantics for the rest
    of joint scoring; this block only reads moment-local queries.
    """
    from .burst import action_position_at, _predicted_enemy_positions
    effect_profile = context.burst_profile()
    radius = effect_profile.blast_radius if effect_profile.geometry_known else 0.
    if radius <= 0.:
        return
    H = max(context.H, 1)
    sources = _death_sources_for_bomber(context, snapshot, actions, report)
    for enemy_id in sorted(sources):
        enemy = snapshot.enemies[enemy_id]
        potential = sum(max(0., result.damage_potential.get(enemy_id) or 0.)
                        for result in report.burst_outcomes.values()
                        if result.opportunity > 0.)
        ordinary = report.damage_known.get(enemy_id, 0.)
        if potential + ordinary < enemy.effective_health \
                and not any(row['damage'] is None for row in sources[enemy_id]):
            continue               # model fire never reaches lethal: no scenario
        scenarios = _bomber_death_scenarios(context, snapshot, sources, enemy_id)
        if not scenarios:
            continue
        # Outcome whose evaluation moment equals the scenario moment can
        # lend its own enemy map; otherwise query the moment directly.
        outcome_by_moment = {}
        for result in report.burst_outcomes.values():
            if result.enemy_positions is not None and result.opportunity > 0.:
                outcome_by_moment.setdefault(
                    round(result.evaluation_after_loops, 6), result)
        per_ally = {}
        unknown_victims = set()
        for moment, contributing in scenarios:
            source_outcome = outcome_by_moment.get(round(moment, 6))
            if source_outcome is not None:
                enemy_map = source_outcome.enemy_positions
                fallbacks = set(source_outcome.static_fallback_enemies)
            else:
                enemy_map, _motion, fallbacks = _predicted_enemy_positions(
                    context, context.snapshot.game_loop + moment)
                fallbacks = set(fallbacks)
            enemy_point = enemy_map.get(enemy_id, enemy.position)
            raw_loss = sum(max(0., row['raw_damage'] or 0.) for row in contributing)
            discounted = sum(max(0., row['damage'] or 0.) for row in contributing)
            for actor, action in actions.items():
                ally = snapshot.allies.get(actor)
                if ally is None or not ally.alive or ally.is_flying:
                    continue
                ally_point, _uncertain = action_position_at(context, actor,
                                                            action, moment)
                amount = burst_damage(context, actor, enemy_target=False)
                if amount is None:
                    unknown_victims.add(actor)
                    continue
                if distance(ally_point, enemy_point) > radius + ally.radius:
                    continue
                weight = min(1., discounted / max(enemy.effective_health, 1.))
                row = per_ally.setdefault(actor, {
                    'max_loss': 0., 'raw_loss': 0., 'moments': []})
                # MAX across mutually exclusive scenarios, never a sum.
                row['max_loss'] = max(row['max_loss'], min(amount, ally.effective_health) * weight)
                row['raw_loss'] = max(row['raw_loss'], min(raw_loss, raw_loss))
                row['moments'].append(round(moment, 3))
        for actor, row in per_ally.items():
            report.enemy_death_burst[actor] = \
                report.enemy_death_burst.get(actor, 0.) + row['max_loss']
        for actor in unknown_victims:
            report.enemy_death_burst_unknown.setdefault(actor, []).append(enemy_id)


