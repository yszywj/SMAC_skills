"""Energy-qualified, bounded rescue opportunities, never inferred heal HP.

The native catalog does not expose a verified healing rate here. An order is
ownership evidence, not proof of an active channel; neither STOP nor an
AlreadyTargeted response proves an interruption. Predictions below only rank
opportunities and never subtract incoming damage or add health to the world.
"""
import math

from .context import GAME_LOOPS_PER_SPEED_SECOND, distance
from .actions import STOP, KEEP_HEAL_ORDER
from .combat_exchange import exposure_window


def healing_energy(context, healer):
    """Exact native energy when supplied; otherwise the known 2dp interval.

    A rounded 1.00 can represent less than one native energy. This is an
    observation-precision boundary, not a newly invented ability cost.
    """
    unit = context.snapshot.allies.get(healer)
    raw = context.snapshot.raw_state.get("allies", {}).get(healer, {})
    exact = "energy_raw" in raw
    value = raw.get("energy_raw") if exact else raw.get("energy")
    source = "native_raw" if exact else "rounded_2dp"
    if not unit or not unit.alive or isinstance(value, bool) \
            or not isinstance(value, (int, float)) or not math.isfinite(value):
        return {"status": "unknown", "source": source, "lower": None, "upper": None}
    lower, upper = (float(value), float(value)) if exact else (float(value) - .005, float(value) + .005)
    status = "sufficient" if lower >= 1. else "insufficient" if upper < 1. else "uncertain"
    return {"status": status, "source": source, "lower": lower, "upper": upper}


def energy_ready(context, healer):
    return healing_energy(context, healer)["status"] == "sufficient"


def eligible_patient(context, patient):
    unit = context.snapshot.allies.get(patient)
    if not unit or not unit.alive or unit.is_flying or not 0 < unit.health < unit.max_health:
        return False
    # The known native roster's biological ground units. Unknown identities
    # require the catalog's Biological attribute (SC2 Attribute enum value 3).
    if unit.type_name in ("marine", "marauder", "zealot", "zergling", "hydralisk", "baneling"):
        return True
    profile = context.snapshot.raw_state.get("unit_type_profiles", {}).get(str(unit.unit_type_id), {})
    return 3 in profile.get("attributes", ()) and not context.capabilities[patient].healer


def observed_patient(context, healer):
    orders = context.snapshot.raw_state.get("allies", {}).get(healer, {}).get("orders") or []
    patient = orders[0].get("target_unit") if orders and orders[0].get("kind") == "heal" else None
    unit = context.snapshot.allies.get(patient)
    return patient if unit and unit.alive else None


def heal_reach(context, healer):
    unit = context.snapshot.allies[healer]
    value = context.snapshot.raw_state.get("action_contract", {}).get(
        "candidate_center_distance_by_type", {}).get(unit.type_name)
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def _near_term_output_weight(context, patient, enemy, reach):
    """Discount scarcity by one possible shot within the shared two-H window.

    Static-target travel is an opportunity estimate, never promised damage.
    Unknown cooldown/speed cannot manufacture a ready or reachable shot.
    """
    unit = context.snapshot.allies[patient]
    raw = context.snapshot.raw_state.get("allies", {}).get(patient, {})
    cooldown = raw.get("weapon_cooldown_loops")
    if isinstance(cooldown, bool) or not isinstance(cooldown, (int, float)) \
            or not math.isfinite(cooldown) or cooldown < 0:
        return 0.
    gap = max(0., distance(unit.position, context.snapshot.enemies[enemy].position) - reach)
    travel = 0.
    if gap > 1e-6:
        speed = context.speed(patient)
        if speed is None or not context.move_destinations(patient):
            return 0.
        travel = gap * GAME_LOOPS_PER_SPEED_SECOND / speed
    elif not context.can_issue_attack(patient, enemy):
        return 0.
    eta = max(travel, float(cooldown))
    return max(0., 1. - eta / (2 * max(context.H, 1)))


def capability_retention(context, patient):
    """Bounded scarcity of a near-term attack opportunity, without unit names."""
    cache = getattr(context, "_heal_capability_cache", None)
    if cache is None:
        cache = context._heal_capability_cache = {}
    if patient in cache:
        return cache[patient]
    value = 0.
    for enemy in context.snapshot.enemies:
        compatible = []
        for actor, unit in context.snapshot.allies.items():
            if not unit.alive or context.capabilities[actor].healer:
                continue
            reach = context.effective_attack_reach(actor, enemy)
            if reach is not None and reach >= 0:
                compatible.append(actor)
        if patient in compatible:
            reach = context.effective_attack_reach(patient, enemy)
            opportunity = _near_term_output_weight(context, patient, enemy, reach)
            value = max(value, opportunity / len(compatible))
    cache[patient] = value
    return value


def rescue_priority(context, patient):
    if not eligible_patient(context, patient):
        return 0.
    unit = context.snapshot.allies[patient]
    need = (unit.max_health - unit.health) / max(unit.max_health, 1.)
    return need * (1. + capability_retention(context, patient))


def healing_opportunity(context, healer, patient, positions=None):
    """Return need/sustain/urgency/opportunity/moving-away, all heuristic.

    Only a legal healing action or explicit retain-order extension can invoke
    this through effects. Energy is checked again to cover historical payloads
    and final-action completion. No HP/s is invented from net health changes.
    """
    if not energy_ready(context, healer) or not eligible_patient(context, patient):
        return None
    h, p = context.snapshot.allies[healer], context.snapshot.allies[patient]
    positions = positions or {}
    hp = positions.get(healer, (h.position, True))[0]
    pp = positions.get(patient, (p.position, True))[0]
    away = distance(hp, pp) > distance(h.position, p.position) + .25
    need = min((p.max_health - p.health) / max(p.max_health, 1.), 1.)
    sustain = .35 if away else .6
    # More imminent exposure raises urgency, but cannot turn a doomed patient
    # into a guaranteed save. A small opportunity remains for direct treatment.
    exposure = exposure_window(context, patient, pp)["amount"]
    viability = min(1., p.effective_health / max(exposure, 1.))
    urgency = min(.6 * need + (.2 if p.health_lost > 0 else 0.)
                  + .1 * min(len(context.threats_to(patient)), 2), 1.)
    opportunity = need * sustain * (.5 + .5 * urgency) * viability
    opportunity *= 1. + capability_retention(context, patient)
    return need, sustain, urgency, min(opportunity, 1.), away


def _service_fact(context, healer, patient, kind, positions=None):
    if not energy_ready(context, healer) or not eligible_patient(context, patient) \
            or context.heal_action(healer, patient) is None:
        return None
    owners = context.heal_order_owners().get(patient, ())
    preferred = context.heal_occupancy().get(patient)
    if preferred not in (None, healer) and not (kind == "observed_retained" and healer in owners):
        return None
    h, p = context.snapshot.allies[healer], context.snapshot.allies[patient]
    positions = positions or {}
    hp, hc = positions.get(healer, (h.position, True))
    pp, pc = positions.get(patient, (p.position, True))
    reach = heal_reach(context, healer)
    # A move/auto-approach with an unmodelled path cannot promise an actual
    # service window. This does not prohibit the ordinary movement candidate.
    if reach is None or not hc or not pc or distance(hp, pp) > reach + 1e-6:
        return None
    value = healing_opportunity(context, healer, patient, positions)
    if value is None:
        return None
    retained = kind == "observed_retained"
    return {"healer": healer, "patient": patient, "kind": kind, "eta_h": 0.,
            "observed_owners": tuple(sorted(owners)),
            "energy_source": healing_energy(context, healer)["source"],
            # Retained STOP service is an observed-order opportunity, not a
            # verified tick. The engineering discount is not a native rate.
            "opportunity": value[3] * (.5 if retained else 1.),
            "evidence": "bounded_service_opportunity_not_heal_hp"}


def potential_healing_services(context):
    """Patient -> finite current legal service candidates, not assignments."""
    services = {}
    for healer, capability in sorted(context.capabilities.items()):
        if not capability.healer:
            continue
        for patient in sorted(context.snapshot.allies):
            fact = _service_fact(context, healer, patient, "potential")
            if fact is not None:
                services.setdefault(patient, []).append(fact)
    return {patient: tuple(facts) for patient, facts in services.items()}


def planned_healing_services(context, actions):
    """Patient -> at most one service supported by FINAL actions and owners.

    Missing actions and MOVE do not retain a promised service. STOP only
    supports a fresh first observed heal order; native auto-targeting a new
    patient is never fabricated. No service adds HP or cancels enemy damage.
    """
    if isinstance(actions, (list, tuple)):
        actions = {f"u{i}": action for i, action in enumerate(actions)}
    candidates, submitted = {}, {}
    for healer, action in sorted(actions.items()):
        capability = context.capabilities.get(healer)
        if not capability or not capability.healer or action not in context.legal_actions(healer):
            continue
        semantic = context.decode(healer, action)
        if semantic.kind in ("heal", KEEP_HEAL_ORDER):
            patient, kind = semantic.target, "submitted_heal"
            submitted.setdefault(patient, []).append(healer)
        elif action == STOP:
            patient, kind = observed_patient(context, healer), "observed_retained"
        else:
            continue
        if patient not in context.snapshot.allies or patient not in actions:
            continue
        positions = {}
        for actor in (healer, patient):
            if actor in actions:
                predicted = context.predict_position(actor, actions[actor])
                positions[actor] = (predicted.position, predicted.certain)
        fact = _service_fact(context, healer, patient, kind, positions)
        if fact is not None:
            candidates.setdefault(patient, []).append(fact)
    return {patient: min(facts, key=lambda fact: (
                fact["kind"] != "submitted_heal",
                fact["healer"] != context.heal_occupancy().get(patient),
                -fact["opportunity"], fact["healer"]))
            for patient, facts in candidates.items() if len(submitted.get(patient, ())) <= 1}


def observed_healing_services(context):
    """Fresh observed services for candidate generation, not final selection."""
    return planned_healing_services(context, {
        actor: STOP for actor, unit in context.snapshot.allies.items() if unit.alive})


def _arrival_h(context, healer, patient, hp, pp):
    reach, speed = heal_reach(context, healer), context.speed(healer)
    if reach is None or speed is None or speed <= 0:
        return math.inf
    return max(0., distance(hp, pp) - reach) / (speed * max(context.H, 1) / GAME_LOOPS_PER_SPEED_SECOND)


def approach_facts(context, healer, patient, actions, max_arrival_h=6.):
    """Re-evaluate the final pair's displacement and conservative arrival.

    A patient's movement needs a real ETA advantage over the same healer move
    alone, no ready legal shot, and no greater exposure. Existing legal heals
    never earn distance-closing credit. Unknown travel speed cannot promise a
    rescue. The six-window bound is the existing S2 approach deadline.
    """
    if not energy_ready(context, healer) or not eligible_patient(context, patient):
        return None
    owner = context.heal_occupancy().get(patient)
    if owner not in (None, healer) or context.heal_action(healer, patient) is not None:
        return None
    service = planned_healing_services(context, actions).get(patient)
    if service is not None and service["healer"] != healer:
        return None             # another final action already serves this rescue
    h, p = context.snapshot.allies[healer], context.snapshot.allies[patient]
    ha = actions.get(healer)
    if ha not in context.move_destinations(healer):
        return None
    hpred = context.predict_position(healer, ha)
    pa = actions.get(patient)
    moving_patient = pa in context.move_destinations(patient)
    ppred = context.predict_position(patient, pa) if pa is not None else None
    pp = ppred.position if ppred else p.position
    before = _arrival_h(context, healer, patient, h.position, p.position)
    alone = _arrival_h(context, healer, patient, hpred.position, p.position)
    after = _arrival_h(context, healer, patient, hpred.position, pp)
    if not math.isfinite(before) or after >= before or 1. + after > max_arrival_h:
        return None
    risk_before = exposure_window(context, patient, horizon_h=min(3., 1. + before))
    risk_after = exposure_window(context, patient, pp, horizon_h=min(3., 1. + after),
                                 certain=ppred.certain if ppred else True)
    # Do not promise that uncalibrated future healing will cancel lethal
    # exposure before arrival. Ordinary attack/escape candidates still compete.
    if risk_after["amount"] >= p.effective_health or risk_after["effect_danger"]:
        return None
    if moving_patient:
        ready_contact = p.weapon_cooldown_loops <= context.H and bool(context.attackable_enemies(patient))
        if alone <= 0. or alone - after < .5 or ready_contact \
                or risk_after["amount"] > risk_before["amount"] + 1e-6:
            return None
    return {"before_h": before, "after_h": after, "alone_h": alone,
            "patient_moves": moving_patient, "evidence": "bounded_rescue_opportunity_uncalibrated"}


def healing_approach_gain(context, actions, plan, report):
    data = plan.goal_data or {}
    healer, patient = data.get("healer"), data.get("patient")
    if healer not in context.snapshot.allies or patient not in context.snapshot.allies:
        return 0.
    deadline = data.get("rescue_deadline_loop")
    remaining = (max(0., (deadline - context.snapshot.game_loop) / max(context.H, 1))
                 if isinstance(deadline, (int, float)) else 6.)
    facts = approach_facts(context, healer, patient, actions, max_arrival_h=remaining)
    if facts is None:
        return 0.
    progress = min(1., max(0., facts["before_h"] - facts["after_h"]))
    return min(1., rescue_priority(context, patient) * progress / (1. + facts["after_h"]))


def rescue_deadline(context, memory, healer, patient):
    old = memory.instance_for("heal_support", healer)
    if old and old.phase == "meet_patient" and old.goal == patient and old.max_loops is not None:
        return old.start_loop + old.max_loops
    return context.snapshot.game_loop + 6 * context.H


def approach_options(context, memory, healers):
    """Primary matching plus one alternative per healer, within the budget.

    Stability is a bounded relative bonus, not lexicographic precedence over
    an urgent feasible rescue. Future patient sets remain disjoint across
    healers; these proposals do not consume an actual healing-capacity slot.
    """
    owners = context.heal_occupancy()
    result, used = {}, set()
    edges = []
    for healer in healers:
        if not energy_ready(context, healer):
            continue
        old = memory.instance_for("heal_support", healer)
        for patient, unit in context.snapshot.allies.items():
            if not eligible_patient(context, patient) or owners.get(patient) not in (None, healer):
                continue
            eta = _arrival_h(context, healer, patient, context.snapshot.allies[healer].position, unit.position)
            remaining = max(0., (rescue_deadline(context, memory, healer, patient)
                                - context.snapshot.game_loop) / max(context.H, 1))
            if not math.isfinite(eta) or eta > remaining:
                continue
            if context.heal_action(healer, patient) is None:
                moves = context.move_destinations(healer)
                if not moves:
                    continue
                solo = min(moves, key=lambda a: (distance(moves[a], unit.position), a))
                feasible = approach_facts(context, healer, patient, {healer: solo}, remaining)
                if feasible is None:
                    origin = context.snapshot.allies[healer].position
                    midpoint = tuple((origin[i] + unit.position[i]) / 2. for i in (0, 1))
                    patient_moves = context.move_destinations(patient)
                    if patient_moves:
                        hm = min(moves, key=lambda a: (distance(moves[a], midpoint), a))
                        pm = min(patient_moves, key=lambda a: (distance(patient_moves[a], midpoint), a))
                        feasible = approach_facts(context, healer, patient, {healer: hm, patient: pm}, remaining)
                if feasible is None:
                    continue
            stable = observed_patient(context, healer) == patient or bool(old and old.goal == patient)
            # Reuse S2's existing switch margin, capped to keep continuation
            # from becoming an absolute assignment lock.
            config = getattr(memory, "encounter_config", {}) or {}
            margin = min(.5, max(0., config.get("heal_support", {}).get("switch_margin", .3)))
            score = rescue_priority(context, patient) / (1. + eta) * (1. + margin if stable else 1.)
            edges.append((-score, eta, healer, patient, stable))
    for _, _, healer, patient, _ in sorted(edges):
        if healer not in result and patient not in used:
            result[healer] = [patient]
            used.add(patient)
    # Prefer a still-feasible continuation as the alternative when the primary
    # now favors a more valuable rescue; otherwise keep the best unused edge.
    for _, _, healer, patient, _ in sorted(edges, key=lambda edge: (not edge[4], *edge[:4])):
        if healer in result and len(result[healer]) < 2 and patient not in used:
            result[healer].append(patient)
            used.add(patient)
    return {healer: tuple(patients) for healer, patients in result.items()}


def approach_assignments(context, memory, healers):
    """Compatibility view of the primary unique allocation."""
    return {healer: patients[0] for healer, patients in approach_options(context, memory, healers).items()}
