"""Shared opening-deployment facts for rally/engage/rejoin/formation skills.

Pure snapshot-local queries cached on the Context. They NEVER create, renew
or commit tasks, never reserve burst budgets, and never read anything beyond
the visible snapshot plus read-only memory history. Skills and the final
duty layer read the SAME phase/readiness facts, so candidate generation,
joint scoring, repair and baseline fallback cannot silently disagree about
whether organization is still required.

Semantics:
  ASSEMBLE       no squad member is engaged; local organization continues;
  COVERED_JOIN   some members are engaged — they keep firing (never STOPped
                 by the organizer); the rest close to protected support;
  ENGAGE         mutually supporting fire/contact conditions hold.
Phase is derived from OBSERVATIONS (contact, damage, threats) and executable
actions only. "Predicted arrival in a few steps" is never counted as current
support. Healers are never required to attack to make a squad ready; melee-
only and ranged-only squads use their own reachable contact/fire bands.
"""
from .context import GAME_LOOPS_PER_SPEED_SECOND, distance

ASSEMBLE, COVERED_JOIN, ENGAGE = "ASSEMBLE", "COVERED_JOIN", "ENGAGE"

# Geometry priors for DEPLOYMENT decisions only (not combat claims): when
# the mean nearest-neighbour gap between ordinary ground allies exceeds this,
# a dispersed opening still needs organization even without visible enemies.
DISPERSED_SPACING = 4.0
# A squad member counts as supportable when it can bring real fire onto a
# shared front within this many control windows.
SUPPORT_ETA_H = 2.0
# Support disconnection threshold: a pending member farther than this from
# EVERY fighting member is genuinely detached. Members within it are already
# mutually supporting — a clustered team keeps advancing (ENGAGE), and is
# never pulled back inward just because a melee ETA exceeds a threshold.
SUPPORT_DISCONNECT = 8.0


def _ordinary(context):
    return tuple(a for a, unit in sorted(context.snapshot.allies.items())
                 if unit.alive and not unit.is_flying
                 and not context.capabilities[a].healer
                 and not context.capabilities[a].suicide_attack)


def geometry_fraction(context, members):
    """One local support neighbourhood, not two disconnected close pairs."""
    positions = [context.snapshot.allies[a].position for a in members]
    if not positions:
        return 1.
    return max(sum(distance(p, q) <= DISPERSED_SPACING + 2.
                   for q in positions) for p in positions) / len(positions)


def engaged_members(context, actors):
    """Members with observed contact, damage or current threat coverage."""
    engaged = []
    for actor in actors:
        unit = context.snapshot.allies.get(actor)
        if unit is None or not unit.alive:
            continue
        if unit.health_lost + unit.shield_lost > 0:
            engaged.append(actor)
            continue
        risk = context.risk_at(actor, unit.position)
        if risk.threats or risk.effect_weapon_near:
            engaged.append(actor)
            continue
        if any(context.contact_status(actor, e) == "in_range"
               for e in context.attackable_enemies(actor)):
            engaged.append(actor)
    return tuple(engaged)


def fighting_members(context, actors):
    """Members with OBSERVED fighting evidence: damage taken or a real
    in-range contact. Merely standing inside somebody's possible weapon
    envelope is a conservative upper bound, not observed engagement —
    treating it as engagement is what pulled clustered approaching teams
    back into a rear anchor."""
    fighting = []
    for actor in actors:
        unit = context.snapshot.allies.get(actor)
        if unit is None or not unit.alive:
            continue
        if unit.health_lost + unit.shield_lost > 0:
            fighting.append(actor)
            continue
        if any(context.contact_status(actor, e) == "in_range"
               for e in context.attackable_enemies(actor)):
            fighting.append(actor)
    return tuple(fighting)


def squads(context):
    """Small mutually-supportable squads, stable within this snapshot.

    Membership is computed from COMPLETE-LINK support groups (no transitive
    chains across the map). Squad identity for diagnostics uses unit tags so
    the coordinator can log membership without task effects.
    """
    cached = getattr(context, "_deployment_squads", None)
    if cached is not None:
        return cached
    ordinary = _ordinary(context)
    memory = getattr(context, 'deployment_memory', None)
    frozen, occupied = [], set()
    for task in sorted(getattr(memory, 'instances', {}).values(), key=lambda t: t.instance_id):
        if task.skill == 'precombat_rally' and not task.termination_reason:
            members = tuple(a for a in task.actors if a in ordinary and a not in occupied)
            if members:
                frozen.append(members)
                occupied.update(members)
    groups = [tuple(a for a in g if a in ordinary and a not in occupied)
              for g in context.support_groups(radius=8.0)]
    groups = [g for g in groups if g]
    # Give isolated actors/pairs a nearby partner. Never require the whole
    # map to converge; established task memberships remain frozen.
    while len(groups) > 1:
        options = []
        for i, left in enumerate(groups):
            if len(left) >= 3:
                continue
            for j, right in enumerate(groups):
                if i == j or len(left) + len(right) > 5:
                    continue
                gap = max(distance(context.snapshot.allies[a].position,
                                   context.snapshot.allies[b].position)
                          for a in left for b in right)
                if gap <= 20.:
                    options.append((gap, i, j))
        if not options:
            break
        _, i, j = min(options)
        merged = tuple(sorted(groups[i] + groups[j]))
        groups = [g for k, g in enumerate(groups) if k not in (i, j)] + [merged]
    groups = frozen + groups
    result = []
    for index, group in enumerate(sorted(groups)):
        result.append({"id": 'squad:' + ':'.join(str(context.snapshot.allies[a].tag) for a in group), "members": group,
                       "ordinary": tuple(a for a in group if a in ordinary)})
    context._deployment_squads = result
    return result


def _fire_eta_h(context, actor, enemies):
    """Windows until this actor can land a shot on one of ``enemies``.

    Uses the real native/command reach and catalog speed; cooldown is part
    of the ETA. Unknown reach/speed is NOT optimistic (infinite).
    """
    unit = context.snapshot.allies[actor]
    best = float("inf")
    speed = context.speed(actor)
    for enemy_id in enemies:
        reach = context.effective_attack_reach(actor, enemy_id)
        if reach is None or reach < 0:
            continue
        gap = max(distance(unit.position,
                           context.snapshot.enemies[enemy_id].position) - reach, 0.)
        if speed:
            stride = speed * context.H / GAME_LOOPS_PER_SPEED_SECOND
            eta = gap / stride if stride > 0 else (0. if gap <= .05 else float("inf"))
        else:
            eta = 0. if gap <= .05 else float("inf")
        best = min(best, max(eta, unit.weapon_cooldown_loops / max(context.H, 1)))
    return best


def _contact_eta_h(context, actor, enemies):
    """Windows until a melee unit has a REACHABLE contact position."""
    unit = context.snapshot.allies[actor]
    speed = context.speed(actor)
    best = float("inf")
    for enemy_id in enemies:
        enemy = context.snapshot.enemies.get(enemy_id)
        if enemy is None or enemy.is_flying:
            continue
        reach = context.effective_attack_reach(actor, enemy_id)
        if reach is None or reach < 0:
            continue
        gap = max(distance(unit.position, enemy.position) - reach, 0.)
        if speed:
            stride = speed * context.H / GAME_LOOPS_PER_SPEED_SECOND
            eta = gap / stride if stride > 0 else (0. if gap <= .05 else float("inf"))
        else:
            eta = 0. if gap <= .05 else float("inf")
        best = min(best, eta)
    return best


def squad_readiness(context, members, fronts=None):
    """Capability-aware readiness: can the rear bring REAL fire soon?

    The legacy geometric radius band stays as one component; the decisive
    test is executable: each non-engaged member must be able to deliver a
    shot (ranged) or reach a contact slot (melee) on a shared front within
    SUPPORT_ETA_H windows. Pure melee and pure ranged squads both satisfy
    this through their own bands; healers are judged by proximity only and
    never need to attack.
    """
    ordinary = tuple(a for a in members
                     if a in context.snapshot.allies
                     and context.snapshot.allies[a].alive
                     and not context.capabilities[a].healer)
    if not ordinary:
        return {"ready": True, "fraction": 1.0, "reason": "no_ordinary_members"}
    enemies = tuple(fronts) if fronts else tuple(context.snapshot.enemies)
    engaged = set(fighting_members(context, ordinary))
    pending = tuple(a for a in ordinary if a not in engaged)
    if not enemies or not engaged:
        # No visible enemy: geometric mutual support only. An enemy-free
        # opening completes organization once bodies are mutually
        # supportable; combat metrics are NOT demanded.
        fraction = geometry_fraction(context, ordinary)
        return {"ready": fraction >= .8 and len(ordinary) >= min(2, len(_ordinary(context))), "fraction": round(fraction, 3),
                "reason": "no_visible_enemy_mutual_support_geometry",
                "engaged": tuple(sorted(engaged))}
    ready = 0
    reasons = {}
    for actor in pending:
        cap = context.capabilities[actor]
        actual = any(e in enemies and context.contact_status(actor, e) == 'in_range'
                     for e in context.attackable_enemies(actor))
        if actual:
            ready += 1
            continue
        # Geometry releases opening control in squad_phase; it must not be
        # reported as executable fire. G2/G6 still need the real ETA below.
        if cap.combat_class == "melee":
            eta = _contact_eta_h(context, actor, enemies)
        else:
            eta = _fire_eta_h(context, actor, enemies)
        # A straight-line ETA alone cannot establish reachable support.
        unit = context.snapshot.allies[actor]
        closer = any(
            any(distance(context.predict_position(actor, action).position,
                         context.snapshot.enemies[e].position) + .05
                < distance(unit.position, context.snapshot.enemies[e].position) for e in enemies)
            and safe_step(context, actor, context.predict_position(actor, action).position)
            for action in context.move_destinations(actor))
        if eta <= SUPPORT_ETA_H and closer:
            ready += 1
        else:
            reasons[actor] = f"eta_{eta:.1f}h"
    fraction = ready / len(pending) if pending else 1.0
    return {"ready": fraction >= .8, "fraction": round(fraction, 3),
            "pending": len(pending), "ready_fraction": fraction,
            "not_ready": reasons, "engaged": tuple(sorted(engaged)),
            "reason": "executable_rear_fire_or_contact"}


def squad_phase(context, squad):
    """ASSEMBLE / COVERED_JOIN / ENGAGE with explicit reasons (cached)."""
    cache = getattr(context, "_deployment_phases", None)
    if cache is None:
        cache = context._deployment_phases = {}
    key = squad["id"] if isinstance(squad, dict) else tuple(squad)
    if key in cache:
        return cache[key]
    members = squad["members"] if isinstance(squad, dict) else tuple(squad)
    ordinary = tuple(a for a in members
                     if a in context.snapshot.allies
                     and not context.capabilities[a].healer
                     and not context.capabilities[a].suicide_attack)
    fighting = fighting_members(context, ordinary)
    fronts = tuple(e for e, enemy in context.snapshot.enemies.items()
                   if any(distance(context.snapshot.allies[a].position,
                                   enemy.position) <= 14. for a in fighting)) \
        if fighting else ()
    readiness = squad_readiness(context, ordinary, fronts or None)
    # Support disconnection is a GEOMETRIC fact, not an ETA threshold: a
    # pending member genuinely detached only when it is beyond one support
    # radius from EVERY fighting member. A clustered team that is merely
    # still approaching (melee not yet in contact, rear outside firing ETA)
    # keeps advancing together as ENGAGE — G2/G6 own its positioning.
    detached = [a for a in ordinary if a not in fighting and all(
        distance(context.snapshot.allies[a].position,
                 context.snapshot.allies[f].position) > SUPPORT_DISCONNECT
        for f in fighting)] if fighting else []
    if fighting and detached and len(fighting) < len(ordinary):
        phase = COVERED_JOIN
    elif fighting:
        phase = ENGAGE
    else:
        phase = ENGAGE if readiness["ready"] else ASSEMBLE
    row = {"phase": phase, "readiness": readiness,
           "engaged": tuple(sorted(fighting)), "detached": tuple(sorted(detached)),
           "squad": key,
           "reason": readiness.get("reason", "")}
    cache[key] = row
    return row


def rally_anchor(context, members):
    """A SAFE, REACHABLE anchor: an existing member position (never a bare
    centroid that can sit inside enemy fire or off-path), preferring the
    position minimizing shared regroup distance among threat-free spots."""
    cached = getattr(context, "_deployment_anchors", None)
    if cached is None:
        cached = context._deployment_anchors = {}
    key = tuple(sorted(members))
    if key in cached:
        return cached[key]
    best = None
    for anchor_actor in sorted(members):
        anchor = context.snapshot.allies.get(anchor_actor)
        if anchor is None or not anchor.alive:
            continue
        risk = context.risk_at(anchor_actor, anchor.position)
        risky = bool(risk.threats or risk.effect_weapon_near)
        gaps = [distance(anchor.position, context.snapshot.allies[a].position)
                for a in members
                if a in context.snapshot.allies
                and context.snapshot.allies[a].alive]
        total = sum(max(gap - DISPERSED_SPACING, 0.) for gap in gaps)
        option = ((risky, total, anchor_actor), tuple(anchor.position))
        if best is None or option[0] < best[0]:
            best = (option[0], tuple(anchor.position), anchor_actor)
    result = {"position": best[1], "anchor_actor": best[2],
              "safe": not best[0][0]} if best else None
    cached[key] = result
    return result


def dispersed(context):
    """Mean nearest-neighbour gap among ordinary ground allies."""
    ordinary = _ordinary(context)
    if len(ordinary) < 2:
        return 0.
    positions = [context.snapshot.allies[a].position for a in ordinary]
    gaps = [min(distance(p, q) for q in positions if q is not p)
            for p in positions]
    return sum(gaps) / len(gaps)


def needs_deployment(context):
    """Opening organization is required without waiting for enemy vision.

    True when the ordinary ground force is dispersed beyond mutual support
    spacing. Enemy vision is NOT required and never fabricated here.
    """
    return dispersed(context) > DISPERSED_SPACING


def safe_step(context, actor, point):
    """Known route exposure, not an assertion that a command will succeed."""
    unit = context.snapshot.allies[actor]
    origin = context.forecast_risk(actor, unit.position)
    for p in (tuple((a + b) / 2. for a, b in zip(unit.position, point)), point):
        risk = context.forecast_risk(actor, p)
        values = [risk.damage_estimates.get(e) for e in risk.reachable_threats]
        if any(v is None for v in values) or sum(values) >= unit.effective_health:
            return False
        if set(risk.effect_weapon_near) - set(origin.effect_weapon_near):
            return False
    return True


def opening_available(context, memory):
    tracker = memory.encounters
    start = tracker.started_loop
    limit = float(memory.encounter_config.get('precombat_rally', {}).get('max_h', 8.))
    return (tracker.can_rally() and start is not None
            and context.snapshot.game_loop - start < limit * context.H)


def deployment_target(context, group, active=None):
    """Shared, stable target for plans, execution duties and observed progress.

    Keep the chosen fighting member until it leaves that front. Small motion
    does not move the waypoint every frame; a front displacement of one
    assembly radius refreshes it. This query never changes persistent state.
    """
    phase = squad_phase(context, group)
    data = active.extra if active else {}
    anchor, front_actor = None, None
    if phase['phase'] == COVERED_JOIN:
        fighting = phase['engaged']
        previous = data.get('front_actor')
        previous_unit = context.snapshot.allies.get(previous)
        valid_previous = previous_unit is not None and previous_unit.alive and any(
            distance(previous_unit.position, context.snapshot.allies[f].position)
            <= SUPPORT_DISCONNECT for f in fighting)
        if valid_previous:
            front_actor = previous
        else:
            front_actor = min(fighting, key=lambda f: (sum(
                distance(context.snapshot.allies[a].position,
                         context.snapshot.allies[f].position)
                for a in phase['detached']), f))
        current = tuple(context.snapshot.allies[front_actor].position)
        frozen = data.get('anchor')
        anchor = (tuple(frozen) if frozen is not None and previous == front_actor
                  and data.get('deployment_phase') == COVERED_JOIN
                  and distance(current, frozen) <= DISPERSED_SPACING else current)
    elif phase['phase'] == ASSEMBLE:
        frozen = data.get('anchor')
        if frozen is not None and data.get('deployment_phase', ASSEMBLE) == ASSEMBLE:
            anchor = tuple(frozen)
        else:
            choice = rally_anchor(context, group['ordinary'])
            if choice and choice['safe']:
                anchor = choice['position']
    return {'phase': phase['phase'], 'anchor': anchor,
            'front_actor': front_actor, 'engaged': phase['engaged']}


def deployment_duty(context, actor, memory):
    """Entry constraint used by complete-plan assessment AND repair seeds.

    Existing fire/rescue/interception duties take precedence. The duty is
    BENEFIT-CONDITIONAL, not proximity-conditional:

    * A unit with a legal window shot is never recalled (output protection).
    * A unit already within one support radius of a FIGHTING member is in a
      reasonable position and is never squeezed toward any anchor centre.
    * COVERED_JOIN anchors on a stable member of the actual fighting front,
      so the required move restores real support for a
      unit taking damage — it is never a retreat to a safe rear point.
    * ASSEMBLE anchors on a frozen task anchor or a safe member position,
      and only during the bounded opening window when no task exists.
    "Closer to the anchor" alone is never a reason: the action must shorten
    support distance to a fighting front (COVERED_JOIN) or close a frozen
    assembly anchor (ASSEMBLE).
    """
    active = next((t for t in memory.instances.values()
                   if t.skill == 'precombat_rally' and actor in t.actors
                   and not t.termination_reason), None)
    if not active and not opening_available(context, memory):
        return None
    group = next((g for g in squads(context) if actor in g['ordinary']), None)
    if group is None or len(group['ordinary']) < 2:
        return None
    # Output protection: a legal, effective shot is a contribution; the
    # organizer never trades it for geometry.
    if any(context.fire_estimate(actor, e).window_shots >= 1
           for e in context.attackable_enemies(actor)):
        return None
    phase = squad_phase(context, group)
    if phase['phase'] == ENGAGE or actor in phase['engaged']:
        return None
    target = deployment_target(context, group, active)
    anchor = target['anchor']
    if anchor is None:
        return None
    fighting = phase['engaged']
    if phase['phase'] == COVERED_JOIN:
        if not fighting:
            return None
        unit_position = context.snapshot.allies[actor].position
        nearest_fighting = min(fighting, key=lambda f: (
            distance(unit_position, context.snapshot.allies[f].position), f))
        # Already mutually supporting this front: reasonable position, no
        # forced inward pull.
        if distance(unit_position,
                    context.snapshot.allies[nearest_fighting].position) \
                <= SUPPORT_DISCONNECT:
            return None
        benefit = 'shorten_support_to_fighting_front'
    else:
        # A member already about to reach its own contact/fire band (e.g. a
        # melee pinning an approaching enemy group) is contributing now —
        # assembly never recalls it backward off that engagement edge.
        nearest = tuple(context.snapshot.enemies)
        eta = (_contact_eta_h(context, actor, nearest)
               if context.capabilities[actor].combat_class == "melee"
               else _fire_eta_h(context, actor, nearest)) if nearest else float('inf')
        if eta <= SUPPORT_ETA_H:
            return None
        benefit = 'close_frozen_or_safe_assembly_anchor'
    from .joint_combat import action_position
    unit = context.snapshot.allies[actor]
    origin_gap = distance(unit.position, anchor)
    if phase['phase'] == ASSEMBLE and origin_gap <= DISPERSED_SPACING:
        return None   # no STOP or inward squeeze for a member already placed
    options = []
    for action in context.legal_actions(actor):
        if context.decode(actor, action).kind not in ('move', 'attack'):
            continue
        point, _ = action_position(context, actor, action)
        gap = distance(point, anchor)
        if gap >= origin_gap - .05 or not safe_step(context, actor, point):
            continue
        if memory.movement_blockage(context, actor, action) >= .5:
            continue
        options.append((gap, action))
    if not options:
        return None
    return {'required': True, 'actions': tuple(a for _, a in sorted(options)),
            'kind': 'entry', 'reason': 'deployment_' + phase['phase'].lower(),
            'squad': group['id'], 'members': group['ordinary'],
            'pending_members': tuple(a for a in group['ordinary'] if a not in fighting),
            'anchor': list(anchor), 'front_actor': target['front_actor'],
            'deployment_phase': phase['phase'], 'benefit': benefit}


def unsupported_cooldown_advance(context, actor, point):
    """Guard for cooldown-window advances (the lone-hydraling failure).

    A cooling unit may not step into ADDITIONAL ordinary enemy weapon
    coverage unless a teammate can actually deliver window fire onto the
    newly covering enemy, or the move is real relief. Checks both explicit
    MOVE and out-of-range ATTACK's implied approach point identically.
    """
    unit = context.snapshot.allies[actor]
    if unit.weapon_cooldown_loops <= context.H:
        return {"blocked": False, "reason": "weapon_ready"}
    origin = context.forecast_risk(actor, unit.position)
    after = context.forecast_risk(actor, tuple(point))
    origin_cover = set(origin.current_threats)
    new_cover = set(after.current_threats) - origin_cover
    if not new_cover:
        return {"blocked": False, "reason": "no_new_weapon_coverage"}
    incoming = 0.
    for enemy in new_cover:
        damage = after.damage_estimates.get(enemy)
        if damage is None:
            return {"blocked": True,
                    "reason": "new_unknown_weapon_coverage_while_cooling"}
        incoming += damage
    if incoming <= 0.:
        return {"blocked": False, "reason": "new_coverage_not_threatening"}
    for peer, other in context.snapshot.allies.items():
        if peer == actor or not other.alive:
            continue
        cap = context.capabilities[peer]
        if cap.healer or cap.suicide_attack:
            continue
        if distance(other.position, tuple(point)) > 10.:
            continue
        if any(enemy in new_cover
               and context.fire_estimate(peer, enemy).window_shots >= 1
               for enemy in context.attackable_enemies(peer)):
            return {"blocked": False,
                    "reason": "teammate_window_fire_covers_new_threat"}
    return {"blocked": True,
            "reason": "solo_advance_into_new_weapon_coverage_while_cooling"}


def recent_roundtrip(context, actor, memory):
    """Observed barren loop: back near a position >=3H old after real walk.

    Uses ONLY recorded move history and observed damage; command acceptance
    is never success. Returns a dict describing the loop, or None.
    """
    history = (getattr(memory, "move_history", {}) or {}).get(actor, [])
    if len(history) < 3:
        return None
    loop_now = context.snapshot.game_loop
    window = [(loop, tuple(position)) for loop, position in history
              if loop_now - 4 * context.H <= loop <= loop_now]
    if len(window) < 3:
        return None
    unit = context.snapshot.allies.get(actor)
    if unit is None or not unit.alive:
        return None
    now = tuple(unit.position)
    walk = sum(distance(a[1], b[1]) for a, b in zip(window, window[1:]))
    old = [(loop, position) for loop, position in window
           if loop_now - loop >= 3 * context.H]
    if walk < 1.2 or not old:
        return None
    near_past = min(((distance(now, position), loop, position)
                     for loop, position in old))
    if near_past[0] > .6:
        return None
    return {"walk": round(walk, 2), "returned_to_loop": near_past[1],
            "position": list(near_past[2]),
            "damaged": unit.health_lost + unit.shield_lost > 0}


def roundtrip_cost(context, actor, action, memory):
    """Bounded selection cost for moves that re-enter a just-abandoned spot.

    A new threat (relief direction) or an emergency override stays free.
    This cost feeds memory.movement_conflict, so it reaches candidate
    filtering, joint scoring, repair and baseline through one channel.
    """
    trip = recent_roundtrip(context, actor, memory)
    if trip is None:
        return 0.
    if not trip["damaged"] and trip["walk"] < 2.:
        return 0.                       # harmless early-game reshuffle
    unit = context.snapshot.allies[actor]
    destination = context.move_destinations(actor).get(action)
    if destination is None:
        return 0.
    origin = context.risk_at(actor, unit.position)
    after = context.risk_at(actor, tuple(destination))
    if len(after.threats) + len(after.effect_weapon_near) < \
            len(origin.threats) + len(origin.effect_weapon_near):
        return 0.                       # genuine relief still allowed
    if distance(tuple(destination), tuple(trip["position"])) > .8:
        return 0.
    return 1.0


def summary(context, memory=None):
    rows = []
    for squad in squads(context)[:4]:
        phase = squad_phase(context, squad)
        rows.append({"squad": squad["id"],
                     "members": list(squad["members"]),
                     "phase": phase["phase"],
                     "engaged": list(phase["engaged"]),
                     "detached": list(phase.get("detached", ())),
                     "ready_fraction": phase["readiness"].get("fraction"),
                     "not_ready": phase["readiness"].get("not_ready", {})})
    return {"dispersion": round(dispersed(context), 2),
            "needs_deployment": needs_deployment(context),
            "squads": rows,
            "semantics": "observed phases; covered_join requires real "
                         "detachment beyond one support radius of a "
                         "fighting member, never a rear-anchor recall"}
