"""Bounded local entry timing from visible geometry and final commands.

This measures uncovered *time*, not another damage/risk estimate. Nearby
enemies form bounded fronts, each charged at most once. Future fire is an
opportunity, never proof of suppression, survival or an enemy death.
"""
import math

from .context import STOP, GAME_LOOPS_PER_SPEED_SECOND, distance
from .combat_exchange import exposure_window
from .fire_opportunities import final_action_budget


WINDOW_H = 3.
WAIT_H = 2.
FRONT_RADIUS = 6.
PEER_RADIUS = 10.


def _ordinary(context, actor):
    unit, cap = context.snapshot.allies.get(actor), context.capabilities.get(actor)
    return bool(unit and unit.alive and cap and not cap.healer and not cap.suicide_attack)


def _fronts(context):
    """Anchor-bounded groups; a long enemy chain cannot join distant fights."""
    pending, fronts = set(context.snapshot.enemies), []
    while pending:
        anchor = min(pending)
        front = tuple(sorted(e for e in pending if distance(
            context.snapshot.enemies[e].position,
            context.snapshot.enemies[anchor].position) <= FRONT_RADIUS))
        pending.difference_update(front)
        fronts.append(front)
    return tuple(fronts)


def _eta(context, actor, target, point, move_h=0.):
    reach = context.effective_attack_reach(actor, target)
    if reach is None or reach < 0:
        return float('inf')
    gap = context.projected_attack_opportunity(actor, target, point).center_gap
    scale = (context.speed(actor) or 0.) * max(context.H, 1) / GAME_LOOPS_PER_SPEED_SECOND
    travel = gap / scale if scale > 0 else (0. if gap <= .05 else float('inf'))
    return max(move_h + travel,
               context.snapshot.allies[actor].weapon_cooldown_loops / max(context.H, 1))


def wait_remaining_loops(context, memory, actor, instance=None, max_h=WAIT_H):
    """Actual consecutive idle STOPs share one TTL across providers.

    Each observed history action belongs to the preceding committed window.
    A new plan id/target does not erase those observations. Real movement or
    fire ends this short idle episode; missing observations invent no history.
    """
    H, loop = max(context.H, 1), context.snapshot.game_loop
    budget = max(0, int(round(min(WAIT_H, max_h) * H)))
    start = loop
    history = getattr(getattr(memory, 'combat_progress', None), 'history', {}).get(actor, ())
    unit = context.snapshot.allies[actor]
    latest = loop
    for row in reversed(history):
        if row['loop'] > loop:
            continue
        if latest - row['loop'] > 2 * H or row.get('tag') != unit.tag \
                or row.get('fired') or row.get('action') != STOP:
            break
        start = min(start, row['loop'] - H)
        latest = row['loop']
    deadline = start + budget
    if instance is not None:
        if instance.max_loops is not None:
            deadline = min(deadline, instance.start_loop + instance.max_loops)
        if instance.phase == 'waiting':
            deadline = min(deadline, instance.phase_start_loop +
                           (instance.extra.get('phase_max_loops') or budget))
    return max(0, deadline - loop)


def _safe_idle(context, actor):
    unit = context.snapshot.allies[actor]
    risk = context.forecast_risk(actor, unit.position)
    return not (risk.reachable_threats or risk.effect_weapon_near
                or unit.health_lost + unit.shield_lost > 0
                or any(context.contact_status(actor, e) == 'in_range'
                       for e in context.attackable_enemies(actor)))


def _entry_routes(context, actor, targets):
    """Finite legal moves that actually close to this same local frontage."""
    cache = getattr(context, '_local_entry_routes', None)
    if cache is None:
        cache = context._local_entry_routes = {}
    key = actor, tuple(sorted(targets))
    if key in cache:
        return cache[key]
    if context.speed(actor) is None:
        cache[key] = []
        return []
    unit = context.snapshot.allies[actor]
    routes = []
    for action in context.move_destinations(actor):
        point = context.predict_position(actor, action).position
        if context.forecast_risk(actor, point).effect_weapon_near:
            continue
        for target in targets:
            before = context.projected_attack_opportunity(actor, target, unit.position).center_gap
            after = context.projected_attack_opportunity(actor, target, point).center_gap
            if after >= before - .05:
                continue
            eta = _eta(context, actor, target, point, 1.)
            if math.isfinite(eta) and eta <= WINDOW_H:
                routes.append((eta, action, target, point))
    cache[key] = routes
    return routes


def _entry_onset(context, actor, routes):
    exposed = [r for r in routes
               if context.forecast_risk(actor, r[3]).reachable_threats]
    if not exposed:
        return None
    point = min(exposed, key=lambda r: (r[0], r[1], r[2]))[3]
    midpoint = tuple((x + y) / 2 for x, y in zip(
        context.snapshot.allies[actor].position, point))
    return .5 if context.forecast_risk(actor, midpoint).reachable_threats else 1.


def wait_opportunity(context, memory, actor, target, instance=None, max_h=WAIT_H):
    """Candidate declaration only; the final branch must deliver its peer."""
    if not _ordinary(context, actor) or target not in context.snapshot.enemies \
            or not _safe_idle(context, actor) or STOP not in context.legal_actions(actor):
        return None
    remaining = wait_remaining_loops(context, memory, actor, instance, max_h)
    if remaining < context.H:
        return None
    fronts = _fronts(context)
    front = next(f for f in fronts if target in f)
    routes = _entry_routes(context, actor, front)
    if not routes:
        return None
    own_eta = _entry_onset(context, actor, routes)
    # Waiting while every useful route is safe has no coordination purpose.
    if own_eta is None:
        return None
    peers = []
    for peer, unit in context.snapshot.allies.items():
        if peer == actor or not _ordinary(context, peer) or distance(
                unit.position, context.snapshot.allies[actor].position) > PEER_RADIUS:
            continue
        possible = [r[0] for r in _entry_routes(context, peer, front)
                    if getattr(memory, 'movement_blockage', lambda *_: 0.)(
                        context, peer, r[1]) < .5
                    if exposure_window(context, peer, r[3], horizon_h=r[0],
                                       certain=False)['viability'] > 0.]
        possible += [_eta(context, peer, enemy, unit.position)
                     for enemy in front if context.can_issue_attack(peer, enemy)
                     and context.contact_status(peer, enemy) == 'in_range'
                     and exposure_window(context, peer, horizon_h=_eta(
                         context, peer, enemy, unit.position))['viability'] > 0.]
        eta = min(possible, default=float('inf'))
        if own_eta + .05 < eta <= min(own_eta + 1., remaining / max(context.H, 1)):
            peers.append(peer)
    if not peers:
        return None
    return {'front': list(front), 'peers': sorted(peers), 'own_entry_eta_h': own_eta,
            'own_fire_eta_h': min(r[0] for r in routes),
            'until_loop': context.snapshot.game_loop + remaining}


def assess_local_entry(context, actions, joint_facts, plans=(), memory=None):
    """Pure final-branch timing cost, bounded to [-1, 1] per local front.

    ``joint_facts`` supplies budgeted primary target ETAs. Contingency fire on
    expected kills cannot promise cover, but is not itself an idle violation.
    No skill/provider label changes a final action's timing or idle facts.
    """
    rows = {a: row for a, row in joint_facts['actors'].items()
            if row.get('ordinary') and _ordinary(context, a)}
    fronts = _fronts(context)
    if not fronts or not rows:
        return {'cost': 0., 'fronts': [], 'waits': {}, 'valid_wait_actors': (),
                'invalid_wait_actors': ()}
    budget = final_action_budget(context, actions)
    before, assigned = {}, {}
    for actor, row in rows.items():
        unit = context.snapshot.allies[actor]
        old = {e: _eta(context, actor, e, unit.position) for e in context.attackable_enemies(actor)
               if context.contact_status(actor, e) == 'in_range'
               and (row.get('observed_attack_target') is None
                    or row['observed_attack_target'] == e)}
        before[actor] = budget.future_targets(actor, old)['primary']
        relevant = set(row.get('target_etas', ())) | set(row.get('contingency_target_etas', ()))
        relevant.update(context.forecast_risk(actor, row['position']).reachable_threats)
        relevant.update(context.forecast_risk(actor, unit.position).reachable_threats)
        if not relevant:
            relevant = {e for e in context.snapshot.enemies
                        if _eta(context, actor, e, unit.position) <= WAIT_H}
        options = [(min(distance(unit.position, context.snapshot.enemies[e].position)
                        for e in front), index) for index, front in enumerate(fronts)
                   if relevant.intersection(front)]
        if options:
            assigned[actor] = min(options)[1]
    positions = {a: r['position'] for a, r in rows.items()}
    after_credible = {a: row.get('cover_after_viability', 0.) > 0.
                     and (row['kind'] != 'move' or memory is None or
                          memory.movement_blockage(context, a, actions[a],
                              joint_actions=actions, joint_positions=positions) < .5)
                     for a, row in rows.items()}
    costs, waits, valid = {}, {}, []
    for actor, index in assigned.items():
        row, front = rows[actor], set(fronts[index])
        unit = context.snapshot.allies[actor]
        peers = [a for a in rows if a != actor and assigned.get(a) == index
                 and distance(unit.position, context.snapshot.allies[a].position) <= PEER_RADIUS]
        def peer_eta(source, viability):
            return min((eta for peer in peers if rows[peer].get(viability, 0.) > 0.
                        and (viability != 'cover_after_viability' or after_credible[peer])
                        for e, eta in source(peer).items() if e in front),
                       default=WINDOW_H)
        old_risk = context.forecast_risk(actor, unit.position)
        new_risk = context.forecast_risk(actor, row['position'])
        was_exposed = bool(front.intersection(old_risk.reachable_threats))
        now_exposed = bool(front.intersection(new_risk.reachable_threats))
        # Real retreat relief is already valued by survival; entry timing
        # must not turn a needed escape into an obligation to stay exposed.
        if now_exposed and not row.get('relief'):
            old_gap = min(WINDOW_H, peer_eta(lambda a: before[a],
                'cover_before_viability')) if was_exposed else 0.
            midpoint = tuple((x + y) / 2 for x, y in zip(unit.position, row['position']))
            onset = (0. if was_exposed else .5 if front.intersection(
                context.forecast_risk(actor, midpoint).reachable_threats) else 1.)
            new_gap = min(WINDOW_H, max(0., peer_eta(lambda a: rows[a]['target_etas'],
                'cover_after_viability') - onset))
            costs.setdefault(index, []).append((actor, (new_gap - old_gap) / WINDOW_H))
        if row['kind'] != 'stop' or row.get('target_etas') or row.get('contingency_target_etas') \
                or not _safe_idle(context, actor):
            continue
        routes = _entry_routes(context, actor, front)
        if not routes:
            continue  # unknown/unreachable fire is not manufactured idleness
        remaining = wait_remaining_loops(context, memory, actor,
            getattr(memory, 'instance_for', lambda *_: None)('coordinated_engage', actor))
        own_eta = _entry_onset(context, actor, routes)
        timely = [peer for peer in peers
                  if own_eta is not None
                  and after_credible[peer]
                  and rows[peer]['kind'] in ('move', 'attack')
                  and (rows[peer]['kind'] == 'attack' or rows[peer].get('closing'))
                  and any(e in front and own_eta + .05 < eta <= min(
                      own_eta + 1., remaining / max(context.H, 1))
                      for e, eta in rows[peer]['target_etas'].items())]
        usable = remaining >= context.H and bool(timely)
        waits[actor] = {'remaining_loops': remaining, 'timely_peers': tuple(sorted(timely)),
                        'valid': usable, 'own_entry_eta_h': own_eta}
        if usable:
            valid.append(actor)
    details = []
    for index, entries in sorted(costs.items()):
        by_actor = {}
        for actor, cost in entries:
            by_actor[actor] = max(by_actor.get(actor, -1.), cost)
        value = max(-1., min(1., sum(by_actor.values()) / max(len(by_actor), 1)))
        details.append({'targets': fronts[index], 'cost': value, 'actor_costs': by_actor})
    return {'cost': sum(f['cost'] for f in details), 'fronts': details,
            'waits': waits, 'valid_wait_actors': tuple(sorted(valid)),
            # Keep ordinary per-actor idleness outside the per-front timing
            # cap. Ten idle actors must not become one discounted idle front.
            'invalid_wait_actors': tuple(sorted(a for a in waits if a not in valid))}
