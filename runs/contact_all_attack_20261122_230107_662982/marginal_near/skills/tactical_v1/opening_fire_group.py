"""One-window alternatives for a dispersed Zerg opening."""
from itertools import combinations

from .context import STOP, distance
from .contracts import Commitment, Dependency, EffectClaim, HEURISTIC, ResourceClaims
from .deployment import (DISPERSED_SPACING, SUPPORT_DISCONNECT, SUPPORT_ETA_H,
                         _fire_eta_h, geometry_fraction, safe_step)
from .skills.common import make_plan


def _step_toward(context, memory, actor, point):
    """Only offer a legal, unblocked step that closes the observed gap."""
    origin = context.snapshot.allies[actor].position
    before = distance(origin, point)
    options = []
    for action in context.move_destinations(actor):
        position = context.predict_position(actor, action).position
        gap = distance(position, point)
        if gap < before - .05 and safe_step(context, actor, position) \
                and memory.movement_blockage(context, actor, action) < .5:
            options.append((gap, action))
    return min(options)[1] if options else None


def _rank_groups(context, hydras, ready):
    units, enemies = context.snapshot.allies, context.snapshot.enemies
    eta = {a: {e: _fire_eta_h(context, a, (e,)) for e in enemies} for a in hydras}
    options = []
    for size in (2, 3):
        for group in combinations(hydras, size):
            diameter = max(distance(units[a].position, units[b].position)
                           for a, b in combinations(group, 2))
            if diameter > SUPPORT_DISCONNECT:
                continue
            choices = []
            for e in enemies:
                arrival = max(eta[a][e] for a in group)
                if arrival <= SUPPORT_ETA_H and all(not ready[a] or e in ready[a] for a in group):
                    shots = sum(e in ready[a] for a in group)
                    choices.append((-shots, arrival, enemies[e].effective_health, e))
            best = min(choices) if choices else None
            rank = (best is None, best[:2] if best else (), -size, diameter, group)
            options.append((rank, group, best[-1] if best else None))
    return sorted(options)


def opening_fire_groups(context, memory, fingerprint):
    units, enemies = context.snapshot.allies, context.snapshot.enemies
    hydras = sorted(a for a, u in units.items() if u.alive and u.type_name == 'hydralisk')
    lings = sorted(a for a, u in units.items() if u.alive and u.type_name == 'zergling')
    if len(hydras) < 2 or not lings or geometry_fraction(context, hydras + lings) >= .8:
        return []
    tracker = memory.encounters
    opening_h = memory.encounter_config.get('precombat_rally', {}).get('max_h', 8.)
    if tracker.current_id != 1 or tracker.started_loop is None or \
            context.snapshot.game_loop - tracker.started_loop >= opening_h * context.H:
        return []

    ready = {a: {e for e in context.attackable_enemies(a)
                 if context.fire_estimate(a, e).window_shots > 0} for a in hydras + lings}

    plans, assigned = [], set()
    for _, group, target in _rank_groups(context, hydras, ready):
        if assigned.intersection(group):
            continue
        # An already firing Hydra is a useful assembly point. In fog, the
        # nearest local lead searches while its partners converge on it.
        nearby = min(enemies, key=lambda e: min(context.distance(a, e) for a in group),
                     default=None)
        front = enemies[target or nearby].position if nearby else context.snapshot.search_location
        lead = min(group, key=lambda a: (not ready[a], distance(units[a].position, front), a))
        anchor = units[lead].position
        formed = all(distance(units[a].position, anchor) <= DISPERSED_SPACING for a in group)
        actions = {}
        for a in group:
            if ready[a]:
                shot = target or min(ready[a], key=lambda e: (enemies[e].effective_health, e))
                actions[a] = context.attack_action(a, shot)
            else:
                goal = front if target or a == lead or formed else anchor
                actions[a] = _step_toward(context, memory, a, goal)
                if actions[a] is None and distance(units[a].position, goal) <= DISPERSED_SPACING:
                    actions[a] = STOP if STOP in context.legal_actions(a) else None
        if any(action is None for action in actions.values()):
            continue
        # With few screens, guard the nearest threatened shooter. Numerous
        # screens can cover two approach lanes and join a verified shared fight.
        scarce = len(lings) < len(hydras)
        guard = min(group, key=lambda a: (min((context.distance(a, e) for e in enemies),
                                              default=0.), a))
        shared_fire = target is not None and sum(target in ready[a] for a in group) >= 2
        release = shared_fire and not context.enemy_capabilities[target].suicide_attack
        escorts = sorted((a for a in lings if a not in assigned),
                         key=lambda a: (distance(units[a].position, units[guard].position), a))
        screen, released = {}, set()
        posts = set(group)
        for a in escorts:
            if len(screen) >= (1 if scarce else 2):
                break
            nearest_rear = min(hydras, key=lambda b: (distance(units[a].position, units[b].position), b))
            local_gap = min(distance(units[a].position, units[b].position) for b in group)
            if nearest_rear not in group and (not scarce or local_gap > SUPPORT_DISCONNECT):
                continue
            if local_gap > SUPPORT_DISCONNECT and \
                    (not nearby or context.distance(a, nearby) > SUPPORT_DISCONNECT):
                continue
            post = nearest_rear if scarce else min(posts or group, key=lambda b: (
                distance(units[a].position, units[b].position), b))
            commit = release and (not scarce or target in context.forecast_risk(
                post, units[post].position).reachable_threats)
            if ready[a]:
                shot = target if target in ready[a] else min(ready[a], key=lambda e: (context.distance(a, e), e))
                screen[a] = context.attack_action(a, shot)
            else:
                # Screen beside the core; commit to enemy contact only once
                # both rear guns can support it in this window.
                if commit:
                    point = enemies[target].position
                else:
                    posts.discard(post)
                    point = units[post].position
                move = _step_toward(context, memory, a, point) if commit or \
                    distance(units[a].position, point) > DISPERSED_SPACING else None
                if move is not None:
                    screen[a] = move
                elif STOP in context.legal_actions(a) and safe_step(context, a, units[a].position):
                    screen[a] = STOP
            if commit and a in screen and screen[a] != STOP and (not ready[a] or target in ready[a]):
                released.add(a)
        # Both the small fire team alone and the screened route are available
        # to ordinary joint scoring. No unit gets a new mandatory duty.
        for final in ([actions, {**actions, **screen}] if screen else [actions]):
            if all(action == STOP for action in final.values()):
                continue
            effects = []
            for a, action in final.items():
                order = context.decode(a, action)
                if order.kind == 'attack':
                    effects.append(EffectClaim('coverage', order.target, None, context.H,
                                               HEURISTIC, 'opening_fire_order'))
            data = {'fire_group': list(group), 'screen': sorted(set(final) - set(group)),
                    'screen_released': bool(released.intersection(final)),
                    'shared_target': target}
            if target:
                data['objective'] = {'kind': 'shared_fire', 'target': target,
                                     'shooters': list(group)}
            elif formed:
                data['objective'] = {'kind': 'waypoint', 'position': list(front)}
            else:
                data.update(objective={'kind': 'rally'}, members=list(group),
                            anchor=list(anchor), ready_fraction=1., ready_radius=DISPERSED_SPACING)
            plans.append(make_plan('coordinated_engage', final, target or lead, final, effects,
                ResourceClaims(controls=tuple(sorted(final))), Commitment('opening_fire_group', 0, context.H),
                {}, context.snapshot.snapshot_id, fingerprint, atomic=True,
                dependencies=(Dependency('target_valid', (target,)),) if target else (),
                goal_kind='enemy' if target else 'ally', goal_data=data,
                rationale='local shared fire' if target else 'assemble local fire group'))
        assigned.update(group)
        assigned.update(screen)
    return plans
