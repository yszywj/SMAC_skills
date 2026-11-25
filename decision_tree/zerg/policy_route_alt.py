"""Heavy-hydra rendezvous then shared-front approach, on certified menus only."""
from math import dist
from statistics import median

from . import policy_distill as cart
from .policy_repair_v2 import v2_choose


def choose(model, raw_pre, candidates, previous_actions=None):
    state = raw_pre.get('state', raw_pre)
    menu = [cart.vector(state, a) for a in candidates]
    base = v2_choose(model, raw_pre, menu, previous_actions)
    original = tuple(base['actions'])
    audit = dict(base['audit'], v2_actions=list(original), v2_reason=base['reason'],
                 route_phase='inactive', v3_triggered=False)
    result = dict(base, audit=audit)
    # no_existing_formation_fix is a request to the runtime to try its native
    # opening generator. Never consume it, even if generation later declines.
    if base['reason'] in ('protect_hydra_support_gap', 'no_existing_formation_fix'):
        return result
    units = {int(k[1:]): u for k, u in state['allies'].items() if u['alive']}
    hydras = {i: u for i, u in units.items() if u['type'].lower() == 'hydralisk'}
    if state['game_loop'] > 82 or len(hydras) < 6 or len(units) < 8:
        return result
    contact = any(r['smac_attack_available'] for rel in state['relations'].values() for r in rel.values())
    if not contact and any(not u['alive'] or u['health'] < u['max_health'] for u in state['allies'].values()):
        return result
    center = tuple(map(median, zip(*(u['position'] for u in hydras.values()))))
    phase = 'shared_front' if contact else 'rendezvous'
    audit['route_phase'] = phase
    if base['unique_count'] < 2:
        return result
    if contact:
        # Fixed front for all candidates; use a target seen by the hydra group,
        # not a private nearest target that would send different flanks apart.
        enemies = state['visible_enemies']
        fronts = [k for k, e in enemies.items() if e['alive'] and all(
            state['relations'].get('u%d' % i, {}).get(k, {}).get('native_weapon_range') is not None
            for i in hydras)]
        if not fronts:
            return result
        front = min(fronts, key=lambda k: (dist(center, enemies[k]['position']), k))
        audit['front'] = front

    def costs(actions):
        rally = approach = 0.
        for i, u in units.items():
            uid = 'u%d' % i
            a = actions[i]
            p = (state['movement_candidates'][uid][str(a)]['destination']
                 if 2 <= a <= 5 else u['position'])
            rally += dist(p, center)
            if contact and i in hydras:
                e = enemies[front]
                reach = state['relations'][uid][front]['native_weapon_range'] + u['radius'] + e['radius']
                approach += max(0., dist(p, e['position']) - reach)
        return rally, approach

    old_rally, old_approach = costs(original)
    ranks = {}
    for v in set(menu) - {original}:
        # All differences must be stop/movement. Invariant attack orders have
        # unknown endpoints; their proxy contributions cancel in comparisons.
        if any(a != b and max(a, b) >= 6 for a, b in zip(original, v)):
            continue
        rally, approach = costs(v)
        if contact:
            if approach <= old_approach - .75 and rally <= old_rally + 1.:
                ranks[v] = (-approach, -rally)
        elif rally <= old_rally - .75:
            ranks[v] = (-rally,)
    if not ranks:
        return result
    best = max(ranks.values())
    scores = [s + 2 * (ranks.get(v) == best) for v, s in zip(menu, base['scores'])]
    unique = dict(zip(menu, scores))
    winner, ties = cart.resolve(cart.tie_salt(state), unique, unique.values())
    audit.update(v3_triggered=True, v2_scores=base['scores'],
                 before_cost=[old_rally, old_approach], after_cost=list(costs(winner)))
    return dict(result, actions=list(winner), scores=scores, score=unique[winner],
                top_ties=ties, reason='route_' + phase)
