"""Conservative ready-volley tie breaker over caller-certified native joint actions."""
from collections import Counter

from . import policy_distill as cart
from .policy_repair_v2 import v2_choose


def volley(state):
    """Known direct shots now, not SMAC acquisition radius or estimated future DPS."""
    out = {}; catalog = state.get('unit_type_profiles', {})
    for uid, ally in state['allies'].items():
        if not ally['alive'] or not ally['weapon_ready']:
            continue
        # Only model ordinary non-bonus, single-hit, unupgraded weapons on this map.
        weapons = catalog.get(str(ally.get('unit_type_id')), {}).get('weapons', [])
        if len(weapons) != 1 or ally.get('attack_upgrade_level') != 0:
            continue
        w, = weapons
        if w['hits_per_attack'] != 1 or w['damage_bonus']:
            continue
        shots = {}
        for e, rel in state['relations'].get(uid, {}).items():
            if rel['in_weapon_range'] is not True or not rel['smac_attack_available']:
                continue
            target = state['visible_enemies'][e]
            armor = catalog.get(str(target['unit_type_id']), {}).get('armor')
            if target['alive'] and armor is not None and target['shield'] == 0 and target['armor_upgrade_level'] == 0:
                shots[6 + int(e[1:])] = max(.5, w['base_damage'] - armor)
        if shots:
            out[int(uid[1:])] = shots
    return out


def profile(state, table, actions):
    loads = Counter(); actors = []
    for i, targets in table.items():
        a = actions[i]
        if a in targets:
            loads[a] += targets[a]; actors.append(i)
    hp = {a: state['visible_enemies']['e%d' % (a-6)]['health'] for a in loads}
    excess = sum(max(0, dmg-hp[a]) for a,dmg in loads.items())
    return dict(actors=sorted(actors), loads=dict(loads), lethal=sum(d>=hp[a] for a,d in loads.items()),
                useful=sum(loads.values())-excess, excess=excess)


def v3_choose(model, raw_pre, candidates, previous_actions=None):
    candidates = list(candidates)
    base = v2_choose(model, raw_pre, candidates, previous_actions)
    base = dict(base, audit=dict(base['audit'], v2_actions=list(base['actions']),
                                 v2_reason=base['reason'], v3_triggered=False))
    if base['reason'] in ('protect_hydra_support_gap', 'no_existing_formation_fix') or base['unique_count'] < 2:
        return base
    state = raw_pre.get('state', raw_pre)
    table = volley(state)
    if not table:
        return base
    original = tuple(base['actions']); old = profile(state, table, original)
    vectors = [cart.vector(state,a) for a in candidates]; gains = {}
    for v in set(vectors)-{original}:
        # Every modified order must become a known ready shot; don't silently
        # redirect escorts, effect-based banelings, or cooldown movement.
        if any(x!=y and y not in table.get(i,{}) for i,(x,y) in enumerate(zip(original,v))):
            continue
        p = profile(state, table, v)
        if (p['lethal'] >= old['lethal'] and p['useful'] >= old['useful'] and
            (p['lethal'] > old['lethal'] or p['useful'] > old['useful'])):
            gains[v] = (p['lethal'], p['useful'])
    if not gains:
        return base
    best = max(gains.values())
    scores = [s + 2 * (gains.get(v)==best) for v,s in zip(vectors,base['scores'])]
    unique = dict(zip(vectors,scores)); winner,ties = cart.resolve(cart.tie_salt(state), unique, unique.values())
    audit = dict(base['audit'], v3_triggered=True, v2_scores=base['scores'],
                 engagement=dict(before=old,after=profile(state,table,winner),ready_targets=table))
    return dict(base,actions=list(winner),scores=scores,score=unique[winner],top_ties=ties,
                reason='improve_ready_volley',audit=audit)
