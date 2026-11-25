"""A one-split, fail-closed router at the first complete plan divergence.

The caller audits plan legality and owns the episode-level sticky route. Neither
scenario identifiers, outcome labels nor policy/candidate names are features.
"""

SCHEMA = 'zerg_route_gate_v1'
# Fixed before reading labels. Each selected stump has one alternative leaf.
CUTS = {'ready_in_range': (0, 2, 4), 'incoming_enemies': (2, 4, 6),
        'live_hydras': (3, 5), 'changed_orders': (0, 2),
        'attack_gain': (0, 2), 'wounded_allies': (0, 2)}


def anchor_features(observable, parent_plan, alt_plan):
    state = observable.get('state', observable)
    units = state['allies']
    before, after = parent_plan['actions'], alt_plan['actions']
    if len(units) != len(before) or len(before) != len(after) or not units:
        raise ValueError('incomplete joint plan')
    if any(type(a) is not int or a < 0 for vec in (before, after) for a in vec):
        raise ValueError('invalid joint action')
    ready = hydras = wounded = 0
    incoming = set()
    for i in range(len(units)):
        uid = 'u%d' % i
        unit = units[uid]
        if not unit['alive']:
            continue
        rels = state['relations'].get(uid, {})
        hydras += unit['type'].lower() == 'hydralisk'
        wounded += unit['health'] < unit['max_health']
        ready += bool(unit['weapon_ready'] and any(r['in_weapon_range'] is True and
                      r['smac_attack_available'] for r in rels.values()))
        incoming.update(e for e, r in rels.items() if r['in_enemy_weapon_range'] is True)
    return dict(ready_in_range=ready, incoming_enemies=len(incoming), live_hydras=hydras,
                wounded_allies=wounded, changed_orders=sum(a != b for a,b in zip(before, after)),
                attack_gain=sum(a >= 6 for a in after) - sum(a >= 6 for a in before))


def valid_features(features):
    return (isinstance(features, dict) and set(features) == set(CUTS) and
            all(type(x) is int and (x >= 0 or k == 'attack_gain') for k,x in features.items()))


def choose_route(model, features):
    """Return 'alternative' only for a well-formed exported stump and full anchor."""
    if (not isinstance(model, dict) or model.get('schema') != SCHEMA or
            model.get('default') != 'v2' or not valid_features(features)):
        return 'v2'
    rule = model.get('tree')
    if not isinstance(rule, dict) or set(rule) != {'feature', 'op', 'threshold'}:
        return 'v2'
    key, op, t = rule['feature'], rule['op'], rule['threshold']
    if (not isinstance(key, str) or type(t) is not int or t not in CUTS.get(key, ()) or
            op not in ('le', 'gt')):
        return 'v2'
    matches = features[key] <= t if op == 'le' else features[key] > t
    return 'alternative' if matches else 'v2'
