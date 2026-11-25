"""Single-tree joint policy inference. Only the pre-observation and actions enter here."""
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


TYPES = ('zergling', 'baneling', 'hydralisk')
CHANNELS = ('alive', 'hp_fraction_sum', 'ready', 'nearest_distance_sum', 'stop',
            'north', 'south', 'east', 'west', 'attack', 'repeat', 'switch_target',
            'reverse_move', 'attack_ready', 'attack_in_range', 'attack_nearest',
            'attack_distance_sum', 'attack_hp_fraction_sum', 'attack_wounded',
            'move_closure_sum', 'move_toward', 'move_away', 'move_with_attack_available',
            'attack_zergling', 'attack_baneling', 'attack_hydralisk')
FEATURES = (['game_loop', 'visible_enemies', 'enemy_hp_fraction_sum', 'distinct_targets',
             'max_focus', 'attack_pairs_same_target', 'previous_available'] +
            [t + '.' + k for t in ('all',) + TYPES for k in CHANNELS])
VERSION = 'zerg_joint_cart_pre_v1'
SCHEMA_HASH = digest([VERSION, FEATURES])


def vector(state, actions):
    size = len(state['allies'])
    if len(actions) != size:
        raise ValueError('incomplete action vector')
    a = tuple(actions['u%d' % i] for i in range(size)) if isinstance(actions, dict) else tuple(actions)
    if any(type(x) is not int or not 0 <= x < 16 for x in a):
        raise ValueError('not a 10v10 zerg action')
    return a


def project(raw_pre, actions, previous_actions=None):
    s = raw_pre.get('state', raw_pre)
    actions = vector(s, actions)
    prev = s['previous_action_ids'] if previous_actions is None else previous_actions
    enemies = s['visible_enemies']
    bins = {t: dict.fromkeys(CHANNELS, 0.) for t in TYPES}
    focus = {}
    for i, act in enumerate(actions):
        uid = 'u%d' % i
        u = s['allies'][uid]
        if not u['alive']:
            if act != 0:
                raise ValueError('dead actor has nonzero action')
            continue
        b = bins[u['type'].lower()]
        rel = s['relations'].get(uid, {})
        nearest = min((r['distance'] for r in rel.values()), default=0.)
        old = int(prev.get(uid, -1) if isinstance(prev, dict) else prev[i])
        b['alive'] += 1
        b['hp_fraction_sum'] += u['health'] / u['max_health']
        b['ready'] += bool(u['weapon_ready'])
        b['nearest_distance_sum'] += nearest
        b['repeat'] += act == old
        if act >= 6:
            key = 'e%d' % (act - 6)
            target, r = enemies[key], rel[key]
            hp = target['health'] / target['max_health']
            focus[key] = focus.get(key, 0) + 1
            b['attack'] += 1
            b['attack_' + target['type'].lower()] += 1
            b['switch_target'] += old >= 6 and act != old
            b['attack_ready'] += bool(u['weapon_ready'])
            b['attack_in_range'] += r['in_weapon_range'] is True
            b['attack_nearest'] += r['distance'] <= nearest
            b['attack_distance_sum'] += r['distance']
            b['attack_hp_fraction_sum'] += hp
            b['attack_wounded'] += hp < 1.
        elif act == 1:
            b['stop'] += 1
        elif 2 <= act <= 5:
            b[('north', 'south', 'east', 'west')[act - 2]] += 1
            b['reverse_move'] += old == (act ^ 1)
            moves = s['movement_candidates'][uid]
            move = moves.get(str(act), moves.get(act))
            distances = move['distances_to_visible_enemies']
            delta = nearest - min(distances.values()) if distances else 0.
            b['move_closure_sum'] += delta
            b['move_toward'] += delta > 0
            b['move_away'] += delta < 0
            b['move_with_attack_available'] += any(r['smac_attack_available'] for r in rel.values())
        else:
            raise ValueError('live actor has zero action')
    return ([s['game_loop'], len(enemies), sum(e['health']/e['max_health'] for e in enemies.values()),
             len(focus), max(focus.values(), default=0), sum(n*(n-1)//2 for n in focus.values()), int(bool(prev))]
            + [sum(b[k] for b in bins.values()) for k in CHANNELS]
            + [bins[t][k] for t in TYPES for k in CHANNELS])


def tie_salt(raw_pre):
    s = raw_pre.get('state', raw_pre)
    return digest([s['game_loop'], [[u['position'], u['health']] for _, u in sorted(s['allies'].items())]])


def resolve(salt, vectors, scores):
    best = max(scores)
    tied = [v for v, x in zip(vectors, scores) if abs(best - x) <= 1e-12]
    return min(tied, key=lambda v: (digest([salt, v]), v)), len(tied)


def score(nodes, x):
    node = nodes[0]
    while len(node) == 4:
        f, cutoff, left, right = node
        node = nodes[left if x[f] <= cutoff else right]
    return node[0]


def choose(model, raw_pre, candidates, previous_actions=None):
    s = raw_pre.get('state', raw_pre)
    original = [vector(s, a) for a in candidates]
    unique = sorted(set(original))
    scores = {a: score(model['nodes'], project(s, a, previous_actions)) for a in unique}
    chosen, ties = resolve(tie_salt(s), unique, [scores[a] for a in unique])
    return dict(actions=list(chosen), score=scores[chosen], top_ties=ties,
                unique_count=len(unique), scores=[scores[a] for a in original])


def load_model(path):
    with open(path, encoding='utf-8') as stream:
        m = json.load(stream)
    if (m['version'] != VERSION or m['schema_hash'] != SCHEMA_HASH or m['features'] != FEATURES
            or any(len(m[k]) != 64 or any(c not in '0123456789abcdef' for c in m[k])
                   for k in ('source_hash', 'ctx_hash', 'split_hash'))):
        raise ValueError('incompatible policy model')
    nodes = m['nodes']
    children = []
    for i, node in enumerate(nodes):
        if len(node) == 1 and math.isfinite(node[0]) and 0 <= node[0] <= 1:
            continue
        if (len(node) != 4 or type(node[0]) is not int or not 0 <= node[0] < len(FEATURES)
                or not math.isfinite(node[1]) or any(type(c) is not int or not i < c < len(nodes) for c in node[2:])):
            raise ValueError('invalid tree node')
        children.extend(node[2:])
    if not nodes or sorted(children) != list(range(1, len(nodes))):
        raise ValueError('invalid tree structure')
    return m
