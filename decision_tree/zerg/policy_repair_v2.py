"""Opening formation protection on existing candidates; no outcome or layout input."""
from math import dist
from . import policy_distill as v1


def formation(state):
    allies = state['allies']
    if state['game_loop'] > 82 or any(not u['alive'] or u['health'] < u['max_health']
                                      for u in allies.values()):
        return {}
    hydras = {int(k[1:]): u['position'] for k, u in allies.items() if u['type'].lower() == 'hydralisk'}
    return hydras if len(hydras) >= 6 else {}


def support_gap(positions):
    """Worst distance to a second supporting hydra (world units)."""
    return max(sorted(dist(p, q) for j, q in positions.items() if j != i)[1]
               for i, p in positions.items())


def endpoint_gap(state, hydras, actions):
    return support_gap({i: state['movement_candidates']['u%d' % i][str(actions[i])]['destination']
                        if 2 <= actions[i] <= 5 else p for i, p in hydras.items()})


def escort_cost(state, isolated, actions):
    endpoints = {i: state['movement_candidates']['u%d' % i][str(act)]['destination']
                 if 2 <= act <= 5 else state['allies']['u%d' % i]['position']
                 for i, act in enumerate(actions)}
    return sum(sorted(dist(endpoints[i], p) for j, p in endpoints.items() if j != i)[1]
               for i in isolated) / len(isolated)


def v2_choose(model, raw_pre, candidates, previous_actions=None):
    candidates = list(candidates)
    base = v1.choose(model, raw_pre, candidates, previous_actions)
    state = raw_pre.get('state', raw_pre)
    hydras = formation(state)
    audit = dict(v1_actions=base['actions'], v1_scores=list(base['scores']))
    result = dict(base, reason='outside_formation_guard', audit=audit)
    if not hydras:
        return result
    before = support_gap(hydras)
    original = base['actions']
    after = endpoint_gap(state, hydras, original)
    audit.update(current_gap=before, v1_gap=after)
    isolated = [i for i,p in hydras.items() if sorted(dist(p,q) for j,q in hydras.items() if i!=j)[1]>=5]
    if not isolated:
        return dict(result, reason='backline_already_supported')
    vectors = [v1.vector(state, a) for a in candidates]
    gaps = [endpoint_gap(state, hydras, a) for a in vectors]
    old_cost = escort_cost(state, isolated, original)
    costs = [escort_cost(state, isolated, a) for a in vectors]
    # Preserve all attack orders and never worsen the worst hydra-to-hydra gap.
    tiers = [g <= after and c <= old_cost-.5 and all(x == y or max(x, y) < 6 for x, y in zip(a, original))
             for g,c,a in zip(gaps,costs,vectors)]
    audit.update(isolated=isolated, v1_escort_cost=old_cost, candidate_escort_costs=costs,
                 candidate_gaps=gaps, protected_tiers=tiers)
    if not any(tiers):
        return dict(result, reason='no_existing_formation_fix')
    scores = [s + 2 * tier for s, tier in zip(base['scores'], tiers)]
    unique = dict(zip(vectors, scores))
    winner, ties = v1.resolve(v1.tie_salt(state), unique, unique.values())
    return dict(result, actions=list(winner), scores=scores, score=unique[winner],
                top_ties=ties, reason='protect_hydra_support_gap')
