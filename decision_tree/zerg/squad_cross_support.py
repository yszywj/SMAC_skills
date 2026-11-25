"""Opening-only Zerg macro policy, one instance per episode.

Projections assume a stationary visible enemy and one completed legal move;
they estimate firing *geometry*, not hits, collision-free paths or enemy intent.
ETA is in legal move lengths, not game loops. No map, seed or fog search goal.
"""
from itertools import combinations
from math import dist, inf

from .squad_rendezvous import SquadRendezvous, _squads


class _Geometry:
    def __init__(self, state, masks):
        self.pos, self.moves, self.reach = [], [], []
        self.targets = {e: u['position'] for e, u in state['visible_enemies'].items() if u['alive']}
        self.masks = masks
        for i, row in enumerate(masks):
            uid = f'u{i}'
            unit = state['allies'][uid]
            self.pos.append(unit['position'])
            self.moves.append({a: m['destination'] for k, m in state['movement_candidates'][uid].items()
                               if 2 <= (a := int(k)) <= 5 and row[a]})
            reach = {}
            for e, p in self.targets.items():
                r = state.get('relations', {}).get(uid, {}).get(e, {})
                parts = (r.get('native_weapon_range'), unit.get('radius'), state['visible_enemies'][e].get('radius'))
                limit = r.get('smac_attack_distance_limit')
                if None in parts or limit is None or r.get('weapon_range_status') == 'no_compatible_weapon':
                    continue
                # Inside today's SMAC envelope, a missing command is not a
                # distance problem. Don't extrapolate through that denial.
                slot = 6 + int(e[1:])
                if slot >= len(row) or (dist(self.pos[i], p) <= limit and not row[slot]):
                    continue
                reach[e] = min(limit, sum(parts))
            self.reach.append(reach)

    def fire(self, i, a):
        p = self.moves[i].get(a, self.pos[i])
        return {e for e, r in self.reach[i].items() if dist(p, self.targets[e]) <= r
                and (a < 6 or a == 6 + int(e[1:]))}

    def eta(self, i, e):
        if e not in self.reach[i]:
            return inf
        gap = dist(self.pos[i], self.targets[e]) - self.reach[i][e]
        if gap <= 0:
            return 0.
        # Require an actual legal first step that closes range; beyond it the
        # remaining straight-line distance is only an optimistic estimate.
        return min((1 + max(0., dist(q, self.targets[e]) - self.reach[i][e]) / dist(q, self.pos[i])
                    for q in self.moves[i].values() if dist(q, self.targets[e]) < gap + self.reach[i][e] - .01),
                   default=inf)


def _guard(g, groups, base, actions):
    before = [g.fire(i, a) for i, a in enumerate(base)]
    after = [g.fire(i, a) for i, a in enumerate(actions)]
    veto = {}
    for left, right in combinations(groups, 2):
        reachable = [set().union(*(after[i] for i in group)) for group in (left, right)]
        shared = set.intersection(*(set().union(*(before[i] for i in group)) for group in (left, right)))
        if not shared or reachable[0] & reachable[1]:
            continue  # Still a jointly reachable engagement, possibly elsewhere.
        members = left + right
        if not any(after[i] for i in members) or any(after[i] - before[i] for i in members):
            continue  # No separate contact, or a compensating new firing lane.
        for i in members:
            lost = shared & (before[i] - after[i])
            if lost:
                row = veto.setdefault(i, dict(actor=i, targets=[], adaptive=actions[i], restored=int(base[i]), witnesses=[]))
                row['targets'] = sorted(set(row['targets']) | lost)
                row['witnesses'].append(dict(squads=[left, right], adaptive_contacts=list(map(sorted, reachable))))
    for i in veto:
        actions[i] = int(base[i])
    return list(veto.values())


def _front(g, groups, actions, locked):
    centers = [tuple(sum(g.pos[i][k] for i in group) / len(group) for k in (0, 1)) for group in groups]
    eta = [{e: min(g.eta(i, e) for i in group) for e in g.targets} for group in groups]
    # Choose at most one neighboring pair. Both must see a near-best visible
    # contact at the same location; never rally across the map to a lone lead.
    options = []
    for a, b in combinations(range(len(groups)), 2):
        d = dist(centers[a], centers[b])
        if all(any(dist(centers[x], c) < d for k, c in enumerate(centers) if k != x) for x in (a, b)):
            continue
        for e in g.targets:
            f, r = sorted((a, b), key=lambda j: eta[j][e])
            early, late = eta[f][e], eta[r][e]
            if (early <= 2 and late <= 5 and late - early >= 1 and
                    not any(times[e] < early + 1 for k, times in enumerate(eta) if k not in (a, b)) and
                    all(eta[j][e] <= min(eta[j].values()) + 1 for j in (a, b))):
                options.append((late, early, d, e, f, r))
    if not options:
        return None
    late, early, _, e, f, r = min(options)
    changes, protected = [], []
    for j in (f, r):
        local_contact = any(sum(bool(g.masks[i][k]) for i in groups[j]) >= 2
                            for k in range(6, len(g.masks[0])))
        if local_contact:
            protected.append(j)
            continue
        for i in groups[j]:
            old = actions[i]
            if i in locked or any(g.masks[i][6:]) or e not in g.reach[i]:
                continue
            p, target = g.pos[i], g.targets[e]
            moves = [(a, q) for a, q in g.moves[i].items() if g.fire(i, old) <= g.fire(i, a)]
            if j == f:
                # Only temper an actual deepening order by the isolated lead.
                if g.eta(i, e) > early + 1 or dist(g.moves[i].get(old, p), target) >= dist(p, target) - .25:
                    continue
                lateral = [(dist(q, centers[r]), abs(dist(q, target) - dist(p, target)), a)
                           for a, q in moves if abs(dist(q, target) - dist(p, target)) <= .5
                           and dist(q, centers[r]) <= dist(p, centers[r]) + .5]
                new = min(lateral)[2] if lateral else 1
                if not g.masks[i][new] or not g.fire(i, old) <= g.fire(i, new):
                    continue
            else:
                if not moves:
                    continue
                best, new = min((dist(q, target), a) for a, q in moves)
                if best >= min(dist(p, target), dist(g.moves[i].get(old, p), target)) - .25:
                    continue
            actions[i] = new
            changes.append(dict(actor=i, role='front' if j == f else 'rear', previous=old, action=new))
    if changes:
        return dict(target=e, front=f, rear=r, eta=[early, late], changes=changes, local_shared_squads=protected)


class CrossSupportRendezvous:
    def __init__(self, mode='guard'):
        if mode not in ('guard', 'front'):
            raise ValueError('mode must be guard or front')
        self.mode = mode
        self.parent = SquadRendezvous('adaptive')
        self.beats = 0
        self.groups = self.exit_reason = None

    def propose(self, payload, masks, base_actions):
        out = self.parent.propose(payload, masks, base_actions)  # Includes legal-vector validation.
        base = list(map(int, base_actions))

        def finish(reason):
            self.exit_reason = reason
            return dict(actions=base, phase='v2', handoff=True, reason=reason)

        if self.exit_reason or self.beats >= 8:
            return finish(self.exit_reason or 'timeout')
        if out['handoff'] and (self.mode == 'guard' or out['reason'] not in ('formed_or_blocked', 'shared_target')):
            return finish(out['reason'])
        s = payload['state']
        if self.mode == 'front':
            # The frozen parent may already be latched on a same-squad mask.
            units = list(s['allies'].values())
            if any(not u['alive'] or any(u.get(k + '_lost_since_previous_decision', 0) > 0
                   for k in ('health', 'shield')) for u in units):
                return finish('casualty_or_damage')
            if any(dist(u['position'], e['position']) <= 2.5 for u in units
                   for e in s['visible_enemies'].values() if e['alive']):
                return finish('close_enemy')
            if self.groups is None and all(dist(u['position'], v['position']) < 12 for u, v in combinations(units, 2)):
                return finish('compact_start')
        if self.groups is None:
            self.groups = self.parent.groups or _squads({i: s['allies'][f'u{i}']['position'] for i in range(len(base))})
        if self.mode == 'front' and any(sum(any(masks[i][a] for i in group) for group in self.groups) >= 2
                                        for a in range(6, len(masks[0]))):
            return finish('cross_squad_shared_target')
        actions = out['actions'].copy()
        g = _Geometry(s, masks)
        veto = _guard(g, self.groups, base, actions)
        front = _front(g, self.groups, actions, {v['actor'] for v in veto}) if self.mode == 'front' else None
        if out['handoff'] and not front:
            return finish(out['reason'])
        self.beats += 1
        if not veto and not front:
            return out
        return dict(actions=actions, phase='rendezvous', handoff=False, squads=self.groups,
                    reason='front_sync' if front else 'veto_split_contact', veto=veto, front=front)
