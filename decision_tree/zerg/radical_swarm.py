"""Experimental 10v10 macro policy. Instantiate once per episode, call once per beat.

No candidate menu, engine object, native micro, or terminal feedback. Enemy slots
are indexed only from this beat's team-visible enemies; last contact is a public
waypoint for a few beats, never a targetable unit behind fog.
"""
from math import dist, hypot
from statistics import median


ROLE = {'hydralisk': 'gunline', 'zergling': 'screen', 'baneling': 'burst'}
THREAT = {'baneling': (4., 3.), 'zergling': (3., 1.6), 'hydralisk': (6., .65)}
# Offsets from the public search location, not inferred enemy spawn points.
PATROL = ((0, 0), (8, 8), (-8, 8), (-8, -8), (8, -8), (0, 0),
          (10, 0), (0, 10), (-10, 0), (0, -10))


def center(points):
    return tuple(median(axis) for axis in zip(*points))


def toward(p, q, length):
    dx, dy = q[0] - p[0], q[1] - p[1]
    ratio = min(1, length / max(0.001, hypot(dx, dy)))
    return p[0] + dx * ratio, p[1] + dy * ratio


def choose_move(i, state, masks, cost, force=False):
    uid = f'u{i}'
    moves = state['movement_candidates'][uid]
    options = [(cost(moves[str(a)]['destination']), a) for a in range(2, 6) if masks[i][a]]
    if masks[i][1] and not (force and options):
        options.append((cost(state['allies'][uid]['position']), 1))
    if not options:
        raise ValueError(f'{uid}: no legal stop/move')
    return min(options)[1]


class RadicalSwarm:
    def __init__(self):
        self.blind = self.leg = self.leg_age = self.stuck = 0
        self.last_contact = self.last_center = None
        self.last_hp = {}
        self.last_gap = None
        self.seen_beats = 0
        self.quiet = self.push = 0

    def _opening(self, s, masks, live, p, core, roles, actions, reasons):
        self.blind += 1
        self.seen_beats = 0
        self.quiet = self.push = 0
        self.last_hp, self.last_gap = {}, None
        anchor = s['search_location']
        if self.last_contact is not None and self.blind <= 3 and dist(core, self.last_contact) > 2:
            goal, phase = self.last_contact, 'reacquire'
        elif self.last_contact is None and self.blind <= 4 and max(dist(p[i], core) for i in live) > 5:
            goal, phase = core, 'muster'
        else:
            self.leg_age += 1
            self.stuck = self.stuck + 1 if self.last_center and dist(core, self.last_center) < .35 else 0
            dx, dy = PATROL[self.leg]
            goal = anchor[0] + dx, anchor[1] + dy
            # Reaching a waypoint, terrain/collisions, and a long leg all end it.
            if dist(core, goal) < 2 or self.stuck >= 3 or self.leg_age >= 16:
                self.leg = (self.leg + 1) % len(PATROL)
                self.leg_age = self.stuck = 0
                dx, dy = PATROL[self.leg]
                goal = anchor[0] + dx, anchor[1] + dy
            phase = 'sweep'
        self.last_center = core
        for i in live:
            # Keep the scouts ahead, but not an entire sight radius ahead.
            reach = 6.0 if roles[i] == 'screen' else 3.8
            waypoint = toward(core, goal, reach)
            actions[i] = choose_move(i, s, masks, lambda q: dist(q, waypoint), force=phase == 'sweep' and dist(p[i], waypoint) > 1)
            reasons[i] = phase
        return phase

    def choose(self, raw_pre, avail_actions):
        s = raw_pre['state']
        units = [s['allies'][f'u{i}'] for i in range(10)]
        masks = avail_actions
        if len(masks) != 10 or any(len(row) != 16 or any(v not in (0, 1) for v in row) for row in masks):
            raise ValueError('expected ten binary masks of width 16')
        live = [i for i, u in enumerate(units) if u['alive']]
        if any(bool(row[0]) == bool(u['alive']) or not any(row) for row, u in zip(masks, units)):
            raise ValueError('mask conflicts with actor life state')
        roles = [ROLE[u['type'].lower()] if u['alive'] else 'dead' for u in units]
        actions, reasons = [0] * 10, ['dead'] * 10
        p = {i: units[i]['position'] for i in live}
        # All target coordinate access descends from this single visibility gate.
        enemy = {int(k[1:]): e for k, e in s['visible_enemies'].items() if e['alive']}
        if not live:
            phase = 'empty'
        else:
            guns = [i for i in live if roles[i] == 'gunline']
            core = center([p[i] for i in (guns or live)])
            if not enemy:
                phase = self._opening(s, masks, live, p, core, roles, actions, reasons)
            else:
                phase = self._fight(s, masks, units, live, p, core, roles, guns, enemy, actions, reasons)
        if any(not masks[i][a] or (a >= 6 and a - 6 not in enemy) for i, a in enumerate(actions)):
            raise ValueError('generated action violates SMAC mask or team visibility')
        return dict(actions=actions, phase=phase, roles=roles, reasons=reasons)

    def _fight(self, s, masks, units, live, p, core, roles, guns, enemy, actions, reasons):
        xy = {j: e['position'] for j, e in enemy.items()}
        hp = {j: e['health'] for j, e in enemy.items()}
        kind = {j: e['type'].lower() for j, e in enemy.items()}
        gap = min(dist(pos, q) for pos in p.values() for q in xy.values())
        progress = (self.last_gap is None or gap + .65 < self.last_gap or
                    any(hp[j] < self.last_hp.get(j, hp[j]) for j in enemy) or bool(self.last_hp.keys() - hp.keys()))
        self.quiet = 0 if progress else self.quiet + 1
        if self.quiet >= 5:
            self.push, self.quiet = 4, 0
        pushing = self.push > 0
        self.push = max(0, self.push - 1)
        self.blind = self.leg_age = self.stuck = 0
        self.seen_beats += 1
        self.last_center = None
        self.last_hp, self.last_gap = hp, gap

        # Pick one reachable seam for units not already trading. Local ready
        # shots are never held waiting for a remote target or a quorum.
        def front_cost(j):
            support = sum(dist(p[i], xy[j]) < 6.5 for i in (guns or live))
            return (dist(xy[j], core) - 1.1 * support - 1.5 * (kind[j] == 'baneling')
                    + 1.2 * hp[j] / enemy[j]['max_health'], j)

        front = min(enemy, key=front_cost)
        self.last_contact = goal = xy[front]
        line = toward(core, goal, 3.2)
        # R2 common severe losses had especially wide first sight lines and
        # remote escorts. Bring the line to a nearby escort before marching
        # everyone at the enemy; local attack orders always take precedence.
        escorts = [i for i in live if i not in guns]
        spread = max(dist(q, core) for q in p.values())
        link = len(guns) >= 2 and self.seen_beats <= 4 and spread >= 7 and not pushing
        if link:
            escort = min(escorts, key=lambda i: dist(p[i], core)) if escorts else None
            line = toward(core, p[escort], 1.8) if escort is not None else core
        reserved = dict.fromkeys(enemy, 0.)
        # Reserve burst footprints before allocating ordinary damage. These
        # are one-beat nominal budgets, not claims of real hits or kills.
        splash = {j: [k for k in enemy if dist(xy[j], xy[k]) <= 2.1] for j in enemy}
        gun_slots = {i: [j for j in enemy if masks[i][6 + j]] for i in guns}
        ready_slots = {j for i in guns if units[i]['weapon_ready'] for j in gun_slots[i]}
        banes = sorted((i for i in live if roles[i] == 'burst'),
                       key=lambda i: min(dist(p[i], q) for q in xy.values()))
        for i in banes:
            def burst_value(j):
                return sum(min(max(0, hp[k] - reserved[k]), 30 if kind[k] == 'zergling' else 16)
                           * (.6 if k != j else 1) for k in splash[j])

            options = [j for j in enemy if masks[i][6 + j]]
            j = max(options, key=lambda k: (burst_value(k), -dist(p[i], xy[k]), -k), default=None)
            weak = j is not None and (burst_value(j) < 12 or
                                      (hp[j] <= 12 and j in ready_slots and len(splash[j]) == 1))
            # Avoid double spending even with no ranged escort. A lone
            # surviving burst unit must still be allowed to finish its target.
            if j is not None and (not weak or units[i]['health'] <= 10 or
                                  (not guns and reserved[j] == 0)):
                actions[i], reasons[i] = 6 + j, 'burst_commit'
                for k in splash[j]:
                    reserved[k] += 30 if kind[k] == 'zergling' else 16
            else:
                aim = max(enemy, key=lambda k: (burst_value(k) - 1.8 * dist(p[i], xy[k])
                                                - .6 * dist(line, xy[k]), -k))
                destination = (core if weak and aim == j else xy[aim])
                if link or (guns and dist(p[i], core) > 5 and dist(p[i], destination) > 3 and not pushing):
                    destination = line
                actions[i] = choose_move(i, s, masks, lambda q: dist(q, destination))
                reasons[i] = 'burst_conserve' if weak else 'burst_stage'

        # Actors with fewer alternatives allocate first, preserving flexible
        # shooters for other targets. Focus wounded targets until budget is met.
        for i in sorted((k for k in live if k not in banes), key=lambda k: (sum(masks[k][6:]), k)):
            is_gun = roles[i] == 'gunline'
            options = [j for j in enemy if masks[i][6 + j]]
            ready = units[i]['weapon_ready']
            def shot_cost(j):
                remaining = hp[j] - reserved[j]
                return (remaining <= 0, max(0, remaining) * .085 + dist(p[i], xy[j]) * .6
                        - 4 * (kind[j] == 'baneling' and dist(core, xy[j]) < 6)
                        - 1.5 * (kind[j] == 'hydralisk') - (j == front), -remaining if remaining <= 0 else 0, j)

            j = min(options, key=shot_cost, default=None)
            if j is not None and (ready or not is_gun):
                actions[i], reasons[i] = 6 + j, 'volley' if is_gun else 'intercept'
                reserved[j] += (12 if is_gun else 5) if ready else 0
                continue

            if is_gun and not ready and options:
                # For near-ready weapons maintain target lock unless a move
                # improves safety. Don't kite out of next-shot range for free.
                def danger(q):
                    return sum(max(0, THREAT[kind[k]][0] - dist(q, xy[k])) * THREAT[kind[k]][1]
                               for k in enemy)

                def cost(q):
                    nearest = min(dist(q, pos) for pos in xy.values())
                    return danger(q) + 1.2 * max(0, nearest - 4.8) + .3 * max(0, dist(q, core) - 3)

                a = choose_move(i, s, masks, cost)
                moves = s['movement_candidates'][f'u{i}']
                if a >= 2 and danger(moves[str(a)]['destination']) + .2 < danger(p[i]) and (
                        units[i]['weapon_cooldown_loops'] > 3 or danger(p[i]) > 4):
                    actions[i], reasons[i] = a, 'cooldown_evade'
                else:
                    actions[i], reasons[i] = 6 + j, 'keep_attack'
                continue

            if is_gun:
                # Enforce contact when the gun is just outside its SMAC attack
                # mask; do not stop exactly at nominal native weapon range.
                destination = goal if pushing or (not link and dist(p[i], core) < 4) else line
                reasons[i] = 'break_stall' if pushing else ('link_up' if link else 'join_volley')
            else:
                threats = [k for k in enemy if any(dist(p[g], xy[k]) <= 5 for g in guns)]
                if threats:
                    dest = min(threats, key=lambda k: (dist(p[i], xy[k]), k))
                    destination = xy[dest]
                else:
                    destination = line if link else (goal if pushing or not guns else toward(core, goal, 5.5))
                reasons[i] = 'screen_advance' if not threats else 'cover_guns'
            actions[i] = choose_move(i, s, masks, lambda q: dist(q, destination), force=pushing)
        return 'break_stall' if pushing else ('link_up' if link else 'engage')
