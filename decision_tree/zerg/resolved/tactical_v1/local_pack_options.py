"""Optional physical comparisons for the frozen local-pack experiment."""
from collections import defaultdict
from math import dist


class PackValue:
    """Score identical complete commands identically, irrespective of plan ownership.

    A projected ray is an option, never damage. Pressure is a discounted count
    of distinct ready commands, not prevented incoming shots. Only a completed
    lower-bound volley earns the threat-removal term.
    """
    def __init__(self, ctx, packs, reference):
        self.ctx = ctx
        self.rays, self.steps = defaultdict(set), defaultdict(set)
        self.front, self.protected = {}, {}
        for p in packs:
            for a, es in p['fire_rays'].items():
                self.rays[a].update(es)
            for a, es in p['one_step_rays'].items():
                self.steps[a].update(es)
            for e in p['enemies']:
                self.front[e] = p['pack_id']
                self.protected[e] = [ctx.snapshot.allies[b] for b, es in p['incoming'].items()
                                     if e in es and ctx.snapshot.allies[b].type_name == 'hydralisk']
        self.orders = {}
        self.baseline = self.measure(reference)

    def order(self, a, cmd):
        key = a, cmd
        if key not in self.orders:
            ctx = self.ctx
            sem = ctx.decode(a, cmd)
            damage, projected = {}, set()
            if sem.kind == 'attack' and sem.target in self.rays[a]:
                est = ctx.fire_estimate(a, sem.target)
                if est.window_shots and est.damage_lo is not None and est.damage_lo > 0:
                    damage[sem.target] = est.damage_lo
            elif sem.kind == 'move' and self.steps[a]:
                end = ctx.predict_position(a, cmd).position
                for e in self.steps[a]:
                    reach = ctx.effective_attack_reach(a, e)
                    if reach is not None and dist(end, ctx.snapshot.enemies[e].position) <= reach:
                        projected.add(e)
            self.orders[key] = damage, projected
        return self.orders[key]

    def measure(self, actions):
        volleys, entry = defaultdict(list), defaultdict(float)
        assigned = {}
        for a, cmd in actions.items():
            fire, steps = self.order(a, cmd)
            for e, damage in fire.items():
                volleys[e].append(damage)
            for e in steps:
                # A move opening two possible shots does not supply two guns.
                entry[e] += 1. / len(steps)
            assigned[a] = set(fire) | steps

        total = 0.
        for e in sorted(volleys.keys() | entry.keys()):
            enemy = self.ctx.snapshot.enemies[e]
            hp = enemy.effective_health
            shots = sorted(volleys[e], reverse=True)
            remaining, pressure = hp, 0.
            for i, dmg in enumerate(shots):
                pressure += min(1., remaining/dmg) / (1+i)
                remaining = max(0., remaining-dmg)
            # New geometry matters most on an uncovered, still-live gun. Do
            # not count the reference's existing entry again in the delta.
            forecast = .55 * min(1., entry[e]) * (remaining/hp) / (1+len(shots))
            total += (1.8 if enemy.type_name == 'hydralisk' else .8) * (pressure + forecast)
            if remaining == 0:
                total += 2.
                if self.protected[e]:
                    urgency = max(1. - u.health_fraction for u in self.protected[e])
                    total += (3. if enemy.type_name == 'zergling' else 2.) * (1.+urgency)
        return total, assigned

    def delta(self, actions):
        total, assigned = self.measure(actions)
        before, old = self.baseline
        # Price diversion from actual reference work on a distinct front.
        # An idle actor, or one whose physical entry still covers its original
        # line, is not charged just for changing pack/plan labels.
        for a, targets in assigned.items():
            if old[a] and targets and not ({self.front[e] for e in old[a]} &
                                           {self.front[e] for e in targets}):
                total -= 1.
        return total - before

    def redirect(self, w, a, e):
        # Called AFTER forced/rescue/lethal-volley/other-front gates and only
        # for window attacks within this pack. Keep a scarce near-ling shot.
        old = w.shot[a]
        if (self.ctx.snapshot.enemies[old].type_name == 'zergling' and self.protected[old]
                and w.ref_lower[old] < self.ctx.snapshot.enemies[old].effective_health):
            return False
        if e not in w.rays[a] or not w.est[a, e].window_shots:
            return False
        return self.delta({**w.ref, a: self.ctx.attack_action(a, e)}) > 0


def home_line(w, a, request, packs):
    """Tagged witnesses near the departing reserve on OTHER contested fronts."""
    ctx, s = w.ctx, w.ctx.snapshot
    here = s.allies[a].position
    watch = {}
    for p in packs:
        if p['pack_id'] == request['pack_id']:
            continue
        for b in p['direct']:
            peer = s.allies[b]
            # A range for the donor side's existing weapon, not a new radius
            # for pack admission. Keep a melee witness on its native line too.
            if not any(dist(here, peer.position) <= (ctx.effective_attack_reach(a, e) or 0.)
                       for e in p['enemies']):
                continue
            rays = {w.enemies[e] for e in p['fire_rays'][b]}
            hp, existing = watch.get(w.allies[b], (peer.effective_health, set()))
            watch[w.allies[b]] = hp, existing | rays
    return {tag: (hp, sorted(rays)) for tag, (hp, rays) in watch.items()}


def near_bid(w, a, pack, live, same, nearby, previous):
    # No new hire during another task's final/failed continuation. No
    # diversion from a locally eligible competing front, even for a lease.
    if nearby - set(pack['enemies']):
        return False
    if not same:
        return live is None and a not in previous.get('prior_claims', {})
    ctx, s = w.ctx, w.ctx.snapshot
    saved = live.extra
    if s.game_loop != saved['last_move_loop'] + ctx.H:
        return False
    # Relative gap alone can shrink because the enemy moved. Also require
    # the helper to have made progress towards the last named position.
    if dist(s.allies[a].position, saved['enemy_position']) >= saved['gap'] - .05:
        return False
    allies = {tag: b for b, tag in w.allies.items()}
    enemies = set(w.enemies.values())
    for tag, (hp, rays) in saved['home_line'].items():
        b = allies.get(tag)
        if b is None or s.allies[b].effective_health < hp:
            return False
        actual = {w.enemies[t] for t in w.rays[b]}
        if (set(rays) & enemies) - actual:
            return False
    return True
