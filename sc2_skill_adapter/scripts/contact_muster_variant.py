"""Install a one-shot opening muster in a private frozen micro8 policy."""
from hashlib import sha256
from inspect import getsource
from pathlib import Path
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/contact_response_micro_20261121_153552_909880'
FROZEN = RUN / 'source_snapshot/skills/tactical_v1'
PACKAGE = '7dbc65b5b74eaa9c76e105b220964f767bdcc39c81fc00788892f85537cc11ed'


class _OpeningMuster:
    """Episode state: the rally point is a coordinate, not a moving leader."""
    radius = 4.0
    limit = 48

    def __init__(self):
        self.start = self.anchor = self.done = None
        self.slots = {}

    @staticmethod
    def _danger(ctx, actor, point):
        risk = ctx.risk_at(actor, point)
        return bool(risk.threats or risk.effect_weapon_near)

    def _choose(self, ctx, dogs, guns, released):
        from math import dist

        allies = ctx.snapshot.allies
        pos = {a: u.position for a, u in allies.items()}
        center = tuple(sum(p[i] for p in pos.values()) / len(pos) for i in (0, 1))
        enemies = [e.position for e in ctx.snapshot.enemies.values()]
        moves = {a: ctx.move_destinations(a) for a in allies}
        choices = []
        for g in guns:
            point = pos[g]
            # Never pick a lead gun closer to a visible enemy than the team's center.
            if (released.get(g) in ('contact', 'support_contact') or
                    any(dist(point, e) < dist(center, e) for e in enemies)):
                continue
            # Without a layout we can establish only local progress, not promise a path.
            if any(dist(p, point) > self.radius and not any(
                    dist(end, point) < dist(p, point) and not self._danger(ctx, a, end)
                    for end in moves[a].values()) for a, p in pos.items() if a not in released):
                continue
            # Clip the slot grid to open movement directions at this observed gun.
            xs = [0] + [i for i in (-2, -1, 1, 2) if (4 if i > 0 else 5) in moves[g]]
            ys = [0] + [i for i in (-2, -1, 1, 2) if (2 if i > 0 else 3) in moves[g]]
            slots = [(point[0] + 1.35*x, point[1] + 1.35*y) for x in xs for y in ys
                     if x*x + y*y <= 5]
            slots = [p for p in slots if not self._danger(ctx, g, p)]
            if len(slots) >= len(allies):
                choices.append((sum(dist(point, pos[a]) for a in guns),
                                sum(dist(point, p) for p in pos.values()), g, slots))
        if not choices:
            return False
        _, _, lead, slots = min(choices)
        self.anchor = pos[lead]
        # Central gun slots, then dogs on the rim. Assign once, near members first.
        slots.sort(key=lambda p: (dist(p, self.anchor), p))
        for group, free in ((guns, slots[:len(guns)]), (dogs, slots[len(guns):])):
            for a in sorted(group, key=lambda a: (dist(pos[a], self.anchor), a)):
                self.slots[a] = min(free, key=lambda p: (dist(pos[a], p), p))
                free.remove(self.slots[a])
        return True

    def apply(self, ctx, actions, summary, plans):
        from math import dist
        from .actions import STOP

        snap = ctx.snapshot
        units = snap.allies
        report = summary['opening_muster'] = dict(active=False, reason=self.done,
                anchor=self.anchor, changes={}, held=[], released={})
        if self.done:
            return plans
        if self.start is None:
            self.start = snap.game_loop
        dogs = [a for a, u in units.items() if u.alive and u.type_name == 'zergling']
        guns = [a for a, u in units.items() if u.alive and u.type_name == 'hydralisk']
        if len(units) != 10 or (len(dogs), len(guns)) not in ((3, 7), (6, 4)):
            self.done = 'outside_roster' if self.anchor is None else 'roster_loss'
        elif snap.game_loop - self.start >= self.limit:
            self.done = 'timeout'
        elif any(all(sum(dist(units[g].position, units[a].position) <= self.radius for a in group)
                         > len(group)//2 for group in (dogs, guns)) for g in guns):
            self.done = 'already_grouped'
        if self.done:
            report['reason'] = self.done
            return plans

        released = report['released']
        engaged = set()
        for a, u in units.items():
            rels = snap.relations.get(a, {})
            if (u.health_lost > 0 or u.shield_lost > 0 or self._danger(ctx, a, u.position)
                    or any(r.in_enemy_weapon_range is True or (r.in_weapon_range is True
                           and ctx.can_issue_attack(a, e)) for e, r in rels.items())):
                engaged.add(a)
                released[a] = 'contact'
            else:
                orders = snap.raw_state['allies'][a].get('orders') or []
                if (ctx.decode(a, actions[u.index]).kind == 'attack'
                        or (orders and orders[0].get('target_unit') in snap.enemies)):
                    released[a] = 'attack'
        # Nearby helpers keep their original response; distant troops still rally.
        for a, u in units.items():
            if any(dist(u.position, units[b].position) <= self.radius for b in engaged):
                released.setdefault(a, 'support_contact')
        if len(engaged) >= 5:
            self.done = 'team_engaged'
        elif len(released) == len(units):
            report['reason'] = 'local_release'
            return plans  # No muster command this window, but not a terminal transition.
        elif self.anchor is None:
            if not self._choose(ctx, dogs, guns, released):
                self.done = 'no_safe_anchor'
        elif self._danger(ctx, guns[0], self.anchor):
            self.done = 'anchor_threatened'
        report.update(reason=self.done or 'gathering', anchor=self.anchor)
        if self.done:
            return plans

        held = report['held']
        reserved = {a: u.position for a, u in units.items()}
        for a in sorted(units, key=lambda a: (dist(units[a].position, self.slots[a]), a)):
            if a in released:
                continue
            u = units[a]
            before = actions[u.index]
            slot = self.slots[a]
            choices = []
            if STOP in ctx.legal_actions(a):
                choices.append((dist(u.position, slot), STOP, u.position))
            for cmd, end in ctx.move_destinations(a).items():
                if self._danger(ctx, a, end) or dist(end, slot) >= dist(u.position, slot) - .05:
                    continue
                # Reserve space for stationary bodies and this window's other destinations.
                if any(dist(p, end) < u.radius + units[b].radius for b, p in reserved.items() if b != a):
                    continue
                choices.append((dist(end, slot), cmd, end))
            if not choices:
                released[a] = 'no_safe_command'
                continue
            _, cmd, reserved[a] = min(choices)
            held.append(a)
            if cmd != before:
                report['changes'][a] = dict(before=before, after=cmd)
            actions[u.index] = summary['actions'][a] = cmd
            summary['units'][a].update(skill='opening_muster', phase='gathering',
                        provider=None, plan=None, goal='muster_slot', goal_kind='area',
                        slot=slot, action=cmd)

        # Ownership, not integer inequality: even an identical move/stop is now
        # a rally command. A partially superseded atomic plan cannot be committed.
        report['active'] = bool(held)
        if not held:
            report['reason'] = 'local_release'
            return plans
        kept = [p for p in plans if set(held).isdisjoint(p.controlled_actors)]
        ids = {p.proposal_id for p in kept}
        summary['chosen_plans'] = [p for p in summary['chosen_plans'] if p['id'] in ids]
        for row in summary['units'].values():
            if row['plan'] is not None and row['plan'] not in ids:
                row['plan'] = None
        return kept


def apply_variant(skill_dir: Path) -> None:
    """Verify package identity before atomically replacing the copy's manager."""
    root = Path(skill_dir).resolve(strict=True)
    if RUN in (root, *root.parents) or ROOT / 'skills' in (root, *root.parents):
        raise ValueError('expected an independent copy outside frozen run and live skills')
    digest = sha256()
    for p in sorted(root.rglob('*')):
        rel = p.relative_to(root)
        if p.is_symlink():
            raise ValueError('expected an independent package without symlinks')
        if p.suffix in ('.py', '.json') and 'tests' not in rel.parts:
            digest.update(rel.as_posix().encode() + b'\0' + sha256(p.read_bytes()).digest())
    if digest.hexdigest() != PACKAGE:
        raise ValueError('expected unmodified frozen contact_response_micro package')
    manager = root / 'skill_manager.py'
    raw = manager.read_bytes()
    text = raw.decode().replace('\r\n', '\n')
    for old, new in (
        ('class SkillManager:', getsource(_OpeningMuster) + '\n\nclass SkillManager:'),
        ('    def reset(self) -> None:\n', '    def reset(self) -> None:\n        self._muster = _OpeningMuster()\n'),
        ('        summary["battle_intent"] =', '        plans = self._muster.apply(context, actions, summary, plans)\n        summary["battle_intent"] ='),
    ):
        if text.count(old) != 1:
            raise ValueError('unexpected manager hook location')
        text = text.replace(old, new)
    compile(text, str(manager), 'exec')
    with NamedTemporaryFile(dir=root, delete=False) as f:
        tmp = Path(f.name)
        try:
            f.write(text.replace('\n', '\r\n' if b'\r\n' in raw else '\n').encode())
            f.close()
            tmp.chmod(manager.stat().st_mode)
            tmp.replace(manager)
        finally:
            tmp.unlink(missing_ok=True)
