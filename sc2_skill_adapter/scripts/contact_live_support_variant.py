"""Install live Zergling-contact support in a private contact_response copy."""
from hashlib import sha256
from inspect import getsource
from pathlib import Path
from tempfile import NamedTemporaryFile


def _live_support(ctx, memory, actions, summary):
    from math import atan2, cos, sin
    from .action_duties import rescue_actions
    from .context import STOP, distance
    from .hydra_support import _safe_moves, support_path_clear

    snap = ctx.snapshot
    units = snap.allies
    if {u.type_name for u in units.values()} != {'zergling', 'hydralisk'}:
        return
    orders = dict(summary['actions'])
    contacts = {}
    for dog, unit in units.items():
        if not unit.alive or unit.type_name != 'zergling':
            continue
        legal = ctx.attackable_enemies(dog)  # visible AND currently issuable
        if not legal:
            continue
        command = ctx.decode(dog, orders[dog])
        native = snap.raw_state['allies'][dog].get('orders') or []
        current = native[0].get('target_unit') if native else None
        if command.kind == 'attack' and command.target in legal:
            target = command.target
        elif current in legal:
            target = current
        else:
            target = legal[0]  # nearest legal contact of this dog, not team focus
        contacts.setdefault(target, []).append(dog)
    if not contacts:
        return

    for actor, gun in units.items():
        if (not gun.alive or gun.type_name != 'hydralisk' or gun.max_health <= 0
                or gun.health_fraction < .60 or any(a >= 6 for a in ctx.legal_actions(actor))):
            continue  # even a cooling Hydra keeps its legal contact
        old = orders[actor]
        row = summary['units'][actor]
        duty = summary['execution_duties']['actors'].get(actor)
        if (duty is None or (old != STOP and old not in ctx.move_destinations(actor))
                or row['skill'] not in ('ranged_fire_formation', 'baseline') or row['plan'] is not None
                or duty.get('kind') not in (None, 'entry') or duty.get('branch_reason')
                or duty.get('joint_rescue') or duty.get('isolated_shooter_support')
                or old in rescue_actions(ctx, actor, memory)):
            continue

        origin = gun.position
        target = min(contacts, key=lambda e: (distance(origin, snap.enemies[e].position), e))
        goal = snap.enemies[target].position
        dog = min(contacts[target], key=lambda d: (distance(origin, units[d].position), d))
        reach = ctx.effective_attack_reach(actor, target)
        if reach is None or reach <= 0:
            continue
        gap = max(0., distance(origin, goal) - reach)
        risk = ctx.risk_at(actor, origin)
        threats = set(risk.threats)
        if old != STOP:
            incumbent = ctx.predict_position(actor, old).position
            if distance(incumbent, goal) < distance(origin, goal) - 1e-6:
                continue  # already closing on this fight; do not optimize its angle
            # A formation escape can be labelled simply 'entry'. Preserve it.
            if any(distance(incumbent, snap.enemies[e].position) >
                   distance(origin, snap.enemies[e].position) + .05
                   for e in (*threats, *risk.effect_weapon_near)):
                continue

        # Aim just inside either side of the dog's contact, not at the dog's body.
        front = units[dog].position
        if distance(front, goal) < .01:
            front = origin
        angle = atan2(front[1] - goal[1], front[0] - goal[0])
        radius = max(0., reach - .15)
        flanks = [(goal[0] + radius * cos(angle + turn), goal[1] + radius * sin(angle + turn))
                  for turn in (-1.05, 1.05)]
        choices = []
        for move, end in _safe_moves(ctx, memory, actor):
            remaining = max(0., distance(end, goal) - reach)
            if remaining >= gap - .01 or not support_path_clear(ctx, actor, end, orders):
                continue
            fresh = set(ctx.risk_at(actor, end).threats) - threats
            if len(fresh) <= 1:
                choices.append((remaining > 0, len(fresh), min(distance(end, p) for p in flanks),
                                remaining, move))
        if choices:
            move = min(choices)[-1]
            orders[actor] = actions[gun.index] = summary['actions'][actor] = row['action'] = move
            row.update(phase='live_support', goal=target, goal_kind='enemy')
            # execution_duties and joint scores describe the scored baseline branch.
            summary.setdefault('live_support', {})[actor] = {
                'before': old, 'after': move, 'target': target, 'dog': dog}


def _screen_fire(ctx, memory, actions, summary):
    from math import dist
    allies, foes = ctx.snapshot.allies, ctx.snapshot.enemies
    if {u.type_name for u in allies.values()} != {'zergling', 'hydralisk'}:
        return
    guns = [u for u in allies.values() if u.alive and u.type_name == 'hydralisk']
    eligible = {a for a, u in allies.items() if u in guns and actions[u.index] >= 6
                and u.weapon_cooldown_loops <= 0}
    targets = sorted((min(dist(e.position, g.position) for g in guns), e.health, t)
                     for t, e in foes.items() if guns and e.type_name == 'zergling')
    for gap, hp, target in targets:
        if gap > 3 or hp <= 0:
            continue
        candidates = []
        for actor in eligible:
            cmd = ctx.attack_action(actor, target)
            if cmd is not None:
                dmg = ctx.fire_estimate(actor, target).damage_lo
                if dmg and dmg > 0:
                    old = actions[allies[actor].index]
                    candidates.append((old != cmd, dist(allies[actor].position, foes[target].position), actor, cmd, dmg))
        if sum(x[-1] for x in candidates) < hp:
            continue
        for _, _, actor, cmd, dmg in sorted(candidates):
            eligible.remove(actor)
            i = allies[actor].index
            old = actions[i]
            if old != cmd:
                actions[i] = summary['actions'][actor] = summary['units'][actor]['action'] = cmd
                summary['units'][actor].update(phase='screen_fire', goal=target, goal_kind='enemy')
                summary.setdefault('live_support', {})[actor] = dict(before=old, after=cmd, target=target)
            hp -= dmg
            if hp <= 0:
                break


def apply_variant(skill_dir: Path, *, screen_fire=False) -> None:
    """Verify the frozen route, then atomically patch only the copy's manager."""
    workspace = Path(__file__).resolve().parents[2]
    frozen = workspace / 'runs/contact_540_20261120_225957_302289'
    root = Path(skill_dir).resolve(strict=True)
    target = root / 'skill_manager.py'
    if (frozen in root.parents or root in (workspace / 'skills/tactical_v1', workspace / 'skills/current')
            or target.resolve(strict=True) != target):
        raise ValueError('expected an independent contact_response copy')
    digests = {
        'coordinator.py': '58927d61c3050600e42f5d1a994e1a70364a7661e744c934637fa5583b1380c2',
        'skills/ranged_fire_formation.py': 'df049f121fdbaeec9459c7c264bb72b7f83f4720120e55f46da3bd898dfb3e14',
    }
    raw = target.read_bytes()
    if (sha256(raw).hexdigest() != '038350fe4c723d60b03034277aca4e369b920f381f8804a11e1a5495275206a5'
            or any(sha256((root / name).read_bytes()).hexdigest() != digest
                   for name, digest in digests.items())):
        raise ValueError('expected unmodified contact_response manager and formation route')
    anchor = '        summary["battle_intent"] ='
    hook = _screen_fire if screen_fire else _live_support
    source = raw.decode().replace('\r\n', '\n').replace(
        'class SkillManager:', getsource(hook) + '\n\nclass SkillManager:').replace(
        anchor, f'        {hook.__name__}(context, self.memory, actions, summary)\n' + anchor)
    compile(source, str(target), 'exec')
    with NamedTemporaryFile(dir=root, delete=False) as handle:
        tmp = Path(handle.name)
        try:
            handle.write(source.replace('\n', '\r\n' if b'\r\n' in raw else '\n').encode())
            handle.flush()
            tmp.chmod(target.stat().st_mode)
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)
