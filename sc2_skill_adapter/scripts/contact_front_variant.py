"""Install the pre-contact Zergling route experiment in a private frozen copy."""
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

if __package__:
    from .contact_early_variant import BATCH, DIGEST, ROOT, ROUTE, SOURCE
else:
    from contact_early_variant import BATCH, DIGEST, ROOT, ROUTE, SOURCE


# Installed inline: a frozen run does not import the experiment harness.
_HOOK = '''
def _front_route(ctx, memory, actions, summary):
    from .context import distance
    from .hydra_support import _safe_moves, support_path_clear
    from .joint_combat import action_position

    snap = ctx.snapshot
    units = snap.allies
    dogs = [a for a, u in units.items() if u.type_name == 'zergling']
    guns = [a for a, u in units.items() if u.alive and u.type_name == 'hydralisk']
    changes = {}
    summary['front_route'] = {'changes': changes}
    if snap.snapshot_id == 1:
        memory.front_open = (len(units) == 10 and all(u.alive for u in units.values())
                             and (len(dogs), len(guns)) in ((3, 7), (6, 4)))
        memory.front_engaged = set()
    if not memory.front_open:
        return
    if any(any(cmd >= 6 for cmd in ctx.legal_actions(a)) for a in dogs):
        memory.front_open = False
        return
    orders = dict(summary['actions'])
    gun_ends = {a: action_position(ctx, a, orders[a])[0] for a in guns}
    ranges = snap.raw_state.get('action_contract', {}).get('candidate_center_distance_by_type', {})
    if not guns or not ranges.get('hydralisk') or not ranges.get('zergling'):
        return
    foes = {e: u for e, u in snap.enemies.items() if u.type_name == 'hydralisk'}
    enemy_reach = {e: ctx._enemy_weapon_range(e, False) for e in foes}
    if None in enemy_reach.values():
        return  # missing catalog is not evidence of zero opposing fire
    for a in dogs:
        u = units[a]
        origin = u.position
        if (not u.alive or u.health < u.max_health or u.weapon_cooldown_loops > 0 or
                memory.last_committed_actions.get(a, 0) >= 6 or any(
                    order.get('kind') == 'attack' for order in snap.raw_state['allies'][a].get('orders', ()))):
            memory.front_engaged.add(a)
        row = summary['units'][a]
        duty = summary['execution_duties']['actors'].get(a, {})
        old_cmd = orders[a]
        approach = ((row['skill'], row['phase']) in (('coordinated_engage', 'approaching'),
                    ('baseline', ''), ('cohesive_search', 'searching')))
        if (a in memory.front_engaged or old_cmd not in ctx.move_destinations(a) or not approach or
                duty.get('kind') != 'entry' or duty.get('joint_rescue') or duty.get('branch_reason') or
                a in summary['joint_rescue_substitutions']):
            continue

        old_end = ctx.predict_position(a, old_cmd).position
        # A potential opponent at the enemy-facing edge of the dog's next
        # legal contact band. Before vision, use the incumbent search heading,
        # never an imagined enemy roster/position. Count coverage, NOT shots.
        def measure(end):
            goal = min((v.position for v in snap.enemies.values()),
                       key=lambda p: distance(p, end), default=None)
            if goal is None:
                dx, dy = (old_end[i] - origin[i] for i in (0, 1))
                length = distance(origin, old_end)
                scale = ranges['zergling'] / max(length, 1.e-6)
            else:
                dx, dy = (goal[i] - end[i] for i in (0, 1))
                scale = min(1., ranges['zergling'] / max(distance(end, goal), 1.e-6))
            zone = end[0] + scale * dx, end[1] + scale * dy
            ally = sum(distance(p, zone) <= ranges['hydralisk'] for p in gun_ends.values())
            enemy = sum(distance(v.position, end) <= enemy_reach[e] + v.radius + u.radius
                        for e, v in foes.items())
            return dict(endpoint=list(end), contact_zone=list(zone), ally_cover=ally, enemy_cover=enemy,
                        local_advantage=ally-enemy, nearest_hydra_gap=min(distance(end, p) for p in gun_ends.values()))

        old = measure(old_end)
        if old['ally_cover'] and old['local_advantage'] >= -1:
            continue  # one extra opposing Hydra is not a clear fire advantage
        origin_gap = min(distance(origin, units[h].position) for h in guns)
        options = []
        for cmd, end in _safe_moves(ctx, memory, a):
            # The action contract has cardinal N/S and E/W. Do not reverse
            # the incumbent route or turn a moving team into a STOP barrier.
            if ((cmd in (2, 3)) == (old_cmd in (2, 3)) or
                    (duty.get('required') and cmd not in duty['actions']) or
                    not support_path_clear(ctx, a, end, orders)):
                continue
            new = measure(end)
            adv = new['local_advantage']
            if new['ally_cover'] and (adv >= 0 or adv > old['local_advantage']):
                rank = (0, -adv, -new['ally_cover'], new['nearest_hydra_gap'], cmd)
                options.append((rank, 'covered_flank', cmd, new))
            elif (not old['ally_cover'] and not new['ally_cover'] and
                  old['nearest_hydra_gap'] > origin_gap + .05 and
                  new['nearest_hydra_gap'] < old['nearest_hydra_gap'] - .05 and
                  new['enemy_cover'] <= old['enemy_cover']):
                rank = (1, new['nearest_hydra_gap'], new['enemy_cover'], cmd)
                options.append((rank, 'limit_separation', cmd, new))
        if options:
            _, reason, cmd, new = min(options)
            orders[a] = actions[u.index] = summary['actions'][a] = row['action'] = cmd
            changes[a] = dict(reason=reason, old_action_id=old_cmd, new_action_id=cmd, old=old, new=new)
            duty['selected_action'] = cmd
            duty['satisfied'] = not duty['required'] or cmd in duty['actions']

'''


def apply_variant(skill_dir: Path) -> None:
    """Atomically replace only skill_manager.py in an unmodified independent copy."""
    root = Path(skill_dir).resolve(strict=True)
    target = root / 'skill_manager.py'
    if BATCH in root.parents or root == ROOT / 'skills/tactical_v1' or target.resolve(strict=True) != target:
        raise ValueError('expected an independent copy of the frozen package')
    if any(sha256((root / name).read_bytes()).hexdigest() != digest for name, digest in ROUTE.items()):
        raise ValueError('unexpected contact_response route')
    raw = target.read_bytes()
    if sha256(raw).hexdigest() != DIGEST:
        raise ValueError('unexpected contact_response manager')
    nl = '\r\n' if b'\r\n' in raw else '\n'
    text = raw.decode().replace('\r\n', '\n').replace('class SkillManager:', _HOOK + '\nclass SkillManager:')
    anchor = '        summary["battle_intent"] ='
    text = text.replace(anchor, '        _front_route(context, self.memory, actions, summary)\n' + anchor)
    compile(text, str(target), 'exec')
    with NamedTemporaryFile(dir=root, delete=False) as handle:
        tmp = Path(handle.name)
        try:
            handle.write(text.replace('\n', nl).encode())
            handle.flush()
            tmp.chmod(target.stat().st_mode)
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)
