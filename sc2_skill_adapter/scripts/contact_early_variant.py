"""Install the opening Hydra approach experiment in a contact_response copy."""
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / 'runs/contact_540_20261120_225957_302289'
SOURCE = BATCH / 'contact_response/skills/tactical_v1'
DIGEST = '038350fe4c723d60b03034277aca4e369b920f381f8804a11e1a5495275206a5'
ROUTE = {'coordinator.py': '58927d61c3050600e42f5d1a994e1a70364a7661e744c934637fa5583b1380c2',
         'skills/ranged_fire_formation.py': 'df049f121fdbaeec9459c7c264bb72b7f83f4720120e55f46da3bd898dfb3e14'}

# Inline in the copied manager: no dependency on the experiment harness at play time.
_HOOK = '''
def _early_approach(ctx, memory, actions, summary):
    from .context import STOP, distance
    from .hydra_support import _safe_moves, support_path_clear

    snap = ctx.snapshot
    units = snap.allies
    if snap.snapshot_id == 1:
        types = [u.type_name for u in units.values() if u.alive]
        memory.early_cohort = (len(types) == len(units) == 10 and
                               (types.count('zergling'), types.count('hydralisk'))
                               in ((3, 7), (6, 4)))
        memory.early_seen = set()
        memory.early_start = None
    if not memory.early_cohort:
        return
    if memory.early_start is None and snap.enemies:
        memory.early_start = snap.game_loop
    if memory.early_start is not None and snap.game_loop - memory.early_start >= 8 * ctx.H:
        return

    orders = {f'u{i}': cmd for i, cmd in enumerate(actions)}
    focus = getattr(ctx, 'hydra_formation_target', None)
    for i, cmd in enumerate(actions):
        actor = f'u{i}'
        gun = units[actor]
        if not gun.alive or gun.type_name != 'hydralisk':
            continue
        # Conservative permanent handoff even at first legal opportunity,
        # regardless of whether a shot from that window actually landed.
        if (any(a >= 6 for a in ctx.legal_actions(actor)) or
                gun.weapon_cooldown_loops > 0 or memory.last_committed_actions.get(actor, 0) >= 6):
            memory.early_seen.add(actor)
        row = summary['units'][actor]
        if (actor in memory.early_seen or focus not in snap.enemies or cmd != STOP or
                row['skill'] != 'ranged_fire_formation' or row['plan'] is not None or
                ctx.formation_orders.get(actor) != STOP):
            continue
        goal = snap.enemies[focus].position  # incumbent focus, from shared team vision
        radius = snap.raw_state['action_contract']['candidate_center_distance_by_type']['hydralisk']
        def gap(point):
            return max(0., distance(point, goal) - radius)
        start = gap(gun.position)
        threats = set(ctx.risk_at(actor, gun.position).threats)
        moves = []
        for move, end in _safe_moves(ctx, memory, actor):
            remaining = gap(end)
            if remaining >= start - .05 or not support_path_clear(ctx, actor, end, orders):
                continue
            fresh = set(ctx.risk_at(actor, end).threats) - threats
            if len(fresh) <= 1:
                moves.append((remaining, len(fresh), move))
        if moves:
            move = min(moves)[-1]
            orders[actor] = actions[i] = summary['actions'][actor] = row['action'] = move
            row['phase'] = 'early_approach'
            summary.setdefault('early_approach', {})[actor] = {
                'scored_action': cmd, 'final_action': move, 'focus': focus}

'''


def apply_variant(skill_dir: Path) -> None:
    """Verify the frozen module and atomically replace only the copy's file."""
    root = Path(skill_dir).resolve(strict=True)
    target = root / 'skill_manager.py'
    if (BATCH in root.parents or root == ROOT / 'skills/tactical_v1' or
            target.resolve(strict=True) != target):
        raise ValueError('expected an independent copy, not a frozen directory or link')
    raw = target.read_bytes()
    if any(sha256((root / name).read_bytes()).hexdigest() != expected
           for name, expected in ROUTE.items()):
        raise ValueError('unexpected formation route; expected unmodified contact_response')
    if sha256(raw).hexdigest() != DIGEST:
        raise ValueError('unexpected skill_manager.py; expected unmodified contact_response')
    text = raw.decode().replace('\r\n', '\n').replace('class SkillManager:', _HOOK + '\nclass SkillManager:')
    anchor = '        summary["battle_intent"] ='
    text = text.replace(anchor, '        _early_approach(context, self.memory, actions, summary)\n' + anchor)
    compile(text, str(target), 'exec')
    with NamedTemporaryFile(dir=root, delete=False) as handle:
        tmp = Path(handle.name)
        try:
            handle.write((text.replace('\n', '\r\n') if b'\r\n' in raw else text).encode())
            handle.flush()
            tmp.chmod(target.stat().st_mode)
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)
