"""Experiment B: extend only Hydra positioning in a copy of contact_response."""
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile


FROZEN = (Path(__file__).resolve().parents[2] / 'runs' /
          'contact_540_20261120_225957_302289/contact_response/skills/tactical_v1')
MODULE = Path('skills/ranged_fire_formation.py')
DIGEST = 'df049f121fdbaeec9459c7c264bb72b7f83f4720120e55f46da3bd898dfb3e14'

_WRAPPER = '''def formation_orders(context, memory, actions):
    units = context.snapshot.allies
    roster = {u.type_name for u in units.values()}
    if 'baneling' not in roster:
        return _line_orders(context, memory, actions)
    if roster != {'zergling', 'hydralisk', 'baneling'}:
        return {}
    # The line is a positioning policy in this cohort, not a target allocator.
    # Decide ownership BEFORE setting formation_orders, which masks actor_duty.
    from ..action_duties import actor_duty
    proposed = _line_orders(context, memory, actions, mixed=True)
    kept = {}
    for actor, cmd in proposed.items():
        if (context.attackable_enemies(actor) or context.forced_hydra_targets(actor)
                or cmd not in context.legal_actions(actor)
                or context.decode(actor, cmd).kind not in ('move', 'stop')):
            continue
        duty = actor_duty(context, actor, memory)
        if not duty['required'] or cmd in duty['actions']:
            kept[actor] = cmd
    # Releasing a gun restores its incumbent trajectory. Do not reserve a
    # partial line if this introduces a collision with any final P0 path.
    combined = {**actions, **kept}
    if any(not support_path_clear(context, a, action_position(context, a, cmd)[0], combined)
           for a, cmd in kept.items()):
        return {}
    return kept


def _line_orders(context, memory, actions, mixed=False):'''


def apply_variant(skill_dir: Path) -> None:
    """Patch one verified copy; replace the file (never write through links)."""
    root = Path(skill_dir).resolve(strict=True)
    target = root / MODULE
    if (root == FROZEN.resolve() or target.resolve() != target
            or FROZEN.resolve() in target.resolve().parents):
        raise ValueError('expected an independent copy of tactical_v1')
    raw = target.read_bytes()
    if sha256(raw).hexdigest() != DIGEST:
        raise ValueError('expected the unmodified frozen formation module')
    newline = '\r\n' if b'\r\n' in raw else '\n'
    text = raw.decode().replace('\r\n', '\n')
    for old, new in (
        ('def formation_orders(context, memory, actions):', _WRAPPER),
        ("or any(u.type_name not in ('hydralisk', 'zergling') for u in units.values()):",
         "or (not mixed and any(u.type_name not in ('hydralisk', 'zergling') for u in units.values())):"),
        ('_screened(end, enemy, outside)',
         "_screened(end, enemy, [(u, p) for u, p in outside if u.type_name == 'zergling'])"),
        ('if memory is not None:\n        memory.last_hydra_target',
         'if memory is not None and not mixed:\n        memory.last_hydra_target'),
    ):
        if text.count(old) != 1:
            raise ValueError(f'unexpected formation patch anchor: {old}')
        text = text.replace(old, new)
    compile(text, str(target), 'exec')
    with NamedTemporaryFile(dir=target.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            f.write(text.replace('\n', newline).encode())
            f.flush()
            tmp.chmod(target.stat().st_mode)
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)
