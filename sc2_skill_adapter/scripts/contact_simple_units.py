"""Install a per-species final-action overlay on a private micro-enabled manager."""
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile


# Same package hashing convention as contact_response_micro_variant.apply_variant,
# but pinned AFTER its hooks have been installed, not to the bare frozen package.
_MICRO_PACKAGE = '7dbc65b5b74eaa9c76e105b220964f767bdcc39c81fc00788892f85537cc11ed'

# Inline in the copy: the runtime must not import scripts from this workspace.
_OVERLAY = '''
def _simple_tag(value):
    tag = str(value)
    return tag if (len(tag) <= 20 and tag.isascii() and tag.isdecimal()
                   and 0 < int(tag) < 2**64) else None


def _simple_units(ctx, actions, summary, plans, selected):
    from .actions import STOP
    from .context import distance

    snap = ctx.snapshot
    raw = snap.raw_state
    enemies = [e for e, enemy in snap.enemies.items() if enemy.health > 0
               and raw['visible_enemies'][e].get('alive') is not False]
    if not enemies:
        return plans  # keep the original search, including its memory contract
    owned = set()
    for actor, unit in snap.allies.items():
        if not unit.alive or unit.type_name not in selected:
            continue
        visible = sorted(enemies, key=lambda e: (distance(unit.position, snap.enemies[e].position),
                                                 snap.enemies[e].index))
        relations = raw['relations'].get(actor, {})
        legal = [e for e in visible if ctx.can_issue_attack(actor, e)
                 and relations[e].get('smac_center_distance_allowed') is True]
        if legal:
            # Only the first native order is effective. ability 3674 is the
            # general Attack ID, serialized with kind='other' by the bridge.
            orders = raw['allies'][actor].get('orders') or []
            first = orders[0] if orders else {}
            target = first.get('target_unit')
            tag = _simple_tag(first.get('target_tag'))
            if (first.get('ability_id') in (23, 3674) and target in legal and tag
                    and tag == str(raw['visible_enemies'][target].get('tag'))):
                phase = 'keep_attack'
            else:
                target, phase = legal[0], 'nearest_attack'
            action = ctx.attack_action(actor, target)
        else:
            target = visible[0]
            pos = snap.enemies[target].position
            best = min(((distance(ctx.predict_position(actor, cmd).position, pos), cmd)
                        for cmd in ctx.move_destinations(actor)), default=None)
            if best and best[0] < distance(unit.position, pos):
                action, phase = best[1], 'approaching'
            else:
                action = STOP if STOP in ctx.legal_actions(actor) else actions[unit.index]
                phase = 'blocked'
        owned.add(actor)
        actions[unit.index] = summary['actions'][actor] = action
        summary['units'][actor] = dict(skill='simple_units', phase=phase, goal=target,
                                       goal_kind='enemy', action=action, plan=None, provider=None)
    # Do not commit a planner task for an actor whose command now belongs to
    # the simple policy, even if its action ID happens to match the plan.
    return [p for p in plans if owned.isdisjoint(p.controlled_actors)]


def _simple_intents(ctx, summary, plans, actions):
    from .context import distance

    snap = ctx.snapshot
    intents = []
    if snap.step_mul != 8:
        return intents
    for actor, row in summary['units'].items():
        unit = snap.allies[actor]
        if (not unit.alive or unit.type_name != 'hydralisk' or row['skill'] != 'simple_units'
                or row['phase'] != 'approaching' or row['action'] != actions[unit.index]
                or row['action'] not in ctx.move_destinations(actor)):
            continue
        target = row['goal']
        enemy = snap.enemies[target]
        raw_enemy = snap.raw_state['visible_enemies'][target]
        rel = snap.raw_state['relations'].get(actor, {}).get(target, {})
        actor_tag = _simple_tag(snap.raw_state['allies'][actor].get('tag'))
        target_tag = _simple_tag(raw_enemy.get('tag'))
        end = ctx.predict_position(actor, row['action']).position
        if (rel.get('smac_center_distance_allowed') is not False
                or not actor_tag or not target_tag or enemy.health <= 0 or raw_enemy.get('alive') is False
                or (ctx.effective_attack_reach(actor, target) or 0) <= 0
                or distance(end, enemy.position) + .05 >= distance(unit.position, enemy.position)):
            continue
        intents.append(dict(actor=actor, actor_tag=actor_tag, target=target, target_tag=target_tag,
                            macro_action_id=row['action'], issued_loop=snap.game_loop,
                            kind='approach_then_attack', skill=row['skill'], phase=row['phase']))
    return intents

'''


def apply_simple(skill_dir: Path, simple_lings: bool, simple_hydras: bool) -> None:
    """Validate the micro-enabled package; atomically replace only its manager."""
    root = Path(skill_dir).resolve(strict=True)
    workspace = Path(__file__).resolve().parents[2]
    protected = (workspace / 'skills', workspace / 'runs/contact_540_20261120_225957_302289',
                 workspace / 'runs/contact_response_micro_20261121_153552_909880')
    if any(p == root or p in root.parents for p in protected):
        raise ValueError('expected an independent copy, not a reference policy')
    digest = sha256()
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if path.is_symlink():
            raise ValueError('expected an independent copy without links')
        if path.suffix in ('.py', '.json') and 'tests' not in rel.parts:
            if path.stat().st_nlink != 1:
                raise ValueError('expected an independent copy without hard links')
            digest.update(rel.as_posix().encode() + b'\0' + sha256(path.read_bytes()).digest())
    if digest.hexdigest() != _MICRO_PACKAGE:
        raise ValueError('expected frozen contact_response with exactly apply_variant installed')
    selected = tuple(t for t, enabled in [('zergling', simple_lings), ('hydralisk', simple_hydras)] if enabled)
    if not selected:
        return
    manager = root / 'skill_manager.py'
    raw = manager.read_bytes()
    text = raw.decode().replace('\r\n', '\n')
    anchor = '        summary["battle_intent"] ='
    text = text.replace(anchor, f'        plans = _simple_units(context, actions, summary, plans, {selected!r})\n' + anchor)
    if simple_hydras:
        # Keep the frozen manager's final-team comparison, clearing rules and
        # record_micro_action. Only the source of hydralisk intent changes.
        text = text.replace('intents = (_micro_intents(', 'intents = (_simple_intents(')
    text += '\n' + _OVERLAY
    compile(text, str(manager), 'exec')
    with NamedTemporaryFile(dir=root, delete=False) as handle:
        tmp = Path(handle.name)
        try:
            handle.write(text.replace('\n', '\r\n' if b'\r\n' in raw else '\n').encode())
            handle.close()
            tmp.chmod(manager.stat().st_mode)
            tmp.replace(manager)
        finally:
            tmp.unlink(missing_ok=True)
