"""Experiment A: patch a copy of the frozen contact_response policy."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / 'runs/contact_540_20261120_225957_302289/contact_response'
          / 'skills/tactical_v1')

# Run after the original formation has computed its orders/reservations. Never
# propose a move, change the shared focus used for positioning, or release a
# formation command that was not a current-window shot on a hydralisk.
_HELPER = '''
def contact_target_options(context, memory):
    from .context import distance
    from .intercept import intercept_plan

    choices = {}
    plans = None
    for a, unit in context.snapshot.allies.items():
        if not unit.alive or unit.type_name != 'hydralisk':
            continue
        ready = [e for e in context.attackable_enemies(a)
                 if context.contact_status(a, e) == 'in_range'
                 and context.fire_estimate(a, e).window_shots > 0]
        enemies = context.snapshot.enemies
        if not any(enemies[e].type_name == 'hydralisk' for e in ready):
            continue
        owned = context.formation_orders.get(a)
        if owned is not None and context.decode(a, owned).target not in [
                e for e in ready if enemies[e].type_name == 'hydralisk']:
            continue
        extra = []
        for e in ready:
            enemy = enemies[e]
            if enemy.type_name == 'zergling' and (
                    context.fire_estimate(a, e).damage_lo or 0) >= enemy.effective_health:
                if any(mate.alive and not mate.is_flying
                       and distance(mate.position, enemy.position) <= 2 + mate.radius + enemy.radius
                       and e in context.forecast_risk(peer, mate.position).current_threats
                       for peer, mate in context.snapshot.allies.items()):
                    extra.append(e)
            elif enemy.type_name == 'baneling':
                if plans is None:
                    plans = intercept_plan(context, memory)
                if any(p['enemy'] == e and p['type'] in ('safe_intercept', 'loss_control_intercept')
                       and any(peer == a for peer, _, _ in p['shooters']) for p in plans):
                    extra.append(e)
        if extra:
            choices[a] = tuple(context.attack_action(a, e) for e in extra)
    return choices
'''


def apply_variant(skill_dir: Path) -> None:
    """Apply to a private copy only; validate all source anchors before writing."""
    root = Path(skill_dir).resolve(strict=True)
    if root == SOURCE.resolve() or SOURCE.resolve() in root.parents:
        raise ValueError('refusing to modify frozen skill')
    changes = {}

    def edit(name, *replacements, append=''):
        path = root / name
        if (path.resolve() != path or path.stat().st_nlink != 1):
            raise ValueError(f'expected private regular copy: {path}')
        raw = path.read_bytes()
        text = raw.decode().replace('\r\n', '\n')
        for old, new in replacements:
            if text.count(old) != 1:
                raise ValueError(f'unexpected {name} source at {old[:70]!r}')
            text = text.replace(old, new)
        text += append
        compile(text, str(path), 'exec')
        changes[path] = text.replace('\n', '\r\n' if b'\r\n' in raw else '\n').encode()

    edit('coordinator.py',
         ('        p0 = self.evaluate_once',
          '        context.contact_targets = contact_target_options(context, memory)\n'
          '        p0 = self.evaluate_once'),
         ('actions.get(a) != cmd for a, cmd in formation.items()',
          'actions.get(a) != cmd and actions.get(a) not in context.contact_targets.get(a, ())\n'
          '               for a, cmd in formation.items()'),
         ('if targets and context.decode(actor, action).target not in targets:',
          'target = context.decode(actor, action).target\n'
          '            exception = (action in context.contact_targets.get(actor, ()) and\n'
          '                         (actor in formation and action != formation[actor] or\n'
          '                          targets and target not in targets))\n'
          "            if exception and context.snapshot.enemies[target].type_name == 'baneling':\n"
          '                if not any((p.goal_data or {}).get("validation_kind") == "bomber_intercept"\n'
          '                           and p.goal_data.get("intercept_type") in ("loss_control_intercept", "safe_intercept")\n'
          '                           and p.goal_data.get("shooter_actions", {}).get(actor) == action for p in plans):\n'
          '                    return None\n'
          '            if (targets and target not in targets\n'
          '                    and action not in context.contact_targets.get(actor, ())):'),
         ('context.formation_orders.keys().isdisjoint(p.controlled_actors)',
          'all(a not in context.formation_orders or p.current_actions.get(a) in\n'
          '                                  context.contact_targets.get(a, ()) for a in p.controlled_actors)'),
         append=_HELPER)
    edit('baseline.py',
         ('if actor in locked_actions:',
          "if actor in locked_actions and not (actor in getattr(context, 'formation_orders', {})\n"
          '                    and locked_actions[actor] == context.formation_orders[actor]\n'
          "                    and action in getattr(context, 'contact_targets', {}).get(actor, ())):"))
    edit('action_duties.py',
         ("'actions': (cmd,),", "'actions': (cmd,) + getattr(context, 'contact_targets', {}).get(actor, ()),"),
         ('tuple(context.attack_action(actor, e) for e in targets),',
          "tuple(context.attack_action(actor, e) for e in targets)\n"
          "                      + getattr(context, 'contact_targets', {}).get(actor, ()),"))
    # Offer finishing lings before the per-unit proposal cap; bombers enter via
    # the existing complete, validated joint intercept proposal, not a solo shot.
    edit('skills/focus_fire.py',
         ('classes = [killable,',
          "finish = [e for e in killable if context.snapshot.enemies[e].type_name == 'zergling'\n"
          "                  and context.attack_action(actor, e) in getattr(context, 'contact_targets', {}).get(actor, ())]\n"
          '        classes = [finish, killable,'))
    for path, content in changes.items():
        path.write_bytes(content)
