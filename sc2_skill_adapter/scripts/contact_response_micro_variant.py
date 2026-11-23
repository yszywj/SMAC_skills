"""Install the micro-window interface in a private copy of contact_response."""
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / 'runs/contact_540_20261120_225957_302289/contact_response/skills/tactical_v1'
# Sorted relative paths and SHA-256 bytes of the frozen Python/JSON inputs.
PACKAGE_DIGEST = 'badafc3c436bffde150c9bbe204b32d6ba39ff3f8e4e0cbbc835a06b52f3aa7a'


# Installed inline so a copied runtime never imports the experiment harness.
_HOOK = '''
def _micro_intents(ctx, summary, plans, actions):
    from .context import distance

    snap = ctx.snapshot
    intents = []
    if snap.step_mul != 8:
        return intents
    duties = summary['execution_duties'].get('actors', {})
    for actor, row in summary['units'].items():
        unit = snap.allies[actor]
        cmd = actions[unit.index]
        target = row['goal']
        enemy = snap.enemies.get(target)
        duty = duties.get(actor, {})
        if (not unit.alive or unit.type_name != 'hydralisk' or row['action'] != cmd
                or cmd not in ctx.move_destinations(actor) or row['goal_kind'] != 'enemy'
                or not enemy or enemy.health <= 0 or duty.get('kind') != 'entry'
                or not duty.get('satisfied') or duty.get('selected_action') != cmd
                or duty.get('joint_rescue') or duty.get('branch_reason')
                or actor in summary['joint_rescue_substitutions'] or ctx.forced_hydra_targets(actor)):
            continue

        end = ctx.predict_position(actor, cmd).position
        role = row['skill'], row['phase']
        if role == ('ranged_fire_formation', 'exclusive_control'):
            # This phase also owns search, alignment, escapes and stationary
            # fallback. Exclude moves that leave predicted coverage, even if
            # the same step incidentally closes on the shared focus.
            model = getattr(ctx, '_ranged_formation_forecast', None)
            if (row['plan'] is not None or ctx.formation_orders.get(actor) != cmd
                    or target != ctx.hydra_formation_target or duty['reason'] != 'exclusive_hydra_formation'
                    or cmd == getattr(ctx, '_debug_stationary', {}).get(actor, cmd)
                    or model is None or model.threats_at(actor, unit.position) - model.threats_at(actor, end)):
                continue
        elif role == ('coordinated_engage', 'approaching'):
            # Do not treat synchronizing/supporting/guarding (even with an
            # enemy goal), or a deployment/follow duty as an attack approach.
            plan = next((p for p in plans if p.proposal_id == row['plan']
                         and p.current_actions.get(actor) == cmd), None)
            if (plan is None or plan.goal != target or plan.goal_data.get('target_id') != target
                    or plan.goal_data.get('objective', {}).get('kind') != 'range_band'
                    or plan.goal_data['objective'].get('target') != target
                    or duty.get('target') != target
                    or duty.get('reason') not in ('short_route_commit', 'join_active_front_now',
                                                  'uncommitted_approach', 'replace_failed_or_overdue_route')):
                continue
        else:
            continue

        # An unmasked public menu can include out-of-range enemies. Use the
        # geometric mask fact at L, not a prospective shot or nearest enemy.
        rel = snap.raw_state['relations'].get(actor, {}).get(target, {})
        if rel.get('smac_center_distance_allowed') is not False:
            continue
        reach = ctx.effective_attack_reach(actor, target)
        before = set(ctx.risk_at(actor, unit.position).threats)
        if (not reach or reach <= 0 or before - set(ctx.risk_at(actor, end).threats)
                or distance(end, enemy.position) + .05 >= distance(unit.position, enemy.position)):
            continue
        raw_enemy = snap.raw_state['visible_enemies'][target]
        tags = [str(snap.raw_state['allies'][actor].get('tag')), str(raw_enemy.get('tag'))]
        if (raw_enemy.get('alive') is False or not all(t.isascii() and t.isdecimal()
                and 0 < int(t) < 2**64 for t in tags)):
            continue
        intents.append(dict(actor=actor, actor_tag=tags[0], target=target, target_tag=tags[1],
                            issued_loop=snap.game_loop, macro_action_id=cmd,
                            kind='approach_then_attack', skill=row['skill'], phase=row['phase']))
    return intents

'''

_RECORD = '''
    def record_micro_action(self, actor: str, action: int) -> None:
        """Runner acknowledgement; do not re-commit tasks, plans or timers."""
        from .actions import attack_action_id
        if any(r['actor'] == actor and attack_action_id(int(r['target'][1:])) == action
               for r in self.last_micro_intents):
            self.memory.last_committed_actions[actor] = action
'''


def apply_variant(skill_dir: Path) -> None:
    """Verify the frozen inputs, then atomically replace just the copy's manager."""
    root = Path(skill_dir).resolve(strict=True)
    if FROZEN.parents[2] in (root, *root.parents) or (ROOT / 'skills') in (root, *root.parents):
        raise ValueError('expected an independent copy outside the frozen batch and workspace skills')
    digest = sha256()
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if path.is_symlink():
            raise ValueError('expected an independent copy without links')
        if path.suffix in ('.py', '.json') and 'tests' not in rel.parts:
            digest.update(rel.as_posix().encode() + b'\0' + sha256(path.read_bytes()).digest())
    if digest.hexdigest() != PACKAGE_DIGEST:
        raise ValueError('expected the unmodified frozen contact_response package')
    manager = root / 'skill_manager.py'
    raw = manager.read_bytes()
    nl = '\r\n' if b'\r\n' in raw else '\n'
    text = raw.decode().replace('\r\n', '\n')
    for old, new in (
        ('class SkillManager:', _HOOK + '\nclass SkillManager:'),
        ('    def reset(self) -> None:\n', '    def reset(self) -> None:\n        self.last_micro_intents = []\n'),
        ('        digest = _digest(', '        self.last_micro_intents = []\n        digest = _digest('),
        ('        if pending is None:', '        self.last_micro_intents = []\n        if pending is None:'),
        ('        self.memory.commit(', '        intents = (_micro_intents(context, self._cache[2], plans, actions)\n'
         '                   if final_actions == actions else [])\n        self.memory.commit('),
    ):
        if text.count(old) != 1:
            raise ValueError('unexpected manager hook location')
        text = text.replace(old, new)
    text += '\n        self.last_micro_intents = intents\n' + _RECORD
    compile(text, str(manager), 'exec')
    with NamedTemporaryFile(dir=root, delete=False) as handle:
        tmp = Path(handle.name)
        try:
            handle.write(text.replace('\n', nl).encode())
            handle.close()
            tmp.chmod(manager.stat().st_mode)
            tmp.replace(manager)
        finally:
            tmp.unlink(missing_ok=True)
