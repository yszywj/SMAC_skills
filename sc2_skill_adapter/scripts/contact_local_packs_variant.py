"""Install the local engagement experiment in a private frozen micro8 policy."""
from hashlib import sha256
from inspect import getsource
from pathlib import Path
from tempfile import NamedTemporaryFile

if __package__:
    from .contact_muster_variant import FROZEN, PACKAGE, ROOT, RUN
else:
    from contact_muster_variant import FROZEN, PACKAGE, ROOT, RUN


# These functions are installed in coordinator.py. The runtime must have no
# dependency on the launcher, another run directory, or a working tree.
def _local_context(ctx, memory, reference, selected, actions):
    """Release only changed orders; never substitute a forced shot or a rescue."""
    import copy
    from .action_duties import actor_duty
    from .joint_combat import action_position
    from .context import distance

    changed = {a for a in actions if actions[a] != reference.actions[a]}
    branch = copy.copy(ctx)
    branch.formation_orders = {a: cmd for a, cmd in ctx.formation_orders.items() if a not in changed}
    branch.formation_reserved_paths = {a: end for a, end in getattr(ctx, 'formation_reserved_paths', {}).items()
                                       if a not in changed}
    branch._execution_actor_duties = dict(getattr(ctx, '_execution_actor_duties', {}))
    branch._execution_assessments = {}
    for plan in selected:
        data = plan.goal_data
        for a in plan.controlled_actors:
            cmd = actions[a]
            target = data['actor_target'].get(a)
            enemy = ctx.snapshot.enemies.get(target)
            tag = ctx.snapshot.raw_state['visible_enemies'].get(target, {}).get('tag')
            if (not enemy or enemy.effective_health <= 0 or tag is None
                    or str(tag) != data['target_tags'].get(target) or cmd not in ctx.legal_actions(a)):
                return None
            sem = ctx.decode(a, cmd)
            if sem.kind == 'attack' and sem.target != target:
                return None  # an unchanged atomic member cannot claim another target
            if a not in changed:
                continue
            forced = ctx.forced_hydra_targets(a)
            if forced and sem.target not in forced:
                return None
            duty = actor_duty(branch, a, memory)
            if duty['required'] and duty['kind'] != 'entry' and cmd not in duty['actions']:
                return None
            pos = ctx.snapshot.allies[a].position
            if sem.kind == 'attack':
                if sem.target != target or not ctx.can_issue_attack(a, target):
                    return None
            elif sem.kind == 'move':
                end = action_position(ctx, a, cmd)[0]
                advancing = distance(end, enemy.position) + .05 < distance(pos, enemy.position)
                protected = [ctx.snapshot.allies[b] for b in data.get('protected', ())
                             if b in ctx.snapshot.allies and ctx.snapshot.allies[b].alive]
                protecting = (any(distance(end, u.position) + .05 < distance(pos, u.position) for u in protected)
                              or len(ctx.risk_at(a, end).threats) < len(ctx.risk_at(a, pos).threats))
                if not (advancing or protecting or duty['kind'] == 'rescue' and cmd in duty['actions']):
                    return None
            else:
                return None  # no blanket exception for STOP/search/unspecified commands
            if not duty['required'] or duty['kind'] == 'entry':
                # Record the actual selected purpose for commit and micro8.
                branch._execution_actor_duties[a, id(memory)] = dict(required=True, kind='entry',
                    actions=(cmd,), reason='local_contact', target=target)
    return branch


def _local_opportunity(ctx, audit, actions):
    """Limited additional credit for reaching a *currently contested* named ray.

    Computed identically on the reference and every complete alternative, and
    only from physical actions. The value is not earned for owning a plan.
    """
    from .context import distance
    from .joint_combat import action_position

    available = {}
    for p in audit['packs']:
        for e in p['enemies']:
            # An actual enemy gun on a named ally, not merely another visible unit.
            if any((rel := ctx.snapshot.relations.get(b, {}).get(e)) and rel.in_enemy_weapon_range is True
                   for b in p['protected']):
                for a in p['members']:
                    available.setdefault(a, set()).add(e)
    for r in audit['requests']:
        if r['target'] in ctx.snapshot.enemies:
            for a in audit.get('idle', ()):
                available.setdefault(a, set()).add(r['target'])
    total = 0.
    for a, enemies in available.items():
        u = ctx.snapshot.allies[a]
        cmd = actions[a]
        sem = ctx.decode(a, cmd)
        end = action_position(ctx, a, cmd)[0]
        best = 0.
        for e in enemies:
            reach = ctx.effective_attack_reach(a, e)
            if reach is None or reach <= 0.:
                continue
            target = ctx.snapshot.enemies[e]
            if sem.kind == 'attack' and sem.target == e and ctx.contact_status(a, e) == 'in_range':
                best = max(best, 1. if ctx.fire_estimate(a, e).window_shots > 0 else .45)
            elif sem.kind == 'move':
                before = max(0., distance(u.position, target.position) - reach)
                after = max(0., distance(end, target.position) - reach)
                best = max(best, min(.75, max(0., before - after) / (1. + after)))
        total += best
    return total


def _survival_margin(ctx, original, trial):
    """Veto moving lethal exposure onto an ally; prefer credible serious relief."""
    relief = 0
    for a, row in trial.items():
        old, new = original[a]['after'], row['after']
        before, after = old['reserve_ratio'], new['reserve_ratio']
        if before < 1 <= after or (after >= .75 and after > before + .12):
            return None
        u = ctx.snapshot.allies[a]
        if (before >= .75 and after < .75 and before - after >= .20
                and not (old['unknown'] or new['unknown'])
                and (u.health_lost + u.shield_lost > 0 or ctx.risk_at(a, u.position).threats)):
            relief += 1
    return relief


def _arbitrate_local(coordinator, ctx, memory, reference, n_agents, dispatch):
    from itertools import combinations
    from .ledger import Ledger
    from .local_engagement import local_conflicts, propose_local
    from .team_exchange import assess_team_exchange
    from .validator import validate_actions, validate_plans

    proposed, audit = propose_local(ctx, memory, dict(reference.actions), dispatch)
    limit = min(64, coordinator.max_evaluations)
    remaining = [limit]  # additional bounded budget, never steals reference evaluations
    audit.update(dispatch=dispatch, selected=[], changes={}, reference_actions=dict(reference.actions),
                 reference_score=reference.score, valid_branches=0, max_joint_plans=0,
                 local_evaluations=0, local_budget=limit, rejection_counts={})

    def reject(reason):
        counts = audit['rejection_counts']
        counts[reason] = counts.get(reason, 0) + 1

    packs = {p['pack_id']: p for p in audit['packs']}
    requests = {r['request_id']: r for r in audit['requests']}
    proposals = []
    for p in proposed:
        data = p.goal_data
        live = memory.instances.get(p.instance_id)
        occupied = any(set(p.controlled_actors).intersection(q.controlled_actors) for q in reference.plans)
        continuation = bool(p.primary_skill == 'reserve_dispatch' and live and not live.termination_reason
                            and live.skill == p.primary_skill and set(live.actors) == set(p.controlled_actors))
        if not any(reference.actions.get(a) != cmd for a, cmd in p.current_actions.items()) and (
                not continuation or occupied):
            reject('reference_command')
            continue
        if (p.snapshot_id != ctx.snapshot.snapshot_id or not p.controlled_actors
                or set(p.current_actions) != set(p.controlled_actors)
                or any(a not in reference.actions or not ctx.snapshot.allies[a].alive
                       or data['reference_actions'].get(a) != reference.actions[a] for a in p.controlled_actors)
                or data['pack_id'] not in packs):
            reject('stale_proposal')
            continue
        request = requests.get(data.get('request_id'))
        pack = packs[data['pack_id']]
        if (p.primary_skill not in ('local_engagement', 'reserve_dispatch')
                or any(data['actor_target'].get(a) not in pack['enemies'] for a in p.controlled_actors)
                or p.primary_skill == 'local_engagement' and (
                    data.get('request_id') or not set(p.controlled_actors).issubset(pack['members']))):
            reject('outside_pack')
            continue
        if p.primary_skill == 'reserve_dispatch' and (not dispatch or not request
                or request['pack_id'] != data['pack_id'] or request['expires_loop'] <= ctx.snapshot.game_loop
                or any(data['actor_target'][a] != request['target'] for a in p.controlled_actors)):
            reject('stale_request')
            continue
        proposals.append(p)
    if not proposals:
        return reference, audit

    ref_opportunity = _local_opportunity(ctx, audit, reference.actions)
    ref_risk = assess_team_exchange(ctx, reference.actions)['actors']
    # Ranking compares the same complete-team duties/utility. Exact ties keep
    # the original, except for a real live reservation still executing its order.
    def rank(entry, local, relief=0):
        continues = sum(p.instance_id in memory.instances for p in local)
        changed = sum(cmd != reference.actions[a] for a, cmd in entry.actions.items())
        priorities = entry.execution['priority']
        return (*priorities[:3], -relief, *priorities[3:], -entry.score, -continues, changed,
                tuple(p.proposal_id for p in local))

    base = (rank(reference, ()), reference, (), ctx)
    best = base
    seen = set()

    def compare(local):
        nonlocal best
        ids = tuple(p.proposal_id for p in local)
        if ids in seen or not remaining[0]:
            return None
        seen.add(ids)
        actors = [a for p in local for a in p.controlled_actors]
        if len(actors) != len(set(actors)):
            reject('double_control')
            return None
        retained = [q for q in reference.plans if set(actors).isdisjoint(q.controlled_actors)]
        plans = [*retained, *local]
        ledger = Ledger()
        for p in plans:
            if ledger.conflicts(p):
                reject('resource_conflict')
                return None
            ledger.reserve(p)
        actions = dict(reference.actions)
        for p in local:
            actions.update(p.current_actions)
        branch = _local_context(ctx, memory, reference, local, actions)
        if branch is None:
            reject('no_local_permission')
            return None
        conflicts = local_conflicts(branch, plans, actions)
        if conflicts:
            reject('local_conflict:' + str(conflicts[0]).split(':')[0])
            return None
        # Reject even profitable diversions that turn an actor's route lethal
        # or materially worsen an already severe exposure. No assumed kill can
        # erase enemy fire during approach. This checks outsiders as well.
        risk = assess_team_exchange(branch, actions)['actors']
        relief = _survival_margin(branch, ref_risk, risk)
        if relief is None:
            reject('survival_regression')
            return None
        # evaluate_once's frozen cache key does not contain the (branch-specific)
        # formation permissions. Never return a cached assessment for another context.
        original_cache, coordinator._cache = coordinator._cache, {}
        try:
            entry = coordinator.evaluate_once(branch, plans, actions, memory, remaining, n_agents)
        finally:
            coordinator._cache = original_cache
        if entry is None:
            reject('joint_validation')
            return None
        # score_joint is the unmodified physical evaluator. Keep its cache
        # immutable: arbitration-only relative opportunity is computed once.
        delta = _local_opportunity(branch, audit, actions) - ref_opportunity
        score = entry.score + delta
        entry = type(entry)(entry.plans, entry.actions, score,
                     {**entry.breakdown, 'local_response_delta': delta, 'total': score}, entry.execution)
        original_cache['local', ids] = entry  # actual alternatives for the final summary
        result = (rank(entry, local, relief), entry, local, branch)
        audit['valid_branches'] += 1
        audit['max_joint_plans'] = max(audit['max_joint_plans'], len(local))
        if result[0] < best[0]:
            best = result
        return result

    # Screen each alternative against the untouched original before extending
    # the beam. Lower-ranked singletons may still participate in a winning set.
    singles = [row for p in proposals if (row := compare((p,))) is not None]
    # A pair can clear a path/HP reservation that neither replacement clears
    # alone. Include pairs even if one or both singletons were invalid.
    order = {row[2][0].proposal_id: row[0] for row in singles}
    pair_order = sorted(proposals, key=lambda p: (p.proposal_id not in order, order.get(p.proposal_id, ()), p.proposal_id))
    pairs = []
    for pair in combinations(pair_order, 2):
        if not remaining[0]:
            break
        if (row := compare(tuple(sorted(pair, key=lambda p: p.proposal_id)))) is not None:
            pairs.append(row)
    beam = sorted(pairs, key=lambda row: row[0])[:8]
    for _ in range(2, min(10, len(proposals))):
        if not remaining[0] or not beam:
            break
        children = []
        for _, _, parent, _ctx in beam:
            for p in pair_order:
                if p not in parent:
                    group = tuple(sorted((*parent, p), key=lambda q: q.proposal_id))
                    if (row := compare(group)) is not None:
                        children.append(row)
        beam = sorted(children, key=lambda row: row[0])[:8]

    _, final, selected, winning_context = best
    audit['local_evaluations'] = limit - remaining[0]
    # Final, explicit noncached validation before handing precisely this same
    # object to summary, the manager's pending commit and the micro8 interface.
    ordered = [final.actions[f'u{i}'] for i in range(n_agents)]
    if (validate_actions(ordered, winning_context, n_agents) or
            validate_plans(final.plans, winning_context, final.actions,
                           coordinator.evaluator.effects(winning_context, final.actions), memory)):
        raise RuntimeError('local arbitration final validation failed')
    if selected:
        for name in ('formation_orders', 'formation_reserved_paths', '_execution_actor_duties', '_execution_assessments'):
            setattr(ctx, name, getattr(winning_context, name))
    for p in selected:
        d = p.goal_data
        for a in p.controlled_actors:
            target = d['actor_target'][a]
            row = dict(actor=a, before=reference.actions[a], after=final.actions[a],
                       changed=reference.actions[a] != final.actions[a], continued=p.instance_id in memory.instances,
                       pack_id=d['pack_id'], target=target, target_tag=d['target_tags'][target],
                       skill=p.primary_skill, plan_id=p.proposal_id, role=d['roles'][a], request_id=d.get('request_id'))
            audit['selected'].append(row)
            if row['changed']:
                audit['changes'][a] = {k: row[k] for k in ('before', 'after', 'pack_id', 'target_tag', 'skill')}
    return final, audit


def _local_micro_target(ctx, summary, plans, actor, row, cmd):
    plan = next((p for p in plans if p.proposal_id == row['plan'] and p.primary_skill == 'local_engagement'
                 and actor in p.controlled_actors and p.current_actions.get(actor) == cmd), None)
    if not plan:
        return None
    data = plan.goal_data
    change = summary['local_coord']['changes'].get(actor, {})
    target = data['actor_target'].get(actor)
    tag = ctx.snapshot.raw_state['visible_enemies'].get(target, {}).get('tag')
    if (data['roles'].get(actor) != 'approach_attack' or data.get('request_id') or tag is None
            or str(tag) != data['target_tags'].get(target) or change.get('target_tag') != str(tag)
            or change.get('after') != cmd or change.get('before') == cmd or change.get('pack_id') != data['pack_id']):
        return None
    return target


def _edit(text, name, edits):
    for old, new in edits:
        if text.count(old) != 1:
            raise ValueError(f'unexpected frozen hook in {name}: {old!r}')
        text = text.replace(old, new)
    return text


def _patched(root, name, edits):
    path = root / name
    raw = path.read_bytes()
    text = _edit(raw.decode().replace('\r\n', '\n'), name, edits)
    compile(text, str(path), 'exec')
    return text.replace('\n', '\r\n' if b'\r\n' in raw else '\n').encode()


def apply_variant(skill_dir: Path, dispatch: bool, *, marginal=False, reserve_horizon=4) -> None:
    """Install on a private frozen copy. Default output matches the released packs."""
    root = Path(skill_dir).resolve(strict=True)
    if type(dispatch) != bool:
        raise TypeError('dispatch must be bool')
    if type(marginal) != bool:
        raise TypeError('marginal must be bool')
    if type(reserve_horizon) != int or reserve_horizon not in (2, 4):
        raise ValueError('reserve_horizon must be 2 or 4 macro steps')
    if RUN in (root, *root.parents) or ROOT / 'skills' in (root, *root.parents):
        raise ValueError('expected an independent copy outside frozen and live skills')
    digest = sha256()
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if path.is_symlink():
            raise ValueError('expected a private package without symlinks')
        if path.suffix in ('.py', '.json') and 'tests' not in rel.parts:
            digest.update(rel.as_posix().encode() + b'\0' + sha256(path.read_bytes()).digest())
    if digest.hexdigest() != PACKAGE:
        raise ValueError('expected unmodified frozen contact_response_micro package')
    source = Path(__file__).with_name('local_engagement_template.py').read_bytes()
    hooks = '\n\n'.join(getsource(fn) for fn in (_local_context, _local_opportunity, _survival_margin, _arbitrate_local))
    short = dispatch and reserve_horizon == 2
    edits = []
    if marginal:
        edits.extend((
            ('    local = [_local_offers', '    from .local_pack_options import PackValue\n'
             '    w.value = PackValue(context, packs, w.ref)\n    local = [_local_offers'),
            ('        return concentrate or surplus', '        return self.value.redirect(self, a, e) or concentrate or surplus'),
            ('    choices.sort()', '    choices.sort(key=lambda row: (-w.value.delta({**w.ref, row[1]: row[3]}), row))'),
        ))
        hooks = _edit(hooks, 'local arbitration', (
            ('    ref_opportunity =', '    from .local_pack_options import PackValue\n'
             "    pack_value = PackValue(ctx, audit['packs'], reference.actions)\n    ref_opportunity ="),
            ('        score = entry.score + delta', '        marginal_delta = pack_value.delta(actions)\n'
             '        score = entry.score + delta + marginal_delta'),
            ("'local_response_delta': delta,", "'local_response_delta': delta, 'pack_marginal_delta': marginal_delta,"),
        ))
    if short:
        edits.extend((
            ('REQUEST_H = 4', 'from .local_pack_options import home_line, near_bid\n\nREQUEST_H = 2'),
            ('    local = [_local_offers', '    w.packs = packs\n    local = [_local_offers'),
            ("        if live:\n            delta", "        extra.update(enemy_position=s.enemies[goal].position,\n"
             "                     home_line=live.extra['home_line'] if live else home_line(w, a, request, w.packs))\n"
             "        if live:\n            delta"),
            ('            if useful[a] - {e}', '            if not near_bid(w, a, p, live, same, nearby[a], previous):\n'
             '                continue\n            if useful[a] - {e}'),
        ))
    if edits:
        source = _edit(source.decode(), 'local_engagement', edits).encode()
    compile(source, 'local_engagement.py', 'exec')
    files = {'local_engagement.py': source, 'coordinator.py': _patched(root, 'coordinator.py', (
        ('class Coordinator:', hooks + '\n\nclass Coordinator:'),
        ('        summary = self._summary(final,',
         f'        final, audit = _arbitrate_local(self, context, memory, final, n_agents, {dispatch})\n'
         '        summary = self._summary(final,'),
        ('        return ([final.actions', '        summary["local_coord"] = audit\n'
         '        summary["plan_evaluations_used"] += audit["local_evaluations"]\n'
         '        return ([final.actions'),
    )), 'skill_manager.py': _patched(root, 'skill_manager.py', (
        ('def _micro_intents(', getsource(_local_micro_target) + '\n\ndef _micro_intents('),
        ("        target = row['goal']", "        local = row['skill'] == 'local_engagement'\n"
         "        target = _local_micro_target(ctx, summary, plans, actor, row, cmd) if local else row['goal']"),
        ("or row['goal_kind'] != 'enemy'", "or (not local and row['goal_kind'] != 'enemy')"),
        ("        if role == ('ranged_fire_formation',", "        if local:\n"
         "            if duty.get('reason') != 'local_contact' or duty.get('target') != target:\n"
         "                continue\n        elif role == ('ranged_fire_formation',"),
        ('        self.last_micro_intents = intents',
         '        from .local_engagement import commit_local\n'
         '        commit_local(context, self.memory, self._cache[2]["local_coord"])\n'
         '        self.last_micro_intents = intents'),
    ))}
    if edits:
        extra = Path(__file__).with_name('local_pack_options_template.py').read_bytes()
        compile(extra, 'local_pack_options.py', 'exec')
        files['local_pack_options.py'] = extra
    # Prepare and compile the whole change before replacing any destination;
    # rename (rather than in-place writes) also protects hard-linked originals.
    temps = {}
    try:
        for name, content in files.items():
            target = root / name
            with NamedTemporaryFile(dir=root, delete=False) as f:
                temps[target] = Path(f.name)
                f.write(content)
            temps[target].chmod(target.stat().st_mode if target.exists() else 0o644)
        for target, tmp in temps.items():
            tmp.replace(target)
    finally:
        for tmp in temps.values():
            tmp.unlink(missing_ok=True)
