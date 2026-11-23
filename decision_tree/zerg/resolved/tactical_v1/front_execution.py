"""Complete local-front commands: keep fire, fill contact, relieve danger.

These are one-observation combinations, not new persistent rally tasks.
Cross-skill contact/output memories retain all deadlines. Geometry prunes
new body overlaps; no position is advertised as certain native pathfinding.
"""
from .action_duties import actor_duty
from .battle_line import battle_line, line_action_rank
from .context import STOP, distance
from .joint_combat import action_position


def _contact_targets(context, actor):
    return tuple(e for e in context.attackable_enemies(actor)
                 if context.contact_status(actor, e) == 'in_range')


def _options(context, actor, lane, memory):
    duty = actor_duty(context, actor, memory)
    contact = _contact_targets(context, actor)
    if duty['required']:
        options = list(duty['actions'])
    elif contact:
        # Cooling weapons retain their real contact; another member may
        # move, but this source need not walk merely to produce a G6 plan.
        options = [context.attack_action(actor, e) for e in contact]
    else:
        ranked = []
        for action in context.legal_actions(actor):
            rank = line_action_rank(context, actor, action, lane['front'], memory)
            if rank is not None:
                ranked.append((rank, action))
        options = [action for _, action in sorted(ranked)]
    options = tuple(dict.fromkeys(action for action in options
                                  if action in context.legal_actions(actor)))
    if not options and STOP in context.legal_actions(actor):
        options = (STOP,)  # No fabricated route: other members can still act.
    return options, duty


def _new_overlap(context, actor, point, peer, other):
    units = context.snapshot.allies
    first, second = units[actor], units[peer]
    if first.is_flying != second.is_flying:
        return False
    origin = distance(first.position, second.position)
    threshold = first.radius + second.radius
    # Existing packed bodies may maintain or increase separation. Never
    # reject stationary shooters just because native observations touch.
    return distance(point, other) < threshold - .01 and distance(point, other) < origin - .01


def _fits(context, actor, action, chosen, members):
    units = context.snapshot.allies
    point, _ = action_position(context, actor, action)
    middle = tuple((x + y) / 2. for x, y in zip(units[actor].position, point))
    for peer, unit in units.items():
        if peer == actor or not unit.alive or (peer in members and peer not in chosen):
            continue
        other = action_position(context, peer, chosen[peer])[0] if peer in chosen else unit.position
        midpoint = tuple((x + y) / 2. for x, y in zip(unit.position, other))
        if _new_overlap(context, actor, point, peer, other) or _new_overlap(
                context, actor, middle, peer, midpoint):
            return False
    return True


def front_commands(context, memory):
    """Bounded joint alternatives for all ordinary members of each pocket.

    A bounded enemy front owns its whole assigned friendly group, including
    a slow rear that is not yet within ten units of its first melee body.
    The shared arrival duties slow unsupported contact and advance the rear
    together; existing shooters keep firing rather than waiting globally.
    """
    cache = getattr(context, '_front_execution_commands', None)
    if cache is not None:
        return cache
    facts, units = battle_line(context), context.snapshot.allies
    proposals = []
    for front in facts['fronts']:
        pending = {a for a in front['members'] if facts['actors'].get(a) is not None
                   and not context.capabilities[a].specialization}
        while pending:
            members = tuple(sorted(pending))
            pending.difference_update(members)
            if len(members) < 2:
                continue
            options, duties = {}, {}
            for actor in members:
                options[actor], duties[actor] = _options(context, actor, facts['actors'][actor], memory)
            if any(not options[a] for a in members):
                continue
            ordered = sorted(members, key=lambda a: (
                0 if duties[a].get('kind') == 'rescue' and duties[a]['required'] else
                1 if _contact_targets(context, a) else 2,
                facts['actors'][a]['eta_h'], a))
            # A small construction beam permits alternate contact directions
            # instead of discarding the full frontage on its first conflict.
            beam = [({}, 0., {})]
            for actor in ordered:
                expanded = []
                for chosen, cost, reserved in beam:
                    for rank, action in enumerate(options[actor][:6]):
                        if not _fits(context, actor, action, chosen, members):
                            continue
                        trial = {**chosen, actor: action}
                        positions = {a: action_position(context, a, command)[0]
                                     for a, command in trial.items()}
                        if context.decode(actor, action).kind == 'move' and memory.movement_blockage(
                                context, actor, action, joint_actions=trial,
                                joint_positions=positions) >= .5:
                            continue
                        semantic = context.decode(actor, action)
                        spent, overflow = dict(reserved), 0.
                        if semantic.kind == 'attack':
                            estimate = context.fire_estimate(actor, semantic.target)
                            amount = max(0., estimate.damage_lo or 0.) if estimate.window_shots else 0.
                            remaining = max(0., context.snapshot.enemies[semantic.target].effective_health
                                            - reserved.get(semantic.target, 0.))
                            overflow = max(0., amount - remaining) / max(amount, 1.)
                            spent[semantic.target] = spent.get(semantic.target, 0.) + amount
                        expanded.append((trial, cost + rank + 3. * overflow, spent))
                beam = sorted(expanded, key=lambda row: (row[1], tuple(sorted(row[0].items()))))[:8]
                if not beam:
                    break
            for actions, _, _ in beam[:2]:
                if len(actions) != len(members):
                    continue
                # Pure all-fire plans are already well represented by G1.
                # Keep this family for an actual simultaneous role change.
                if not any(distance(units[a].position, action_position(context, a, command)[0]) > .01
                           for a, command in actions.items()):
                    continue
                proposals.append({'members': members, 'actions': actions,
                                  'front': front['targets'],
                                  'duties': {a: duties[a]['reason'] for a in members},
                                  'arrival_coordination': {a: memory.contact_progress.coordination(
                                      context, a, memory) for a in members
                                      if memory.contact_progress.coordination(context, a, memory)},
                                  'roles': {a: facts['actors'][a]['role'] for a in members}})
    context._front_execution_commands = proposals
    return proposals
