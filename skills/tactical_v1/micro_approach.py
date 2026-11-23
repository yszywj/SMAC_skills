"""One-window permissions from the winning attack route, never target selection."""
from math import dist, isfinite

from .actions import attack_action_id


def _native_tag(raw):
    text = str(raw.get('tag'))
    if (len(text) <= 20 and text.isascii() and text.isdecimal()
            and not text.startswith('0') and int(text) < 2**64):
        return text
    return None


def _named_approach(ctx, actor, cmd, plan, duty, local):
    """Read the provider's own target; a duty or a closing vector alone is not a plan."""
    data = plan.goal_data
    target = data.get('target_id')
    phase = plan.commitment.phase
    reason = duty.get('reason')
    unit = ctx.snapshot.allies[actor]
    semantic = ctx.decode(actor, cmd)
    if plan.primary_skill == 'local_engagement':
        # In a joint plan the actor's enemy need not be the group's first enemy.
        target = data.get('actor_target', {}).get(actor)
        tag = _native_tag(ctx.snapshot.raw_state['visible_enemies'].get(target, {}))
        change = local.get('changes', {}).get(actor, {})
        if (semantic.kind == 'move' and phase == 'support' and reason == 'local_contact'
                and duty.get('kind') == 'entry' and duty.get('target') == target
                and not data.get('request_id') and data.get('roles', {}).get(actor) == 'approach_attack'
                and data.get('actor_tags', {}).get(actor) == str(unit.tag)
                and tag and data.get('target_tags', {}).get(target) == change.get('target_tag') == tag
                and change.get('skill') == plan.primary_skill and data.get('pack_id')
                and change.get('pack_id') == data['pack_id'] and change.get('after') == cmd
                and isinstance(change.get('before'), int) and change['before'] != cmd
                and change['before'] == data.get('reference_actions', {}).get(actor)):
            return target
        return None

    if target != plan.goal or plan.controlled_actors != (actor,):
        return None
    if plan.primary_skill == 'burst_engage':
        # A fixed AoE region has an anchor enemy but is not a command to
        # attack that enemy. Only the direct-contact branch names a victim.
        if (unit.type_name == 'baneling' and phase == 'approaching' and semantic.kind == 'move'
                and duty.get('kind') == 'burst' and reason == 'cashout_direct_contact'
                and duty.get('target') == target and cmd in duty.get('actions', ())
                and (data.get('execution_duty') or {}).get('reason') == reason):
            return target
        return None

    if duty.get('kind') != 'entry':
        return None
    objective = data.get('objective', {})
    melee = unit.type_name in ('zergling', 'zealot')
    g2 = (plan.primary_skill, phase) == ('coordinated_engage', 'approaching')
    if semantic.kind == 'attack':
        # The public mask can open before a melee weapon makes contact.
        # G1 can also own the same attack while the FINAL duty is exactly
        # native melee approach. Ordinary fire, including on cooldown, cannot.
        rel = ctx.snapshot.relations.get(actor, {}).get(target)
        if (melee and semantic.target == target and duty.get('required') and cmd in duty.get('actions', ())
                and reason == 'native_contact_commit' and duty.get('target') == target
                and rel is not None and rel.in_weapon_range is False
                and ((plan.primary_skill, phase) == ('focus_fire', 'attacking') or (
                    g2 and data.get('contact_execution') == reason and objective.get('kind') == 'range_band'
                    and objective.get('target') == target))):
            return target
        return None
    if not g2 or semantic.kind != 'move':
        return None
    if not ((reason in ('uncommitted_approach', 'short_route_commit', 'join_active_front_now',
                        'replace_failed_or_overdue_route') and duty.get('target') == target)
            or (reason == 'no_visible_contact_duty' and not duty.get('required') and not duty.get('target'))):
        return None
    if melee:
        # G2 reserves a contact slot, not a ranged unit's range_band.
        if objective.get('kind') == 'contact_point' and any(
                c.actor == actor and (c.slot_id or '').startswith(f'slot:{target}:') for c in plan.claims.space):
            return target
    elif objective.get('kind') == 'range_band' and objective.get('target') == target:
        return target
    return None


def extra_approaches(ctx, summary, plans, actions):
    """Capture stable identities at decision time; publish only after a matching commit."""
    snap = ctx.snapshot
    if snap.step_mul != 8:
        return []
    raw = snap.raw_state
    duties = summary['execution_duties']['actors']
    by_id = {p.proposal_id: p for p in plans if p.snapshot_id == snap.snapshot_id}
    grants = []
    for actor, row in summary['units'].items():
        unit = snap.allies[actor]
        cmd = actions[unit.index]
        duty = duties.get(actor, {})
        # Legacy hydra authorizations pass through unchanged. No new permissions
        # for a healer, an unknown type, or a branch satisfying another duty.
        if (unit.type_name not in ('zergling', 'baneling', 'zealot', 'stalker', 'colossus', 'marine', 'marauder')
                or not unit.alive or unit.health <= 0 or row['action'] != cmd
                or cmd not in ctx.legal_actions(actor) or duty.get('selected_action') != cmd
                or not duty.get('satisfied') or duty.get('branch_reason') or duty.get('joint_rescue')
                or duty.get('isolated_shooter_support') or actor in summary['joint_rescue_substitutions']):
            continue
        plan = by_id.get(row['plan'])
        if plan is None:
            # The live all-Zerg profile completes Banelings without CandidatePlans.
            # Require BOTH the actual directed fallback and the direct-contact
            # duty; an unlabelled baseline or a region/wait duty is not enough.
            target = duty.get('target')
            roster = getattr(ctx, 'zerg_policy', None)
            if not (unit.type_name == 'baneling' and row['plan'] is None and row['skill'] == 'baseline'
                    and roster and roster.named_moves.get(actor) == (cmd, target)
                    and ctx.decode(actor, cmd).kind == 'move' and duty.get('kind') == 'burst'
                    and duty.get('reason') == 'cashout_direct_contact' and cmd in duty.get('actions', ())):
                continue
            skill, phase = 'baseline', 'cashout_direct_contact'
        elif (row['skill'] != plan.primary_skill or row['phase'] != plan.commitment.phase
                or row['goal'] != plan.goal or row['goal_kind'] != 'enemy' or plan.goal_kind != 'enemy'
                or actor not in plan.controlled_actors or plan.current_actions.get(actor) != cmd):
            continue
        else:
            skill, phase = row['skill'], row['phase']
            target = _named_approach(ctx, actor, cmd, plan, duty, summary.get('local_coord', {}))
        enemy = snap.enemies.get(target)
        if not enemy or enemy.health <= 0:
            continue
        actor_tag = _native_tag(raw['allies'][actor])
        raw_enemy = raw['visible_enemies'][target]
        target_tag = _native_tag(raw_enemy)
        if (not actor_tag or not target_tag or actor_tag == target_tag or actor_tag != str(unit.tag)
                or raw_enemy.get('alive') is False):
            continue
        if unit.type_name == 'baneling':
            # Effect-based ground contact, no native catalog weapon. Never use
            # splash radius or issue Detonate. Execution rechecks body contact.
            if enemy.is_flying or unit.unit_type_id != 9:
                continue
            reach = unit.radius + enemy.radius
        else:
            reach = ctx.effective_attack_reach(actor, target)
        before = dist(unit.position, enemy.position)
        if not reach or not isfinite(reach) or reach <= 0 or not isfinite(before) or before <= reach:
            continue
        if ctx.decode(actor, cmd).kind == 'move':
            if cmd not in ctx.move_destinations(actor):
                continue
            end = ctx.predict_position(actor, cmd).position
            if not dist(end, enemy.position) + .05 < before:
                continue
        grants.append(dict(actor=actor, actor_tag=actor_tag, target=target, target_tag=target_tag,
                           issued_loop=snap.game_loop, macro_action_id=cmd, kind='approach_then_attack',
                           skill=skill, phase=phase))
    return grants


def merge_approaches(original, extra):
    actors, tags, merged = set(), set(), []
    for r in (*original, *extra):
        if r['actor'] not in actors and r['actor_tag'] not in tags:
            merged.append(r)
            actors.add(r['actor'])
            tags.add(r['actor_tag'])
    return merged


def record_approach(memory, intents, actor, action):
    """Runner calls only for an accepted interior attack. No plan/timer commit."""
    if type(action) is int and any(
            r['actor'] == actor and memory.last_committed_actions.get(actor) == r['macro_action_id']
            and r['macro_action_id'] != action and attack_action_id(int(r['target'][1:])) == action
            for r in intents):
        memory.last_committed_actions[actor] = action
