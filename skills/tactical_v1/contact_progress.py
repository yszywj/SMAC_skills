"""Cross-skill participation debt discharged by an observed weapon cycle.

Empty recommendations, brief contact and tiny gap reductions do not complete
a duty. Debt ranks alternatives; it never authorizes ignoring an observed
lethal threat or entering a known explosion.
"""
import math

from .battle_intent import contacts, enemy_identity
from .combat_exchange import exposure_window, danger_evidence
from .context import STOP, distance, GAME_LOOPS_PER_SPEED_SECOND
from .joint_combat import action_position


class ContactProgress:
    def __init__(self, config=None):
        settings = (config or {}).get('contact_progress', {})
        self.stall_h = min(2., settings.get('stall_h', 2.))
        self.deadline_h = settings.get('deadline_h', 6.)
        self.records, self.pending, self.outcomes = {}, {}, []
        self.loop = None

    def __eq__(self, other):
        return isinstance(other, ContactProgress) and self.__dict__ == other.__dict__

    @staticmethod
    def targets(context, actor):
        return tuple(e for e in context.snapshot.enemies
                     if context.effective_attack_reach(actor, e) is not None
                     and context.effective_attack_reach(actor, e) >= 0.)

    @staticmethod
    def gap(context, actor, target, point):
        return max(0., distance(point, context.snapshot.enemies[target].position)
                   - context.effective_attack_reach(actor, target))

    def _finish(self, actor, loop, reason):
        record = self.records.pop(actor, None)
        if record is not None:
            self.outcomes.append({'actor': actor, 'start_loop': record['start_loop'],
                'end_loop': loop, 'reason': reason, 'first_contact_loop': record['first_contact_loop']})
            self.outcomes[:] = self.outcomes[-24:]

    def observe(self, context):
        loop, H = context.snapshot.game_loop, max(context.H, 1)
        if self.loop is not None and loop <= self.loop:
            return
        self.loop = loop
        for actor in set(self.records) | set(self.pending):
            unit = context.snapshot.allies.get(actor)
            record = self.records.get(actor) or self.pending.get(actor)
            if unit is None or not unit.alive or unit.tag != record['tag']:
                self._finish(actor, loop, 'identity_or_death')
                self.pending.pop(actor, None)
        for actor, unit in context.snapshot.allies.items():
            cap = context.capabilities[actor]
            if not unit.alive or cap.healer or cap.suicide_attack:
                continue
            pending = self.pending.pop(actor, None)
            elapsed = loop - pending['loop'] if pending else 0
            consecutive = bool(pending and pending['tag'] == unit.tag and 0 < elapsed <= 2 * H)
            active = contacts(context, actor)
            fired = bool(consecutive and (pending['contact'] or pending['attack_command'] or active)
                         and unit.weapon_cooldown_loops > max(0., pending['cooldown'] - elapsed) + .5)
            record = self.records.get(actor)
            if fired:
                self._finish(actor, loop, 'observed_weapon_cycle')
                continue
            local = tuple(e for e in self.targets(context, actor) if distance(
                unit.position, context.snapshot.enemies[e].position) <= 12.)
            if not local:
                if record and loop - record['last_local_loop'] >= 4 * H:
                    self._finish(actor, loop, 'encounter_left')
                continue
            if record is None:
                if active:
                    continue  # Established shooters have separate output duties.
                record = self.records[actor] = {
                    'tag': unit.tag, 'start_loop': loop, 'last_progress_loop': loop,
                    'last_local_loop': loop, 'first_contact_loop': None,
                    'target': None, 'target_tag': None, 'best_gap': float('inf'),
                    'failed_targets': {}, 'failed_routes': {}, 'last_replan_loop': loop,
                    'route_action': None, 'route_until_loop': loop,
                    'route_reference': None, 'history': [], 'last_failure': None,
                    'synchronization_wait_loops': 0}
            if consecutive and pending.get('coordination') and pending['action'] == STOP:
                record['synchronization_wait_loops'] += elapsed
            record['last_local_loop'] = loop
            if active and record['first_contact_loop'] is None:
                record['first_contact_loop'] = loop
            record['failed_targets'] = {key: until for key, until in record['failed_targets'].items()
                                        if until > loop}
            record['failed_routes'] = {key: value for key, value in record['failed_routes'].items()
                                       if value['until_loop'] > loop}
            target = record['target']
            if target not in local or enemy_identity(context, target) != record['target_tag']:
                record['target'] = target = None
                record['route_until_loop'] = loop
            history = record['history']
            history.append({'loop': loop, 'position': tuple(unit.position),
                            'contact': bool(active), 'action': pending['action'] if consecutive else None})
            history[:] = [row for row in history[-5:] if loop - row['loop'] <= 4 * H]
            repeated = None
            if not active and consecutive:
                for index, old in enumerate(history[:-1]):
                    segment = history[index:]
                    if loop - old['loop'] < 2 * H or any(row['contact'] for row in segment):
                        continue
                    walked = sum(distance(a['position'], b['position'])
                                 for a, b in zip(segment, segment[1:]))
                    if walked >= 1. and distance(old['position'], unit.position) <= .5:
                        repeated = segment
                        break
            if consecutive and target is not None and pending['target'] == target \
                    and pending['target_tag'] == record['target_tag']:
                gap = self.gap(context, actor, target, unit.position)
                # Measure own motion against the previous enemy position;
                # enemy pursuit alone cannot count as successful routing.
                reference = record['route_reference'] or pending['target_position']
                own_progress = (distance(pending['position'], reference)
                                - distance(unit.position, reference))
                if own_progress > .1 and gap < record['best_gap'] - .1:
                    record['best_gap'] = gap
                    record['last_progress_loop'] = loop
                stalled = loop - record['last_progress_loop'] >= self.stall_h * H
                if not active and (stalled or repeated is not None) \
                        and loop - record['last_replan_loop'] >= self.stall_h * H:
                    record['failed_targets'][(target, record['target_tag'])] = loop + 2 * H
                    record['failed_routes'][(target, pending['action'])] = {
                        'position': pending['position'], 'until_loop': loop + 2 * H}
                    if repeated:
                        # Each observed command owns its departure point. Ban
                        # the return leg where it occurred, across providers.
                        for a, b in zip(repeated, repeated[1:]):
                            if b['action'] is not None:
                                record['failed_routes'][(None, b['action'])] = {
                                    'position': a['position'], 'until_loop': loop + 2 * H}
                    record['last_replan_loop'] = loop
                    record['last_failure'] = 'observed_roundtrip' if repeated else 'no_net_contact_progress'
                    record['route_until_loop'] = loop
                    record['target'] = None  # Does not renew the debt deadline.

    def _route_failed(self, context, actor, target, action):
        record = self.records.get(actor)
        if record is None:
            return False
        unit = context.snapshot.allies[actor]
        specific = record['failed_routes'].get((target, action)) if target is not None else None
        local = record['failed_routes'].get((None, action))
        return any(row and row['until_loop'] > context.snapshot.game_loop
                   and distance(unit.position, row['position']) <= radius
                   for row, radius in ((specific, 2.), (local, max(.25, unit.radius))))

    def native_approach_actions(self, context, actor, memory, targets=None):
        """Reserve near-term melee contact capacity before direction scoring.

        One snapshot-local assignment is shared by G2, rejoin and adaptive.
        Current contact is never redirected. The final joint scorer still
        owns collision/fire budgets; this is a physical approach preference.
        """
        cache = getattr(context, '_native_contact_assignments', None)
        if cache is None:
            from .battle_line import actor_line
            cache = context._native_contact_assignments = {}
            occupied, candidates = {}, {}
            for peer, unit in context.snapshot.allies.items():
                cap = context.capabilities[peer]
                if not unit.alive or cap.healer or cap.suicide_attack or cap.combat_class != 'melee':
                    continue
                active = contacts(context, peer)
                if active:
                    chosen = min(active, key=lambda e: (distance(unit.position,
                        context.snapshot.enemies[e].position), e))
                    occupied[chosen] = occupied.get(chosen, 0) + 1
                    continue
                stride = (context.speed(peer) or 0.) * max(context.H, 1) / GAME_LOOPS_PER_SPEED_SECOND
                if stride <= 0. or danger_evidence(context, peer, horizon_h=2.)['immediate']:
                    continue
                options = []
                lane = actor_line(context, peer)
                compatible = self.targets(context, peer)
                peer_targets = tuple(e for e in compatible if e in lane['front']) if lane else compatible
                for enemy in peer_targets:
                    if not context.can_issue_attack(peer, enemy):
                        continue
                    gap = self.gap(context, peer, enemy, unit.position)
                    if gap / stride > 2.:
                        continue
                    action = context.attack_action(peer, enemy)
                    if self._route_failed(context, peer, enemy, action) or memory.movement_blockage(
                            context, peer, action) >= .5:
                        continue
                    point, _ = action_position(context, peer, action)
                    evidence = danger_evidence(context, peer, point, horizon_h=2.)
                    if evidence['immediate']:
                        continue
                    target = context.snapshot.enemies[enemy]
                    reach = context.effective_attack_reach(peer, enemy)
                    chord = 2 * unit.radius + .05
                    # Near-side semicircle, scaled by actual body size. No
                    # fixed unit count or race/composition special case.
                    capacity = max(1, int(math.pi * max(reach, unit.radius + target.radius) / chord))
                    options.append((gap / stride, enemy, action, capacity))
                if options:
                    candidates[peer] = options
            for peer in sorted(candidates, key=lambda a: (min(r[0] for r in candidates[a]), a)):
                options = candidates[peer]
                available = [r for r in options if occupied.get(r[1], 0) < r[3]]
                if not available:
                    continue  # full frontage needs another route, not forced crowding
                record = self.records.get(peer)
                committed = record['target'] if record and record['route_until_loop'] > context.snapshot.game_loop else None
                best = min(available, key=lambda r: (r[1] != committed if committed is not None else False,
                    r[0] + .5 * occupied.get(r[1], 0) / r[3],
                    context.snapshot.enemies[r[1]].effective_health, r[1]))
                cache[peer] = (best[2],)
                occupied[best[1]] = occupied.get(best[1], 0) + 1
        actions = cache.get(actor, ())
        return tuple(a for a in actions if targets is None or context.decode(actor, a).target in targets)

    def recovery(self, context, actor, memory):
        record = self.records.get(actor)
        if record is None:
            return None
        cache = getattr(context, '_contact_recovery', None)
        if cache is None:
            cache = context._contact_recovery = {}
        if actor in cache:
            return cache[actor]
        from .battle_line import actor_line
        unit, H = context.snapshot.allies[actor], max(context.H, 1)
        lane = actor_line(context, actor)
        targets = tuple(lane['front']) if lane else self.targets(context, actor)
        if not targets:
            return None
        loop = context.snapshot.game_loop
        stalled, elapsed = ((loop - record['last_progress_loop']) / H,
                            (loop - record['start_loop']) / H)
        failed = lambda e: record['failed_targets'].get((e, enemy_identity(context, e)), -1) > loop
        target = record['target']
        if target not in targets or failed(target):
            target = min(targets, key=lambda e: (failed(e), self.gap(context, actor, e, unit.position), e))
        native = self.native_approach_actions(context, actor, memory, targets)
        if native:
            target = context.decode(actor, native[0]).target
        stride = (context.speed(actor) or 0.) * H / GAME_LOOPS_PER_SPEED_SECOND
        before = danger_evidence(context, actor, horizon_h=2.)
        options = []
        for action in context.legal_actions(actor):
            semantic = context.decode(actor, action)
            if semantic.kind not in ('move', 'attack'):
                continue
            if semantic.kind == 'attack' and semantic.target not in targets:
                continue
            point, certain = action_position(context, actor, action)
            chosen = semantic.target if semantic.kind == 'attack' else target
            if memory.movement_blockage(context, actor, action) >= .5 \
                    or self._route_failed(context, actor, chosen, action):
                continue
            old_gap = self.gap(context, actor, chosen, unit.position)
            gap = self.gap(context, actor, chosen, point)
            if gap >= old_gap - .05 and not (old_gap <= .05 and semantic.kind == 'attack'):
                continue
            after = danger_evidence(context, actor, point, horizon_h=2.)
            relief = before['supported_amount'] - after['supported_amount'] >= max(
                .1, .15 * before['supported_amount'])
            # Potential all-enemy fire remains a ranking bound, not a veto.
            if after['immediate'] and not relief:
                continue
            distance_gap = old_gap if semantic.kind == 'attack' else gap
            eta = distance_gap / stride if stride > 0. else (0. if distance_gap <= .05 else float('inf'))
            if semantic.kind == 'move':
                eta += 1.
            eta = max(eta, unit.weapon_cooldown_loops / H)
            if not math.isfinite(eta):
                continue
            bound = exposure_window(context, actor, point, horizon_h=2., certain=certain)
            options.append((bool(native) and action not in native, failed(chosen),
                            eta,
                            bound['reserve_ratio'], action))
        options.sort()
        result = {'target': target, 'stalled_h': stalled, 'no_contact_h': elapsed,
                  'unfulfilled_h': elapsed,
                  'exhausted': stalled >= self.stall_h or elapsed >= self.deadline_h,
                  'preferred_actions': tuple(row[-1] for row in options[:2]),
                  'action_etas_h': {row[-1]: row[2] for row in options},
                  'contact_observed': bool(contacts(context, actor)),
                  'first_contact_loop': record['first_contact_loop'],
                  'failed_routes': len(record['failed_routes']),
                  'last_failure': record['last_failure'],
                  'route_until_loop': record['route_until_loop'],
                  'deadline_loop': record['start_loop'] + self.deadline_h * H,
                  'completion': 'awaiting_observed_weapon_cycle'}
        cache[actor] = result
        return result

    def coordination(self, context, actor, memory):
        """Shared arrival schedule, before individual approach commitments.

        A melee unit does not buy an uncovered first contact with its life.
        Half of the compatible rear volley must be able to arrive within
        its uncovered exchange allowance. This is a prospective scheduling
        bound, not a claim that all enemies target this unit. Existing
        contact keeps firing; lagging rear members actively close the gap.
        """
        cached = getattr(context, '_contact_coordination', None)
        if cached is not None:
            return cached.get(actor)
        from .battle_line import battle_line, _eta, _volley
        facts, units = battle_line(context), context.snapshot.allies
        H = max(context.H, 1)
        result, supporting = {}, {}
        for front in facts['fronts']:
            rear = [a for a in front['members'] if context.capabilities[a].combat_class != 'melee'
                    and not context.capabilities[a].healer and not context.capabilities[a].suicide_attack]
            for melee in front['members']:
                cap, unit = context.capabilities[melee], units[melee]
                lane = facts['actors'].get(melee)
                if cap.combat_class != 'melee' or lane is None or contacts(context, melee):
                    continue
                target = lane['target']
                choices = []
                for peer in rear:
                    eta = _eta(context, peer, target, units[peer].position)
                    value = _volley(context, peer, target)
                    established = set(contacts(context, peer)).intersection(front['targets'])
                    if established:
                        # Already fighting this bounded front counts through
                        # its real contact/cooldown, not an invented march to
                        # the melee actor's exact individual enemy.
                        eta = min(_eta(context, peer, e, units[peer].position) for e in established)
                        value = max(_volley(context, peer, e) for e in established)
                    if math.isfinite(eta) and value > 0.:
                        choices.append((eta, peer, value))
                if not choices:
                    continue  # An all-melee group must not wait for nonexistent rear fire.
                choices.sort()
                total, cumulative, support_eta = sum(r[2] for r in choices), 0., 0.
                supporters = []
                for eta, peer, value in choices:
                    cumulative += value
                    supporters.append(peer)
                    support_eta = eta
                    if cumulative >= .5 * total:
                        break
                own_eta = _eta(context, melee, target, unit.position)
                lead = support_eta - own_eta
                if lead <= 1.:
                    continue
                # Use one plausible hostile source, not every enemy's full
                # volley counted against this same body. Actual pressure is
                # retained as an independent lower bound.
                rates = []
                for enemy in context.risk_at(melee, lane['goal']).threats:
                    if context.enemy_capabilities[enemy].suicide_attack:
                        continue
                    damage = context.enemy_attack_damage(enemy, melee)
                    profile = context.snapshot.raw_state.get('unit_type_profiles', {}).get(
                        str(context.snapshot.enemies[enemy].unit_type_id), {})
                    periods = [w.get('attack_period_loops') for w in profile.get('weapons', ())
                               if w.get('target_type') in ('air' if unit.is_flying else 'ground', 'any')]
                    periods = [p for p in periods if isinstance(p, (int, float)) and math.isfinite(p) and p > 0.]
                    if damage is not None and periods:
                        rates.append(damage * H / min(periods))
                pressure = danger_evidence(context, melee, horizon_h=1.)['supported_amount']
                rate = max([pressure, *rates], default=0.)
                if rate <= 0. or lead * rate < .35 * unit.effective_health:
                    continue
                # Help already engaged bodies too: the rear should move to
                # this fight rather than each chasing a different target.
                anchor = min(supporters, key=lambda p: (distance(unit.position, units[p].position), p))
                origin_gap = distance(unit.position, units[anchor].position)
                before = danger_evidence(context, melee, horizon_h=2.)
                origin_risk = context.forecast_risk(melee, unit.position)
                routes = []
                for action in context.move_destinations(melee):
                    if self._route_failed(context, melee, None, action) or memory.movement_blockage(context, melee, action) >= .5:
                        continue
                    point, _ = action_position(context, melee, action)
                    gap = distance(point, units[anchor].position)
                    after = danger_evidence(context, melee, point, horizon_h=2.)
                    middle = tuple((x + y) / 2. for x, y in zip(unit.position, point))
                    midway = danger_evidence(context, melee, middle, horizon_h=2.)
                    risk = context.forecast_risk(melee, point)
                    if gap >= origin_gap - .1 or set(risk.effect_weapon_near) - set(origin_risk.effect_weapon_near):
                        continue
                    if after['supported_amount'] > before['supported_amount'] + 1.:
                        continue
                    if max(after['supported_amount'], midway['supported_amount']) >= unit.effective_health:
                        continue  # A dying body's less-lethal retreat is not coordination.
                    if len(risk.current_threats) > len(origin_risk.current_threats):
                        continue
                    routes.append((after['supported_amount'], gap, action))
                options = tuple(r[-1] for r in sorted(routes)[:2])
                record = self.records.get(melee)
                if not options and record is not None and not origin_risk.current_threats \
                        and not origin_risk.effect_weapon_near and STOP in context.legal_actions(melee) \
                        and record.get('synchronization_wait_loops', 0) < H:
                    options = (STOP,)  # One actual window, never renewed by another provider.
                result[melee] = {'required': bool(options), 'actions': options,
                    'reason': 'synchronize_with_rear_fire' if options else 'no_safe_synchronization_route',
                    'target': target, 'supporters': tuple(supporters), 'support_eta_h': support_eta,
                    'own_eta_h': own_eta, 'uncovered_h': lead,
                    'uncovered_damage_estimate': lead * rate,
                    'wait_used_loops': record.get('synchronization_wait_loops', 0) if record else 0}
                for peer in supporters:
                    supporting.setdefault(peer, []).append((melee, target))
        # Existing melee contact also requests support; it must not wait for
        # a fresh contact debt or a multi-window no-progress timeout.
        for front in facts['fronts']:
            engaged = [(a, e) for a in front['members']
                       if context.capabilities[a].combat_class == 'melee'
                       for e in contacts(context, a) if e in front['targets']]
            if engaged:
                for peer in front['members']:
                    if context.capabilities[peer].combat_class != 'melee':
                        supporting.setdefault(peer, []).extend(engaged)
        for peer, requests in supporting.items():
            if contacts(context, peer):
                continue
            unit = units[peer]
            before = danger_evidence(context, peer, horizon_h=2.)
            candidates = []
            for melee, target in requests:
                old_eta = _eta(context, peer, target, unit.position)
                for action in context.legal_actions(peer):
                    semantic = context.decode(peer, action)
                    if semantic.kind not in ('move', 'attack') or semantic.kind == 'attack' and semantic.target != target:
                        continue
                    if self._route_failed(context, peer, target, action) or memory.movement_blockage(context, peer, action) >= .5:
                        continue
                    point, _ = action_position(context, peer, action)
                    eta = _eta(context, peer, target, point)
                    if eta >= old_eta - .05:
                        continue
                    after = danger_evidence(context, peer, point, horizon_h=2.)
                    if after['immediate'] and (after['supported_amount'] >= unit.effective_health
                            or after['supported_amount'] >= before['supported_amount']):
                        continue
                    already_fighting = bool(contacts(context, melee)) or (
                        units[melee].health_lost + units[melee].shield_lost > 0.)
                    candidates.append((not already_fighting, eta, after['supported_amount'], action, target, melee))
            if candidates:
                candidates.sort()
                best = candidates[0]
                result[peer] = {'required': True, 'actions': tuple(dict.fromkeys(r[3] for r in candidates[:2])),
                    'reason': 'advance_rear_to_shared_contact', 'target': best[4],
                    'supporting_actors': tuple(sorted(set(r[0] for r in requests)))}
        context._contact_coordination = result
        return result.get(actor)

    def execution(self, context, actor, memory):
        """Required first actions when a concrete feasible route exists.

        Queries never renew commitment or erase a failed route. A final
        action from another provider receives exactly the same obligation.
        Rescue and current-fire duties are resolved by the coordinator first.
        """
        cache = getattr(context, '_contact_execution', None)
        if cache is None:
            cache = context._contact_execution = {}
        if actor in cache:
            return cache[actor]
        result = {'required': False, 'actions': (), 'reason': 'no_visible_contact_duty', 'target': None}
        unit, cap = context.snapshot.allies.get(actor), context.capabilities.get(actor)
        if unit is None or not unit.alive or cap is None or cap.healer or cap.suicide_attack:
            cache[actor] = result
            return result
        active = contacts(context, actor)
        coordinated = self.coordination(context, actor, memory)
        if active:
            result.update(actions=tuple(context.attack_action(actor, e) for e in active),
                          reason='current_contact_keeps_fire')
        elif coordinated and coordinated['required']:
            result.update(coordinated)
        elif danger_evidence(context, actor, horizon_h=2.)['immediate']:
            result['reason'] = 'observed_danger_requires_rescue_comparison'
        else:
            recovery = self.recovery(context, actor, memory)
            record = self.records.get(actor)
            if recovery is not None and record is not None:
                feasible = recovery['preferred_actions']
                native = self.native_approach_actions(context, actor, memory)
                native = tuple(a for a in native if a in feasible)
                held = record['route_action']
                if native:
                    result.update(required=True, actions=native, reason='native_contact_commit',
                                  target=context.decode(actor, native[0]).target)
                elif record['route_until_loop'] > context.snapshot.game_loop \
                        and held in recovery['action_etas_h']:
                    result.update(required=True, actions=(held,), reason='short_route_commit',
                                  target=recovery['target'])
                elif recovery['exhausted'] or record['failed_routes']:
                    result.update(required=bool(feasible), actions=feasible,
                        reason='replace_failed_or_overdue_route' if feasible else 'no_feasible_replacement',
                        target=recovery['target'])
                else:
                    from .battle_line import actor_line
                    lane = actor_line(context, actor)
                    front = set(lane['front']) if lane else set()
                    peers = lane['peers'] if lane else ()
                    active_front = any(context.snapshot.allies[p].alive and (
                        front.intersection(contacts(context, p)) or
                        context.snapshot.allies[p].health_lost + context.snapshot.allies[p].shield_lost > 0.)
                        for p in peers if distance(unit.position, context.snapshot.allies[p].position) <= 10.)
                    timely = tuple(a for a in feasible if recovery['action_etas_h'][a] <= 3.)
                    result.update(required=bool(active_front and timely),
                        actions=timely if active_front and timely else feasible,
                        reason='join_active_front_now' if active_front and timely else 'uncommitted_approach',
                        target=recovery['target'])
        cache[actor] = result
        return result

    def action_reason(self, context, actor, action, memory):
        cache = getattr(context, '_contact_action_costs', None)
        if cache is None:
            cache = context._contact_action_costs = {}
        key = actor, action
        if key not in cache:
            cache[key] = self._action_reason(context, actor, action, memory)
        return cache[key]

    def _action_reason(self, context, actor, action, memory):
        recovery = self.recovery(context, actor, memory)
        if not recovery or not recovery['exhausted']:
            return {'cost': 0., 'reason': 'no_overdue_visible_duty'}
        unit = context.snapshot.allies[actor]
        point, certain = action_position(context, actor, action)
        before = danger_evidence(context, actor, horizon_h=2.)
        after = danger_evidence(context, actor, point, horizon_h=2.)
        relief = before['supported_amount'] - after['supported_amount'] >= max(
            .1, .15 * before['supported_amount'])
        if before['immediate'] and (relief or before['effect_danger'] and not after['effect_danger']):
            return {'cost': 0., 'reason': 'observed_emergency_relief'}
        semantic = context.decode(actor, action)
        target = semantic.target if semantic.kind == 'attack' else recovery['target']
        if semantic.kind == 'attack' and target in contacts(context, actor):
            # Permit current fire; only observation, not this command, closes
            # the record. A cooling contact is not a newly completed duty.
            return {'cost': 0., 'reason': 'current_contact_attack_pending_observation'}
        base = min(2., max(1., recovery['stalled_h'] / self.stall_h,
                          recovery['no_contact_h'] / self.deadline_h))
        factor, reason = 1., 'overdue_without_contact'
        if action in recovery['preferred_actions']:
            old_gap = self.gap(context, actor, target, unit.position)
            new_gap = self.gap(context, actor, target, point)
            stride = (context.speed(actor) or 0.) * max(context.H, 1) / GAME_LOOPS_PER_SPEED_SECOND
            progress = max(0., old_gap - new_gap) / max(stride, .1)
            factor = max(.25, 1. - .5 * min(1., progress))
            reason = 'bounded_route_credit_still_unfulfilled'
        if not recovery['preferred_actions']:
            reason = 'no_recommendation_does_not_cancel_debt'
        return {'cost': base * factor, 'reason': reason,
                'no_contact_h': recovery['no_contact_h'],
                'stalled_h': recovery['stalled_h'], 'target': target}

    def action_cost(self, context, actor, action, memory):
        return self.action_reason(context, actor, action, memory)['cost']

    def commit(self, context, actions, memory=None):
        """Only final commands install a route; candidate queries are pure."""
        for actor, action in actions.items():
            unit, cap = context.snapshot.allies.get(actor), context.capabilities.get(actor)
            if unit is None or not unit.alive or cap is None or cap.healer or cap.suicide_attack:
                continue
            semantic = context.decode(actor, action)
            record = self.records.get(actor)
            recovery = self.recovery(context, actor, memory) if record and memory is not None else None
            target = semantic.target if semantic.kind == 'attack' else recovery['target'] if recovery else None
            target = target if target in self.targets(context, actor) else None
            identity = enemy_identity(context, target) if target is not None else None
            if record is not None and target is not None and (
                    target != record['target'] or identity != record['target_tag']):
                record.update(target=target, target_tag=identity,
                              best_gap=self.gap(context, actor, target, unit.position))
                record['route_until_loop'] = context.snapshot.game_loop
            if record is not None and target is not None and semantic.kind in ('move', 'attack') \
                    and record['route_until_loop'] <= context.snapshot.game_loop:
                # Only the actually submitted route starts its short lock.
                # Repeated commits of the same route do not slide the deadline.
                record.update(route_action=action,
                    route_until_loop=context.snapshot.game_loop + 2 * max(context.H, 1),
                    route_reference=tuple(context.snapshot.enemies[target].position))
            self.pending[actor] = {
                'tag': unit.tag, 'loop': context.snapshot.game_loop,
                'position': unit.position, 'cooldown': unit.weapon_cooldown_loops,
                'contact': bool(contacts(context, actor)), 'action': action,
                'attack_command': semantic.kind == 'attack',
                'coordination': bool(memory is not None and (self.coordination(context, actor, memory) or {}).get('required')),
                'target': target, 'target_tag': identity,
                'target_position': context.snapshot.enemies[target].position if target is not None else None}

    def summary(self, context):
        loop, H = context.snapshot.game_loop, max(context.H, 1)
        return {'active': {actor: {
                    'start_loop': row['start_loop'], 'deadline_loop': row['start_loop'] + self.deadline_h * H,
                    'no_contact_h': (loop - row['start_loop']) / H,
                    'stalled_h': (loop - row['last_progress_loop']) / H,
                    'first_contact_loop': row['first_contact_loop'], 'target': row['target'],
                    'failed_routes': len(row['failed_routes']), 'last_failure': row['last_failure'],
                    'route_action': row['route_action'], 'route_until_loop': row['route_until_loop']}
                for actor, row in self.records.items()}, 'outcomes': list(self.outcomes)}
