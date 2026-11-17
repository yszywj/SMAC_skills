"""Observed fixed-region duties for one-use attackers; commands are not hits."""
import math


def _enemy_tag(context, enemy_id):
    return context.snapshot.raw_state.get('visible_enemies', {}).get(enemy_id, {}).get('tag')


class BurstProgress:
    def __init__(self, config=None):
        self.records, self.pending, self.last_actions = {}, {}, {}
        self.loop = None

    def __eq__(self, other):
        return isinstance(other, BurstProgress) and self.__dict__ == other.__dict__

    @staticmethod
    def _clear_goal(record):
        record.update(goal=None, goal_tag=None, region=None, lock_until_loop=0,
                      best_distance=float('inf'))

    def observe(self, context):
        context.burst_progress = self
        loop, H = context.snapshot.game_loop, max(context.H, 1)
        if self.loop is not None and loop <= self.loop:
            return
        self.loop = loop
        for actor in set(self.records) | set(self.pending):
            unit = context.snapshot.allies.get(actor)
            record = self.records.get(actor) or self.pending[actor]
            if unit is None or not unit.alive or unit.tag != record['tag']:
                self.records.pop(actor, None)
                self.pending.pop(actor, None)
        for actor, unit in context.snapshot.allies.items():
            if not unit.alive or not context.capabilities[actor].suicide_attack:
                continue
            record = self.records.setdefault(actor, {
                'tag': unit.tag, 'start_loop': loop, 'last_progress_loop': loop,
                'approach_start_loop': None, 'wait_loops': 0, 'goal': None,
                'goal_tag': None, 'region': None, 'best_distance': float('inf'),
                'lock_until_loop': 0, 'last_visible_loop': loop, 'blocked': {},
                'failed_regions': [], 'failed_edges': [], 'history': [],
                'last_action': None, 'encounter_active': False,
                'escape_used': False, 'detour_used': False, 'last_failure': None,
                'last_execution': None, 'deadline_expired': False})
            visible = any(not e.is_flying for e in context.snapshot.enemies.values())
            if visible:
                if not record['encounter_active']:
                    record.update(start_loop=loop, last_progress_loop=loop,
                                  approach_start_loop=None, wait_loops=0,
                                  escape_used=False, detour_used=False,
                                  last_failure=None, encounter_active=True, deadline_expired=False)
                record['last_visible_loop'] = loop
            elif loop - record['last_visible_loop'] > 4 * H:
                self._clear_goal(record)
                record.update(encounter_active=False, approach_start_loop=None,
                              history=[], failed_edges=[], failed_regions=[])
            record['blocked'] = {a: end for a, end in record['blocked'].items() if end > loop}
            for field in ('failed_regions', 'failed_edges'):
                record[field] = [r for r in record[field] if r['until_loop'] > loop]
            pending = self.pending.pop(actor, None)
            consecutive = pending and pending['tag'] == unit.tag and 0 < loop - pending['loop'] <= 2 * H
            if consecutive:
                elapsed = loop - pending['loop']
                action = pending['action']
                kind = context.decode(actor, action).kind
                record['last_action'] = action
                if pending['visible'] and kind == 'stop':
                    record['wait_loops'] += elapsed
                displacement = math.dist(unit.position, pending['position'])
                if kind in ('move', 'attack') and pending['expected_motion'] and displacement < .2:
                    record['blocked'][action] = loop + 2 * H
                if pending['goal_position'] is not None and record['region'] == pending['goal_position']:
                    gap = math.dist(unit.position, record['region'])
                    # Best distance to ONE fixed reference, not each move's
                    # individually convenient target or enemy pursuit.
                    if gap < record['best_distance'] - .25:
                        recovered_route = record['best_distance'] - gap >= .5
                        record['best_distance'] = gap
                        record['last_progress_loop'] = loop
                        if pending['execution_reason'] != 'observed_emergency_one_window':
                            record['escape_used'] = False
                        if recovered_route and pending['execution_reason'] != 'one_window_obstacle_detour':
                            record['detour_used'] = False
                if pending['execution_reason'] == 'observed_emergency_one_window':
                    self._clear_goal(record)
            history = record['history']
            if not consecutive:
                history[:] = []
            history.append((loop, tuple(unit.position)))
            history[:] = [r for r in history[-6:] if loop - r[0] <= 4 * H]
            revisited = None
            for index, (old_loop, point) in enumerate(history[:-1]):
                if loop - old_loop < 2 * H or math.dist(point, unit.position) > .6:
                    continue
                walk = sum(math.dist(a[1], b[1]) for a, b in zip(history[index:], history[index+1:]))
                if walk >= 1.:
                    revisited = index
                    break
            stalled = record['region'] is not None and loop - record['last_progress_loop'] >= 2 * H
            if record['region'] is not None and (stalled or revisited is not None):
                record['failed_regions'].append({'target': record['goal'], 'position': record['region'],
                                                 'until_loop': loop + 2 * H})
                segments = history[revisited:] if revisited is not None else history[-2:]
                for before, after in zip(segments, segments[1:]):
                    if math.dist(before[1], after[1]) > .2:
                        record['failed_edges'].append({'start': before[1], 'end': after[1],
                                                       'until_loop': loop + 2 * H})
                record['last_failure'] = 'observed_roundtrip' if revisited is not None else 'no_net_approach_2h'
                self._clear_goal(record)  # never renew approach_start_loop
            target = context.snapshot.enemies.get(record['goal'])
            if record['goal'] is not None and (target is None or target.is_flying or
                    _enemy_tag(context, record['goal']) != record['goal_tag']):
                self._clear_goal(record)
            if (not record['deadline_expired'] and record['approach_start_loop'] is not None
                    and loop - record['approach_start_loop'] >= 4 * H):
                # Expiry changes the reference and admissible horizon, not
                # just its log label. Never restart the encounter deadline.
                if record['region'] is not None:
                    record['failed_regions'].append({'target': record['goal'],
                        'position': record['region'], 'until_loop': loop + 2 * H})
                self._clear_goal(record)
                record.update(deadline_expired=True, last_failure='approach_deadline_expired')

    def _edge_failed(self, context, actor, action):
        from .burst import action_position
        record = self.records.get(actor)
        if record is None:
            return False
        if action in record['blocked']:
            return True
        unit = context.snapshot.allies[actor]
        point = action_position(context, actor, action)
        vx, vy = point[0] - unit.position[0], point[1] - unit.position[1]
        length = math.hypot(vx, vy)
        if length <= .05:
            return False
        for edge in record['failed_edges']:
            if math.dist(unit.position, edge['start']) > max(.25, unit.radius):
                continue
            dx, dy = edge['end'][0] - edge['start'][0], edge['end'][1] - edge['start'][1]
            if (vx * dx + vy * dy) / max(length * math.hypot(dx, dy), .01) > .9:
                return True
        return False

    def region(self, context, actor):
        """Select all regions once BEFORE actions. Reservations only diversify
        planning; final-branch residual damage allocation remains independent.
        """
        cache = getattr(context, '_burst_fixed_regions', None)
        if cache is not None:
            return cache.get(actor)
        from .burst import profile, useful_area_value, area_damage, action_position, exchange_value
        effect = profile(context)
        selected, reserved, budgets = {}, {}, {}
        H, loop = max(context.H, 1), context.snapshot.game_loop
        if effect.trigger_reach is None:
            context._burst_fixed_regions = selected
            return None
        def arrival_order(item):
            source, unit = item
            speed = context.speed(source) or 0.
            earliest = min((max(0., math.dist(unit.position, enemy.position) -
                unit.radius - enemy.radius - effect.trigger_reach) * 16. / speed
                for enemy in context.snapshot.enemies.values() if not enemy.is_flying),
                default=float('inf')) if speed > 0. else float('inf')
            return earliest, source
        sources = [(source, unit) for source, unit in context.snapshot.allies.items()
                   if unit.alive and context.capabilities[source].suicide_attack]
        for source, unit in sorted(sources, key=arrival_order):
            if not unit.alive or not context.capabilities[source].suicide_attack:
                continue
            record = self.records.get(source)
            speed = context.speed(source)
            if speed is None or speed <= 0.:
                continue
            remaining = {e: max(0., enemy.effective_health - reserved.get(e, 0.))
                         for e, enemy in context.snapshot.enemies.items()}
            budgets[source] = remaining
            def approachable(target, point):
                old_gap = math.dist(unit.position, point)
                for action in context.legal_actions(source):
                    semantic = context.decode(source, action)
                    if semantic.kind not in ('move', 'attack') or self._edge_failed(context, source, action):
                        continue
                    if semantic.kind == 'attack' and semantic.target != target:
                        continue
                    gap = math.dist(action_position(context, source, action), point)
                    if gap < old_gap - .05 or old_gap <= .05 and semantic.kind == 'attack':
                        return True
                    if semantic.kind == 'move' and gap <= old_gap + .5 and not (record and record['detour_used']):
                        return True
                return False
            options = []
            locked = record and record['region'] is not None and loop < record['lock_until_loop']
            retained = locked and record['goal'] in context.snapshot.enemies and useful_area_value(
                    context, record['region'], remaining) > 0. and approachable(record['goal'], record['region']) \
                    and exchange_value(context, source, record['region'], remaining)['worthwhile']
            if retained:
                options = [(record['goal'], record['region'])]
            else:
                for enemy_id, enemy in sorted(context.snapshot.enemies.items()):
                    if enemy.is_flying:
                        continue
                    angle = math.atan2(unit.position[1] - enemy.position[1],
                                       unit.position[0] - enemy.position[0])
                    reach = unit.radius + enemy.radius + effect.trigger_reach
                    for turn in (0., -.6, .6):
                        point = (enemy.position[0] + math.cos(angle + turn) * reach,
                                 enemy.position[1] + math.sin(angle + turn) * reach)
                        if record and any(r['target'] == enemy_id and math.dist(point, r['position']) < .4
                                          for r in record['failed_regions']):
                            continue
                        if approachable(enemy_id, point):
                            options.append((enemy_id, point))
            ranked = []
            for target, point in options:
                useful = useful_area_value(context, point)
                uncertain_value = any(amount is None for amount in area_damage(context, point).values())
                if useful <= 0. and not uncertain_value:
                    continue
                residual = useful_area_value(context, point, remaining)
                arrival = math.dist(unit.position, point) * 16. / speed
                exchange = exchange_value(context, source, point, remaining)
                value = max(0., exchange['useful_hp'] - exchange['counter_burst_hp']) / (
                    exchange['source_cost_hp'] * (1. + arrival / H))
                # Unknown native damage permits geometric approach, without
                # a numeric damage reward or mandatory useful-burst claim.
                # Once the original approach expired, a fresh long detour
                # cannot inherit its mandatory status. Prefer a genuinely
                # reachable replacement; retain other routes as soft options.
                actionable = exchange['worthwhile'] and (not record or not record['deadline_expired']
                                                        or arrival <= H)
                ranked.append((actionable, residual > 0., value, -arrival, target,
                               point, useful > 0., exchange))
            if not ranked:
                continue
            actionable, _, value, negative_arrival, target, point, known_value, exchange = max(
                ranked, key=lambda row: row[:6])
            selected[source] = {'target_id': target, 'position': tuple(point),
                                'target_tag': _enemy_tag(context, target), 'value': value,
                                'arrival_loops': -negative_arrival, 'locked': bool(retained),
                                'known_value': known_value, 'exchange': exchange,
                                'compulsory_approach': bool(actionable)}
            for enemy, amount in area_damage(context, point).items():
                if amount is not None:
                    reserved[enemy] = reserved.get(enemy, 0.) + amount
        context._burst_fixed_regions = selected
        context._burst_region_budgets = budgets
        return selected.get(actor)

    def preferred_goal(self, context, actor):
        region = self.region(context, actor)
        return region['target_id'] if region is not None else None

    def overdue(self, context, actor):
        record = self.records.get(actor)
        if not record or not record['encounter_active']:
            return False
        loop, H = context.snapshot.game_loop, max(context.H, 1)
        return bool(record['approach_start_loop'] is not None and loop - record['approach_start_loop'] >= 4 * H
                    or record['region'] is not None and loop - record['last_progress_loop'] >= 2 * H
                    or record['last_failure'] is not None)

    def wait_available(self, context, actor):
        record = self.records.get(actor)
        return bool(record and record['encounter_active'] and record['wait_loops'] < context.H
                    and not self.overdue(context, actor))

    def release_goal(self, actor, goal):
        record = self.records.get(actor)
        if record and record['goal'] == goal:
            self._clear_goal(record)

    def execution(self, context, actor):
        """Pure final-command duty. No feasible alternative means no empty veto."""
        cache = getattr(context, '_burst_execution', None)
        if cache is None:
            cache = context._burst_execution = {}
        if actor not in cache:
            cache[actor] = self._execution(context, actor)
        return cache[actor]

    def _execution(self, context, actor):
        from .burst import (action_position, outcome, useful_area_value, cover_window,
                            trigger_assignments, exchange_value)
        from .combat_exchange import danger_evidence
        record, unit = self.records.get(actor), context.snapshot.allies.get(actor)
        empty = {'required': False, 'actions': (), 'reason': 'no_visible_useful_region'}
        if record is None or unit is None or not unit.alive:
            return empty
        region = self.region(context, actor)
        legal = tuple(context.legal_actions(actor))
        assignments = trigger_assignments(context)
        remaining = getattr(context, '_burst_trigger_remaining', {})
        def answer(actions, reason):
            return {'required': bool(actions), 'actions': tuple(actions), 'reason': reason,
                    'target': assignments[actor]['target'] if actor in assignments
                        else region['target_id'] if region else None,
                    'region': assignments[actor]['position'] if actor in assignments
                        else region['position'] if region else None,
                    'exchange': assignments.get(actor, {}).get('exchange'),
                    'overdue': self.overdue(context, actor),
                    'deadline_loop': record['approach_start_loop'] + 4 * max(context.H, 1)
                        if record['approach_start_loop'] is not None else None}
        if actor in assignments:
            return answer([assignments[actor]['action']], 'execute_positive_marginal_exchange')
        native_options = [a for a in legal if context.decode(actor, a).kind == 'attack'
                          and outcome(context, actor, a).trigger == 'active_contact_opportunity']
        if native_options:
            # No compulsory poor trade, and no compulsory approach that
            # silently contains that same rejected trigger. Keep all legal
            # alternatives for the final exchange scorer instead.
            return answer([], 'marginal_trigger_requires_joint_comparison')
        before = danger_evidence(context, actor, horizon_h=1.)
        if before['immediate'] and not record['escape_used'] and useful_area_value(context, unit.position) <= 0.:
            escapes = []
            for action in context.move_destinations(actor):
                if self._edge_failed(context, actor, action):
                    continue
                point = action_position(context, actor, action)
                after = danger_evidence(context, actor, point, horizon_h=1.)
                if (after['supported_amount'] < unit.effective_health and not after['effect_danger']
                        and (before['supported_amount'] > 0.
                             and after['supported_amount'] <= .6 * before['supported_amount']
                             or before['effect_danger'])):
                    escapes.append((after['supported_amount'], action))
            if escapes:
                return answer([a for _, a in sorted(escapes)[:2]], 'observed_emergency_one_window')
        if region is None:
            return answer([], 'blocked_no_reachable_region' if record['failed_edges'] or record['blocked']
                          else 'no_visible_useful_region')
        if not region['known_value']:
            return answer([], 'unknown_damage_geometric_fallback')
        if not region['compulsory_approach']:
            return answer([], 'expired_approach_returned_to_joint_search' if record['deadline_expired']
                          else 'regional_exchange_requires_joint_comparison')
        origin_gap = math.dist(unit.position, region['position'])
        forward, lateral = [], []
        for action in legal:
            semantic = context.decode(actor, action)
            if semantic.kind not in ('move', 'attack') or self._edge_failed(context, actor, action):
                continue
            if semantic.kind == 'attack' and semantic.target != region['target_id']:
                continue
            result = outcome(context, actor, action)
            if result.opportunity > 0. and not exchange_value(context, actor, result.position,
                    remaining, opportunity=result.opportunity, consumed=assignments)['worthwhile']:
                continue  # An attractive endpoint cannot require a poor path-death trade.
            point = action_position(context, actor, action)
            gap = math.dist(point, region['position'])
            if gap < origin_gap - .05:
                forward.append((semantic.kind != 'attack', gap, action))
            elif semantic.kind == 'move' and gap <= origin_gap + .5:
                lateral.append((gap, action))
        if forward:
            actions = [a for _, _, a in sorted(forward)[:2]]
            if self.wait_available(context, actor):
                ready, arriving = cover_window(context, actor, region['target_id'])
                if ready == 0 and arriving > 0:
                    actions += [a for a in legal if context.decode(actor, a).kind == 'stop']
            return answer(actions, 'deadline_short_replacement' if record['deadline_expired']
                          else 'fixed_region_approach')
        if lateral and not record['detour_used']:
            return answer([min(lateral)[1]], 'one_window_obstacle_detour')
        return answer([], 'blocked_no_progressing_action')

    def action_cost(self, context, actor, action):
        from .burst import outcome, exchange_value
        duty = self.execution(context, actor)
        if duty['required']:
            return 0. if action in duty['actions'] else 1.6
        record = self.records.get(actor)
        if record is None or not record['encounter_active']:
            return 0.
        # Weak trades remain legal. Keep their exchange deficit visible to
        # baseline/final scoring instead of relabelling any positive damage
        # as a productive commitment.
        cost = .8 if self._edge_failed(context, actor, action) else 0.
        result = outcome(context, actor, action)
        if result.opportunity > 0.:
            exchange = exchange_value(context, actor, result.position,
                opportunity=result.opportunity)
            if exchange['known']:
                cost += min(2., max(0., -exchange['net_hp']) / exchange['source_cost_hp'])
        return cost

    def commit(self, context, actions):
        from .burst import action_position
        self.last_actions = dict(actions)
        for actor, action in actions.items():
            unit, record = context.snapshot.allies.get(actor), self.records.get(actor)
            if record is None or unit is None or not unit.alive:
                continue
            region, duty = self.region(context, actor), self.execution(context, actor)
            loop, H = context.snapshot.game_loop, max(context.H, 1)
            following = region is not None and math.dist(action_position(context, actor, action),
                region['position']) < math.dist(unit.position, region['position']) - .05
            if following and (record['goal'] != region['target_id'] or
                                      record['region'] != region['position']):
                record.update(goal=region['target_id'], goal_tag=region['target_tag'],
                              region=region['position'], lock_until_loop=loop + 2 * H,
                              best_distance=math.dist(unit.position, region['position']),
                              last_progress_loop=loop)
            if following and region['arrival_loops'] <= 4 * H and record['approach_start_loop'] is None:
                record['approach_start_loop'] = loop
            record['last_execution'] = dict(duty, selected_action=action,
                violated=bool(duty['required'] and action not in duty['actions']))
            followed = duty['required'] and action in duty['actions']
            if followed and duty['reason'] == 'observed_emergency_one_window':
                record['escape_used'] = True
            if followed and duty['reason'] == 'one_window_obstacle_detour':
                record['detour_used'] = True
            self.pending[actor] = {
                'tag': unit.tag, 'loop': loop, 'position': tuple(unit.position),
                'action': action, 'visible': bool(context.snapshot.enemies),
                'expected_motion': math.dist(unit.position, action_position(context, actor, action)) > .2,
                'goal_position': record['region'],
                'execution_reason': duty['reason'] if followed else None}

    def summary(self):
        return {actor: {key: record[key] for key in (
            'goal', 'region', 'start_loop', 'approach_start_loop', 'lock_until_loop',
            'last_progress_loop', 'wait_loops', 'last_action', 'last_failure', 'last_execution',
            'deadline_expired')}
                for actor, record in self.records.items()}
