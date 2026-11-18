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
                'last_execution': None, 'deadline_expired': False,
                'cashout_reason': None, 'cashout_target': None, 'cashout_tag': None,
                'ordinary_start': 0})
            ordinary_alive = sum(u.alive and not context.capabilities[a].suicide_attack
                                 and not context.capabilities[a].healer
                                 for a, u in context.snapshot.allies.items())
            visible = any(not e.is_flying for e in context.snapshot.enemies.values())
            if visible:
                if not record['encounter_active']:
                    record.update(start_loop=loop, last_progress_loop=loop,
                                  approach_start_loop=None, wait_loops=0,
                                  escape_used=False, detour_used=False,
                                  last_failure=None, encounter_active=True, deadline_expired=False,
                                  cashout_reason=None, cashout_target=None, cashout_tag=None,
                                  ordinary_start=ordinary_alive)
                record['last_visible_loop'] = loop
            elif loop - record['last_visible_loop'] > 4 * H:
                self._clear_goal(record)
                record.update(encounter_active=False, approach_start_loop=None,
                              history=[], failed_edges=[], failed_regions=[],
                              cashout_reason=None, cashout_target=None, cashout_tag=None)
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
                if record['approach_start_loop'] is not None:
                    record['cashout_reason'] = record['cashout_reason'] or record['last_failure']
                self._clear_goal(record)  # never renew approach_start_loop
            target = context.snapshot.enemies.get(record['goal'])
            if record['goal'] is not None and (target is None or target.is_flying or
                    _enemy_tag(context, record['goal']) != record['goal_tag']):
                self._clear_goal(record)
            if record['goal'] is not None and target is not None \
                    and record.get('region_target_position') is not None \
                    and math.dist(target.position, record['region_target_position']) > 1.5:
                # The cluster plainly moved away: release the locked region
                # early WITHOUT a failure penalty and WITHOUT resetting the
                # total approach deadline, wait budget or detour allowance.
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
                record['cashout_reason'] = record['cashout_reason'] or 'approach_deadline_expired'
            if visible and (ordinary_alive == 0 or record['ordinary_start'] > 0
                            and ordinary_alive * 2 <= record['ordinary_start']):
                record['cashout_reason'] = record['cashout_reason'] or 'frontline_depleted'
            # A moving/invalid goal must not hide perpetual approach before
            # approach_start_loop was ever installed by a selected region.
            if visible and loop - record['start_loop'] >= 6 * H:
                record['cashout_reason'] = record['cashout_reason'] or 'visible_encounter_budget_expired'

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
            if retained and record.get('region_value') is not None:
                # Early release when the cluster plainly moved away or its
                # effective coverage collapsed. The total approach deadline
                # and wait budget are NOT refreshed by a re-lock elsewhere.
                retained = useful_area_value(context, record['region'], remaining) \
                    >= .5 * record['region_value']
            if retained:
                options = [(record['goal'], record['region'])]
            else:
                effect_radius = effect.blast_radius_upper \
                    if effect.blast_radius_upper is not None else 0.
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
                # Bounded multi-victim contact points: pairs of ground enemies
                # whose separation is coverable from one trigger position add
                # the short side-shift point between them. Small fixed set —
                # no grid search, no unbounded candidate growth.
                ground = [e for e, enemy in sorted(context.snapshot.enemies.items())
                          if not enemy.is_flying]
                pair_points = []
                for index, first in enumerate(ground):
                    for second in ground[index + 1:]:
                        gap = math.dist(context.snapshot.enemies[first].position,
                                        context.snapshot.enemies[second].position)
                        if gap > 2. * effect_radius + 1.:
                            continue
                        for anchor, other in ((first, second), (second, first)):
                            a, b = (context.snapshot.enemies[anchor].position,
                                    context.snapshot.enemies[other].position)
                            direction = math.atan2(b[1] - a[1], b[0] - a[0])
                            reach2 = unit.radius + context.snapshot.enemies[anchor].radius \
                                + effect.trigger_reach
                            point = (a[0] + math.cos(direction) * reach2,
                                     a[1] + math.sin(direction) * reach2)
                            if record and any(r['target'] == anchor
                                              and math.dist(point, r['position']) < .4
                                              for r in record['failed_regions']):
                                continue
                            if approachable(anchor, point):
                                pair_points.append((anchor, point))
                # Keep the bounded best side-shift points by coverable value.
                pair_points.sort(key=lambda item: -useful_area_value(context, item[1]))
                # Reserve useful multi-target candidates before trimming;
                # enemy ID order must not erase the new candidate family.
                pairs = list(dict.fromkeys(pair_points))[:4]
                ordinary = sorted(set(options) - set(pairs), key=lambda item: (
                    -useful_area_value(context, item[1], remaining),
                    math.dist(unit.position, item[1]), item))
                options = pairs + ordinary[:12 - len(pairs)]
            ranked = []
            for target, point in options:
                useful = useful_area_value(context, point)
                uncertain_value = any(amount is None for amount in area_damage(context, point).values())
                if useful <= 0. and not uncertain_value:
                    continue
                arrival = math.dist(unit.position, point) * 16. / speed
                # Only supported lethal danger can replace the chosen region
                # with an earlier death center. A whole-army volley bound
                # must not erase every reachable region before joint search.
                from .burst import supported_death_point
                eval_point = point
                horizon = max(1., min(arrival, 4. * H))
                death = supported_death_point(context, source, point, horizon)
                if death is not None:
                    eval_point = death[0]
                residual = useful_area_value(context, eval_point, remaining)
                exchange = exchange_value(context, source, eval_point, remaining)
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
                               point, useful > 0., exchange, eval_point))
            if not ranked:
                continue
            actionable, _, value, negative_arrival, target, point, known_value, exchange, reserved_point = max(
                ranked, key=lambda row: row[:6])
            selected[source] = {'target_id': target, 'position': tuple(point),
                                'target_tag': _enemy_tag(context, target), 'value': value,
                                'arrival_loops': -negative_arrival, 'locked': bool(retained),
                                'known_value': known_value, 'exchange': exchange,
                                'compulsory_approach': bool(actionable)}
            for enemy, amount in area_damage(context, reserved_point).items():
                if amount is not None:
                    reserved[enemy] = reserved.get(enemy, 0.) + amount
        context._burst_fixed_regions = selected
        context._burst_region_budgets = budgets
        return selected.get(actor)

    def _low_value_harvest(self, context, actor, exchange, remaining, result):
        """Reject LOW-RESIDUAL-VALUE harvests only — never opportunity doubt.

        Two concepts are deliberately separated:
          A. the target itself offers little remaining value (this check);
          B. the current trigger/approach opportunity is uncertain — that
             is route/exchange evaluation with its own discounts and is NOT
             a "harvest" rejection.
        Value is therefore measured on an UNDISCOUNTED exchange
        (opportunity=1.0): a full-health zergling or a healthy hydra is
        never "low value" just because the death-discount shrank the
        estimate. A rejection additionally requires a genuinely REACHABLE
        alternative (known burst damage and arrival within 4H), or an
        ordinary teammate able to harvest the残血 target this window.
        """
        target = result.target_id
        if target is None or target not in context.snapshot.enemies:
            return None
        unit = context.snapshot.allies[actor]
        from .burst import (exchange_value as _exchange, damage as burst_damage,
                            trigger_position, action_position, supported_death_point)
        undiscounted = _exchange(context, actor, result.position, remaining)
        if not undiscounted['known']:
            return None                      # unknown value is concept B
        # This exception is specifically an isolated residual cleanup. A
        # low damage matchup or several small victims is not that case.
        victims = [e for e, amount in result.damage_potential.items()
                   if amount is not None and amount > 0. and remaining.get(e, 0.) > 0.]
        if victims != [target] or remaining.get(target, 0.) >= .5 * max(
                undiscounted['source_cost_hp'], 1.):
            return None
        if undiscounted['useful_hp'] >= .5 * max(undiscounted['source_cost_hp'], 1.):
            return None                      # clearly worth the source
        enemy = context.snapshot.enemies[target]
        if context.enemy_capabilities[target].suicide_attack:
            return None                      # exception: block enemy bomber
        if len(context.snapshot.enemies) <= 2:
            return None                      # exception: endgame clearance
        for ally, other in context.snapshot.allies.items():
            if other.alive and other.health_fraction < .35 \
                    and math.dist(other.position, enemy.position) <= 6.:
                return None                  # exception: rescue critical ally
        # Alternative must have native reach/speed, a legal progressing
        # first step, and useful coverage at any supported path-death point.
        # Do not recurse into viable_step: it calls this residual-value check.
        speed = context.speed(actor)
        reachable = []
        for enemy_id, other in context.snapshot.enemies.items():
            if enemy_id == target or other.is_flying \
                    or remaining.get(enemy_id, 0.) <= 0. \
                    or burst_damage(context, enemy_id) is None or not speed or speed <= 0.:
                continue
            point, can_arrive = trigger_position(context, actor, enemy_id, 4 * max(context.H, 1))
            if not can_arrive:
                continue
            death = supported_death_point(context, actor, point, 4 * max(context.H, 1))
            alternative = _exchange(context, actor, death[0] if death else point, remaining)
            if not alternative['known'] or alternative['endangered_peers'] \
                    or alternative['useful_hp'] <= max(undiscounted['useful_hp'],
                        alternative['counter_burst_hp'], .5 * alternative['source_cost_hp']):
                continue
            old_gap = math.dist(unit.position, point)
            for action in context.legal_actions(actor):
                semantic = context.decode(actor, action)
                if semantic.kind not in ('move', 'attack') or self._edge_failed(context, actor, action) \
                        or semantic.kind == 'attack' and semantic.target != enemy_id:
                    continue
                step = action_position(context, actor, action)
                if math.dist(step, point) >= old_gap - .05 and not (
                        old_gap <= .05 and semantic.kind == 'attack'):
                    continue
                first_death = supported_death_point(context, actor, step)
                if first_death is not None:
                    first_exchange = _exchange(context, actor, first_death[0], remaining)
                    if not first_exchange['known'] or first_exchange['endangered_peers'] \
                            or first_exchange['useful_hp'] <= first_exchange['counter_burst_hp']:
                        continue
                reachable.append(enemy_id)
                break
        # A teammate able to finish this残血 target this window also makes
        # the bomber's重复投入 redundant — reported with its own reason.
        for peer, mate in context.snapshot.allies.items():
            if peer == actor or not mate.alive \
                    or context.capabilities[peer].healer \
                    or context.capabilities[peer].suicide_attack:
                continue
            estimate = context.fire_estimate(peer, target)
            if context.contact_status(peer, target) == 'in_range' \
                    and estimate.window_shots >= 1 and estimate.damage_lo is not None \
                    and estimate.damage_lo >= max(remaining.get(target, 0.), 0.):
                return {"reason": "declined_harvestable_by_teammates",
                        "target": target,
                        "useful_hp": round(undiscounted['useful_hp'], 2),
                        "harvester": peer,
                        "reachable_alternatives": sorted(reachable)[:3]}
        if not reachable:
            return None                      # no actually feasible alternative
        return {"reason": "declined_low_value_harvest",
                "target": target,
                "useful_hp": round(undiscounted['useful_hp'], 2),
                "source_cost_hp": round(undiscounted['source_cost_hp'], 2),
                "reachable_alternatives": sorted(reachable)[:3]}

    def viable_step(self, context, actor, action, remaining=None):
        """One-window route guard, shared by forced approach and fallback.

        Death inside an effective blast is allowed. A distant attractive
        target cannot pay for death before entering any useful coverage.
        This query never calls cashout/region allocation.
        """
        from .burst import outcome, action_position, supported_death_point, exchange_value
        unit = context.snapshot.allies[actor]
        if remaining is None:
            remaining = {e: enemy.effective_health for e, enemy in context.snapshot.enemies.items()}
        result = outcome(context, actor, action)
        if result.opportunity > 0.:
            exchange = exchange_value(context, actor, result.position, remaining,
                                      opportunity=result.opportunity,
                                      enemy_positions=result.enemy_positions)
            if exchange['known']:
                return (exchange['useful_hp'] > exchange['counter_burst_hp']
                        and not exchange['endangered_peers']
                        and not self._low_value_harvest(context, actor, exchange, remaining, result))
        point = action_position(context, actor, action)
        return supported_death_point(context, actor, point) is None

    def guarded_actions(self, context, actor):
        """A feasible safe alternative makes bad approach an actual violation."""
        cache = getattr(context, '_burst_guard_actions', None)
        if cache is None:
            cache = context._burst_guard_actions = {}
        if actor not in cache:
            legal = tuple(context.legal_actions(actor))
            valid = tuple(a for a in legal if self.viable_step(context, actor, a))
            # No empty veto: preserve legal exchange comparison when trapped.
            cache[actor] = valid or legal
        return cache[actor]

    def joint_allowed(self, context, actions):
        """Branch-local residual budgets; never reserve across candidates."""
        from .burst import outcome
        key = tuple(sorted(actions.items()))
        cache = getattr(context, '_burst_joint_allowed', None)
        if cache is None:
            cache = context._burst_joint_allowed = {}
        if key in cache:
            return cache[key]
        sources = [(actor, outcome(context, actor, action))
                   for actor, action in actions.items()
                   if context.snapshot.allies[actor].alive
                   and context.capabilities[actor].suicide_attack]
        reservations, allowed = [], {}
        for actor, result in sorted(sources, key=lambda row: (row[1].trigger_loops, row[0])):
            def residual_for(candidate):
                reserved = {}
                for trigger_loop, source, damage in reservations:
                    if (trigger_loop, source) >= (candidate.trigger_loops, actor):
                        continue
                    for enemy, amount in damage.items():
                        reserved[enemy] = reserved.get(enemy, 0.) + amount
                remaining = {e: max(0., unit.effective_health - reserved.get(e, 0.))
                             for e, unit in context.snapshot.enemies.items()}
                # The candidate's own trigger time determines which known
                # ordinary shots happen first. A later selected action must
                # not erase an earlier alternative's remaining damage.
                for peer, action in actions.items():
                    if not context.snapshot.allies[peer].alive or context.capabilities[peer].suicide_attack:
                        continue
                    semantic = context.decode(peer, action)
                    if semantic.kind != 'attack' or semantic.target not in remaining:
                        continue
                    estimate = context.fire_estimate(peer, semantic.target)
                    if (estimate.window_shots and estimate.damage_lo is not None
                            and estimate.delay_known and estimate.effective_after_loops is not None
                            and estimate.effective_after_loops < candidate.trigger_loops):
                        remaining[semantic.target] = max(0., remaining[semantic.target] - estimate.damage_lo)
                return remaining
            valid = tuple(a for a in self.guarded_actions(context, actor)
                          if self.viable_step(context, actor, a,
                                              residual_for(outcome(context, actor, a))))
            allowed[actor] = valid
            if actions[actor] in valid and result.opportunity > 0.:
                remaining = residual_for(result)
                reserved = {}
                for e, damage in result.damage_potential.items():
                    if damage is not None:
                        reserved[e] = min(remaining.get(e, 0.), damage * result.opportunity)
                reservations.append((result.trigger_loops, actor, reserved))
        cache[key] = allowed
        return allowed

    def cashout_plan(self, context, actor):
        """After bounded staging, require a real exchange or directed contact.

        This is a read-only decision, not a fresh approach task. Positive
        immediate assignments still win; otherwise a consumed unit may take
        the best remaining damage rather than preserve itself forever.
        """
        cache = getattr(context, '_burst_cashout_plans', None)
        if cache is None:
            cache = context._burst_cashout_plans = {}
        if actor in cache:
            return cache[actor]
        record = self.records.get(actor)
        if not record or not record.get('cashout_reason'):
            cache[actor] = None
            return None
        from .burst import (profile, action_position, outcome, exchange_value, damage,
                            trigger_assignments)
        effect, unit = profile(context), context.snapshot.allies[actor]
        if effect.trigger_reach is None:
            cache[actor] = None
            return None
        trigger_assignments(context)
        remaining = dict(getattr(context, '_burst_trigger_remaining', {}))
        # Region allocation already diversifies targets across our bombers;
        # respect that residual budget without inventing any landed damage.
        self.region(context, actor)
        regional = getattr(context, '_burst_region_budgets', {}).get(actor, {})
        remaining = {e: min(remaining.get(e, enemy.effective_health),
                            regional.get(e, enemy.effective_health))
                     for e, enemy in context.snapshot.enemies.items()}
        legal = tuple(context.legal_actions(actor))
        strikes = []
        for action in legal:
            semantic = context.decode(actor, action)
            if semantic.kind not in ('move', 'attack'):
                continue
            result = outcome(context, actor, action)
            if result.opportunity <= 0. or result.target_id not in context.snapshot.enemies:
                continue
            exchange = exchange_value(context, actor, result.position, remaining,
                                      opportunity=result.opportunity,
                                      enemy_positions=result.enemy_positions)
            if not exchange['known'] or exchange['useful_hp'] <= exchange['counter_burst_hp'] \
                    or exchange['endangered_peers']:
                continue
            rejection = self._low_value_harvest(context, actor, exchange,
                                                remaining, result)
            if rejection is not None:
                if not hasattr(context, '_burst_decline_log'):
                    context._burst_decline_log = {}
                context._burst_decline_log[actor] = rejection
                continue
            value = exchange['useful_hp'] - exchange['counter_burst_hp']
            value /= 1. + result.trigger_loops / max(context.H, 1)
            strikes.append((value, result.trigger == 'active_contact_opportunity',
                            -result.trigger_loops, -action, action, result))
        if strikes:
            *_, action, result = max(strikes, key=lambda item: item[:4])
            plan = {'actions': (action,), 'target_id': result.target_id,
                    'position': tuple(result.position), 'reason': 'cashout_execute_available_burst',
                    'cause': record['cashout_reason']}
            cache[actor] = plan
            return plan
        targets = []
        for enemy_id, enemy in context.snapshot.enemies.items():
            if enemy.is_flying or remaining.get(enemy_id, 0.) <= 0. or damage(context, enemy_id) is None:
                continue
            reach = unit.radius + enemy.radius + effect.trigger_reach
            origin_gap = max(0., math.dist(unit.position, enemy.position) - reach)
            forward = []
            for action in legal:
                semantic = context.decode(actor, action)
                if semantic.kind not in ('move', 'attack') or self._edge_failed(context, actor, action):
                    continue
                if semantic.kind == 'attack' and semantic.target != enemy_id:
                    continue
                result = outcome(context, actor, action)
                if result.opportunity > 0.:
                    exchange = exchange_value(context, actor, result.position, remaining,
                                              opportunity=result.opportunity,
                                              enemy_positions=result.enemy_positions)
                    if exchange['counter_burst_hp'] >= exchange['useful_hp'] and exchange['counter_burst_hp'] > 0. \
                            or exchange['endangered_peers']:
                        continue
                    if self._low_value_harvest(context, actor, exchange, remaining, result):
                        continue
                if not self.viable_step(context, actor, action, remaining):
                    continue
                point = action_position(context, actor, action)
                gap = max(0., math.dist(point, enemy.position) - reach)
                if gap < origin_gap - .05 or semantic.kind == 'attack' and origin_gap <= .05:
                    forward.append((gap, semantic.kind != 'attack', action))
            if not forward:
                continue
            retained = enemy_id == record.get('cashout_target') and _enemy_tag(context, enemy_id) == record.get('cashout_tag')
            best = min(forward)
            targets.append((not retained, origin_gap, best[0], enemy_id, best[2]))
        if targets:
            _, _, _, enemy, action = min(targets)
            plan = {'actions': (action,), 'target_id': enemy,
                    'position': tuple(context.snapshot.enemies[enemy].position),
                    'reason': 'cashout_direct_contact', 'cause': record['cashout_reason']}
        else:
            # Timeout must still end with a CONCRETE task, never a permanent
            # STOP: with no viable target, advance with the fire team and
            # wait for a shared pressure window. This is squad-timing
            # participation, not occupying an ordinary formation slot, and
            # the same window-preservation checks as the route guard apply.
            follow = self._justified_follow(context, actor)
            plan = ({'actions': (follow,), 'target_id': None,
                     'position': tuple(action_position(context, actor, follow)),
                     'reason': 'cashout_advance_with_fire_team',
                     'cause': record['cashout_reason']}
                    if follow is not None else None)
        cache[actor] = plan
        return plan

    def preferred_goal(self, context, actor):
        cashout = self.cashout_plan(context, actor)
        if cashout is not None:
            return cashout['target_id']
        region = self.region(context, actor)
        return region['target_id'] if region is not None else None

    def follow_fire_team(self, context, actor, region=None):
        """Participate in squad timing WITHOUT occupying a formation slot.

        A legal move toward an ordinary ally with ready fire/live contact,
        while retaining a useful entry or interception window. Used when a solo approach is not
        survivable and after cashout with no viable target: the bomber
        advances with the fire team and waits for a shared pressure window
        instead of charging alone or standing still forever. Pure query.
        """
        from .burst import action_position, damage as burst_damage, profile
        record, unit = self.records.get(actor), context.snapshot.allies.get(actor)
        if record is None or unit is None or not unit.alive:
            return None
        anchors = []
        for peer, other in sorted(context.snapshot.allies.items()):
            if peer == actor or not other.alive:
                continue
            cap = context.capabilities[peer]
            if cap.healer or cap.suicide_attack:
                continue
            targets = tuple(e for e in context.attackable_enemies(peer)
                            if not context.snapshot.enemies[e].is_flying
                            and burst_damage(context, e) is not None
                            and (context.contact_status(peer, e) == 'in_range'
                                 or context.fire_estimate(peer, e).window_shots >= 1))
            if targets:
                anchors.append((peer, targets))
        if not anchors:
            return None
        effect, speed = profile(context), context.speed(actor)
        if effect.trigger_reach is None or not speed or speed <= 0.:
            return None
        def contact_gap(point, target):
            enemy = context.snapshot.enemies[target]
            return max(0., math.dist(point, enemy.position) - unit.radius
                       - enemy.radius - effect.trigger_reach)
        # Retain nearby interception only if this actor is a leading bomber
        # and the candidate would actually move it away. The mere presence
        # of an enemy bomber does not prohibit all fire-team movement.
        intercepts = []
        for enemy, cap in context.enemy_capabilities.items():
            if not cap.suicide_attack or enemy not in context.snapshot.enemies:
                continue
            if not any(mate.alive and not mate.is_flying and math.dist(mate.position,
                       context.snapshot.enemies[enemy].position) <= 10.
                       for mate in context.snapshot.allies.values()):
                continue
            leading = sorted((peer for peer, mate in context.snapshot.allies.items()
                              if mate.alive and context.capabilities[peer].suicide_attack),
                             key=lambda peer: (math.dist(context.snapshot.allies[peer].position,
                                           context.snapshot.enemies[enemy].position), peer))[:2]
            if actor in leading:
                intercepts.append(enemy)
        locked_target = region['target_id'] if region is not None else record.get('goal')
        best = None
        for action in sorted(context.legal_actions(actor)):
            semantic = context.decode(actor, action)
            if semantic.kind not in ('move', 'stop') \
                    or self._edge_failed(context, actor, action):
                continue
            if semantic.kind == 'stop':
                continue
            if not self.viable_step(context, actor, action):
                continue
            point = action_position(context, actor, action)
            if any(contact_gap(point, target) > contact_gap(unit.position, target) + .05
                   for target in intercepts):
                continue
            if locked_target in context.snapshot.enemies \
                    and contact_gap(point, locked_target) > contact_gap(unit.position, locked_target) + .05:
                continue
            for anchor, targets in anchors:
                target_position = context.snapshot.allies[anchor].position
                gap = math.dist(point, target_position)
                if gap >= math.dist(unit.position, target_position) - .05:
                    continue
                windows = [(contact_gap(point, target), target) for target in targets
                           if contact_gap(point, target) <= speed * 4 * max(context.H, 1) / 16.
                           and contact_gap(point, target) <= contact_gap(unit.position, target) + .05]
                if not windows:
                    continue
                # Prefer the shortest preserved entry window, not simply
                # the closest friendly body on the opposite side of combat.
                candidate = (min(windows)[0], gap, anchor, action)
                if best is None or candidate < best:
                    best = candidate
        return best[-1] if best is not None else None

    def _justified_follow(self, context, actor, region=None):
        """Follow the fire team ONLY when the move keeps real value.

        A follow move is rejected when it would abandon an interception
        window on an approaching enemy bomber (this bomber is among the two
        nearest suicide-capable interceptors), or when it drags the bomber
        away from its own locked region. Otherwise the anchored teammate
        has ready fire/contact, so following approaches a valuable enemy
        area and can be a compulsory alternative.
        """
        return self.follow_fire_team(context, actor, region)

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
                    and not self.overdue(context, actor) and not record.get('cashout_reason'))

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
            duty = self._execution(context, actor)
            allowed = self.guarded_actions(context, actor)
            kept = tuple(a for a in duty.get('actions', ()) if a in allowed)
            if duty.get('required') and not kept:
                duty = dict(duty, actions=allowed, required=bool(allowed),
                            reason='route_or_residual_guard_replacement')
            elif duty.get('required'):
                duty = dict(duty, actions=kept)
            elif set(allowed) != set(context.legal_actions(actor)):
                duty = dict(duty, actions=allowed, required=True,
                            reason='route_or_residual_guard')
            cache[actor] = duty
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
        cashout = self.cashout_plan(context, actor)
        if cashout is not None:
            result = answer(cashout['actions'], cashout['reason'])
            result.update(target=cashout['target_id'], region=cashout['position'],
                          cashout_cause=cashout['cause'])
            return result
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
        # A supported death INSIDE an effective blast area is a valid trade.
        # The all-enemy upper bound alone never mandates following/retreat.
        from .burst import supported_death_point
        horizon = max(1, int(min(region['arrival_loops'], 4 * context.H)))
        death = supported_death_point(context, actor, region['position'], horizon)
        if death is not None:
            if exchange_value(context, actor, death[0], remaining)['worthwhile']:
                pass    # valid death-burst exchange: continue the approach
            else:
                follow = self._justified_follow(context, actor, region)
                if follow is not None:
                    duty = answer([follow], 'path_death_without_value_follow_fire_team')
                    duty.update(target=None, region=tuple(action_position(context, actor, follow)))
                    return duty
                return answer([], 'path_death_without_value_no_alternative')
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
                    remaining, opportunity=result.opportunity, consumed=assignments,
                    enemy_positions=result.enemy_positions)['worthwhile']:
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
                from .burst import useful_area_value
                record.update(goal=region['target_id'], goal_tag=region['target_tag'],
                              region=region['position'], lock_until_loop=loop + 2 * H,
                              best_distance=math.dist(unit.position, region['position']),
                              last_progress_loop=loop,
                              region_value=useful_area_value(context, region['position']),
                              region_target_position=tuple(
                                  context.snapshot.enemies[region['target_id']].position)
                              if region['target_id'] in context.snapshot.enemies else None)
            if following and region['arrival_loops'] <= 4 * H and record['approach_start_loop'] is None:
                record['approach_start_loop'] = loop
            record['last_execution'] = dict(duty, selected_action=action,
                violated=bool(duty['required'] and action not in duty['actions']))
            followed = duty['required'] and action in duty['actions']
            if followed and duty['reason'].startswith('cashout_'):
                record['cashout_target'] = duty['target']
                record['cashout_tag'] = _enemy_tag(context, duty['target'])
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
            'deadline_expired', 'cashout_reason', 'cashout_target')}
                for actor, record in self.records.items()}
