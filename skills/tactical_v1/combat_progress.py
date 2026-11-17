"""Short observed combat paths and interruptible return-to-fire duties.

Positions and weapon cycles are observations. Predicted return costs are
bounded opportunities, never damage or proof that a future shot lands.
"""
import math

from .battle_intent import contacts, enemy_identity
from .combat_exchange import (attack_approach_position, attack_approach_relief,
                              attack_requires_approach, escape_urgency,
                              imminent_exchange_danger, next_fire_delay)
from .context import distance, GAME_LOOPS_PER_SPEED_SECOND


def exposure_at(context, actor, point):
    risk = context.forecast_risk(actor, point)
    current = set(risk.current_threats)
    return sum((1. if e in current else .5) * (
        max(context.enemy_attack_damage(e, actor), 0.) / 14.
        if context.enemy_attack_damage(e, actor) is not None else 1.)
        for e in risk.reachable_threats) + len(risk.effect_weapon_near)


class CombatProgress:
    def __init__(self, config=None):
        # The existing zero-commitment control also disables these duties;
        # each query is otherwise separately usable for offline ablation.
        self.enabled = (config or {}).get('lifecycle', {}).get('combat_cycle_commit_h', 2.) > 0
        self.next_fire_enabled = (config or {}).get('common_combat', {}).get(
            'next_fire_budget_enabled', True)
        self.joint_enabled = (config or {}).get('common_combat', {}).get(
            'joint_assessment_enabled', True)
        self.pressure_memory_enabled = (config or {}).get('common_combat', {}).get(
            'pressure_memory_enabled', True)
        self.critical_fraction = (config or {}).get('disengage_reenter', {}).get(
            'critical_health_fraction', .35)
        self.history = {}
        self.pending = {}
        self.return_duties = {}
        self.output_duties = {}
        self.closed_departures = {}
        self.output_outcomes = []
        self.fired = set()
        self.loop = None

    def __eq__(self, other):
        return isinstance(other, CombatProgress) and self.__dict__ == other.__dict__

    def observe(self, context):
        loop, units, H = context.snapshot.game_loop, context.snapshot.allies, max(context.H, 1)
        if self.loop is not None and loop <= self.loop:
            return
        self.loop, self.fired = loop, set()
        for actor in set(self.history) | set(self.return_duties) | set(self.output_duties) | set(self.closed_departures):
            unit = units.get(actor)
            record = (self.output_duties.get(actor) or self.closed_departures.get(actor)
                      or self.return_duties.get(actor) or (self.history.get(actor) or [{}])[-1])
            if unit is None or not unit.alive or unit.tag != record.get('tag'):
                self.history.pop(actor, None)
                self.return_duties.pop(actor, None)
                self._finish_output(actor, loop, 'identity_or_death', close=False)
                self.closed_departures.pop(actor, None)
        for actor, unit in units.items():
            cap = context.capabilities[actor]
            if not unit.alive or cap.healer or cap.suicide_attack:
                continue
            pending = self.pending.get(actor)
            elapsed = loop - pending['loop'] if pending else 0
            consecutive = pending and pending['tag'] == unit.tag and 0 < elapsed <= 2 * H
            fired = bool(consecutive and unit.weapon_cooldown_loops >
                         max(0., pending['cooldown'] - elapsed) + .5)
            if fired:
                self.fired.add(actor)
            active = contacts(context, actor)
            self._observe_output(context, actor, pending if consecutive else None,
                                 active, fired)
            duty = self.return_duties.get(actor)
            if duty:
                targets = self._live_targets(context, duty, active if self.joint_enabled else ())
                if fired or loop >= duty['until_loop'] or not targets:
                    self.return_duties.pop(actor, None)
                else:
                    duty['targets'] = targets  # local replacement, same deadline
            # Arrival creates one bounded duty. Staying in contact or issuing
            # another attack never refreshes its deadline.
            if self.enabled and not fired and not duty and consecutive and active \
                    and not pending['contacts'] and (pending['kind'] == 'move'
                        or self.joint_enabled and pending.get('approaching', False)):
                wait = min(3 * H, max(H, math.ceil(unit.weapon_cooldown_loops) + H))
                self.return_duties[actor] = {
                    'tag': unit.tag, 'start_loop': loop, 'until_loop': loop + wait,
                    'targets': {e: enemy_identity(context, e) for e in active}}
            history = self.history.setdefault(actor, [])
            if history and (loop - history[-1]['loop'] > 2 * H or history[-1]['tag'] != unit.tag):
                history.clear()
            history.append({'tag': unit.tag, 'loop': loop, 'position': unit.position,
                            'fired': fired, 'exposure': exposure_at(context, actor, unit.position),
                            'action': pending['action'] if consecutive else None,
                            'targets': {e: enemy_identity(context, e) for e in context.snapshot.enemies}})
            history[:] = [r for r in history[-5:] if loop - r['loop'] <= 4 * H]
        self.pending = {}

    def commit(self, context, actions):
        for actor, action in actions.items():
            unit = context.snapshot.allies.get(actor)
            if unit is None or not unit.alive:
                continue
            self.pending[actor] = {'tag': unit.tag, 'loop': context.snapshot.game_loop,
                'cooldown': unit.weapon_cooldown_loops, 'contacts': contacts(context, actor),
                'kind': context.decode(actor, action).kind, 'action': action,
                'approaching': self.joint_enabled and attack_requires_approach(context, actor, action),
                'position': unit.position, 'exposure': exposure_at(context, actor, unit.position),
                'threats': tuple(context.risk_at(actor, unit.position).threats),
                'targets': {e: enemy_identity(context, e) for e in context.snapshot.enemies
                            if context.effective_attack_reach(actor, e) is not None
                            and context.effective_attack_reach(actor, e) >= 0.},
                'gap': self._gap(context, actor, unit.position)}

    @staticmethod
    def _gap(context, actor, point, targets=None):
        return min((context.projected_attack_opportunity(actor, e, point).center_gap
                    for e in (context.snapshot.enemies if targets is None else targets)
                    if e in context.snapshot.enemies), default=float('inf'))

    def _finish_output(self, actor, loop, reason, *, close=True):
        duty = self.output_duties.pop(actor, None)
        if duty is None:
            return
        if close:
            # A continuous retreat cannot obtain another budget merely because
            # its old task expired or another skill was selected.
            self.closed_departures[actor] = {'tag': duty['tag'], 'targets': duty['targets']}
        self.output_outcomes.append({'actor': actor, 'start_loop': duty['start_loop'],
            'end_loop': loop, 'until_loop': duty['until_loop'], 'cycles': duty['cycles'],
            'reason': reason})
        self.output_outcomes[:] = self.output_outcomes[-24:]

    @staticmethod
    def _live_targets(context, record, active):
        """Preserve identities; only present legal contacts can replace them."""
        targets = {e: tag for e, tag in record.get('targets', {}).items()
                   if e in context.snapshot.enemies and enemy_identity(context, e) == tag}
        targets.update({e: enemy_identity(context, e) for e in active})
        return targets

    def _observe_output(self, context, actor, pending, active, fired):
        if not self.joint_enabled or not self.enabled:
            return
        loop, H = context.snapshot.game_loop, max(context.H, 1)
        unit = context.snapshot.allies[actor]
        matches = lambda record: any(e in context.snapshot.enemies and
            enemy_identity(context, e) == tag for e, tag in record['targets'].items())
        closed = self.closed_departures.get(actor)
        if closed and (fired or not matches(closed)):
            self.closed_departures.pop(actor, None)
        duty = self.output_duties.get(actor)
        if duty is not None:
            # A late observation/shot cannot retroactively fulfill an expired
            # obligation. Neither providers nor target replacements renew it.
            if loop >= duty['until_loop']:
                self._finish_output(actor, loop, 'deadline')
                return
            targets = self._live_targets(context, duty, active)
            if not targets:
                self._finish_output(actor, loop, 'target_lost', close=False)
                return
            duty['targets'] = targets
            if fired:
                duty['cycles'] += 1
                duty.setdefault('first_fire_loop', loop)
                duty['last_progress_loop'] = loop
                if duty['cycles'] >= 2:
                    self._finish_output(actor, loop, 'second_cycle', close=False)
                    self.closed_departures.pop(actor, None)
                    return
            gap = self._gap(context, actor, unit.position, duty['targets'])
            if gap < duty['best_gap'] - .05:
                duty['best_gap'], duty['last_approach_loop'] = gap, loop
            duty['phase'] = ('second_cycle' if duty['cycles'] else
                             'first_cycle' if active else 'return')
            return
        if actor in self.closed_departures:
            return
        arrived = bool(active) and (pending is None or not pending['contacts'])
        if arrived:
            targets = {e: enemy_identity(context, e) for e in active}
            start, reason = loop, 'observed_contact'
        else:
            if pending is None or not (pending['kind'] == 'move'
                    or pending.get('approaching', False)) \
                    or distance(unit.position, pending.get('position', unit.position)) <= .1:
                return
            # An old distant visible enemy must not keep an unrelated front
            # alive. Start from previous contact or actual local threats.
            relevant = pending['contacts'] or pending.get('threats', ())
            targets = {e: tag for e, tag in pending.get('targets', {}).items()
                       if e in relevant and e in context.snapshot.enemies
                       and enemy_identity(context, e) == tag}
            if not targets:
                return
            gap = self._gap(context, actor, unit.position, targets)
            left_contact = bool(pending['contacts']) and not active
            left_pressure = pending.get('exposure', 0.) > 0. and \
                gap > pending.get('gap', float('inf')) + .05
            if not left_contact and not left_pressure:
                return
            start, reason = pending['loop'], 'observed_departure'
        gap = self._gap(context, actor, unit.position, targets)
        speed = context.speed(actor)
        if not math.isfinite(gap) or (gap > .05 and (speed is None or speed <= 0)):
            return
        travel = gap / (speed * H / GAME_LOOPS_PER_SPEED_SECOND) if speed else 0.
        period = getattr(context, 'weapon_period', lambda *_: None)(actor)
        known_period = period is not None and math.isfinite(period) and period > 0
        # Unknown period means an explicit bounded duty budget, not a fabricated
        # fire rate. A verified period only sizes the horizon; no damage credit.
        cycle_budget = 2 * period if known_period else 4 * H
        duration = min(12 * H, max(3 * H,
            math.ceil(2 * H + travel * H + max(unit.weapon_cooldown_loops, 0.) + cycle_budget)))
        self.output_duties[actor] = {'tag': unit.tag, 'targets': targets,
            'start_loop': start, 'until_loop': start + duration,
            'last_progress_loop': loop if fired else start,
            'last_approach_loop': loop, 'best_gap': gap, 'cycles': int(fired),
            'phase': 'second_cycle' if fired else 'first_cycle' if active else 'return',
            'period_known': known_period, 'started_by': reason}
        if fired:
            self.output_duties[actor]['first_fire_loop'] = loop

    def output_duty(self, context, actor):
        """Pure cross-skill obligation, never a command or permanent target lock."""
        duty = self.output_duties.get(actor)
        unit, loop = context.snapshot.allies.get(actor), context.snapshot.game_loop
        if not self.joint_enabled or not self.enabled or duty is None or unit is None \
                or not unit.alive or unit.tag != duty['tag'] or loop >= duty['until_loop']:
            return None
        targets = tuple(self._live_targets(context, duty, contacts(context, actor)))
        if not targets:
            return None
        return {key: duty[key] for key in ('phase', 'start_loop', 'until_loop', 'cycles', 'period_known')} | {
            'targets': targets, 'remaining_loops': duty['until_loop'] - loop,
            'stalled_h': (loop - duty['last_progress_loop']) / max(context.H, 1),
            'overdue': False}

    def _survival_emergency(self, context, actor, memory):
        if self.joint_enabled:
            # combat_emergency also signals newly encountered threats for task
            # interruption. That alone cannot erase the next-shot obligation.
            return escape_urgency(context, actor,
                critical_fraction=self.critical_fraction,
                pressure_memory_enabled=self.pressure_memory_enabled)['immediate']
        return memory.combat_emergency(context, actor) or imminent_exchange_danger(context, actor)

    def _safety_override(self, context, actor, action, memory):
        if self._survival_emergency(context, actor, memory):
            if not self.joint_enabled or not attack_requires_approach(context, actor, action) \
                    or attack_approach_relief(context, actor, action,
                        pressure_memory_enabled=self.pressure_memory_enabled):
                return True
        if not self.joint_enabled:
            unit = context.snapshot.allies[actor]
            point = context.predict_position(actor, action).position
            before = exposure_at(context, actor, unit.position)
            after = exposure_at(context, actor, point)
            if before > 0. and before - after >= max(.1, .15 * before):
                return True
        # Ordinary pressure reduction still earns the risk term's benefit;
        # it does not also make an otherwise late return shot cost nothing.
        rows = self.history.get(actor, ())
        last_action = rows[-1].get('action') if rows else None
        return bool(last_action is not None and memory.movement_blockage(
            context, actor, last_action) > .5)

    def _moves(self, context, actor, action):
        return context.decode(actor, action).kind == 'move' or (
            self.joint_enabled and attack_requires_approach(context, actor, action))

    def _position(self, context, actor, action):
        """Same straight-line approach estimate as joint action positions.

        This is a prospective endpoint only. Observed displacement and fire
        remain the authorities for progress; no path or hit is guaranteed.
        """
        if not self.joint_enabled or not attack_requires_approach(context, actor, action):
            return context.predict_position(actor, action).position
        return attack_approach_position(context, actor, action)

    def cycle_cost(self, context, actor, action, memory):
        if not self.enabled or not memory.battle_state(context).get('engaged'):
            return 0.
        unit = context.snapshot.allies[actor]
        if not unit.alive or context.capabilities[actor].healer \
                or context.capabilities[actor].suicide_attack \
                or context.speed(actor) is None \
                or not self._moves(context, actor, action) \
                or self._safety_override(context, actor, action, memory):
            return 0.
        rows = self.history.get(actor, ())
        if len(rows) < 2 or rows[-1]['loop'] != context.snapshot.game_loop:
            return 0.
        point = self._position(context, actor, action)
        stride = distance(unit.position, point)
        if stride <= .4:
            return 0.
        for index, old in enumerate(rows[:-1]):
            interval = rows[index+1:]
            if any(r['fired'] for r in interval):
                continue  # real shoot-and-move kiting is not a barren loop
            if not any(e in context.snapshot.enemies and enemy_identity(context, e) == tag
                       for e, tag in old['targets'].items()):
                continue  # a wholly different visible encounter is not the same duty
            # A real escape counts if the candidate retains its net safety
            # gain. Briefly leaving danger then immediately returning to the
            # same exposure without firing is precisely a barren cycle.
            drop = max(.1, .15 * old['exposure'])
            if any(old['exposure'] - r['exposure'] >= drop for r in interval) \
                    and old['exposure'] - exposure_at(context, actor, point) >= drop:
                continue
            back = distance(point, old['position'])
            if distance(unit.position, old['position']) > .4 and back <= max(.25, .35 * stride):
                return min(1., max(0., 1. - back / max(stride, .25)))
        return 0.

    def _duty(self, context, actor, memory, action=None):
        duty = self.return_duties.get(actor)
        unit = context.snapshot.allies.get(actor)
        if not self.enabled or not duty or unit is None or not unit.alive \
                or duty['tag'] != unit.tag or context.snapshot.game_loop >= duty['until_loop']:
            return None
        if (self._survival_emergency(context, actor, memory) if action is None else
                self._safety_override(context, actor, action, memory)):
            return None
        return duty

    def return_pending(self, context, actor, memory):
        if self._duty(context, actor, memory) is None:
            return False
        # G4 may still offer a real escape. This does not mutate/renew duty.
        return not any(self._safety_override(context, actor, move, memory)
                       for move in context.move_destinations(actor))

    def return_cost(self, context, actor, action, memory):
        duty = self._duty(context, actor, memory, action)
        if duty is None or not self._moves(context, actor, action):
            return 0.
        if self.joint_enabled and attack_requires_approach(context, actor, action):
            return next_fire_delay(context, actor, action,
                eligible_targets=tuple(self._live_targets(
                    context, duty, contacts(context, actor))), allow_outside=True)
        unit = context.snapshot.allies[actor]
        targets = [e for e, tag in duty['targets'].items() if e in context.snapshot.enemies
                   and enemy_identity(context, e) == tag]
        if not any(e in contacts(context, actor) for e in targets):
            return 0.  # closing distance remains the approach scorer's concern
        speed = context.speed(actor)
        if speed is None:
            return 0.  # do not invent a return time
        point = context.predict_position(actor, action).position
        gap = min((context.projected_attack_opportunity(actor, e, point).center_gap
                   for e in targets), default=float('inf'))
        if not math.isfinite(gap) or gap <= .05:
            return 0.
        travel = gap / (speed * max(context.H, 1) / GAME_LOOPS_PER_SPEED_SECOND)
        # Cooldown provides a real allowance for range-preserving kiting.
        slack = max(0., unit.weapon_cooldown_loops / max(context.H, 1) - 1.)
        return min(1., max(0., travel - slack))

    def next_fire_cost(self, context, actor, action, memory, *, eligible_targets=None):
        """Price the first departure too, without requiring an arrival duty."""
        if not self.next_fire_enabled or not self._moves(context, actor, action) \
                or self._safety_override(context, actor, action, memory):
            return 0.
        return next_fire_delay(context, actor, action,
            eligible_targets=eligible_targets if self.joint_enabled else None,
            allow_outside=self.output_duty(context, actor) is not None,
            use_native_period=self.joint_enabled)

    def summary(self):
        return {'observed_weapon_cycles': sorted(self.fired),
                'return_fire_duties': {a: {'start_loop': r['start_loop'],
                                          'until_loop': r['until_loop']}
                                       for a, r in self.return_duties.items()},
                'output_duties': {a: {k: r[k] for k in ('start_loop', 'until_loop', 'phase', 'cycles')}
                                  for a, r in self.output_duties.items()},
                'output_duty_outcomes': list(self.output_outcomes)}
