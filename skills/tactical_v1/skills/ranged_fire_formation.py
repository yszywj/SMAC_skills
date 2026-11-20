"""Local firing lines for Hydra mirror fights. All orders last one observation."""
import math
from ..context import IN_RANGE, STOP, distance
from ..contracts import Commitment, Dependency, EffectClaim, HEURISTIC, ResourceClaims
from ..deployment import safe_step
from ..enemy_motion import predicted_position
from ..hydra_support import (SUPPORT_RADIUS, _can_fire_now, _enemy_covers, _safe_moves,
                            support_path_clear, support_paths_overlap)
from ..joint_combat import action_position
from ..ranged_targets import support_attack
from .common import make_plan, space_claim


class _VolleyForecast:
    """Snapshot-local physical facts, shared by proposal and final branch scoring."""

    def __init__(self, context):
        self.context = context
        self.units = context.snapshot.allies
        self.members = [a for a, u in self.units.items() if u.alive and u.type_name == 'hydralisk']
        self.enemies = {e: u for e, u in context.snapshot.enemies.items() if u.type_name == 'hydralisk'}
        self.predicted = {
            e: predicted_position(context, e, context.snapshot.game_loop + context.H) or u.position
            for e, u in self.enemies.items()}
        self.positions = {}
        self.threats = {}
        self.guns = {}
        self.followup_safe = {}
        self.fronts = None
        self.moves = {}
        self.possible_shooters = {}
        self.crossfire_entries = {}
        self.current_threats = {a: {e for e in self.enemies if _enemy_covers(context, a, e)}
                                for a in self.members}
        # The adapter's loss is the observed delta since the previous decision.
        # A longer decision interval must not be charged in full every H.
        previous = getattr(getattr(context, 'deployment_memory', None), 'last_committed_loop', None)
        elapsed = context.snapshot.game_loop - previous if previous is not None else context.H
        self.loss_per_window = {
            a: (self.units[a].health_lost + self.units[a].shield_lost) * context.H / elapsed
            for a in self.members if elapsed > 0 and self.units[a].health_lost + self.units[a].shield_lost > 0.
            and self.current_threats[a]}

    def positions_for(self, actor, order):
        key = actor, order
        if key not in self.positions:
            ctx = self.context
            first = action_position(ctx, actor, order)[0]
            # The same MOVE can continue only to its real command destination.
            second = (ctx.predict_position(actor, order, 2 * ctx.H).position
                      if ctx.decode(actor, order).kind == 'move' else first)
            if ctx.decode(actor, order).kind == 'move' and ctx.speed(actor) is None:
                first = second = self.units[actor].position
            self.positions[key] = first, second
        return self.positions[key]

    def covering(self, actor, order):
        key = actor, order
        if key not in self.threats:
            self.threats[key] = tuple(self.threats_at(actor, p) for p in self.positions_for(actor, order))
        return self.threats[key]

    def threats_at(self, actor, point):
        unit = self.units[actor]
        targets = set()
        for eid, enemy in self.enemies.items():
            reach = self.context._enemy_weapon_range(eid, unit.is_flying)
            if reach is not None and distance(point, self.predicted[eid]) <= reach + enemy.radius + unit.radius:
                targets.add(eid)
        return targets

    def safe_moves(self, actor):
        if actor not in self.moves:
            memory = getattr(self.context, 'deployment_memory', None)
            self.moves[actor] = tuple(action for action, _ in _safe_moves(self.context, memory, actor))
        return self.moves[actor]

    def crossfire_entry(self, actor, order):
        """Threats caused by this advance, and legal orders with less exposure."""
        key = actor, order
        if key in self.crossfire_entries:
            return self.crossfire_entries[key]
        ctx, unit = self.context, self.units[actor]
        command = ctx.decode(actor, order)
        threats, alternatives = set(), []
        if command.kind == 'move' or (command.kind == 'attack' and
                ctx.contact_status(actor, command.target) != IN_RANGE):
            point = self.positions_for(actor, order)[0]
            after = self.covering(actor, order)[0]
            if distance(unit.position, point) > .05 and len(after) >= 2:
                # Compare at the same future time: enemy advances also cover a stationary gun.
                stationary = self.threats_at(actor, unit.position)
                deeper = any(distance(point, self.predicted[e]) + .05 <
                             distance(unit.position, self.predicted[e]) for e in after & stationary)
                if len(after) >= len(stationary) and (after - stationary or deeper):
                    threats = after - stationary or after
                    candidates = list(self.safe_moves(actor))
                    if STOP in ctx.legal_actions(actor) and safe_step(ctx, actor, unit.position):
                        candidates.append(STOP)
                    alternatives = [cmd for cmd in candidates if len(self.covering(actor, cmd)[0]) < len(after)]
        self.crossfire_entries[key] = threats, alternatives
        return threats, alternatives

    def can_join(self, actor, target):
        key = actor, target
        if key not in self.possible_shooters:
            ctx = self.context
            if _can_fire_now(ctx, actor, target):
                self.possible_shooters[key] = True
            else:
                base = ctx.baseline_actions[actor]
                can_move = not any(_can_fire_now(ctx, actor, e) for e in ctx.attackable_enemies(actor))
                options = (base, *self.safe_moves(actor)) if can_move else (base,)
                self.possible_shooters[key] = any(any(self.firing(actor, cmd, target)) for cmd in options)
        return self.possible_shooters[key]

    def can_finish_approach(self, actor, order, target, reach):
        """Could one more equal, speed-limited move actually open the firing lane?"""
        if target not in self.predicted:
            return False
        ctx = self.context
        origin = self.units[actor].position
        first = self.positions_for(actor, order)[0]
        end = tuple(2. * b - a for a, b in zip(origin, first))
        enemy = self.predicted[target]
        if distance(first, enemy) >= distance(origin, enemy) or distance(end, enemy) > reach:
            return False
        key = actor, order
        if key not in self.followup_safe:
            bounds = ctx.snapshot.raw_state.get('scenario_brief', {}).get('map_size', ())
            radius = self.units[actor].radius
            mid = tuple((a + b) * .5 for a, b in zip(first, end))
            # Only the first order is legal now. The next window must confirm
            # a second command, path and attack mask; never claim damage here.
            self.followup_safe[key] = (
                isinstance(bounds, (tuple, list)) and len(bounds) == 2
                and all(isinstance(size, (int, float)) and radius <= p <= size - radius
                        for p, size in zip(end, bounds))
                and len(self.threats_at(actor, end) - self.current_threats[actor]) <= 1
                and all(safe_step(ctx, actor, p) for p in (first, mid, end)))
        return self.followup_safe[key]

    def firing(self, actor, order, target, allow_followup=True):
        key = actor, order, target, allow_followup
        if key not in self.guns:
            if target not in self.predicted:
                # Non-hydra targets (e.g. lings when no enemy hydra is
                # visible) are outside the formation model: their value is
                # owned by the ordinary target scoring, not by firing lines.
                self.guns[key] = (0., 0.)
                return self.guns[key]
            ctx, unit = self.context, self.units[actor]
            reach = ctx.effective_attack_reach(actor, target)
            command = ctx.decode(actor, order)
            now = command.kind == 'attack' and _can_fire_now(ctx, actor, command.target)
            ready = unit.weapon_cooldown_loops
            if now:
                period = ctx.weapon_period(actor, command.target)
                ready += period if period is not None else float('inf')
            row = []
            loss = self.loss_per_window.get(actor, 0.)
            for h, point in enumerate(self.positions_for(actor, order), start=1):
                # Predict enemy motion for H only. Hold that estimate for the
                # second horizon, rather than extrapolating unobserved acceleration.
                future = (reach is not None and reach >= 0. and ready <= h * ctx.H
                          and distance(point, self.predicted[target]) <= reach)
                if point == unit.position and self.predicted[target] == self.enemies[target].position:
                    future = future and ctx.can_issue_attack(actor, target) and ctx.contact_status(actor, target) == IN_RANGE
                credit = float(future)
                possible_second_move = (h == 2 and allow_followup and not credit and command.kind == 'move'
                                        and ready <= 2 * ctx.H and reach is not None and reach >= 0.
                                        and self.can_finish_approach(actor, order, target, reach))
                if possible_second_move:
                    credit = .5
                if credit and loss and (possible_second_move or self.covering(actor, order)[h - 1]):
                    # Half credit when a repeat of the observed loss would
                    # consume the remaining life by this horizon. It is a
                    # risk forecast, so a threatened shooter keeps some credit.
                    credit *= min(1., max(.25, unit.effective_health / (2. * h * loss)))
                row.append(1. if h == 1 and now and command.target == target else credit)
            self.guns[key] = tuple(row)
        return self.guns[key]

    def value(self, anchor, target, actions):
        threats = self.covering(anchor, actions[anchor])
        guns = [self.firing(a, actions[a], target, allow_followup=a != anchor)
                for a in self.members if a in actions]
        result = 0.
        for h, weight in enumerate((2. / 3., 1. / 3.)):
            friendly = sum(row[h] for row in guns)
            hostile = len(threats[h])
            if is_opening_formation(self.context):
                hostile = max((len(self.covering(a, actions[a])[h]) for a in self.members
                               if a in actions), default=hostile)
            result += weight * (friendly - hostile) / (1. + friendly + hostile)
        if is_opening_formation(self.context):
            # The second gun's distance to legal contact also distinguishes
            # initial steps before either firing lane can be opened.
            gaps = []
            for a in self.members:
                reach = self.context.effective_attack_reach(a, target)
                if a in actions and reach is not None and reach >= 0.:
                    gap = distance(self.positions_for(a, actions[a])[0], self.predicted[target]) - reach
                    gaps.append(max(0., gap))
            if len(gaps) >= 2:
                result -= .25 * min(1., sorted(gaps)[1] / SUPPORT_RADIUS)
        return result


def _forecast(context):
    if not hasattr(context, '_ranged_formation_forecast'):
        context._ranged_formation_forecast = _VolleyForecast(context)
    return context._ranged_formation_forecast


def formation_value(context, anchor, target, actions):
    """Bounded next-volley balance, evaluated using the entire final team's commands."""
    return _forecast(context).value(anchor, target, actions)


def _screened(point, target, lings):
    """A friendly Ling is ahead of the gun and close to its approach line."""
    dx, dy = target[0] - point[0], target[1] - point[1]
    length = math.hypot(dx, dy)
    if length < .01:
        return False
    for ling, end in lings:
        x, y = end[0] - point[0], end[1] - point[1]
        along = (x * dx + y * dy) / length
        if ling.radius < along < length and abs(x * dy - y * dx) / length <= 2.:
            return True
    return False


def _counter_positions(context, memory, model, shots, outside):
    """Offer a short route to a real enemy gun behind a screen or covering a Hydra."""
    routes = {e: {} for e in model.enemies}
    pressured = set().union(*model.current_threats.values())
    for a in model.members:
        unit = model.units[a]
        if unit.health_fraction < .5 or shots[a].intersection(model.enemies):
            continue
        origin = unit.position
        current = model.threats_at(a, origin)
        for cmd, end in _safe_moves(context, memory, a):
            threats = model.threats_at(a, end)
            if len(threats - current) > 1 or any(
                    context.snapshot.enemies[e].type_name == 'zergling'
                    for e in context.risk_at(a, end).threats):
                continue
            if any(support_paths_overlap(unit, end, ling, dest) for ling, dest in outside):
                continue
            for e, enemy in model.predicted.items():
                if distance(end, enemy) >= distance(origin, enemy) - .05:
                    continue
                if e not in pressured and not _screened(end, enemy, outside):
                    continue
                if context.can_issue_attack(a, e) and context.contact_status(a, e) == IN_RANGE:
                    continue
                first, second = model.firing(a, cmd, e)
                if first or second:
                    routes[e].setdefault(a, {})[cmd] = 1 if first else 2
    return routes


def formation_orders(context, memory, actions):
    """One joint order for the Hydra line, from search through first and later volleys."""
    units = context.snapshot.allies
    members = sorted(a for a, u in units.items() if u.alive and u.type_name == 'hydralisk')
    if len(members) < 2 or any(u.type_name not in ('hydralisk', 'zergling') for u in units.values()):
        return {}
    enemies = context.snapshot.enemies
    model = _forecast(context)
    shots = {a: set(context.forced_hydra_targets(a)) or {
        e for e in context.attackable_enemies(a) if _can_fire_now(context, a, e)} for a in members}
    from ..deployment import _fire_eta_h
    outside = [(u, action_position(context, a, actions[a])[0]) for a, u in units.items()
               if a not in members and u.alive and not u.is_flying]
    routes = _counter_positions(context, memory, model, shots, outside)

    experimental = (memory.encounter_config or {}).get(
        'hydra_micro', {}).get('enabled', False)

    def target_rank(e):
        if experimental:
            # [3] A flank hydra nobody can shoot YET still ranks through the
            # members that are one legal move from a firing position against
            # it (can_join covers shooters and one-step fills), plus
            # continuity so similar candidates keep the current focus.
            shooters = sum(e in shots[a] for a in members)
            joiners = (sum(model.can_join(a, e) and e not in shots[a]
                           for a in members) if e in model.enemies else 0)
            continuity = .4 if e == getattr(memory, 'last_hydra_target', None) else 0.
            eta = sorted(_fire_eta_h(context, a, (e,)) for a in members)
            return (-(shooters + .75 * joiners), -continuity,
                    -sum(e in model.current_threats[a] for a in members),
                    enemies[e].effective_health, eta[1], eta[0], e)
        # Non-hydra enemies (e.g. lings during a hydra-less view) rank by
        # current shooters and threat coverage only.
        shooters = sum(e in shots[a] for a in members)
        eta = sorted(_fire_eta_h(context, a, (e,)) for a in members)
        threats = sum(e in model.current_threats[a] for a in members)
        fillers = routes.get(e, {})
        next_guns = sum(1 in moves.values() for moves in fillers.values())
        responding = bool(threats and (shooters or next_guns or len(fillers) >= 2))
        return (-shooters, -responding, -threats, -next_guns, -len(fillers),
                enemies[e].effective_health, eta[0], e)

    ranked = sorted(model.enemies or enemies, key=target_rank)
    target = ranked[0] if ranked else None
    context.hydra_formation_target = target
    if memory is not None:
        memory.last_hydra_target = target
    context._debug_shots = {a: sorted(shots.get(a, ())) for a in members}
    if target is not None:
        goal = enemies[target].position
    else:
        from .cohesive_search import CohesiveSearchSkill
        search = CohesiveSearchSkill(memory.encounter_config)
        first_goal = search._waypoint(context, memory, members, None)
        points = [first_goal, *search._exploration_points(context)]
        # Formation now owns these search orders, so reaching a waypoint need
        # not create a cohesive_search task. Use observed arrivals of two guns.
        def last_arrival(point):
            visits = [max((loop for loop, p in memory.move_history.get(a, ())
                           if distance(p, point) <= search.arrival_tolerance), default=-1)
                      for a in members]
            return sorted(visits)[-2]
        goal = min(points, key=last_arrival)

    # Each member keeps a lane on the near edge of the shared firing band.
    # During search the same ordering produces a line across the direction of travel.
    centre = tuple(sum(units[a].position[i] for a in members) / len(members) for i in (0, 1))
    angle = math.atan2(centre[1] - goal[1], centre[0] - goal[0])
    tangent = (-math.sin(angle), math.cos(angle))
    ordered = sorted(members, key=lambda a: (sum(units[a].position[i] * tangent[i] for i in (0, 1)), a))
    spacing = 2 * max(units[a].radius for a in members) + .15
    options, stationary, facts = {}, {}, {}
    # Previous focus, derived read-only from the last COMMITTED attacks: the
    # finishing reserve and target continuity below must not mutate memory.
    previous = None
    last = getattr(memory, 'last_committed_actions', {}) or {}
    for a in members:
        cmd = last.get(a)
        if cmd is not None:
            semantic = context.decode(a, cmd)
            if semantic.kind == 'attack' and semantic.target in enemies:
                previous = semantic.target
                break
    # A member whose own volley kills the previous focus finishes it; the
    # released members re-rank against the (possibly new) shared target.
    finishers = set()
    if previous is not None and previous in enemies:
        for a in members:
            estimate = context.fire_estimate(a, previous)
            if previous in shots[a] and estimate.damage_lo is not None \
                    and estimate.damage_lo >= enemies[previous].effective_health:
                finishers.add(a)
                break
    for i, a in enumerate(ordered):
        offset = (i - (len(ordered) - 1) / 2.) * spacing
        known_reach = context.effective_attack_reach(a, target) if target else None
        may_close = not target or (known_reach is not None and known_reach >= 0.)
        reach = max(0., (known_reach or 0.) - .1)
        if reach > 0.:
            bearing = angle + offset / reach
            slot = (goal[0] + reach * math.cos(bearing), goal[1] + reach * math.sin(bearing))
        else:
            slot = tuple(goal[k] + offset * tangent[k] for k in (0, 1))
        if shots[a]:
            # [3] Finishing reserve: a member that kills the previous focus
            # in this window keeps that aim; everyone else re-ranks.
            if a in finishers and previous is not None:
                aim = previous
            else:
                preferred = next((e for e in ranked if e in shots[a]), None)
                aim = preferred if preferred is not None else min(shots[a], key=lambda e: (
                    -sum(e in other for other in shots.values()), enemies[e].effective_health, e))
            wait = context.attack_action(a, aim)
        else:
            # Queue the next shot even if the shared focus is still out of reach.
            contact = next((e for e in ranked if context.can_issue_attack(a, e)
                            and context.contact_status(a, e) == IN_RANGE), None)
            wait = context.attack_action(a, contact) if contact is not None else None
        legal = context.legal_actions(a)
        if wait is None or wait not in legal:
            fallback = next(iter(sorted(legal)), None)
            wait = STOP if STOP in legal else (
                actions[a] if actions.get(a) is not None and actions[a] in legal
                else fallback)
        stationary[a] = wait
        start = units[a].position
        current_gap = max(0., distance(start, goal) - reach)
        exposure = len(model.threats_at(a, start))
        commands = [(wait, action_position(context, a, wait)[0])]
        safe_grab = None
        if not shots[a]:
            for cmd, end in _safe_moves(context, memory, a):
                gap = max(0., distance(end, goal) - reach)
                opens_lane = gap < current_gap - .05
                aligns = distance(end, slot) < distance(start, slot) - .05 and gap <= current_gap + .05
                escapes = exposure > 0 and len(model.threats_at(a, end)) < exposure
                detour = not target and all(distance(end, p) > .35 for _, p in memory.move_history.get(a, ())[-4:])
                if ((may_close and (opens_lane or aligns)) or escapes or detour) and not any(
                        support_paths_overlap(units[a], end, u, p) for u, p in outside):
                    commands.append((cmd, end))
        elif ((memory.encounter_config or {}).get('hydra_micro', {}).get('enabled', False) and any(context.snapshot.enemies[e].type_name == 'zergling' for e in shots[a])
                and target and target in enemies):
            # [2] Ling-only shooter: keep the zergling shot AND add the legal
            # moves that open a firing position on the shared HYDRA target
            # next window. The fill competes through line_rank, which counts
            # next-window guns, so a ready hydralisk shot is only given up
            # when the move buys a genuine second firing lane. A cooling gun
            # may reposition while its next shot stays deliverable.
            cooldown_active = units[a].weapon_cooldown_loops > 0.
            covered_now = len(model.threats_at(a, units[a].position)) > 0
            for cmd, end in _safe_moves(context, memory, a):
                # A covered filler must reach the band in ONE step (h1):
                # a two-step chase eats enemy fire while dealing none. The
                # endpoint's new lines must not outnumber friendly guns that
                # can answer them next window (local fire parity).
                credits = model.firing(a, cmd, target)
                opens = credits[0] > 0. if covered_now else any(credits)
                fresh_lines = len(model.threats_at(a, end) - model.threats_at(a, units[a].position))
                parity_ok = (not covered_now) or fresh_lines <= 1 + sum(
                    any(model.firing(peer, mcmd, e)[0]
                        for mcmd, _ in _safe_moves(context, memory, peer)
                        for e in model.threats_at(a, end) - model.threats_at(a, units[a].position))
                    for peer in model.members if peer != a)
                if (opens and parity_ok and not any(
                        support_paths_overlap(units[a], end, u, p) for u, p in outside)):
                    commands.append((cmd, end))
        if (not shots[a] and exposure > 0
                and (memory.encounter_config or {}).get('hydra_micro', {}).get('enabled', False)):
            # [1] Covered but cannot legally designate the attacker yet: a
            # one-step GRAB of a firing position on the covering enemy is
            # compared against stepping out. Grab conditions: the endpoint
            # opens the shot (mask/range/cooldown), at most ONE new firing
            # line, health is safe for a single trade, AND local fire parity —
            # the lines covering the endpoint must not outnumber the friendly
            # guns (self included) that can answer them next window.
            healthy = context.snapshot.allies[a].health_fraction >= .5
            for cmd, end in _safe_moves(context, memory, a):
                end_lines = model.threats_at(a, end)
                fresh = end_lines - model.threats_at(a, start)
                if len(fresh) > 1 or not healthy:
                    continue        # into crossfire or too wounded: keep retreating
                if not any(model.firing(a, cmd, e)[0] > 0. for e in model.current_threats[a]):
                    continue        # no shot actually opens
                friendly_guns = 1 + sum(
                    any(model.firing(peer, mcmd, e)[0]
                        for mcmd, _ in _safe_moves(context, memory, peer)
                        for e in end_lines)
                    for peer in model.members if peer != a)
                if len(end_lines) > friendly_guns:
                    continue        # outnumbered locally: not a fair trade
                safe_grab = cmd
                commands.append((cmd, end))
                break
        origin_threats = len(model.threats_at(a, start))
        counter_moves = routes.get(target, {}).get(a, {}) if a not in finishers else {}
        for cmd in counter_moves:
            if not any(candidate == cmd for candidate, _ in commands):
                commands.append((cmd, model.positions_for(a, cmd)[0]))
        hold = context.decode(a, wait)
        if (not shots[a] and hold.kind == 'attack' and hold.target in model.current_threats[a]
                and len(model.current_threats[a]) == 1 and units[a].health_fraction >= .5
                and safe_step(context, a, start)):
            # In a live duel a cooldown interval does not end the engagement.
            commands = [(wait, start)]
        for cmd, end in commands:
            gap = max(0., distance(end, goal) - reach)
            opening_grab = safe_grab == cmd
            advancing = cmd != wait and (gap < current_gap - .05 or opening_grab)
            facts[a, cmd] = (end, gap, distance(end, slot), len(model.threats_at(a, end)),
                             advancing, bool(opening_grab),
                             any(model.firing(a, cmd, target)) if target else 0.,
                              origin_threats, counter_moves.get(cmd, 0))
        options[a] = [cmd for cmd, _ in commands]
        # [3] Covered with NO counter available (nobody can fire at the
        # threats now, and nobody is one step from doing so): an aligns-only
        # walk wastes this hydra's windows under fire. Keep the ready shot,
        # real escapes and safe grabs; drop pure formation alignment.
        if not shots[a] and exposure > 0 and model.current_threats[a]:
            countered = any(
                _can_fire_now(context, peer, e)
                or any(model.firing(peer, mcmd, e)[0]
                       for mcmd, _ in _safe_moves(context, memory, peer))
                for peer in members if peer != a
                for e in model.current_threats[a])
            if not countered:
                options[a] = [cmd for cmd in options[a]
                              if cmd == stationary[a]
                              or len(model.threats_at(a, facts[a, cmd][0])) < exposure
                               or cmd == safe_grab or cmd in counter_moves]

    # Score whole lines. A lone lead gains nothing by moving one stride
    # ahead of the second gun while a rear gun can close that gap instead.
    stride = max((context.speed(a) or 0.) * context.H / 16. for a in members)
    def line_rank(orders):
        rows = [facts[a, orders[a]] for a in members]
        # row = (end, gap, slot_dist, threats, advancing, safe_grab, gun)
        first, second = sorted(row[1] for row in rows)[:2]
        lone = second > first + max(.1, stride)
        # A first gun must not voluntarily close into enemy fire when no
        # second gun has a lane, even if it is the only unit able to move.
        # Progress elsewhere is irrelevant to granting the lead permission.
        closing_together = sum(row[4] and row[1] <= first + stride for row in rows) >= 2
        unsupported_entry = bool(target and second > stride and not closing_together
            and any(row[4] and row[3] > row[7] and row[1] == first and not row[5]
                    for row in rows))
        exposed = sum(row[3] for a, row in zip(members, rows)
                      if lone and row[1] == first and not shots[a] and not row[5])
        advance = any(row[4] for row in rows)
        outruns = lone and any(row[4] and row[1] == first and not row[5] for row in rows)
        moves = sum(orders[a] != stationary[a] for a in members)
        stalled_search = target is None and moves == 0 and distance(centre, goal) > 2.
        # [2] More members holding a next-window firing position against the
        # shared hydralisk target ranks first among otherwise equal lines.
        guns = sum(row[6] for row in rows)
        # Only count reinforcement from orders in THIS joint branch. An exposed
        # solo gun may close a mask gap in one step for an even exchange. A
        # longer approach, fresh exposure or crossfire requires real support.
        counters = sum(bool(row[8]) or target in shots[a] for a, row in zip(members, rows))
        unsafe = any(row[8] and (row[3] > counters or
                     ((row[8] > 1 or row[3] > row[7]) and counters < 2))
                     for row in rows)
        immediate = sum(row[8] == 1 for row in rows)
        later = sum(row[8] == 2 for row in rows)
        return (unsafe, -immediate, -later, -guns, unsupported_entry, exposed, not advance, stalled_search,
                outruns, second,
                sum(row[1] for row in rows), sum(row[2] for row in rows), moves)

    # Reserve complete synchronized paths, allowing a column to move together.
    # Preserve bounded work for larger compositions. Four Hydras have at most
    # four cardinal moves plus hold each; the last expansion picks the final line.
    branches = [stationary]
    assigned = []
    conflicts = {}
    for a in sorted(members, key=lambda a: (len(options[a]), a)):
        expanded = []
        for branch in branches:
            for cmd in options[a]:
                blocked = False
                for peer in assigned:
                    key = a, cmd, peer, branch[peer]
                    if key not in conflicts:
                        conflicts[key] = support_paths_overlap(units[a], facts[a, cmd][0],
                            units[peer], facts[peer, branch[peer]][0])
                    if conflicts[key]:
                        blocked = True
                        break
                if not blocked:
                    expanded.append({**branch, a: cmd})
        if not expanded:
            return stationary
        branches = sorted(expanded, key=line_rank)[:64]
        assigned.append(a)
    result = min((branches[0], stationary), key=line_rank)
    context._debug_branches = [(dict(b), line_rank(b), line_rank(b) == line_rank(branches[0]))
                               for b in branches[:64]]
    context._debug_stationary = dict(stationary)
    context._debug_best = dict(branches[0])
    context._debug_options = {k: list(v) for k, v in options.items()}
    # Exclusive-contract guard: a member whose forced-hydra set is non-empty
    # must ATTACK one of them this window (the final validator rejects any
    # other command for it, including formation moves). A line that
    # repositions such a member is repaired so the joint branch survives.
    for a in members:
        forced = context.forced_hydra_targets(a)
        if not forced:
            continue
        order = result.get(a)
        if order is not None and context.decode(a, order).kind == 'attack'                 and context.decode(a, order).target in forced:
            continue
        shot = context.attack_action(a, forced[0])
        if shot is not None:
            result[a] = shot
        else:
            result.pop(a, None)   # no legal mandated attack: release to duties
    return result


def is_opening_formation(context):
    memory = getattr(context, 'deployment_memory', None)
    start = getattr(memory, 'first_hydra_sighting', None)
    return (start is not None and not memory.shared_hydra_contact
            and context.snapshot.game_loop - start < 8 * context.H)


def formation_fronts(context):
    """Candidate (anchor, main target) pairs, decided by fire exchange.

    For each outgunned lead, EVERY accessible threatening enemy is a main
    target candidate — the incumbent approach direction, the left side and
    the right side are compared per target by the anchor's own orders, so
    the main target is never fixed to nearest/farthest. Targets with two
    or more rear hydras that can actually reach a firing position rank
    first: at least two legal firing lines on ONE target is the priority.
    Up to three pairs survive (top target of the worst lead, its second
    target, then the next distinct pocket), fixed before any branch.
    """
    model = _forecast(context)
    if model.fronts is not None:
        return model.fronts
    incumbent = getattr(context, 'baseline_actions', None)
    if not incumbent or not model.enemies or len(model.members) < 2:
        return ()
    options = []
    for anchor in model.members:
        peers = [a for a in model.members if a != anchor and
                 distance(model.units[a].position, model.units[anchor].position) <= SUPPORT_RADIUS]
        if not peers:
            continue
        exposed = model.current_threats[anchor].union(*model.covering(anchor, incumbent[anchor]))
        opening = is_opening_formation(context)
        if not exposed and not opening:
            continue
        ranked = []
        for eid in (model.enemies if opening else exposed):
            gap = float('inf')
            feasible = 0
            for peer in (anchor, *peers):
                reach = context.effective_attack_reach(peer, eid)
                if reach is None or reach < 0.:
                    continue
                feasible += int(model.can_join(peer, eid))
                peer_gap = max(0., distance(model.units[peer].position,
                                            model.predicted[eid]) - reach)
                gap = min(gap, peer_gap)
            if gap == float('inf'):
                continue            # no rear shooter can ever reach this line
            value = model.value(anchor, eid, incumbent)
            # Two reachable firing lines on the same target outrank a
            # marginally closer single line; among accessible targets the
            # BEST fire exchange wins (descending value) — the lead itself
            # is still picked as the most outgunned one.
            ranked.append((feasible < 2, -feasible, -value, gap,
                           model.enemies[eid].effective_health, eid))
        if not ranked:
            continue
        ranked.sort()
        # Spend the bounded candidate budget on the most outgunned lead.
        lead_cost = (model.value(anchor, ranked[0][-1], incumbent) if exposed else
                     distance(model.units[anchor].position, model.predicted[ranked[0][-1]]))
        options.append((not bool(exposed), lead_cost, -len(exposed),
                        model.units[anchor].health_fraction, anchor, ranked))
    selected = []
    for _, _, _, _, anchor, ranked in sorted(options):
        if any(distance(model.units[anchor].position, model.units[b].position) <= SUPPORT_RADIUS
               for b, _ in selected):
            continue
        for row in ranked[:2]:
            target = row[-1]
            if selected and row[0] and not is_opening_formation(context):
                continue
            if all(target != other for _, other in selected):
                selected.append((anchor, target))
            if len(selected) == 3:
                break
        if len(selected) == 3:
            break
    model.fronts = tuple(selected)
    return model.fronts


def formation_hold_relief(context, actor):
    """Duty relief for a formation anchor walking into a crossfire.

    True only when this hydra is an anchor of a computed front, its
    incumbent action would enter TWO or more fresh firing lines, and no
    teammate's incumbent order can counter the front target next window —
    exactly the case where a plain entry duty would eliminate the better
    joint hold/sidestep branch. Any rescuer, interceptor or ready gun keeps
    its own precedence; this never relaxes those.
    """
    incumbent = getattr(context, 'baseline_actions', None)
    if not incumbent or actor not in incumbent or not getattr(context, 'ranged_fire_formation_enabled', True):
        return False
    fronts = formation_fronts(context)
    target = next((t for a, t in fronts if a == actor), None)
    if target is None:
        return False
    model = _forecast(context)
    threats = model.covering(actor, incumbent[actor])[0]
    if len(threats) < 2 or len(threats) <= len(model.current_threats[actor]):
        return False
    countered = any(model.firing(peer, incumbent[peer], target)[0]
                    for peer in model.members if peer != actor)
    return not countered


def _crossfire_alternatives(context, actions):
    """Yield an unsupported advance with its less exposed, unobstructed alternatives."""
    if (getattr(context, 'formation_orders', {}) or
            not getattr(context, 'ranged_fire_formation_enabled', False)):
        return
    model = _forecast(context)
    if len(model.enemies) < 2 or len(model.members) < 2:
        return

    for actor in model.members:
        order = actions[actor]
        threats, alternatives = model.crossfire_entry(actor, order)
        if not alternatives:
            continue
        supported = False
        for peer in model.members:
            if peer == actor:
                continue
            cmd = actions[peer]
            command = context.decode(peer, cmd)
            if command.kind == 'attack' and command.target in threats and (
                    context.contact_status(peer, command.target) == IN_RANGE and _can_fire_now(context, peer, command.target)):
                supported = True
                break
            if command.kind != 'move' or cmd not in model.safe_moves(peer):
                continue
            end = model.positions_for(peer, cmd)[0]
            if any(model.firing(actor, order, e)[0] and model.firing(peer, cmd, e)[0] and
                   distance(end, model.predicted[e]) + .05 < distance(model.units[peer].position, model.predicted[e])
                   for e in threats) and support_path_clear(context, peer, end, actions):
                supported = True
                break
        if not supported:
            clear = tuple(alt for alt in alternatives
                          if support_path_clear(context, actor, model.positions_for(actor, alt)[0], actions))
            if clear:
                yield actor, clear


def opening_crossfire(context, actions):
    """Unsupported advances into overlapping Hydra fire in the final joint orders."""
    return tuple(actor for actor, _ in _crossfire_alternatives(context, actions))


def repair_crossfire(context, actions, duties):
    """Seed the same safer routes the final-action check actually accepts."""
    while True:
        for actor, options in _crossfire_alternatives(context, actions):
            duty = duties[actor]
            if duty['required'] and duty['kind'] in ('rescue', 'intercept', 'burst', 'fire'):
                options = [cmd for cmd in options if cmd in duty['actions']]
            if not options:
                continue
            model = _forecast(context)
            peers = [model.positions_for(peer, actions[peer])[0] for peer in model.members if peer != actor]

            def rank(cmd):
                end = model.positions_for(actor, cmd)[0]
                return len(model.covering(actor, cmd)[0]), min(distance(end, p) for p in peers), cmd

            actions[actor] = min(options, key=rank)
            # Every repair strictly reduces this actor's exposure. Recalculate
            # partner support and collision checks against the changed orders.
            break
        else:
            return actions


def opening_entry_action(context, actor, actions):
    """A coordinated legal approach can satisfy a unit's ordinary entry duty."""
    if not getattr(context, 'ranged_fire_formation_enabled', False) or not is_opening_formation(context):
        return False
    model = _forecast(context)
    base = getattr(context, 'baseline_actions', None)
    if actor not in model.members or not base:
        return False
    cmd = actions[actor]
    for anchor, target in formation_fronts(context):
        if distance(model.units[actor].position, model.units[anchor].position) > SUPPORT_RADIUS:
            continue
        if model.value(anchor, target, actions) <= model.value(anchor, target, base):
            continue
        if actor == anchor and formation_hold_relief(context, actor):
            if len(model.covering(actor, cmd)[0]) < len(model.covering(actor, base[actor])[0]):
                return True
        if context.decode(actor, cmd).kind == 'move' and cmd in model.safe_moves(actor):
            first = model.positions_for(actor, cmd)[0]
            if distance(first, model.predicted[target]) < distance(model.units[actor].position, model.predicted[target]):
                return True
    return False


class RangedFireFormationSkill:
    name = 'ranged_fire_formation'

    def __init__(self, config):
        self.config = config

    def _opening_arcs(self, context, model, anchor, target, shots, base):
        """Assign the near-side arc's slots together, then issue only the first legal step."""
        members = [a for a in model.members if distance(model.units[a].position,
                   model.units[anchor].position) <= SUPPORT_RADIUS
                   and (context.effective_attack_reach(a, target) or 0.) > 0.]
        if len(members) < 2:
            return []
        seed = getattr(context, 'formation_seed_actions', base)
        outside_paths = [(unit, action_position(context, b, seed[b])[0])
                         for b, unit in model.units.items()
                         if unit.alive and not unit.is_flying and b not in members and b in seed]
        enemy = model.predicted[target]
        center = tuple(sum(model.units[a].position[i] for a in members) / len(members) for i in (0, 1))
        bearing = math.atan2(center[1] - enemy[1], center[0] - enemy[0])
        choices = {}
        for turn in (0., -.65, .65):
            angle = bearing + turn
            tangent = (-math.sin(angle), math.cos(angle))
            ordered = sorted(members, key=lambda a: (
                sum((model.units[a].position[i] - enemy[i]) * tangent[i] for i in (0, 1)), a))
            spacing = 2 * max(model.units[a].radius for a in members) + .12
            legal = {}
            slots = {}
            for index, a in enumerate(ordered):
                reach = context.effective_attack_reach(a, target) - .15
                offset = (index - (len(ordered) - 1) / 2.) * spacing / max(reach, .1)
                slots[a] = (enemy[0] + reach * math.cos(angle + offset),
                            enemy[1] + reach * math.sin(angle + offset))
                if shots[a]:
                    shot = support_attack(context, a, target, anchor, base)
                    legal[a] = [shot if shot is not None else
                                (base[a] if base[a] in shots[a] else shots[a][0])]
                else:
                    legal[a] = [cmd for cmd in model.safe_moves(a)
                                if not any(support_paths_overlap(model.units[a],
                                       model.positions_for(a, cmd)[0], unit, end)
                                           for unit, end in outside_paths)]
                    if STOP in context.legal_actions(a) and safe_step(context, a, model.units[a].position):
                        legal[a].append(STOP)
            # Place fixed firing orders first and reject paths intersecting
            # existing reservations. The complete coordinator rechecks outsiders.
            orders = {}
            for a in sorted(ordered, key=lambda a: (not shots[a], distance(model.units[a].position, enemy), a)):
                valid = []
                for cmd in legal[a]:
                    end = model.positions_for(a, cmd)[0]
                    if any(support_paths_overlap(model.units[a], end, model.units[b],
                               model.positions_for(b, other)[0]) for b, other in orders.items()):
                        continue
                    valid.append((distance(end, slots[a]), cmd != base[a], cmd))
                if not valid:
                    break
                orders[a] = min(valid)[2]
            if len(orders) != len(members) or all(orders[a] == base[a] for a in members):
                continue
            if all(context.decode(a, cmd).kind == 'stop' for a, cmd in orders.items()):
                continue
            key = tuple(sorted(orders.items()))
            if key not in choices:
                plan = self._plan(context, anchor, target, orders)
                plan.goal_data['first_engagement'] = True
                choices[key] = plan
        return list(choices.values())

    def propose(self, context, memory, telemetry, ledger=None):
        if not self.config.get('common_combat', {}).get('joint_assessment_enabled', True):
            return []
        fronts = formation_fronts(context)
        if not fronts:
            return []
        telemetry.count(self.name, 'applicable', len(fronts))
        base = context.baseline_actions
        model = _forecast(context)
        shots = {a: [context.attack_action(a, e) for e in context.attackable_enemies(a)
                     if _can_fire_now(context, a, e)] for a in model.members}
        moves = {a: [m for m, _ in _safe_moves(context, memory, a)] for a in model.members if not shots[a]}
        plans = []
        for anchor, target in fronts:
            if is_opening_formation(context):
                plans.extend(self._opening_arcs(context, model, anchor, target, shots, base))
            incumbent = model.value(anchor, target, base)
            alternatives = {}
            if shots[anchor]:
                anchor_orders = [base[anchor] if base[anchor] in shots[anchor] else shots[anchor][0]]
            else:
                anchor_orders = list(moves[anchor])
                if context.decode(anchor, base[anchor]).kind != 'move':
                    anchor_orders.append(base[anchor])
                elif STOP in context.legal_actions(anchor) and safe_step(context, anchor, model.units[anchor].position):
                    anchor_orders.append(STOP)
            for order in dict.fromkeys(anchor_orders):
                orders = {anchor: order}
                # Add at most two rear shooters. Each extra member must improve
                # the candidate with its actual order; original assignments
                # outside this local set are supplied by the complete branch.
                for _ in range(2):
                    best = None
                    for peer in model.members:
                        if peer in orders or distance(
                                model.units[anchor].position, model.units[peer].position) > SUPPORT_RADIUS:
                            continue
                        if shots[peer]:
                            fire = support_attack(context, peer, target, anchor, base)
                            options = [] if fire is None else [fire]
                        else:
                            options = moves[peer]
                        for fill in options:
                            if not any(model.firing(peer, fill, target)):
                                continue
                            end = model.positions_for(peer, fill)[0]
                            if any(support_paths_overlap(model.units[peer], end, model.units[a],
                                   model.positions_for(a, cmd)[0]) for a, cmd in orders.items()):
                                continue
                            # Do not solve the lead's problem by sending a
                            # rear shooter into an even larger crossfire.
                            exposure = len(model.covering(peer, fill)[0])
                            if exposure > len(model.covering(peer, base[peer])[0]) + 1:
                                continue
                            trial = {**base, **orders, peer: fill}
                            value = model.value(anchor, target, trial)
                            rank = (value, -exposure, -distance(end, model.predicted[target]))
                            if best is None or rank > best[0]:
                                best = rank, peer, fill
                    if best is None:
                        break
                    rank, peer, fill = best
                    if len(orders) > 1 and rank[0] <= model.value(anchor, target, {**base, **orders}):
                        break
                    orders[peer] = fill
                    if rank[0] > incumbent and any(base[a] != cmd for a, cmd in orders.items()):
                        if order == STOP and base[anchor] != STOP and not any(
                                model.firing(a, cmd, target) > model.firing(a, base[a], target)
                                for a, cmd in orders.items() if a != anchor):
                            continue   # waiting must buy actual supporting fire
                        new_threats = model.covering(anchor, order)[0] - model.current_threats[anchor]
                        # Stepping into TWO or more fresh firing lines needs a
                        # teammate that can genuinely counter the target next
                        # window — otherwise sidestep, switch side or target,
                        # or hold; when every route is already engaged the
                        # incumbent stays the better executable exchange.
                        counters = any(model.firing(peer, cmd, target)[0]
                                       for peer, cmd in orders.items() if peer != anchor)
                        if len(new_threats) > 1 and (rank[0] < 0. or not counters):
                            continue
                        key = tuple(sorted(orders.items()))
                        alternatives[key] = rank[0]
            for key in sorted(alternatives, key=lambda k: (-alternatives[k], len(k), k))[:3]:
                plans.append(self._plan(context, anchor, target, dict(key)))
        telemetry.count(self.name, 'proposed', len(plans))
        return plans

    def _plan(self, context, anchor, target, orders):
        effects, fire = [], []
        for actor, order in orders.items():
            cmd = context.decode(actor, order)
            if cmd.kind == 'attack':
                estimate = context.fire_estimate(actor, cmd.target)
                if estimate.window_shots >= 1:
                    effects.append(EffectClaim('damage', cmd.target, estimate.damage_est, context.H,
                                               estimate.evidence, 'shared_damage_model_window_shot',
                                               lower=estimate.damage_lo, upper=estimate.damage_hi))
                    if estimate.damage_lo is not None:
                        fire.append((cmd.target, estimate.damage_lo))
                else:
                    effects.append(EffectClaim('coverage', cmd.target, None, context.H, HEURISTIC,
                                               'approach_not_confirmed_fire'))
            elif cmd.kind == 'move':
                effects.append(EffectClaim('position', actor, None, context.H, HEURISTIC,
                                           'projected_firing_position'))
        members = [anchor] + sorted(a for a in orders if a != anchor)
        return make_plan(self.name, members, target, orders, effects,
                         ResourceClaims(controls=tuple(members), fire=tuple(fire),
                             space=tuple(space_claim(context, a, cmd) for a, cmd in orders.items()
                                         if context.decode(a, cmd).kind == 'move')),
                         Commitment('covering', 0, context.H), {}, context.snapshot.snapshot_id,
                         self.config.get('_fingerprint', ''), atomic=True, goal_kind='enemy',
                         dependencies=(Dependency('member', tuple(members)), Dependency('target_valid', (target,))),
                         goal_data={'objective': {'kind': self.name, 'target': target},
                                    'support_pair': members},
                         rationale='form a shared firing line against a threat to ' + anchor)
