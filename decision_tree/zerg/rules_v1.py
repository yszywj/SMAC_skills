"""Experimental second proposer for a single pending native Zerg decision.

This module neither chooses the winning arm nor commits, registers or renders
it. The caller owns prefix verification, live engine masks, commit and receipts.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import math
from pathlib import Path

RULE_ID = 'zerg_dual_lane_clearance_v1'
_RULE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_MOVE = frozenset((2, 3, 4, 5))
_EPS = 1e-8  # world units; strict improvements, not numerical ties

# Only ordinary registered skills; local_engagement and reserve_dispatch need
# branch-private permissions. An unknown skill is not an ordinary baseline row.
_SKILLS = frozenset({"focus_fire", "coordinated_engage", "combat_reposition",
                     "disengage_reenter", "ranged_fire_formation", "burst_engage",
                     "regroup", "local_formation", "cohesive_search", "precombat_rally",
                     "combat_rejoin", "splash_fire", "heal_support"})


def rule_hash() -> str:
    """Bind exact rule code and parameters. Native source/config is bound separately."""
    return _RULE_SHA256


def _native(module, name):
    return importlib.import_module(f"{module.__package__}.{name}")


def may_trigger(payload, base_actions) -> bool:
    """Necessary menu-only filter. No manager, history, returns or partition inputs.

    Callers are responsible for admitting only training episodes to their scan.
    The menu is a list/dict of action IDs, NOT a binary availability vector.
    """
    try:
        state = payload['state']
        allies = state['allies']
        actions = ({f'u{i}': a for i, a in enumerate(base_actions)}
                   if isinstance(base_actions, (list, tuple)) else base_actions)
        if not isinstance(actions, dict) or actions.keys() != allies.keys():
            return False
        movers, fighting = set(), False
        for actor, unit in allies.items():
            cmd = actions[actor]
            if type(cmd) is not int:
                return False
            if unit['alive'] is False and cmd == 0:
                continue
            if unit['alive'] is not True or not unit['health'] > 0:
                return False
            menu = state.get('legal_actions', {}).get(actor)
            if menu is None:
                menu = payload['questions'][actor]['criteria']
            legal = {int(k) for k in menu}
            if cmd not in legal:
                return False
            kind = str(unit['type']).lower()
            if kind not in ('baneling', 'zergling', 'hydralisk'):
                return False
            if cmd in _MOVE and not unit['is_flying'] and max(legal) < 6:
                movers.add(kind)
            if cmd >= 6:
                e = f'e{cmd - 6}'
                enemy = state['visible_enemies'].get(e, {})
                if (enemy.get('alive') is True and enemy.get('health', 0) > 0
                        and state['relations'][actor][e]['smac_attack_available'] is True):
                    fighting = True
        return fighting and {'baneling', 'zergling'} <= movers
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return False


def _pending(manager, summary, payload):
    pending, cache = manager._pending_commit, manager._cache
    if not pending or not cache or manager.memory.committed:
        return None
    snap, plans, vector, _, ctx = pending
    actions = {f"u{i}": a for i, a in enumerate(vector)}
    mod = importlib.import_module(type(manager).__module__)
    old = manager.coordinator._cache
    if (snap is not ctx.snapshot or snap is not cache[4] or plans is not cache[3]
            or snap.n_allies != len(vector) or set(snap.allies) != set(actions)
            or cache[0] != mod._digest(payload, len(vector), manager.config["_fingerprint"])
            or summary.get("snapshot_id") != snap.snapshot_id
            or summary.get("snapshot_game_loop") != snap.game_loop
            or cache[1] != vector or summary.get("actions") != actions
            or summary.get("local_coord") != cache[2]["local_coord"]
            or summary["local_coord"]["selected"] or not old
            or len(old) != cache[2]["cache_entries"]
            or summary.get("chosen_plans") != cache[2]["chosen_plans"]
            or [p.proposal_id for p in plans] != [r['id'] for r in summary['chosen_plans']]):
        return None
    return mod, ctx, plans, actions


def _assemble(mod, config, evaluator, ctx, memory, base_plans, base, replacements,
              retained, skills):
    """Mirror native local assembly: fixed A backdrop + whole plan substitutions.

    ``ctx``/``memory``/``evaluator`` are private. No arbitrary vector is an
    input to scoring: every changed order must be owned by a replacement plan.
    ``retained`` is the read-only, UNTRIMMED old coordinator cache.
    """
    if not replacements:
        return None
    ledger = _native(mod, 'ledger').Ledger()
    plan_type = _native(mod, 'contracts').CandidatePlan
    supported = _SKILLS.intersection(k for k, skill in skills.items() if skill.enabled)
    snap = ctx.snapshot
    touched = {a for p in replacements for a in p.controlled_actors}
    if not touched or touched & ctx.formation_orders.keys():
        return None
    plans = []
    for p in [*base_plans, *replacements]:
        if (not isinstance(p, plan_type) or p.primary_skill not in supported
                or p.specialization_provider is not None and p.specialization_provider not in supported
                or p.snapshot_id != snap.snapshot_id or p.config_fingerprint != config['_fingerprint']
                or not p.controlled_actors or set(p.controlled_actors) != set(p.current_actions)
                or any(a not in snap.allies or not snap.allies[a].alive for a in p.controlled_actors)):
            return None
    for p in base_plans:
        actors = set(p.controlled_actors)
        if any(p.current_actions[a] != base[a] for a in actors):
            return None
        if actors & touched:
            if not actors <= touched:  # never silently orphan part of an atomic A task
                return None
        else:
            plans.append(p)
    plans.extend(replacements)
    actions = dict(base)
    for p in plans:
        if p.proposal_id in ledger.plans or ledger.conflicts(p):
            return None
        ledger.reserve(p)
        actions.update(p.current_actions)
    if (actions == base or any(entry.actions == actions for entry in retained.values())
            or any(type(a) is not int or a not in snap.legal[u] for u, a in actions.items())):
        return None
    # Complete ALL free actors together under genuine locked contracts and
    # formation orders. If this would change any outsider (including a duty
    # seed action), abstain instead of patching that actor back into the map.
    baseline = _native(mod, 'baseline')
    try:
        completed = baseline.complete_plan(ctx, memory, skills, locked_plans=plans)
    except baseline.LockedPlanConflict:
        return None
    if completed != actions:
        return None
    # evaluate_once checks fixed formation orders / paths, forced targets,
    # native action & plan validation, effects and full execution duties. A
    # fresh coordinator guarantees a cache miss and exactly one private score.
    coord = _native(mod, 'coordinator').Coordinator(config, evaluator)
    return coord.evaluate_once(ctx, plans, completed, memory, [1], snap.n_allies)


def _overlap(u, end, v, other_end):
    """Maximum synchronous overlap depth of two linearly moving ground discs."""
    x, y = u.position
    a, b = v.position
    dx, dy = (end[0] - x) - (other_end[0] - a), (end[1] - y) - (other_end[1] - b)
    x, y = x - a, y - b
    speed2 = dx * dx + dy * dy
    t = min(1., max(0., -(x * dx + y * dy) / speed2)) if speed2 else 0.
    return max(0., u.radius + v.radius - math.hypot(x + t * dx, y + t * dy))


def _lanes(mod, ctx, base_plans, base, retained):
    """Rank genuine plan pairs by geometry, not native score or future return."""
    snap = ctx.snapshot
    ground = {i: u for i, u in sorted(snap.allies.items()) if u.alive and not u.is_flying}
    if any(not all(math.isfinite(x) for x in (*u.position, u.radius)) or u.radius <= 0
           for u in ground.values()):
        return []
    pos = _native(mod, 'joint_combat').action_position
    endpoints = {i: pos(ctx, i, base[i])[0] for i in ground}
    if any(not all(map(math.isfinite, pt)) for pt in endpoints.values()):
        return []
    attacked = {ctx.decode(i, cmd).target for i, cmd in base.items() if cmd >= 6}
    roles = {'baneling': 'burst_engage', 'zergling': 'coordinated_engage'}
    owners = {}
    for p in base_plans:
        if len(p.controlled_actors) != 1:
            continue
        i, = p.controlled_actors
        u = ground.get(i)
        if (u and p.primary_skill == roles.get(u.type_name) and base[i] in _MOVE
                and snap.legal[i] and max(snap.legal[i]) < 6 and p.goal_kind == 'enemy'
                and p.goal in attacked and p.commitment.phase == 'approaching'
                and p.goal in snap.enemies and not snap.enemies[p.goal].is_flying
                and i not in ctx.formation_orders):
            owners[i] = p

    def slots(plan):
        return tuple(sorted((s.actor, s.slot_id) for s in plan.claims.space if s.slot_id))

    # No re-propose on the live or copied manager: reuse actual singletons
    # found ANYWHERE in scored native branches (not just the top-k summary).
    by_type = {'baneling': [], 'zergling': []}
    seen = set()
    for entry in retained.values():
        if any(p.primary_skill not in _SKILLS for p in entry.plans):
            continue  # branch-private permissions cannot travel with the move
        for p in entry.plans:
            if len(p.controlled_actors) != 1:
                continue
            i, = p.controlled_actors
            old = owners.get(i)
            if old is None or p.proposal_id in seen:
                continue
            cmd = p.current_actions.get(i)
            if (p.primary_skill != old.primary_skill or p.goal != old.goal
                    or p.goal_kind != old.goal_kind or p.commitment.phase != old.commitment.phase
                    or cmd not in _MOVE or cmd == base[i] or cmd != entry.actions[i]
                    or cmd not in snap.legal[i] or slots(p) != slots(old)):
                continue
            seen.add(p.proposal_id)
            u, target = ground[i], snap.enemies[p.goal].position
            end = pos(ctx, i, cmd)[0]
            if (not all(map(math.isfinite, (*end, *target))) or
                    math.dist(end, target) + _EPS >= math.dist(u.position, target)):
                continue
            if u.type_name == 'baneling':
                # Swept path vs observed teammate body is a clearance proxy,
                # not a claim the teammate will remain stationary.
                def depth(pt):
                    return max((_overlap(u, pt, v, v.position)
                                for j, v in ground.items() if j != i), default=0.)
                blocked, clear = depth(endpoints[i]), depth(end)
                if blocked <= _EPS or clear > _EPS:
                    continue
            by_type[u.type_name].append((i, p, end))

    ranked = []
    for i, bomber, end in by_type['baneling']:
        for j, ling, other_end in by_type['zergling']:
            if bomber.goal != ling.goal:
                continue
            # Hold all other paths fixed, including the proposed Baneling
            # lane. Only the Zergling's incident edges can change here.
            peers = [(v, end if k == i else endpoints[k]) for k, v in ground.items() if k != j]
            before = sum(_overlap(ground[j], endpoints[j], v, pt) for v, pt in peers)
            after = sum(_overlap(ground[j], other_end, v, pt) for v, pt in peers)
            if before <= _EPS or after > _EPS:
                continue
            # Require C to improve team overlap relative to BOTH A and
            # the isolated bomber move. Only edges touching i or j differ.
            bomber_delta = sum(_overlap(ground[i], end, v, endpoints[k]) -
                               _overlap(ground[i], endpoints[i], v, endpoints[k])
                               for k, v in ground.items() if k != i)
            improvement = before - after - bomber_delta
            if improvement > _EPS:
                ranked.append(((-round(improvement, 8), -round(before - after, 8),
                                i, j, bomber.proposal_id, ling.proposal_id), (bomber, ling)))
    return sorted(ranked, key=lambda row: row[0])


def make_entry(manager, base_summary, payload):
    """Return one validated native _CompletePlan or abstain; never modify A."""
    if not isinstance(base_summary, dict) or not may_trigger(payload, base_summary.get('actions')):
        return None
    pending = _pending(manager, base_summary, payload)
    if pending is None:
        return None
    mod, original_ctx, original_plans, base = pending
    retained = manager.coordinator._cache
    # Scoring AND geometry lazily populate context/memory caches. Copy them
    # as one object graph to preserve internal references. Neither evaluator
    # nor proposal collector nor coordinator telemetry is used on the parent.
    ctx, memory, plans, evaluator, config = copy.deepcopy(
        (original_ctx, manager.memory, original_plans, manager.evaluator, manager.config))
    # Fresh native skill instances contain only the actual configuration;
    # never call a live shadow-wrapped propose method, even on a deepcopy.
    skills = _native(mod, 'registry').build_skills(config)
    if {k for k,s in skills.items() if s.enabled} != {k for k,s in manager.skills.items() if s.enabled}:
        return None
    for _, pair in _lanes(mod, ctx, plans, base, retained):
        entry = _assemble(mod, config, evaluator, ctx, memory, plans, base,
                          copy.deepcopy(pair), retained, skills)
        if entry is not None:
            return entry
    return None
