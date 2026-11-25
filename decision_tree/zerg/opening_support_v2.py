"""One-macro, pre-contact support controller. Not an output of the fitted CART."""
import copy
import hashlib
from math import dist
from pathlib import Path

from . import rules_v1 as native
from .policy_repair_v2 import endpoint_gap, escort_cost, formation, support_gap

RULE_ID = 'opening_hydra_support_v2'


def rule_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def ranked_moves(payload, base):
    """Geometric proposals only. A proposal is not a certified native branch."""
    s = payload['state']
    h = formation(s)
    if not h or support_gap(h) < 5:
        return []
    if any(r['in_enemy_weapon_range'] is True or r['in_weapon_range'] is True
           for targets in s['relations'].values() for r in targets.values()):
        return []
    isolated = [i for i,p in h.items() if sorted(dist(p,q) for j,q in h.items() if j!=i)[1] >= 5]
    v = [base[f'u{i}'] for i in range(len(base))]
    old_gap = endpoint_gap(s,h,v)
    old_costs = [escort_cost(s,[i],v) for i in isolated]
    out = []
    for i in range(len(v)):
        actor = f'u{i}'
        if v[i] not in (2,3,4,5) or any(int(a)>=6 for a in payload['questions'][actor]['criteria']):
            continue
        for a in s['movement_candidates'][actor]:
            if a not in payload['questions'][actor]['criteria'] or int(a) == v[i]:
                continue
            trial = v.copy(); trial[i] = int(a)
            gap = endpoint_gap(s,h,trial)
            costs = [escort_cost(s,[j],trial) for j in isolated]
            gains = [x-y for x,y in zip(old_costs,costs)]
            if gap <= old_gap and min(gains) >= 0 and max(old_gap-gap,max(gains)) >= .5:
                out.append((sum(costs)/len(costs),gap,actor,int(a)))
    return sorted(out)


def make_entry(manager, base_summary, payload, *, reference=None):
    """Only the opening snapshot; no new live task or persistent controller state."""
    if base_summary['snapshot_id'] != 1:
        return None
    base = reference.actions if reference is not None else base_summary['actions']
    moves = ranked_moves(payload, base)
    if not moves:
        return None
    pending = native._pending(manager, base_summary, payload)
    if pending is None:
        return None
    mod, live_ctx, plans, _ = pending
    cache = manager.coordinator._cache
    if reference is not None:
        if not any(e is reference for e in cache.values()):
            return None
        plans = reference.plans
    ctx, memory, evaluator = copy.deepcopy((live_ctx, manager.memory, manager.evaluator))
    common = native._native(mod,'skills.common')
    support = native._native(mod,'hydra_support')
    contracts = native._native(mod,'contracts')
    ledger_type = native._native(mod,'ledger').Ledger
    coord = native._native(mod,'coordinator').Coordinator(manager.config,evaluator)
    for cost,gap,actor,cmd in moves:
        owned = [p for p in plans if actor in p.controlled_actors]
        if (any(tuple(p.controlled_actors) != (actor,) for p in owned)
                or actor in ctx.formation_orders or actor in getattr(ctx,'formation_reserved_paths',{})
                or any(t >= 6 for t in ctx.legal_actions(actor))):
            continue
        actions = dict(base, **{actor: cmd})
        if any(e.actions == actions for e in cache.values()):
            continue
        safe = dict(support._safe_moves(ctx,memory,actor))
        if cmd not in safe or not support.support_path_clear(ctx,actor,safe[cmd],actions):
            continue
        claim = common.space_claim(ctx,actor,cmd)
        move = common.make_plan(RULE_ID,(actor,), 'backline_support', {actor:cmd}, [],
            contracts.ResourceClaims(controls=(actor,),space=(claim,)),
            contracts.Commitment('opening',max_loops=ctx.H), {}, ctx.snapshot.snapshot_id,
            manager.config['_fingerprint'], rationale='One macro to reduce isolated hydra support distance',
            goal_data=dict(projected_escort=cost, projected_hydra_gap=gap))
        trial = [p for p in plans if actor not in p.controlled_actors] + [move]
        ledger = ledger_type()
        for p in trial:
            if p.proposal_id in ledger.plans or ledger.conflicts(p):
                break
            ledger.reserve(p)
        else:
            entry = coord.evaluate_once(ctx,trial,actions,memory,[1],len(base))
            if entry is not None:
                return entry
    return None

