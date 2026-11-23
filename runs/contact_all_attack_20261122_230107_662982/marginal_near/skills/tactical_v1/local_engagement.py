"""Local engagement and reserve dispatch for the frozen 10-unit SMAC policy.

Only visible, tagged SC2 units participate. A pack is a *direct* contest,
not a connected component of nearby units. Reach after a move is a forecast,
never counted as damage. Offers are read-only; TacticalMemory.commit owns
all dispatch instances. commit_local remembers observations, not control.
"""
from collections import Counter, defaultdict
from dataclasses import replace
from math import ceil, dist

from .action_duties import rescue_actions
from .context import GAME_LOOPS_PER_SPEED_SECOND
from .contracts import CandidatePlan, Commitment, Dependency, EffectClaim, HEURISTIC, ResourceClaims
from .hydra_support import support_path_clear
from .skills.common import make_plan, space_claim


LOCAL = 'local_engagement'
RESERVE = 'reserve_dispatch'
from .local_pack_options import home_line, near_bid

REQUEST_H = 2
MAX_OFFERS = 16


def _tag(raw):
    value = str(raw.get('tag', ''))
    return value if value.isascii() and value.isdecimal() and 0 < int(value) < 2**64 else None


def _joint(snap, actions):
    return actions if isinstance(actions, dict) else {a: actions[u.index] for a, u in snap.allies.items()}


def _previous(ctx, memory):
    s = ctx.snapshot
    state = getattr(memory, '_local_observed', {})
    if (state.get('episode') == s.episode and state.get('loop') < s.game_loop
            and state.get('snapshot_id') < s.snapshot_id):
        return state
    return {}


def commit_local(context, memory, audit):
    """Called after the runner's real commit. Never create a claim from audit."""
    s = context.snapshot
    if not audit.get('in_scope') or not memory.committed or memory.last_committed_loop != s.game_loop:
        return
    # Keep clock/anchor facts, and only the actually committed claim IDs for
    # release telemetry (native retirement may precede the next proposal).
    memory._local_observed = dict(episode=s.episode, loop=s.game_loop,snapshot_id=s.snapshot_id,
        anchors=[(p['anchor_enemy_tag'], p['anchor_ally_tag']) for p in audit['packs']],
        clocks={r['key']: (r['created_loop'], r['expires_loop']) for r in audit['requests']},
        cooldowns=dict(audit['request_cooldowns']),
        prior_claims={i.actors[0]:{k:i.extra[k] for k in ('request_id','expires_loop','actor_tag')}
                      for i in memory.instances.values() if i.skill==RESERVE})


class _Window:
    def __init__(self, ctx, memory, reference):
        self.ctx, self.memory, self.ref = ctx, memory, reference
        s = ctx.snapshot
        self.allies = {a: t for a, u in s.allies.items() if u.alive and
                       (t := _tag(s.raw_state['allies'].get(a, {})))}
        self.enemies = {e: t for e, u in s.enemies.items() if u.effective_health > 0 and
                        (t := _tag(s.raw_state['visible_enemies'].get(e, {})))}
        self.rays, self.threats, self.est = {}, {}, {}
        self.steps, self.moves, self.shot, self.locked = {}, {}, {}, set()
        self.ref_lower = defaultdict(float)
        for a in self.allies:
            u = s.allies[a]
            self.rays[a] = {e for e in self.enemies if ctx.can_issue_attack(a, e)
                            and ctx.contact_status(a, e) == 'in_range'}
            self.threats[a] = {e for e, r in s.relations.get(a, {}).items() if
                               e in self.enemies and r.in_enemy_weapon_range is True}
            for e in self.rays[a]:
                self.est[a, e] = ctx.fire_estimate(a, e)
            sem = ctx.decode(a, reference[a])
            old = sem.target if sem.kind == 'attack' and sem.target in self.rays[a] else None
            self.shot[a] = old
            if old:
                self.ref_lower[old] += self.est[a, old].damage_lo or 0.
            end = ctx.predict_position(a, reference[a]).position
            retreat = sem.kind == 'move' and self.threats[a] - set(ctx.risk_at(a,end).threats)
            if retreat or reference[a] in rescue_actions(ctx, a, memory) or (self.threats[a] and
                    (u.health_fraction <= .4 or u.health_lost >= u.health)):
                self.locked.add(a)
            self.moves[a] = []
            if a not in self.locked and ctx.speed(a) is not None:
                for move in sorted(ctx.move_destinations(a)):
                    if memory.movement_blockage(ctx, a, move) >= .5:
                        continue
                    end = ctx.predict_position(a, move).position
                    if dist(u.position, end) <= .05:
                        continue
                    mid = tuple((x+y)/2 for x,y in zip(u.position, end))
                    # Do not open a path through an unrelated firing line.
                    dangers = set()
                    for pt in (mid, end):
                        dangers.update(ctx.risk_at(a, pt).threats)
                    self.moves[a].append((move, end, dangers - self.threats[a]))
            self.steps[a] = {}
            for e in self.enemies.keys() - self.rays[a]:
                route = self.route(a, e, 1)
                if route is not None:
                    self.steps[a][e] = route

    def route(self, a, e, horizon):
        """Test the legal first step; subsequent straight-line ETA is an estimate."""
        ctx, s = self.ctx, self.ctx.snapshot
        reach, speed = ctx.effective_attack_reach(a, e), ctx.speed(a)
        if reach is None or reach <= 0 or not speed:
            return None
        start, enemy = s.allies[a].position, s.enemies[e].position
        gap = dist(start, enemy) - reach
        if gap <= .01:
            return None  # menu closed for a reason other than predicted distance
        options = []
        for cmd, end, extra in self.moves[a]:
            remain = max(0., dist(end, enemy) - reach)
            if extra - {e} or remain >= gap - .05:
                continue
            eta = max(ctx.H + ceil(remain * GAME_LOOPS_PER_SPEED_SECOND / speed),
                      ceil(s.allies[a].weapon_cooldown_loops))
            if eta <= horizon * ctx.H and support_path_clear(ctx, a, end, self.ref):
                options.append((eta, remain, cmd))
        return min(options) if options else None

    def can_redirect(self, a, e, pack):
        """Keep useful fire: only actual attacks compete for an existing shot."""
        ctx, s = self.ctx, self.ctx.snapshot
        old = self.shot[a]
        if a in self.locked or (ctx.forced_hydra_targets(a) and e not in ctx.forced_hydra_targets(a)):
            return False
        if not old or old == e:
            return True
        if e not in self.rays[a] or not self.est[a,e].window_shots:
            return False
        old_hp = s.enemies[old].effective_health
        lower = self.est[a,old].damage_lo or 0.
        # A whole-team finishing volley is protected too, unless this actor
        # is genuinely surplus. Recheck this condition on the final branch.
        if lower >= old_hp or (self.ref_lower[old] >= old_hp > self.ref_lower[old]-lower):
            return False
        if any(old in self.threats[b] and s.allies[b].health_fraction <= .4 for b in self.allies):
            return False
        surplus = self.ref_lower[old]-lower >= old_hp
        if old not in pack['enemies']:
            return surplus  # never borrow productive fire from a different fight
        # Within the SAME compact contest, concentrating on the visibly more
        # damaged attacker with a second return-fire ray can bring removal
        # forward. No useful shot is lent to another front on this basis.
        concentrate = (old in pack['enemies'] and s.enemies[e].effective_health < old_hp
            and any(b != a and e in self.rays[b] and s.allies[b].weapon_cooldown_loops <= 2*ctx.H
                    for b in pack['members'])
            and any(e in self.threats[b] for b in pack['protected']))
        return self.value.redirect(self, a, e) or concentrate or surplus or any(b != a and e in self.threats[b]
                       and old not in self.threats[b] and s.allies[b].health_fraction < .7 for b in pack['protected'])


def _packs(w, previous):
    s = w.ctx.snapshot
    contacts = {e: {a for a in w.allies if e in w.rays[a] or e in w.threats[a]} for e in w.enemies}
    prior = {t: (i, u) for i, (t, u) in enumerate(previous.get('anchors', ()))}
    order = sorted((e for e in contacts if contacts[e]), key=lambda e: (
        prior.get(w.enemies[e], (100, ''))[0], w.enemies[e]))
    packs = []
    while order:
        e = order.pop(0)
        old_anchor = prior.get(w.enemies[e], (None, None))[1]
        anchor = min(contacts[e], key=lambda a:(w.allies[a] != old_anchor,
                     e not in w.threats[a], w.allies[a]))
        # Complete-link on enemy positions, single shared *actual* anchor.
        # A shooter who sees two fronts never bridges them. Opposing bearings
        # around the same protected ally remain separate even at melee range.
        enemies = [e]
        origin = s.allies[anchor].position
        def bearing_ok(x, y):
            p, q = s.enemies[x].position, s.enemies[y].position
            return sum((i-k)*(j-k) for i,j,k in zip(p,q,origin)) >= 0
        for t in order[:]:
            if w.enemies[t] in prior or anchor not in contacts[t]:
                continue
            if all(dist(s.enemies[x].position, s.enemies[t].position) <= 3.0 and
                   bearing_ok(x, t) for x in enemies):
                enemies.append(t)
                order.remove(t)
        targets = set(enemies)
        direct = set.union(*(contacts[x] for x in enemies))
        protected = sorted(a for a in direct if targets & w.threats[a])
        members = sorted(direct | {a for a in w.allies if targets.intersection(w.steps[a])})
        packs.append(dict(pack_id=f'{w.enemies[e]}/{w.allies[anchor]}', anchor_enemy_tag=w.enemies[e],
            anchor_ally_tag=w.allies[anchor], members=members, direct=sorted(direct), enemies=enemies,
            protected=protected, fire_rays={a:sorted(targets & w.rays[a]) for a in members},
            one_step_rays={a:sorted(targets.intersection(w.steps[a])) for a in members},
            incoming={a:sorted(targets & w.threats[a]) for a in protected}))
    return packs


def _plan(w, pack, orders, *, request=None, live=None, eta=None):
    ctx, s = w.ctx, w.ctx.snapshot
    actors = tuple(sorted(orders))
    targets = {a:orders[a][0] for a in actors}
    actions = {a:orders[a][1] for a in actors}
    attack = {a:ctx.decode(a, actions[a]).kind == 'attack' for a in actors}
    roles = {a:('counter_fire' if attack[a] else 'reserve_approach' if request else 'approach_attack') for a in actors}
    d = {key:pack[key] for key in ('pack_id','anchor_enemy_tag','anchor_ally_tag','members','protected')}
    d.update(actor_target=targets, target_tags={e:w.enemies[e] for e in targets.values()},
             actor_tags={a:w.allies[a] for a in actors}, roles=roles, explicit_attack=attack,
             reference_actions={a:w.ref[a] for a in actors}, preserve_volley={},
             witnesses={a:sorted(b for b in pack['direct'] if b != a and
                        (targets[a] in w.rays[b] or targets[a] in w.threats[b])) for a in actors})
    skill = RESERVE if request else LOCAL
    effects, fire, spaces = [], Counter(), []
    for a, e in targets.items():
        spaces.append(space_claim(ctx, a, actions[a]))
        if attack[a]:
            est = w.est[a,e]
            effects.append(EffectClaim('damage', e, est.damage_est, ctx.H, est.evidence,
                           'own window shot estimate', est.damage_lo, est.damage_hi))
            if est.damage_est is not None:
                fire[e] += est.damage_est
        else:
            effects.append(EffectClaim('coverage', e, None, ctx.H, HEURISTIC, 'projected support route'))
        old = w.shot[a]
        if old and old != e and w.ref_lower[old] >= s.enemies[old].effective_health:
            d['preserve_volley'][old] = s.enemies[old].effective_health
    goal = targets[actors[0]]
    delta = {}
    phase = 'engage' if all(attack.values()) else 'support'
    ttl = None
    if request:
        rid = request['request_id']
        ttl = request['expires_loop'] - s.game_loop
        phase = 'reinforcing'
        d.update(request_id=rid, seat=0, eta_loops=eta, expires_loop=request['expires_loop'],
                 request_role=request['role'])
        spaces[0] = replace(spaces[0], slot_id=rid+':0')
        a = actors[0]
        extra = dict(pack_id=pack['pack_id'], request_id=rid, target_tag=w.enemies[goal],
                     actor_tag=w.allies[a], expires_loop=request['expires_loop'],
                     gap=ctx.distance(a,goal), last_move_loop=s.game_loop,
                     retry_key=(skill, a, rid))
        extra.update(enemy_position=s.enemies[goal].position,
                     home_line=live.extra['home_line'] if live else home_line(w, a, request, w.packs))
        if live:
            delta = dict(kind='continue', instance_id=live.instance_id, phase=phase, extra=extra)
        else:
            # Instance goal is tag-based, not an observation slot. Native
            # _retire_switched releases absent/mismatched continuations.
            delta = dict(kind='create', skill=skill, actors=actors, goal=rid, goal_kind='internal',
                         phase=phase, max_loops=ttl, extra=extra)
    p = make_plan(skill, actors, goal, actions, effects,
        ResourceClaims(controls=actors, fire=tuple(sorted(fire.items())), space=tuple(spaces)),
        Commitment(phase, max_loops=ttl), {}, s.snapshot_id, w.memory.encounter_config.get('_fingerprint',''),
        instance_id=live.instance_id if live else None, beneficiaries=sorted(set(pack['protected']) |
            {b for mates in d['witnesses'].values() for b in mates}), dependencies=(Dependency('member', actors),
            Dependency('target_valid', tuple(sorted(set(targets.values()))))), atomic=len(actors)>1,
        goal_kind='enemy', goal_data=d, memory_delta=delta, rationale='tagged local contest',
        task_key=f"{skill}|{pack['pack_id']}|{','.join(w.allies[a] for a in actors)}")
    # Same action toward two named requests/targets is not the same contract.
    p.proposal_id += ':' + pack['pack_id'] + ':' + ','.join(w.enemies[e] for e in targets.values())
    return p


def _local_offers(w, p):
    s, ctx = w.ctx.snapshot, w.ctx
    choices = []
    for a in p['members']:
        for e in p['enemies']:
            peers = [b for b in p['direct'] if b != a and (e in w.threats[b] or e in w.rays[b])]
            if not peers or not w.can_redirect(a,e,p):
                continue
            # Holding contact across the last part of cooldown requires an
            # attack command, but this is not an available window volley.
            shot = e in w.rays[a] and s.allies[a].weapon_cooldown_loops <= 2*ctx.H
            move = w.steps[a].get(e)
            if shot:
                cmd = ctx.attack_action(a,e)
            elif move and not w.shot[a] and not w.rays[a]:
                cmd = move[-1]
            else:
                continue
            if cmd == w.ref[a] or w.ref_lower[e] >= s.enemies[e].effective_health:
                continue
            protected_health = min((s.allies[b].health_fraction for b in peers if e in w.threats[b]),default=2.)
            choices.append(((not shot, protected_health, s.enemies[e].effective_health,
                           ctx.distance(a,e),w.allies[a],w.enemies[e]), a, e, cmd))
    choices.sort(key=lambda row: (-w.value.delta({**w.ref, row[1]: row[3]}), row))
    if not choices:
        return []
    # Offer single interventions AND one joint allocation; don't force a
    # witness to take an unrelated order just to make the intervention atomic.
    group, budget = {}, Counter(w.ref_lower)
    for _, a, e, cmd in choices:
        if a in group or budget[e] >= s.enemies[e].effective_health:
            continue
        group[a] = (e, cmd)
        if w.shot[a]:
            budget[w.shot[a]] -= w.est[a,w.shot[a]].damage_lo or 0.
        if (a,e) in w.est:
            budget[e] += w.est[a,e].damage_lo or 0.
        if len(group) == 3:
            break
    offers = [_plan(w,p,{a:(e,cmd)}) for _,a,e,cmd in choices[:3]]
    if len(group)>1:
        offers.insert(0,_plan(w,p,group))
    return offers


def _requests(w, packs, previous, audit):
    s = w.ctx.snapshot
    requests = audit['requests']
    cooldowns = audit['request_cooldowns']
    cooldowns.update({k:v for k,v in previous.get('cooldowns', {}).items() if s.game_loop < v})
    for p in packs:
        needs = []
        for e in p['enemies']:
            receivers = [a for a in p['direct'] if e in w.threats[a]]
            if not receivers or w.ref_lower[e] >= s.enemies[e].effective_health:
                continue
            # A lone shooting line can need a second gun, or a threatened
            # gun may need a body against a ling. Never request a whole squad.
            inbound = {i.actors[0] for i in w.memory.instances.values() if i.skill==RESERVE and
                       i.extra.get('pack_id')==p['pack_id'] and i.extra.get('target_tag')==w.enemies[e]
                       and e not in w.rays.get(i.actors[0],())}
            guns = [a for a in p['members'] if a not in inbound and s.allies[a].type_name=='hydralisk' and
                    (e in w.rays[a] or e in w.steps[a])]
            dogs = [a for a in p['members'] if a not in inbound and s.allies[a].type_name=='zergling' and
                    (e in w.rays[a] or e in w.steps[a])]
            screen = s.enemies[e].type_name=='zergling' and not dogs and any(
                     s.allies[a].type_name=='hydralisk' for a in receivers)
            if not screen and len(guns)>=2:
                continue
            role = 'melee_screen' if screen else 'ranged_fire'
            key = f"{p['pack_id']}:{w.enemies[e]}:{role}"
            created, expires = previous.get('clocks', {}).get(key, (s.game_loop,s.game_loop+w.ctx.H*REQUEST_H))
            if s.game_loop >= expires:
                if s.game_loop < expires + 2*w.ctx.H:
                    cooldowns[key] = expires + 2*w.ctx.H
                else:
                    created, expires = s.game_loop, s.game_loop + REQUEST_H*w.ctx.H
            if key in cooldowns:
                continue
            needs.append(((key not in previous.get('clocks', {}), min(s.allies[a].health_fraction for a in receivers),
                           len(guns),w.enemies[e]), dict(key=key, pack_id=p['pack_id'],request_id=f'{key}@{created}',
                  target=e, target_tag=w.enemies[e], protected=sorted(receivers), slots=1,
                  role=role, created_loop=created,expires_loop=expires,claimed_by=[])))
        if needs:
            requests.append(min(needs,key=lambda r:r[0])[1])
    return requests


def _reserve_offers(w, packs, requests, audit, previous):
    s, ctx = w.ctx.snapshot, w.ctx
    by_pack = {p['pack_id']:p for p in packs}
    by_id = {r['request_id']:r for r in requests}
    prior = {i.actors[0]:i for i in w.memory.instances.values() if i.skill==RESERVE and not i.termination_reason}
    # Only unengaged actors without useful fire, local contact or a useful
    # reference route into another contested ray count as reserves. Handoff
    # at the FIRST directly actionable ray, not after the next move.
    idle = [a for a in sorted(w.allies) if not (a in w.locked or w.rays[a] or w.threats[a])
            and s.allies[a].health_fraction >= .55 and ctx.decode(a,w.ref[a]).kind in ('move','stop')]
    contested = {e for p in packs for e in p['enemies']}
    nearby = {a:contested.intersection(w.steps[a]) for a in idle}
    useful = {a:{e for e in nearby[a] if dist(ctx.predict_position(a,w.ref[a]).position,
                 s.enemies[e].position) <= ctx.effective_attack_reach(a,e)} for a in idle}
    audit['idle'] = [a for a in idle if not nearby[a] or a in prior]
    candidates = defaultdict(list)
    for r in requests:
        e, rid = r['target'], r['request_id']
        p = by_pack[r['pack_id']]
        for a in idle:
            live = prior.get(a)
            same = live and live.extra.get('request_id') == rid and live.extra.get('actor_tag')==w.allies[a]
            if not near_bid(w, a, p, live, same, nearby[a], previous):
                continue
            if useful[a] - {e} or (not same and (a in p['members'] or nearby[a])):
                continue
            if (s.allies[a].type_name=='hydralisk') != (r['role']=='ranged_fire'):
                continue
            if same and (live.extra.get('target_tag') != w.enemies[e] or
                         ctx.distance(a,e) >= live.extra['gap']-.05):
                continue
            if w.memory.retry_blocked(RESERVE,a,rid,s.game_loop):
                continue
            route = w.route(a,e,REQUEST_H)
            if route is None:
                continue
            eta, _, cmd = route
            if s.game_loop+eta > r['expires_loop'] or (cmd==w.ref[a] and not same):
                continue
            candidates[rid].append((not bool(same),eta,ctx.distance(a,e),a,cmd))

    # A claimant's lease survives only while it actually has an executable
    # continuation. A new pack cannot steal it on an ETA tie. Peers may bid
    # for a vacated seat immediately (handoff / route blocked / timeout).
    owned = {a:rid for rid, rows in candidates.items() for new,eta,gap,a,cmd in rows if not new}
    observed = {**previous.get('prior_claims',{}), **{a:i.extra for a,i in prior.items()}}
    for a, extra in observed.items():
        rid = extra['request_id']
        if owned.get(a)==rid:
            by_id[rid]['claimed_by'].append(a)
            continue
        if s.game_loop >= extra['expires_loop']:
            reason='expired'
        elif a not in w.allies or w.allies[a] != extra['actor_tag']:
            reason='actor_lost'
        elif w.rays[a] or w.threats[a]:
            reason='local_contact_handoff'
        elif rid not in by_id:
            reason='request_withdrawn'
        else:
            reason='blocked_or_preempted'
        audit['releases'].append(dict(actor=a,request_id=rid,reason=reason))
    groups = []
    for rid, rows in candidates.items():
        r, group = by_id[rid], []
        for new,eta,_,a,cmd in sorted(rows):
            if a in owned and owned[a] != rid or new and r['claimed_by']:
                continue
            group.append(_plan(w,by_pack[r['pack_id']],{a:(r['target'],cmd)},request=r,
                               live=None if new else prior[a],eta=eta))
            if len(group)==2:
                break
        if group:
            groups.append(group)
    return groups


def propose_local(context, memory, reference_actions, dispatch: bool) -> tuple[list[CandidatePlan], dict]:
    s = context.snapshot
    kinds = Counter(u.type_name for u in s.allies.values())  # includes dead original roster members
    n_enemies = s.raw_state.get('scenario_brief',{}).get('n_enemies',s.n_enemies)
    scope = (s.step_mul==8 and n_enemies==10 and len(s.allies)==10 and
             (kinds['zergling'],kinds['hydralisk']) in ((3,7),(6,4)) and
             all(u.type_name in ('zergling','hydralisk') for u in s.enemies.values()))
    audit = dict(in_scope=scope,packs=[],requests=[],idle=[],releases=[],candidate_count=0,
                 request_cooldowns={})
    if not scope:
        return [], audit
    w = _Window(context, memory, _joint(s,reference_actions))
    previous = _previous(context,memory)
    packs = audit['packs'] = _packs(w,previous)
    from .local_pack_options import PackValue
    w.value = PackValue(context, packs, w.ref)
    w.packs = packs
    local = [_local_offers(w,p) for p in packs]
    reserves = _reserve_offers(w,packs,_requests(w,packs,previous,audit),audit,previous) if dispatch else []
    # Give each front and each kind of duty a first option before adding
    # alternatives. IDs distinguish competing goals on identical commands.
    groups = reserves + local
    plans = [rows[k] for k in range(max(map(len,groups),default=0)) for rows in groups if k<len(rows)][:MAX_OFFERS]
    audit['candidate_count'] = len(plans)
    audit['candidate_skills'] = dict(Counter(p.primary_skill for p in plans))
    return plans, audit


def local_conflicts(context, plans, actions) -> list[str]:
    """Check FINAL complete commands. Only each real shooter can spend its shot.

    An estimate of damage from a witness or a projected move is never
    deposited. Unchanged baseline fire has first call on the remaining HP;
    the last (possibly slightly oversized) volley is allowed, redundant
    new volleys are not. The frozen evaluator prices the complete branch.
    """
    ctx, s = context, context.snapshot
    joint = _joint(s,actions)
    local = [p for p in plans if p.primary_skill in (LOCAL,RESERVE)]
    if not local:
        return []
    changed_shots, must_keep, seats, controls = defaultdict(list), {}, set(), set()
    for p in local:
        d = p.goal_data
        if p.primary_skill==RESERVE:
            seat = d['request_id'],d['seat']
            if seat in seats or d['expires_loop'] <= s.game_loop:
                return ['seat_or_expiry']
            seats.add(seat)
        must_keep.update(d['preserve_volley'])
        for a,cmd in p.current_actions.items():
            if a in controls or cmd != joint[a]:
                return ['control:'+a]
            controls.add(a)
            sem = ctx.decode(a,cmd)
            if sem.kind=='move':
                end = ctx.predict_position(a,cmd).position
                if not support_path_clear(ctx,a,end,joint):
                    return ['path:'+a]
            elif sem.kind=='attack' and cmd != d['reference_actions'][a]:
                changed_shots[sem.target].append(a)
    seen = defaultdict(dict)
    for a,cmd in joint.items():
        sem = ctx.decode(a,cmd)
        if sem.kind=='attack' and (sem.target in changed_shots or sem.target in must_keep) and ctx.can_issue_attack(a,sem.target):
            seen[sem.target][a] = ctx.fire_estimate(a,sem.target).damage_lo or 0.
    for e,needed in must_keep.items():
        if sum(seen[e].values()) < needed:
            return ['stolen_finisher:'+e]
    for e,added in changed_shots.items():
        total = sum(seen[e].values())
        if any(total-seen[e][a] >= s.enemies[e].effective_health for a in added):
            return ['redundant_fire:'+e]
    return []
