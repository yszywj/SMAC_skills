"""Offline replay/decision audit of 150 matched baseline/packs/packs_reserve scenarios."""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

from compare_protoss_parallel import digest, validate_manifest, write_json
from contact_early_report import COMPOSITIONS, SIDES, _check, _deaths, _run
from contact_factorial_report import MODE
from contact_front_report import METHODOLOGY as FRONT, _layout, _since, _stats, _triples
from contact_micro_report import OUTCOMES, TYPES, _read, _shard
from contact_muster_report import METRICS, _behavior as _opening_behavior

LABELS = ('baseline', 'packs', 'packs_reserve')
HORIZONS = (8, 16)


def _snapshot(batch, label, files):
    root = batch / label
    for name, checksum in files.items():
        path = root / name
        _check(Path(name).parts and Path(name).parts[0] in ('skills', 'sc2_skill_adapter', 'smacv2') and
               path.resolve().is_relative_to(root.resolve()) and path.is_file() and
               hashlib.sha256(path.read_bytes()).hexdigest() == checksum, path, 'snapshot path/hash mismatch')
    actual = {p.relative_to(root).as_posix() for tree in ('skills', 'sc2_skill_adapter', 'smacv2')
              for p in (root / tree).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    _check(actual == set(files), root, 'snapshot coverage differs from design')


def _submission(actor, action, receipt, state, where):
    unit = state['allies'][actor]
    cmd, status = receipt['command'], receipt['submission']
    _check(receipt['requested_action_id'] == str(action), where, f'{actor}: receipt action mismatch')
    if not unit['alive']:
        _check(action == 0 and status == 'dead_noop' and cmd is None and receipt['response'] is None,
               where, f'{actor}: invalid dead receipt')
        return
    kind, ability = ('attack', 23) if action >= 6 else ('stop', 4) if action == 1 else ('move', 16)
    response = receipt['response']
    code = response.get('code') if isinstance(response, dict) else None
    _check(isinstance(cmd, dict) and cmd['kind'] == kind and cmd['ability_id'] == ability and
           ((response is None and status == 'result_unmapped') or
            (type(code) is int and status == ('accepted' if code == 1 else 'rejected'))), where,
           f'{actor}: inconsistent submission')
    target = state['visible_enemies'].get(f'e{action-6}') if action >= 6 else None
    _check((action < 6 or (target and target['alive'])) and cmd['target_tag'] == (target and target['tag']),
           where, f'{actor}: submitted target tag mismatch')


def _frame(obs, decision, step, label, enemy_roster, deaths, where):
    payload = obs['payload']
    state = payload['state']
    loop = state['game_loop']
    allies, enemies = state['allies'], state['visible_enemies']
    tags = {e: u['tag'] for e, u in enemies.items() if u['alive']}
    _check(len(set(tags.values())) == len(tags) and all(enemy_roster.get(t) == enemies[e]['type'] and
           deaths.get(t, math.inf) > loop for e, t in tags.items()), where, 'visible enemy roster/death mismatch')
    actions = decision['actions']
    feedback = step['feedback']
    _check(actions == {f'u{i}': a for i, a in enumerate(step['actions'])} and
           actions.keys() == allies.keys() == feedback['actors'].keys() and
           feedback['from_game_loop'] == loop and feedback['to_game_loop'] == loop+8,
           where, 'submitted vector/receipt window mismatch')
    units, reverse = {}, {}
    for a, u in allies.items():
        action = actions[a]
        _check(type(action) is int, where, 'invalid action')
        receipt = feedback['actors'][a]
        _submission(a, action, receipt, state, where)
        if not u['alive']:
            continue
        menu = payload['questions'][a]['criteria']
        rel = state['relations'][a]
        _check(str(action) in menu and all(type(rel[e]['smac_attack_available']) is bool for e in tags) and
               {e for e in tags if rel[e]['smac_attack_available']} == {f'e{int(k)-6}' for k in menu if int(k)>=6},
               where, f'{a}: attack mask/menu mismatch')
        rays = {t: rel[e]['smac_attack_available'] and rel[e]['in_weapon_range'] for e,t in tags.items()}
        reverse[a] = {t: rel[e]['in_enemy_weapon_range'] for e,t in tags.items()}
        orders = u['orders']
        order = orders[0] if orders else {}
        # Generic SC2 Attack (3674) may be labelled "other" by the frozen recorder.
        active = order.get('target_tag') if order.get('ability_id') in (23, 3674) else None
        units[a] = dict(tag=u['tag'], position=u['position'], cd=u['weapon_cooldown_loops'], rays=rays,
                        active=active, requested=receipt['command']['target_tag'], status=receipt['submission'])
    packs, requests, selected = _audit(decision, label, payload['questions'], units, tags, where)
    return dict(loop=loop, units=units, tags=tags, enemies={t: enemies[e]['position'] for e, t in tags.items()},
                reverse=reverse, packs=packs, requests=requests, selected=selected)


def _audit(decision, label, questions, units, tags, where):
    actions = decision['actions']
    if label == 'baseline':
        _check('local_coord' not in decision, where, 'unexpected baseline local_coord')
        return {}, {}, {}
    trace = decision.get('local_coord')
    _check(isinstance(trace, dict) and all(isinstance(trace.get(k), list) for k in ('packs','requests','selected'))
           and isinstance(trace.get('changes'), dict), where, 'missing local_coord audit')
    reference = trace.get('reference_actions')
    _check(trace.get('dispatch') is (label == 'packs_reserve') and isinstance(reference, dict) and
           reference.keys() == actions.keys(), where, 'dispatch/reference mismatch')
    packs, requests, selected = {}, {}, {}
    for p in trace['packs']:
        pid = p['pack_id']
        _check(pid not in packs and p['members'] and len(set(p['members'])) == len(p['members']) and
               set(p['members']) <= units.keys() and set(p['enemies']) <= tags.keys() and
               p['anchor_enemy_tag'] in tags.values() and p['anchor_ally_tag'] in {u['tag'] for u in units.values()},
               where, 'invalid pack identity/members')
        packs[pid] = p
    for r in trace['requests']:
        rid = r['request_id']
        _check(rid not in requests and r['pack_id'] in packs and r['target'] in packs[r['pack_id']]['enemies'] and
               r['target_tag'] == tags[r['target']] and type(r['slots']) is int and r['slots'] > 0 and
               isinstance(r.get('claimed_by'), list) and set(r['claimed_by']) <= units.keys(),
               where, 'invalid local request')
        requests[rid] = r
    claims = Counter()
    for s in trace['selected']:
        a, pid, e, rid = (s[k] for k in ('actor','pack_id','target','request_id'))
        before, after = s['before'], s['after']
        _check(a in units and a not in selected and pid in packs and e in tags and s['target_tag'] == tags[e] and
               s['plan_id'] and s['skill'] and type(before) is int and str(before) in questions[a]['criteria']
               and reference[a] == before and decision['units'][a]['plan'] == s['plan_id'] and
               decision['units'][a]['skill'] == s['skill'] and decision['units'][a]['action'] == after
               and type(after) is int and after == actions[a] and (after < 6 or e == f'e{after-6}') and
               s['changed'] is (before != after), where, 'invalid selected action/target')
        if rid is not None:
            _check(label == 'packs_reserve' and rid in requests and requests[rid]['pack_id'] == pid and
                   requests[rid]['target_tag'] == tags[e] and s['skill'] == 'reserve_dispatch' and
                   requests[rid]['expires_loop'] > decision['game_loop'], where, 'invalid reserve claim')
            claims[rid] += 1
        else:
            _check(s['skill'] == 'local_engagement' and a in packs[pid]['members'] and e in packs[pid]['enemies'],
                   where, 'selected actor/target not in named pack')
        selected[a] = s
    _check(all(n <= requests[r]['slots'] for r,n in claims.items()), where, 'request capacity exceeded')
    changes = trace['changes']
    _check(changes.keys() == {a for a,s in selected.items() if s['changed']} ==
           {a for a, cmd in actions.items() if cmd != reference[a]} and all(
           all(c.get(k) == selected[a][k] for k in ('before','after','pack_id','target_tag','skill'))
           for a,c in changes.items()), where, 'changes differ from selected actions')
    return packs, requests, selected


def _fronts(frame):
    """Two disconnected engagements, or incoming contacts on opposite bearings around one ally."""
    units, enemies = frame['units'], frame['enemies']
    edges, graph, pincer = [], defaultdict(set), None
    for a,u in units.items():
        incoming = []
        for t in enemies:
            threat = frame['reverse'][a][t] is True
            if threat:
                incoming.append(t)
            if threat or u['rays'][t] is True:
                graph[a].add(t)
                graph[t].add(a)
                edges.append((a,t))
        if pincer is None:
            x,y = u['position']
            for t,v in combinations(incoming, 2):
                ax,ay = enemies[t][0]-x, enemies[t][1]-y
                bx,by = enemies[v][0]-x, enemies[v][1]-y
                scale = math.hypot(ax,ay)*math.hypot(bx,by)
                if scale and (ax*bx+ay*by)/scale <= -.5:
                    pincer = dict(actor=a, enemy_tags=[t,v])
                    break
    seen, components = set(), []
    for a,t in edges:
        if a not in seen:
            group, todo = set(), [a]
            while todo:
                node = todo.pop()
                if node not in group:
                    group.add(node)
                    todo.extend(graph[node] - group)
            seen.update(group)
            components.append(dict(allies=sorted(group & units.keys()), enemies=sorted(group & enemies.keys())))
    def positions(component):
        return [units[a]['position'] for a in component['allies']] + [enemies[t] for t in component['enemies']]
    separate = None
    for left,right in combinations(components, 2):
        gap = min(math.dist(p,q) for p in positions(left) for q in positions(right))
        if gap >= 4:
            separate = dict(components=[left,right], min_gap=gap)
            break
    return dict(multiple_fronts=separate, incoming_pincer=pincer)


def _sample(frames, i, a, actor_tag, target, deaths):
    if i >= len(frames):
        return dict(status='truncated')
    f = frames[i]
    if deaths.get(target, math.inf) <= f['loop']:
        return dict(status='target_dead')
    if deaths.get(actor_tag, math.inf) <= f['loop']:
        return dict(status='actor_dead')
    u = f['units'].get(a)
    if not u or u['tag'] != actor_tag:
        return dict(status='actor_unavailable')
    if target not in f['enemies']:
        return dict(status='target_unseen')
    old = frames[i-1]['units'].get(a) if i else None
    cd = u['cd'] > max(0, old['cd']-8)+.02 if old and old['tag'] == actor_tag else None
    ray = u['rays'][target]
    active = u['active'] == target
    return dict(status='observed', ray=ray, active_named_order=active, accepted_named_order=
                u['requested'] == target and u['status'] == 'accepted', cd_restart=cd,
                named_cd=(ray and active and cd))


def _followup(frames, i, a, target, deaths):
    tag = frames[i]['units'][a]['tag']
    base, *later = [_sample(frames, i+j, a, tag, target, deaths) for j in range(3)]
    windows = {}
    for j,h in enumerate(HORIZONS, 1):
        end = later[j-1]
        samples = later[:j]
        # Report positive observed evidence even if a subsequent sample is unavailable;
        # negative evidence requires *all* samples through that horizon.
        def within(key):
            values = [s.get(key) for s in samples]
            return True if True in values else None if None in values else False
        had_named_line = base['ray'] and base['active_named_order']
        windows[str(h)] = dict(
            endpoint=end, new_ray_by_h=not base['ray'] and within('ray') if base['ray'] is not None else None,
            new_named_cd_by_h=not had_named_line and within('named_cd') if had_named_line is not None else None,
            named_cd_by_h=within('named_cd'))
    return dict(actor=a, actor_tag=tag, target_tag=target, at_select=base, windows=windows,
                actor_death_loop=deaths.get(tag), target_death_loop=deaths.get(target))


def _behavior(triples, game, label, enemy_roster, deaths, where):
    frames = []
    def observe():
        for obs, decision, step in triples:
            frames.append(_frame(obs, decision, step, label, enemy_roster, deaths, f"{where}: step {obs['step']}"))
            yield obs, decision, step
    # All three policies have the same contact metrics and *no* opening_muster override.
    _opening_behavior(observe(), game, 'baseline', where)
    game.pop('opening_muster_changes')
    counts, events, first, proposals = Counter(), [], {}, []
    prev_packs, prev_requests, prev_claims = set(), set(), set()
    for i,f in enumerate(frames):
        fronts = _fronts(f)
        for key,value in fronts.items():
            if value:
                first.setdefault(key, dict(loop=f['loop'], evidence=value))
                counts[key+'_frames'] += 1
        packs, requests, selected = (f[k] for k in ('packs','requests','selected'))
        counts['pack_frames'] += len(packs)
        counts['pack_starts'] += len(packs.keys() - prev_packs)
        prev_packs = packs.keys()
        counts['request_frames'] += len(requests)
        counts['request_starts'] += len(requests.keys() - prev_requests)
        counts['prior_claimant_frames'] += sum(len(r['claimed_by']) for r in requests.values())
        prev_requests = requests.keys()
        if packs or requests:
            proposals.append(dict(loop=f['loop'], packs=list(packs.values()), requests=list(requests.values())))
        counts['selected_pack_frames'] += len({s['pack_id'] for s in selected.values()})
        claims = {(s['request_id'],a) for a,s in selected.items() if s['request_id'] is not None}
        counts['claimed_request_frames'] += len({rid for rid,a in claims})
        counts['reserve_claim_starts'] += len(claims-prev_claims)
        prev_claims = claims
        # Audit original members once per selected pack + named target per frame,
        # including unmodified members. Selection of two helpers isn't two lost members.
        common = {}
        for a,s in selected.items():
            target = s['target_tag']
            group = 'reserve' if s['request_id'] is not None else 'pack'
            receipt = f['units'][a]['status']
            counts[f'{group}_selected_actor_frames'] += 1
            counts[f'{group}_changed_actions'] += int(s['changed'])
            counts[f'{group}_receipt_{receipt}'] += 1
            for k, evidence in fronts.items():
                counts[f'{group}_selected_in_{k}'] += int(bool(evidence))
            track = _followup(frames,i,a,target,deaths)
            donor = None
            if group == 'reserve':
                # Foregone reference attack, or previously active named attack: a possible
                # donor cost, not proof that the counterfactual would have hit.
                old = f['tags'].get(f"e{s['before']-6}") if s['before'] >= 6 else f['units'][a]['active']
                if old and old != target and f['units'][a]['rays'].get(old) is True:
                    donor = _followup(frames,i,a,old,deaths)
                    counts['reserve_left_other_named_opportunity'] += 1
            events.append(dict(loop=f['loop'], kind=group, selection=s, receipt=receipt, donor=donor,
                               geometry=[k for k, v in fronts.items() if v], **track))
            if receipt == 'accepted':
                for h,w in track['windows'].items():
                    counts[f'{group}_status_{h}_{w["endpoint"]["status"]}'] += 1
                    for key in ('new_ray_by_h','new_named_cd_by_h'):
                        val = w[key]
                        counts[f'{group}_{key}_{h}_{"unknown" if val is None else str(val).lower()}'] += 1
            common.setdefault((s['pack_id'],target),set()).add(a)
        for (pid,target), controllers in common.items():
            for a in packs[pid]['members']:
                track = _followup(frames,i,a,target,deaths)
                base = track['at_select']
                if not (base.get('active_named_order') and base.get('ray') is True):
                    continue
                counts['original_named_active'] += 1
                for h,w in track['windows'].items():
                    end = w['endpoint']
                    if end['status'] != 'observed':
                        result = end['status']
                    elif end['ray'] is None:
                        result = 'range_unknown'
                    elif not end['ray'] or not end['active_named_order']:
                        result = 'ray_or_order_lost'
                    else:
                        result = 'ray_and_order_retained'
                    w['continuity'] = result
                    counts[f'original_{h}_{result}'] += 1
                    if base['named_cd'] is True:
                        participation = w['named_cd_by_h']
                        status = 'unknown' if participation is None else 'seen' if participation else 'not_seen'
                        counts[f'original_pre_cd_{h}_named_cd_{status}'] += 1
                events.append(dict(loop=f['loop'], kind='original_member', pack_id=pid,
                              selected_actors=sorted(controllers), **track))
    game['local_coord'] = dict(counts=dict(counts), first_geometry=first, proposals=proposals, events=events)


def _games(work, manifest, name, label, files, *, mode=MODE, audit_triples=None):
    games, contract = _shard(work, manifest, mode, files)
    verified = _run(work, name, manifest)
    with (work/'events.jsonl').open('rb') as stream:
        for ep,(g,v) in enumerate(zip(games,verified),1):
            stem = f'episode-{ep:04d}'
            where = work/stem
            _check(all(g[k] == v[k] for k in ('initial_state_sha256','outcome','damage_by_type')),
                   where, 'replay/episode mismatch')
            g.update(v)
            d = _read(work/'damage_by_type'/(stem+'.json'))
            rosters = {p:r['initial_roster_by_api_tag'] for p,r in d['players'].items()}
            deaths = _deaths(work/'replays'/(stem+'.SC2Replay'), rosters, d['begin_request_loop'], d['terminal_loop'])
            triples = _triples(stream, ep, where, with_micro=audit_triples is not None)
            if audit_triples is not None:
                triples = audit_triples(triples, ep, g, rosters, where)
            _behavior(triples,g,label,rosters['2'],deaths,where)
    return games, contract


def _paired(rows, left, right):
    pairs = [(r['versions'][left], r['versions'][right]) for r in rows]
    table = Counter((a['outcome'],b['outcome']) for a,b in pairs)
    return dict(pairs=len(pairs), left=left, right=right,
                gained_wins=sum(table[o,'win'] for o in OUTCOMES if o != 'win'),
                dropped_wins=sum(table['win',o] for o in OUTCOMES if o != 'win'),
                paired_outcomes={a:{b:table[a,b] for b in OUTCOMES} for a in OUTCOMES},
                damage_delta={side:{u:sum(b['damage_by_type'][side][u]-a['damage_by_type'][side][u]
                                          for a,b in pairs) for u in TYPES} for side in SIDES},
                metrics_delta={k:_stats([_since(b[k],a[k]) for a,b in pairs]) for k in METRICS})


def _versions(rows, labels):
    versions = {}
    for label in labels:
        games = [r['versions'][label] for r in rows]
        counts = Counter(g['outcome'] for g in games)
        local = Counter()
        for g in games:
            local.update(g['local_coord']['counts'])
        games_with = {k: sum(g['local_coord']['counts'].get(k, 0) > 0 for g in games) for k in
                      ('pack_starts', 'selected_pack_frames', 'reserve_claim_starts', 'multiple_fronts_frames',
                       'incoming_pincer_frames')}
        versions[label] = dict(
            games=len(games), outcomes={o: counts[o] for o in OUTCOMES}, win_rate=counts['win'] / len(games),
            damage_by_type={s: {u: sum(g['damage_by_type'][s][u] for g in games) for u in TYPES} for s in SIDES},
            first_hydra_death_side={s: sum(g['first_hydra_death_side'] == s for g in games)
                                    for s in (*SIDES, 'same_frame', 'none')},
            metrics={k: _stats([g[k] for g in games]) for k in METRICS},
            hydra_cd_window_status=dict(Counter(g['hydra_cd_window_status'] for g in games)),
            micro_trigger={k: sum(g['micro_trigger'][k] for g in games) for k in ('windows','authorizations','submitted')},
            local_coord=dict(counts=dict(local), games_with=games_with))
    return versions


def _summary(rows):
    return dict(pairs=len(rows), versions=_versions(rows, LABELS), paired_vs_baseline={v:_paired(rows,'baseline',v) for v in LABELS[1:]},
                reserve_vs_packs=_paired(rows,'packs','packs_reserve'))


def _by_layout(rows):
    return {k:_summary([r for r in rows if r['layout']==k]) for k in sorted({r['layout'] for r in rows})}


METHODOLOGY = {**{k:FRONT[k] for k in ('contact','support','participation','deaths_damage','aggregation','layout')},
    'stages':'Pack/request starts are first appearance or reappearance after an absent frame in one episode. '
             'Frame counts are not unique packages. claimed_by records earlier commitments; selected is final ownership '
             '(including unchanged actions); changes are distinct legal before/after actions. Each selected actor is '
             'bound to the submitted action and its SC2 receipt. Result_unmapped is not accepted.',
    'followup':'For each selected actor and target tag, observe the real legal mask AND native range at L, L+8, L+16. '
               'A new ray requires absence at L. New named CD evidence requires ray + active first attack order on that '
               'tag + sampled CD restart observed in (L,L+h], with no active named line at L (a shooter between CD '
               'restarts is not new). This is participation evidence, not a '
               'target-attributed shot or hit. Unknown range, hidden/dead target, unavailable actor, and truncated '
               'windows supply no positive or negative endpoint evidence. Interval positives observed earlier remain valid.',
    'member_cost':'Original members with a ray and active first attack order to the selected common target at L are '
                  'audited once per pack/target/frame, including members not controlled. Losing ray or order is not '
                  'a proven missed shot; endpoint CD is independent. Reserve donors leaving another named opportunity '
                  'are tracked separately. Only replay-verified game totals measure actual damage.',
    'geometry':'Independent of spawn layout and policy proposals. Positive multiple fronts: disconnected simultaneous '
               'visible bipartite contact components (ally legal+native or enemy native range), with minimum inter-unit '
               'distance between components >=4. Incoming pincer: two visible enemies covering the same ally from '
               'bearings >=120 degrees apart. These are observed opportunities, not proof of intent. No evidence is '
               'not proof of absence beyond vision. Geometry strata use the same baseline game for all three versions.',
    'reserve':'reserve_vs_packs is the matched incremental comparison; stratifying on selected reserve actors is '
             'post-selection descriptive, not causal attribution to one order. Damage deltas are right minus left.'}


def _matched_rows(batch, design, labels, load_games):
    """Common frozen cohort and runner contract. The first label is the pairing reference."""
    files, shards = design['policy_files'], design['shards']
    _check(tuple(design['labels']) == labels and set(files) == set(labels) and type(design['seed']) is int,
           batch, 'invalid local packs batch design')
    _check(set(shards) == {f'{n}_{i:02d}' for n in COMPOSITIONS for i in range(0,75,5)}, batch,
           'expected 15 five-game shards per composition')
    def shared(source):
        return {k:v for k,v in source.items() if not k.startswith('skills/tactical_v1/')}
    ref = labels[0]
    for label in labels:
        _snapshot(batch,label,files[label])
        _check(shared(files[label]) == shared(files[ref]), label, 'non-policy sources differ')
    rows, contracts = [], {}
    for group,(name,roster) in enumerate(COMPOSITIONS.items()):
        team = {u.lower():n for u,n in roster.items()}
        for offset in range(0,75,5):
            stem = f'{name}_{offset:02d}'
            m = _read(batch/'manifests'/(stem+'.json'))
            validate_manifest(m)
            episodes = m['episodes']
            start = design['seed'] + 75*group
            _check(shards[stem]['composition'] == name and shards[stem]['manifest_sha256'] == m['sha256'] and
                   m['seed'] == start and [e['index'] for e in episodes] == list(range(offset,offset+5)) and
                   [e['sc2_seed'] for e in episodes] == list(range(start+offset,start+offset+5)) and
                   all(Counter(e['episode_config']['team_gen'][s]) == team for s in ('ally_team','enemy_team')
                       for e in episodes), stem, 'unexpected manifest, seed, count or composition')
            runs = {}
            for label in labels:
                runs[label], contract = load_games(batch/label/stem,m,name,label,files[label])
                _check(contract == contracts.setdefault(label,contract) and contract[1:] == contracts[ref][1:]
                       and shared(contract[0]) == shared(contracts[ref][0]), stem, 'runner contract mismatch')
            for i,e in enumerate(episodes):
                v = {label:runs[label][i] for label in labels}
                states = {g['initial_state_sha256'] for g in v.values()}
                _check(len(states)==1, stem, f'initial state mismatch: episode {i+1}')
                rows.append(dict(shard=stem, composition=name, layout=_layout(e['episode_config'],stem), episode=i+1,
                                 sc2_seed=e['sc2_seed'], scenario_sha256=e['sha256'], manifest_sha256=m['sha256'],
                                  initial_state_sha256=states.pop(), versions=v))
    return rows


def summarize(batch: Path):
    """Validate 30 five-game shards x three policies, then write and return comparison.json."""
    batch = Path(batch)
    design = _read(batch/'design.json')
    files = design['policy_files']
    _check(design['execution_mode'] == MODE, batch, 'invalid local packs batch design')
    rows = _matched_rows(batch, design, LABELS, _games)
    report = dict(seed=design['seed'], execution_mode=MODE, labels=list(LABELS), methodology=METHODOLOGY,
                  policy_sha256={label:digest(files[label]) for label in LABELS}, episodes=rows, total=_summary(rows),
                  by_layout=_by_layout(rows), by_composition={})
    for n in COMPOSITIONS:
        group = [r for r in rows if r['composition']==n]
        report['by_composition'][n] = dict(**_summary(group), by_layout=_by_layout(group))
    strata = defaultdict(list)
    for r in rows:
        evidence = r['versions']['baseline']['local_coord']['first_geometry']
        strata['+'.join(sorted(evidence)) or 'no_positive_evidence'].append(r)
    report['by_baseline_contact_geometry'] = {k:_summary(v) for k,v in strata.items()}
    for claimed in (False, True):
        group = [r for r in rows if bool(r['versions']['packs_reserve']['local_coord']['counts']['reserve_claim_starts']) == claimed]
        report.setdefault('reserve_selection_descriptive',{})['claimed' if claimed else 'no_claim'] = _paired(group,'packs','packs_reserve')
    _write_report(batch, report)
    return report


def _write_report(batch, report):
    # Detailed per-selection evidence can be large. Do not construct a second
    # full JSON string in memory or truncate an existing report on a write failure.
    with NamedTemporaryFile(mode='w', encoding='utf-8', dir=batch, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.close()
            tmp.replace(batch / 'comparison.json')
        finally:
            tmp.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    summarize(parser.parse_args().batch)
