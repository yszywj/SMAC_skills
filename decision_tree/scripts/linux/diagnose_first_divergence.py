"""Offline paired prefix audit. No simulator imports; outputs are deterministic."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

RUNS = Path(__file__).resolve().parents[3] / 'runs'
BATCHES = ['zerg_policy_eval_20261124_162949', 'zerg_policy_eval_6x40x2_20261124_164422',
           'zerg_policy_eval_random_20261124_165149']
OUT = RUNS / 'zerg_first_divergence_v1'


def load(path):
    return json.loads(path.read_bytes())


def beat(stream):
    pre = None
    for line in stream:
        if line.startswith(b'{"event": "skill_observation",'):
            assert pre is None
            obs = json.loads(line)
            pre = obs['payload']
        elif line.startswith(b'{"event": "step",'):
            e = json.loads(line)
            assert obs['step'] + 1 == e['step'] and pre['state']['game_loop'] == e['game_loop']
            return pre, e
    raise ValueError('missing step')


def features(state, a, b):
    allies = state['allies']; enemies = state['visible_enemies']; rel = state['relations']
    changes = []
    def distance(p, positions):
        return min((math.dist(p, q) for q in positions), default=None)
    for i, (old, new) in enumerate(zip(a, b)):
        if old == new:
            continue
        u = f'u{i}'; unit = allies[u]; p = unit['position']; moves = state['movement_candidates'][u]
        ep = [v['position'] for v in enemies.values() if v['alive']]
        hp = [v['position'] for k,v in allies.items() if k != u and v['alive'] and v['type']=='Hydralisk']
        def dest(k):
            return moves[str(k)]['destination'] if 2 <= k <= 5 else p
        changes.append(dict(actor=u, type=unit['type'], old=old, new=new, ready=unit['weapon_ready'],
                            hp=unit['health'], nearest=distance(p,ep),
                            enemy_delta=round(distance(dest(new),ep)-distance(dest(old),ep),3) if ep else None,
                            hydra_delta=round(distance(dest(new),hp)-distance(dest(old),hp),3) if hp else None,
                            legal=sum(v['smac_attack_available'] for v in rel.get(u,{}).values())))
    return dict(visible=len(enemies), alive=sum(x['alive'] for x in allies.values()),
                min_distance=min((v['distance'] for rs in rel.values() for v in rs.values()), default=None),
                legal_actors=sum(any(v['smac_attack_available'] for v in rs.values()) for rs in rel.values()),
                actions=[a,b],changes=changes)


def pair(batch, p):
    root=RUNS/batch/p['pair_id']; arms=[root/a/'runner' for a in ('baseline','tree')]
    x,y=[load(d/'episodes.json')[0] for d in arms]
    for d,e in zip(arms,(x,y)):
        assert load(d/'manifest.json')['sha256']==p['manifest_sha256'] and e['scenario_sha256']==p['scenario_sha256']
    assert x['initial_state_sha256']==y['initial_state_sha256']
    receipt=load(arms[1]/'distill/episode-0001.json'); assert receipt['seed']==p['seed']
    r=dict(batch=batch, pair=p['pair_id'], roster=p['roster'],seed=p['seed'],outcomes=[x['outcome'],y['outcome']],
           group=''.join({'win':'W','loss':'L','draw':'D'}[e['outcome']] for e in (x,y)),steps=[x['steps'],y['steps']])
    with (arms[0]/'events.jsonl').open('rb') as left,(arms[1]/'events.jsonl').open('rb') as right:
        for i in range(min(r['steps'])):
            (s,e),(t,f)=beat(left),beat(right); a,b=e['actions'],f['actions']; dec=receipt['decisions'][i]
            assert i+1==e['step']==f['step'] and list(b)==[dec['final_actions'][f'u{j}'] for j in range(len(b))]
            r.update(step=i,loop=e['game_loop'])
            if s != t:
                return dict(r,status='pre_mismatch',different_fields=[k for k in s['state'] if s['state'][k]!=t['state'].get(k)])
            if a!=b:
                z=features(s['state'],a,b)
                digest=hashlib.sha256(json.dumps(s,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                return dict(r,status='macro',pre_sha256=digest,mode=dec['mode'],candidate=dec['selected_candidate'],
                            parent_actions=[dec['parent_actions'][f'u{j}'] for j in range(len(a))],
                            parent_matches=all(dec['parent_actions'][f'u{j}']==v for j,v in enumerate(a)),
                            accepted=all(g['feedback']['actors'][c['actor']]['submission']=='accepted' for c in z['changes'] for g in (e,f)),**z)
            if e['feedback']['micro_submissions']!=f['feedback']['micro_submissions']:
                return dict(r,status='micro_feedback_first',micro=[g['feedback']['micro_submissions'] for g in (e,f)])
            if e['feedback']!=f['feedback']:
                return dict(r,status='feedback_mismatch')
    return dict(r,status='identical' if x['steps']==y['steps'] else 'unequal_length')


def signals(r):
    cs=r['changes']; moving=[c for c in cs if 2<=c['old']<=5 and 2<=c['new']<=5]
    return dict(opening=r['step']==0, no_legal=r['legal_actors']==0,
                no_visible=r['visible']==0, single=len(cs)==1, multi=len(cs)>1,
                moves_only=len(moving)==len(cs),
                nearer_enemy=any((c['enemy_delta'] or 0)<-0.5 for c in moving),
                farther_enemy=any((c['enemy_delta'] or 0)>0.5 for c in moving),
                breaks_attack=any(c['old']>=6>c['new'] for c in cs),
                starts_attack=any(c['new']>=6>c['old'] for c in cs),
                retarget=any(min(c['old'],c['new'])>=6 for c in cs),
                changes_hydra=any(c['type']=='Hydralisk' for c in cs),
                changes_bane=any(c['type']=='Baneling' for c in cs),
                changes_ling=any(c['type']=='Zergling' for c in cs),
                ling_leaves_hydra=any(c['type']=='Zergling' and (c['hydra_delta'] or 0)>0.5 for c in moving),
                hydra_nearer=any(c['type']=='Hydralisk' and (c['enemy_delta'] or 0)<-0.5 for c in moving))


def table(rows):
    return dict(sorted(Counter(r['group'] for r in rows).items()))


def summarize(rows):
    m=[r for r in rows if r['status']=='macro']; strata=sorted({(r['batch'],r['roster']) for r in m})
    output={}
    for key in signals(m[0]):
        hit=[r for r in m if signals(r)[key]]
        # Mantel-Haenszel common odds ratio: loss->win vs win->loss, among discordant pairs only.
        num=den=informative=0
        for b,t in strata:
            both=[r for r in m if (r['batch'],r['roster'])==(b,t) and r['group'] in ('LW','WL')]
            yes=Counter(r['group'] for r in both if signals(r)[key]); no=Counter(r['group'] for r in both if not signals(r)[key]); n=len(both)
            if n:
                ad=yes['LW']*no['WL']; bc=yes['WL']*no['LW']; num+=ad/n; den+=bc/n; informative+=bool(ad+bc)
        output[key]=dict(total=table(hit),by_batch={b:table([r for r in hit if r['batch']==b]) for b in BATCHES},
                         within_batch_roster_flip_OR=num/den if den else None, informative_strata=informative)
    clusters=defaultdict(list)
    for r in m:
        clusters[(r['pre_sha256'],tuple(map(tuple,r['actions'])))].append(r)
    opposite=[dict(pre_sha256=k[0],outcomes=table(rs),pairs=[(r['batch'],r['pair']) for r in rs])
              for k,rs in clusters.items() if {'WL','LW'}<=set(table(rs))]
    return dict(group_definition='First letter baseline, second letter tree. W win; L literal loss; D draw.',
                prefix_definition='Start at step 0; compare full parsed payload and every prior submitted vector and complete feedback incl micro_submissions. No claim about hidden engine state.',
                or_definition='Exploratory Mantel-Haenszel odds ratio of LW versus WL with signal present versus absent, stratified by batch and ally roster, literal flips among macro pairs only. Not causal.',
                distinct_prestates=len({r['pre_sha256'] for r in m}),opposite_flips_with_identical_pre_and_vectors=opposite,
                distinct_pre_and_vector_pairs=len(clusters),parent_mismatch=[(r['batch'],r['pair']) for r in m if not r['parent_matches']],
                status_counts=dict(Counter(r['status'] for r in rows)),all=table(rows),macro=table(m),by_batch={b:table([r for r in rows if r['batch']==b]) for b in BATCHES},
                signals_among_macro=output,by_batch_roster={b:{t:table([r for r in rows if r['batch']==b and r['roster']==t]) for k,t in strata if k==b} for b in BATCHES})


def main():
    rows=[]
    for batch in BATCHES:
        ps=load(RUNS/batch/'design.json')['pairs']; assert len(ps)==len({p['pair_id'] for p in ps})==240
        rows.extend(pair(batch,p) for p in ps)
    assert len({r['seed'] for r in rows})==720
    OUT.mkdir(exist_ok=True)
    (OUT/'pairs.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    (OUT/'summary.json').write_text(json.dumps(summarize(rows),sort_keys=True,indent=2)+'\n')
    for b in BATCHES:
        print(b,Counter((r['status'],r['group']) for r in rows if r['batch']==b))


if __name__=='__main__':
    main()
