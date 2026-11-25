"""Offline macro-observation diagnostic; no policy/environment imports. Run from repo root."""
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from statistics import mean, median

ROOT = Path('runs/zerg_R1_R6_R7_random1000_20261125_155323')
STAGES = ['contact', 'death', 'plus4', 'plus8']


def extract(p):
    pid = p['pair_id']
    config = json.loads((ROOT/p['manifest']).read_bytes())['episodes'][0]['episode_config']
    a, b = ([tuple(x) for x in config[k+'_start_positions']['item']] for k in ('ally', 'enemy'))
    teams = [sorted(config['team_gen'][k]) for k in ('ally_team', 'enemy_team')]
    assert teams[0] == teams[1]
    # Equal weight per distance family: rotation/reflection invariant, not unit-labelled.
    features = [sorted(min(math.dist(x,y) for y in b) for x in a),
                sorted(min(math.dist(x,y) for x in a) for y in b)]
    features += [sorted(math.dist(x,y) for x,y in combinations(side,2)) for side in (a,b)]
    frames = []
    with (ROOT/pid/'R6/runner/events.jsonl').open('rb') as f:
        for line in f:
            if line.startswith(b'{"event": "skill_observation"'):
                e=json.loads(line); s=e['payload']['state']; q=e['payload']['questions']; allies=s['allies']
                assert e['step']==len(frames) and s['step_mul']==8
                legal={k:{int(i) for i in q[k]['criteria'] if int(i)>=6} for k in allies if allies[k]['alive']}
                hp=sum(allies[k]['health'] for k in legal)
                if not frames: hp0=hp
                frames.append(dict(t=e['step'],loop=s['game_loop'],alive=len(legal),hp=round(hp,2),hp_pct=round(100*hp/hp0,3),can=sum(map(bool,legal.values()))))
            elif line.startswith(b'{"event": "step"'):
                e=json.loads(line); r=frames[-1]; counts=Counter(); accepted=0
                assert e['game_loop']==r['loop'] and e['step']==r['t']+1
                for k,acts in legal.items():
                    i=e['actions'][int(k[1:])]; fb=e['feedback']['actors'][k]
                    if i in acts and fb['submission']=='accepted':
                        accepted+=1; target='e'+str(i-6)
                        if s['relations'][k][target]['in_weapon_range'] is True: counts[target]+=1
                r.update(accepted=accepted,shared=max(counts.values(),default=0))
            elif line.startswith(b'{"event": "episode_completed"'):
                end=json.loads(line)
    assert len(frames)==end['steps'] and end['scenario_sha256']==p['scenario_sha256'] and end['outcome']==p['R6']['outcome'] and all('shared' in r for r in frames)
    c=next((r['t'] for r in frames if r['can']),None); d=next((r['t'] for r in frames if r['alive']<10),None)
    both=all(p[k]['outcome']=='loss' for k in ('R6','R7'))
    sev=both and all(p[k]['enemy_elimination_fraction']<=.5 for k in ('R6','R7'))
    # Full death-to-+8 windows only. Count established -> absent transitions, not absence throughout.
    breaks=None if d is None or d+8>=len(frames) else sum(frames[i-1]['shared']>=2>frames[i]['shared'] for i in range(d+1,d+9))
    return dict(id=pid,roster=p['roster'],geometry='center' if len(set(a))==1 else 'spread',features=features,
                group='severe' if sev else 'otherLL' if both else 'win',
                stages={k:frames[t] if t is not None and t<len(frames) else None for k,t in zip(STAGES,[c,d,None if d is None else d+4,None if d is None else d+8])},
                first_accepted=next((r['t'] for r in frames if r['accepted']),None),breaks=breaks,
                terminal=end['steps'],window=None if d is None else [r['shared'] for r in frames[d:d+9]])


def distance(r,w):
    return math.sqrt(mean(mean((x-y)**2 for x,y in zip(a,b)) for a,b in zip(r['features'],w['features'])))


def summary(rows):
    out={}
    for k in STAGES:
        rs=[r['stages'][k] for r in rows if r['stages'][k] is not None]; out[k]={'n':len(rs)}
        if rs: out[k].update({v:round(mean(s[v] for s in rs),3) for v in ('t','can','alive','hp','hp_pct','accepted','shared')})
    full=[r for r in rows if r['breaks'] is not None]; out['breaks']=[sum(r['breaks']>0 for r in full),len(full)]
    out['never_shared']=[sum(max(r['window'])<2 for r in full),len(full)]
    return out


if __name__=='__main__':
    ps=json.loads((ROOT/'report.json').read_bytes())['paired_outcomes']; ps=[p for p in ps if p['R6']['outcome']=='win' or all(p[k]['outcome']=='loss' for k in ('R6','R7'))]
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv)>1 else 8) as pool: rows=list(pool.map(extract,ps,chunksize=4))
    wins=[r for r in rows if r['group']=='win']; losses=[r for r in rows if r not in wins]; severe=[r for r in losses if r['group']=='severe']; assert (len(severe),len(losses),len(wins))==(81,701,283)
    pairs=[]; lookup={r['id']:r for r in rows}
    for r in losses:
        ds=sorted((distance(r,w),w['id']) for w in wins if (r['geometry'],r['roster'])==(w['geometry'],w['roster']))
        if ds and ds[0][0]<=2: pairs.append([r['id'],ds[0][1],ds[0][0]])
    stats={k:summary(v) for k,v in [('severe',severe),('LL',losses),('otherLL',[r for r in losses if r not in severe])]}
    for label,sub in [('matchedLL',pairs),('matchedSevere',[p for p in pairs if lookup[p[0]]['group']=='severe'])]:
        for i,k in enumerate(['loss','win']): stats[label+k]=summary([lookup[p[i]] for p in sub])
    (ROOT/'r6_contact_windows.json').write_text(json.dumps(dict(rows=rows,pairs=pairs,stats=stats),separators=(',',':'))+'\n'); print(json.dumps(stats))
