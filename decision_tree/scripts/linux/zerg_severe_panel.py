"""Describe all three historical panels, and optionally join a V1/V2 replay by scenario.

Outcome and initial-layout groupings are diagnostics, never inputs to a policy.
"""
import argparse
from collections import Counter, defaultdict
import json
from math import isclose
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT/'runs'
BATCHES = ['zerg_policy_eval_20261124_162949','zerg_policy_eval_6x40x2_20261124_164422',
           'zerg_policy_eval_random_20261124_165149']


def read(path):
    return json.loads(path.read_text())


def outcome(folder):
    ep, = read(folder/'runner/episodes.json')
    return dict(result=ep['outcome'], enemy_left=10-ep['info']['dead_enemies'],
                ally_left=10-ep['info']['dead_allies'], reward=ep['reward'], initial=ep['initial_state_sha256'],
                scenario=ep['scenario_sha256'])


def partition(rows):
    counts = Counter((a['result'],b['result']) for a,b,_ in rows)
    severe = lambda a:a['result']=='loss' and a['enemy_left']>=5
    win = lambda a:a['result']=='win'
    bs = [b['result']=='loss' and a['result']=='loss' for a,b,_ in rows]
    return dict(pairs=len(rows),hand_wins=sum(win(a) for a,b,m in rows),tree_wins=sum(win(b) for a,b,m in rows),
        both_won=counts['win','win'],both_lost=sum(bs),both_severe=sum(severe(a) and severe(b) for a,b,_ in rows),
        hand_severe=sum(severe(a) for a,_,_ in rows),tree_severe=sum(severe(b) for _,b,_ in rows),
        loss_to_win=counts['loss','win'],win_to_loss=counts['win','loss'],
        nonwin_to_win=sum(not win(a) and win(b) for a,b,_ in rows),win_to_nonwin=sum(win(a) and not win(b) for a,b,_ in rows),
        transitions={f'{a}->{b}':n for (a,b),n in sorted(counts.items())},
        mean_return_delta=fmean(b['reward']-a['reward'] for a,b,_ in rows))


def panel():
    allrows, by_panel, by_geometry = [], {}, defaultdict(list)
    for name in BATCHES:
        run = RUNS/name; report=read(run/'report.json'); design=read(run/'design.json'); rows=[]
        if len(report['paired_outcomes'])!=240 or len(design['pairs'])!=240: raise ValueError(name)
        for p,score in zip(design['pairs'],report['paired_outcomes']):
            folder=run/p['pair_id']; a,b=(outcome(folder/arm) for arm in ('baseline','tree'))
            if not (a['scenario']==b['scenario']==p['scenario_sha256'] and a['initial']==b['initial'] and
                    score['pair_id']==p['pair_id'] and a['result']==score['baseline']['outcome'] and b['result']==score['tree']['outcome']):
                raise ValueError(f'mismatched pair {folder}')
            manifest=read(folder/'manifest.json')['episodes'][0]['episode_config']
            allies=manifest['ally_start_positions']['item']; enemies=manifest['enemy_start_positions']['item']
            geom='x_reflection' if all(isclose(ay,ey,abs_tol=1e-6) and isclose(ax+ex,32,abs_tol=1e-6)
                                          for (ax,ay),(ex,ey) in zip(allies,enemies)) else 'other_start_geometry'
            row=(a,b,dict(batch=name,seed=p['seed'],pair_id=p['pair_id'],roster=p['roster'],geometry=geom))
            rows.append(row);by_geometry[geom].append(row)
        allrows.extend(rows); by_panel[name]=partition(rows)
    if len({a['scenario'] for a,_,_ in allrows})!=720: raise ValueError('overlap in historical panels')
    return allrows, dict(all=partition(allrows),by_panel=by_panel,by_geometry={k:partition(v) for k,v in by_geometry.items()})


def join_replay(rows, run):
    """Only compare to archived labels when the old V1 reproduces the archived endpoint."""
    hist={a['scenario']:(a,b,meta) for a,b,meta in rows}
    design, score=read(run/'design.json'),read(run/'report.json')
    if len(score['paired_outcomes'])!=len(design['pairs']):raise ValueError('incomplete experiment')
    events=[];misses=[];no_overlap=0;matched=set()
    for p in design['pairs']:
        key=p['scenario_sha256']
        if key not in hist:no_overlap+=1;continue
        if key in matched:raise ValueError('repeated historical scenario')
        matched.add(key)
        hand,old,meta=hist[key];v1,v2=(outcome(run/p['pair_id']/arm) for arm in ('v1','v2'))
        if any(v1[k]!=v2[k] or v1[k]!=old[k] for k in ('initial','scenario')):raise ValueError('unmatched initial state')
        if (v1['result']!=old['result'] or v1['enemy_left']!=old['enemy_left'] or
            v1['ally_left']!=old['ally_left'] or not isclose(v1['reward'],old['reward'],abs_tol=1e-8,rel_tol=0)):
            misses.append(meta);continue
        events.append((hand,v1,v2))
    def groups(items):
        return dict(pairs=len(items),v2_wins=sum(c['result']=='win' for _,b,c in items),
                    v2_still_severe=sum(c['result']=='loss' and c['enemy_left']>=5 for _,b,c in items),
                    v1_severe=sum(b['result']=='loss' and b['enemy_left']>=5 for _,b,c in items),
                    mean_v2_minus_v1_return=fmean(c['reward']-b['reward'] for _,b,c in items) if items else None)
    return dict(scope='historical endpoints reproduced, not an independent holdout',planned=len(design['pairs']),
        overlapping=len(matched),nonoverlap=no_overlap,v1_changed_endpoint=misses,all_reproduced=groups(events),
        historical_joint_losses=groups([x for x in events if x[0]['result']==x[1]['result']=='loss']),
        historical_joint_severe=groups([x for x in events if x[0]['result']==x[1]['result']=='loss' and
                                            min(x[0]['enemy_left'],x[1]['enemy_left'])>=5]))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--out',type=Path)
    cli.add_argument('--join-replay',type=Path,help='completed V1/V2 replay; annotate prior joint loss strata')
    args=cli.parse_args();rows,report=panel()
    if args.join_replay:report['replay']=join_replay(rows,args.join_replay)
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('x') as f:f.write(text)
    else:print(text)
