"""Measure paired +4/+8 windows after the first accepted, same-observation R8 macro change."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from statistics import fmean

PARENT = Path(__file__).resolve().parents[3]/'runs/zerg_R1_R6_R7_random1000_20261125_155323'


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def frames(path):
    obs = relay = None
    with path.open() as f:
        for line in f:
            if line.startswith('{"event": "skill_observation",'):
                obs = json.loads(line)
            elif line.startswith('{"event": "combat_relay_decision",'):
                relay = json.loads(line)
            elif line.startswith('{"event": "step",'):
                e = json.loads(line)
                if obs is None or obs['step']+1 != e['step']:raise ValueError(f'{path}: missing observation')
                s = obs['payload']['state'];fb = e['feedback'];shots=Counter();capable=0
                for u,unit in s['allies'].items():
                    if not unit['alive']:continue
                    rels=s['relations'].get(u,{})
                    capable+=any(r['smac_attack_available'] for r in rels.values())
                    act=e['actions'][int(u[1:])];target=f'e{act-6}'
                    if act>=6 and rels.get(target,{}).get('in_weapon_range') and fb['actors'][u]['submission']=='accepted': shots[target]+=1
                yield dict(loop=e['game_loop'],obs=digest(obs['payload']),actions=e['actions'],relay=relay,accepted=[fb['actors'][f'u{i}']['submission']=='accepted' for i in range(len(e['actions']))],
                           fb=digest(fb),metrics=dict(capable=capable,shared=max(shots.values(),default=0),alive=sum(u['alive'] for u in s['allies'].values()),
                           hp=sum(u['health'] for u in s['allies'].values() if u['alive'])))
                obs=relay=None


def match(a,b):
    from itertools import zip_longest
    changed=None; windows={}
    for i,(pa,ch) in enumerate(zip_longest(a,b)):
        if pa is None or ch is None:return ('different_length' if changed is None else 'macro',windows)
        if pa['loop']!=ch['loop']:return 'different_clock',windows
        if changed is None:
            if pa['obs']!=ch['obs']:return 'different_pre_observation',{}
            ids=[i for i,(j,k) in enumerate(zip(pa['actions'],ch['actions'])) if j!=k]
            if not ids:
                if pa['fb']!=ch['fb']:return 'different_execution_feedback',{}
                continue
            proposal=ch.get('relay') or {}
            if not (proposal.get('owner') in ('menu','custom') and proposal.get('r6_actions')==pa['actions'] and proposal.get('actions')==ch['actions']):
                return 'different_macro_not_attributable_to_relay',{}
            if not all(pa['accepted'][i] and ch['accepted'][i] for i in ids):return 'rejected_macro',{}
            changed=i
        if i-changed in (4,8):
            windows[str(i-changed)]={k:ch['metrics'][k]-v for k,v in pa['metrics'].items()}
        if i-changed==8:break
    return ('identical' if changed is None else 'macro',windows)


def one(task):
    mode,pid,path,parent_events,groups=task
    status,window=match(frames(parent_events),frames(path))
    return mode,pid,groups,status,window


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',required=True,type=Path);args=ap.parse_args()
    run=args.run;design=json.loads((run/'design.json').read_text());summary=json.loads((run/'report.json').read_text())
    reference=json.loads((PARENT/'report.json').read_text());table={p['pair_id']:p for p in reference['paired_outcomes']}
    tasks=[]
    for p in design['pairs']:
        pid=p['pair_id'];r=table[pid];groups=['all']
        if p['scenario_sha256']!=r['scenario_sha256']:raise ValueError(f'not the same R6 replay panel: {pid}')
        if all(r[k]['outcome']=='loss' for k in ('R6','R7')):
            groups.append('common_loss')
            if max(r[k]['enemy_elimination_fraction'] for k in ('R6','R7'))<=.5:groups.append('common_severe')
            if min(r[k]['enemy_elimination_fraction'] for k in ('R6','R7'))>=.8:groups.append('common_close_loss')
        if r['R6']['outcome']=='win':groups.append('parent_win')
        for mode in ('menu','full'):
            path=run/pid/mode/'runner/events.jsonl'
            tasks.append((mode,pid,path,PARENT/pid/'R6/runner/events.jsonl',groups))
    del summary  # report must exist (all engine receipts audited) before measuring secondary endpoints
    by=defaultdict(list)
    with ProcessPoolExecutor(8) as pool:
        for mode,pid,groups,status,window in pool.map(one,tasks,chunksize=6):
            for group in groups:by[mode,group].append((status,window))
    out={}
    for (mode,group),rows in by.items():
        result={'statuses':Counter(s for s,_ in rows)}
        for h in ('4','8'):
            matched=[w[h] for s,w in rows if s=='macro' and h in w]
            result['plus'+h]={'matched':len(matched),'mean_R8_minus_R6':{k:fmean(e[k] for e in matched) for k in ('hp','alive','capable','shared')} if matched else {}}
        out.setdefault(mode,{})[group]=result
    print(json.dumps(out,indent=2,allow_nan=False))


if __name__=='__main__':main()
