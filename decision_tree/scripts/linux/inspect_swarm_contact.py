"""Compare observable timing markers for two played whole-game policies, not causal labels."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from statistics import median


def markers(path):
    result={}; event_name='policy_observation' if path.name=='swarm' else 'skill_observation'
    with (path/'runner/events.jsonl').open() as f:
        for line in f:
            if '"event": "'+event_name+'"' not in line:continue
            e=json.loads(line)
            if e['event']!=event_name:continue
            s=e['payload']['state'];loop=s['game_loop'];units=s['allies'];alive=[u for u in units.values() if u['alive']]
            if s['visible_enemies']:result.setdefault('first_seen',loop)
            if len(alive)<len(units) or any(u['health']<u['max_health']-1e-6 for u in alive):result.setdefault('first_ally_damage',loop)
            if len(alive)<len(units):result.setdefault('first_ally_dead',loop)
            shooters=sum(any(r['smac_attack_available'] for r in rel.values()) for rel in s['relations'].values())
            if shooters and 'first_attack_option' not in result:
                result.update(first_attack_option=loop,legal_shooters_on_contact=shooters,alive_on_contact=len(alive))
            if shooters>=3:result.setdefault('three_legal_shooters',loop)
    return result


def one(pair):return {arm:markers(pair/arm) for arm in ('v2','swarm')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    r=json.loads((args.run/'report.json').read_text());jobs=[args.run/q['pair_id'] for q in r['paired_outcomes']]
    with ProcessPoolExecutor(6) as pool:rows=list(pool.map(one,jobs,chunksize=8))
    summary={};keys=set(k for r in rows for d in r.values() for k in d)
    for k in sorted(keys):
        valid=[r for r in rows if k in r['swarm'] and k in r['v2']]
        summary[k]={'observed_in_both':len(valid),'missing_in_v2':sum(k not in r['v2'] for r in rows),'missing_in_swarm':sum(k not in r['swarm'] for r in rows)}
        if valid:
            summary[k].update({arm:median(r[arm][k] for r in valid) for arm in ('swarm','v2')},
                              paired_median_delta=median(r['swarm'][k]-r['v2'][k] for r in valid))
    print(json.dumps({'units':'game loops or number of units; range availability is not damage','metrics':summary},indent=2))
