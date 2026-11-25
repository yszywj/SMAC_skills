"""Compare actual adaptive/fixed handoffs to each other and to the archived online router."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'runs/zerg_router_old900_live_20261125_105505'


def contact(path):
    first=None
    with path.open() as stream:
        for line in stream:
            if '"skill_observation"' not in line:continue
            e=json.loads(line)
            if e['event']!='skill_observation':continue
            s=e['payload']['state']; loop=s['game_loop']
            targets=Counter(t for rel in s['relations'].values() for t,r in rel.items() if r['smac_attack_available'])
            if targets and first is None:first=loop
            if first is not None and loop==first+32:return max(targets.values(),default=0)
    return None


def evidence(args):
    f,a,pid=args
    return tuple(contact(root/pid/arm/'runner/events.jsonl') for root,arm in
                 [(BASE,'online_router'),(f,'squad_rendezvous'),(a,'squad_rendezvous')])


def effect(rows,left,right):
    t=Counter((r[left],r[right]) for r in rows)
    return dict(pairs=len(rows),net_wins=sum(n*((b=='win')-(a=='win')) for (a,b),n in t.items()),
                gained=sum(n for (a,b),n in t.items() if a!='win' and b=='win'),
                lost=sum(n for (a,b),n in t.items() if a=='win' and b!='win'),
                transitions={a+'_to_'+b:n for (a,b),n in sorted(t.items())})


def compare(fixed,adaptive):
    read=lambda path:json.loads(path.read_text())
    reports=[read(d/'report.json') for d in (fixed,adaptive)]
    designs=[read(d/'design.json') for d in (fixed,adaptive)]
    if [r['mode'] for r in reports]!=['fixed','adaptive']:raise ValueError('expected fixed, adaptive runs')
    if designs[0]['pairs']!=designs[1]['pairs'] or designs[0]['sources']!=designs[1]['sources']:
        raise ValueError('different panel or different worker code; not a timing ablation')
    table=lambda report:{r['pair_id']:r for r in report['paired_outcomes']}
    f,a=map(table,reports); orig=table(read(BASE/'report.json'))
    ids=[p['pair_id'] for p in designs[0]['pairs']]
    if not ids or len(ids)!=len(set(ids)) or set(ids)!=f.keys() or f.keys()!=a.keys():raise ValueError('incomplete runs')
    with ProcessPoolExecutor(6) as pool:
        contacts=list(pool.map(evidence,[(fixed,adaptive,i) for i in ids],chunksize=6))
    groups=defaultdict(list)
    for pid,counts in zip(ids,contacts):
        old=orig[pid]; fr,ar=f[pid],a[pid]
        for key in ('scenario_sha256','initial','seed','manifest_sha256'):
            if old[key]!=fr[key] or old[key]!=ar[key]:raise ValueError(f'{pid}: inconsistent {key}')
        row={'parent':old['online_outcome'],'v2':old['archived_v2'],'adaptive':ar['outcome'],'fixed':fr['outcome'],'contact':counts}
        if fr['strata']!=ar['strata']:raise ValueError(f'{pid}: different historical strata')
        for group in set(fr['strata'])|{'all'}:groups[group].append(row)
    out={}
    for k, rows in groups.items():
        matched=[r['contact'] for r in rows if None not in r['contact']]
        out[k]=dict(parent_to_fixed=effect(rows,'parent','fixed'),parent_to_adaptive=effect(rows,'parent','adaptive'),
            archived_v2_to_adaptive=effect(rows,'v2','adaptive'),
            fixed_to_adaptive=effect(rows,'fixed','adaptive'),contact4_common=dict(matched=len(matched),total=len(rows),
            means=dict(zip(('parent','fixed','adaptive'),map(fmean,zip(*matched)))) if matched else {}))
    return out


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--fixed',required=True,type=Path);cli.add_argument('--adaptive',required=True,type=Path)
    args=cli.parse_args(); print(json.dumps(compare(args.fixed,args.adaptive),indent=2,allow_nan=False))
