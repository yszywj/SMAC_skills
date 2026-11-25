"""Read a completed V2/V3 comparison, separate research regressions from confirmation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from decision_tree.scripts.linux.evaluate_zerg_v2_gate import assess


def summarize(report, parent_report=None, resamples=10000):
    rows=report['paired_outcomes']
    # Reuse the exact paired statistical gate from the prior iteration, under its
    # old local argument names. Do not pool an adaptive development panel with fresh games.
    if len({r['scenario_sha256'] for r in rows})!=len(rows) or len(rows)!=report['pairs']:
        raise ValueError('duplicate or incomplete paired scenarios')
    if not rows or any(r['v2']['initial']!=r['v3']['initial'] for r in rows):
        raise ValueError('missing or unmatched pair')
    fresh=report['replay_batch'] is None
    metrics=assess([{'v1':r['v2'],'v2':r['v3']} for r in rows],fresh=fresh,resamples=resamples)
    metrics['severe_parent_v2']=metrics.pop('severe_v1');metrics['severe_candidate_v3']=metrics.pop('severe_v2')
    transitions=Counter((r['v2']['outcome'],r['v3']['outcome']) for r in rows)
    metrics['transitions']={f'{a}_to_{b}':n for (a,b),n in sorted(transitions.items())}
    if parent_report is not None:
        if fresh: raise ValueError('no historical join for a fresh panel')
        hist={r['scenario_sha256']:r for r in parent_report['paired_outcomes']}; excluded=[]; cohorts={}
        for r in rows:
            old=hist[r['scenario_sha256']]; a,b,c=old['v1'],old['v2'],r['v3'];now=r['v2']
            if now['initial']!=b['initial']:raise ValueError('historical initial state differs')
            if now['outcome']!=b['outcome'] or abs(now['reward']-b['reward'])>1e-8 or now['enemy_elimination_fraction']!=b['enemy_elimination_fraction']:
                excluded.append(r['pair_id']);continue
            ll=a['outcome']==b['outcome']=='loss'
            severe=ll and max(a['enemy_elimination_fraction'],b['enemy_elimination_fraction'])<=.5
            for group,eligible in [('all_reproduced',True),('historic_both_lost',ll),('historic_both_severe',severe),
                    ('old_v2_preserved_win',b['outcome']=='win')]:
                if eligible:
                    totals=cohorts.setdefault(group,Counter());totals['pairs']+=1;totals['v3_win']+=int(c['outcome']=='win')
                    totals['v3_loss']+=int(c['outcome']=='loss');totals['v3_severe']+=int(c['outcome']=='loss' and c['enemy_elimination_fraction']<=.5)
        metrics['historical_cohorts']=cohorts;metrics['historical_v2_endpoints_not_reproduced']=excluded
    return metrics


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--run',type=Path,required=True)
    run=cli.parse_args().run
    report=json.loads((run/'report.json').read_text()); prior=report.get('replay_batch')
    historical=json.loads((ROOT/'runs'/prior/'report.json').read_text()) if prior else None
    print(json.dumps(summarize(report,historical),indent=2,allow_nan=False))
