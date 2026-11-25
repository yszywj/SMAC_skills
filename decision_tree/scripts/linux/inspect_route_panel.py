"""Separate observed controller complementarity from learned routing on a paired panel."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(rows):
    counts=Counter(); cohorts={k:Counter() for k in ('observed_disagreement','no_verified_disagreement')}
    for r in rows:
        a,b=r['v2'],r['alternative'];stat=cohorts['observed_disagreement' if r.get('anchor') else 'no_verified_disagreement']
        aw,bw=(x['outcome']=='win' for x in (a,b))
        severe=lambda x:x['outcome']=='loss' and x['enemy_elimination_fraction']<=.5
        for obj in (counts,stat):
            for key,value in dict(pairs=1,default_win=aw,alternative_win=bw,ideal_posthoc_win=aw or bw,
                                  rescued=not aw and bw,lost=aw and not bw,default_severe=severe(a),
                                  alternative_severe=severe(b),severe_to_win=severe(a) and bw,
                                  severe_to_nonsevere=severe(a) and not severe(b)).items():obj[key]+=value
    return dict(all_pairs=counts,cohorts=cohorts,
                note='ideal_posthoc_win uses outcomes, not a deployable tree; no verified disagreement is never a switch label')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',required=True,type=Path)
    r=json.loads((ap.parse_args().run/'report.json').read_text());print(json.dumps(summarize(r['paired_outcomes']),indent=2))
