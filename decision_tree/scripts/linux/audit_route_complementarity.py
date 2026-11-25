"""Historical policy-library ceilings. Post-outcome choices are not routing accuracy."""
from collections import Counter
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parents[3] / 'runs'


def index(name, arm):
    root = RUNS/name
    design = json.loads((root/'design.json').read_text())
    report = json.loads((root/'report.json').read_text())
    by_pair = {r['pair_id']: r for r in report['paired_outcomes']}
    table = {}
    for p in design['pairs']:
        row = by_pair[p['pair_id']]
        ep, = json.loads((root/p['pair_id']/arm/'runner/episodes.json').read_text())
        assert ep['scenario_sha256']==p['scenario_sha256'] and ep['outcome']==row[arm]['outcome']
        key=(ep['scenario_sha256'],ep['initial_state_sha256'],p['seed'])
        assert key not in table
        table[key]=dict(win=ep['outcome']=='win',severe=ep['outcome']=='loss' and ep['info']['dead_enemies']<=5)
    return table


def compare(parent, panels):
    k=set(parent).intersection(*panels.values()); out={}
    for name,alt in panels.items():
        c=Counter()
        for key in k:
            p,a=parent[key],alt[key]
            c['parent_wins']+=p['win']; c['alt_wins']+=a['win']; c['opportunity']+=not p['win'] and a['win']
            c['risk']+=p['win'] and not a['win']; c['parent_severe']+=p['severe']; c['rescued_severe']+=p['severe'] and a['win']
        out[name]=dict(c)
    return dict(common=len(k),by_arm=out,posthoc_oracle_wins=sum(parent[key]['win'] or any(t[key]['win'] for t in panels.values()) for key in k))


if __name__=='__main__':
    history='zerg_policy_eval_6x40x2_20261124_164422'; old='zerg_v1_v2_regression_20261124_185519'
    new='zerg_v2_v3_replay_20261124_193542'; fresh='zerg_v1_v2_fresh_20261124_190338'
    print(json.dumps(dict(regression=compare(index(old,'v2'),dict(hand=index(history,'baseline'),v1=index(old,'v1'))),
        random=compare(index(new,'v2'),dict(old_v3=index(new,'v3'),previous_run_v2=index(fresh,'v2')))),indent=2))
