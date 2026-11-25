"""Fit on completed development pairs, then score one sticky choice per archived test game. No SC2 launch."""
import argparse
import json
from pathlib import Path

from decision_tree.scripts.linux.eval_zerg_swarm import compare, ev
from decision_tree.scripts.linux.fit_zerg_route_gate import fit
from decision_tree.zerg.policy_route_gate import anchor_features, choose_route


def load(folder):
    result = compare(folder)
    v2, swarm = (result['policies'][key] for key in ('v2', 'swarm'))
    hashes = {k: v for k, v in v2.items() if k.endswith('sha256') or k == 'execution_context_hash'}
    hashes['radical_swarm'] = swarm['controller_sha256']
    rows = []
    for item in result['paired_outcomes']:
        diff = item['first_order_difference']
        row = dict(scenario_sha256=item['scenario_sha256'], seed=item['seed'], v2=item['v2'], alternative=item['swarm'], anchor=None,
                   prefix_verified=False, micro_verified=False, exclusion_reason=diff['status'])
        if diff['status'] == 'macro':
            with (folder/item['pair_id']/'v2/runner/events.jsonl').open() as f:
                payload = next(e['payload'] for line in f if line.startswith('{"event": "skill_observation",')
                               and (e := json.loads(line))['step'] == diff['step'])
            a,b = ({'actions':diff[k]} for k in ('v2','swarm'))
            row.update(anchor=dict(step=diff['step'],parent_plan=a,alt_plan=b,features=anchor_features(payload,a,b)),
                       prefix_verified=True,micro_verified=True)
        rows.append(row)
    return hashes, rows


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--train',type=Path,nargs='+',required=True)
    cli.add_argument('--test',type=Path,required=True);cli.add_argument('--out',type=Path,required=True)
    args=cli.parse_args()
    if args.out.exists():cli.error('--out must be new')
    hashes=None; dev=[]
    for p in args.train:
        h,rs=load(p)
        if hashes is not None and h!=hashes:raise ValueError('development policies differ')
        hashes=h;dev.extend(rs)
    gate,diagnostics=fit(dev,hashes)
    h,test=load(args.test)
    if h!=hashes or any({r[k] for r in dev} & {r[k] for r in test} for k in ('seed','scenario_sha256')):
        raise ValueError('test overlap or policy mismatch')
    decisions=[]
    for r in test:
        if not r['prefix_verified']:raise ValueError('cannot score unverified test divergence')
        choice=choose_route(gate,r['anchor']['features'])
        decisions.append(dict(scenario=r['scenario_sha256'],choice='swarm' if choice=='alternative' else 'v2',
                              outcome=r[choice]['outcome'],baseline=r['v2']['outcome']))
    report=dict(mode='offline paired replay, one sticky choice at first macro divergence',train_pairs=len(dev),
                test_pairs=len(test),tree=gate['tree'],swarm_selections=sum(r['choice']=='swarm' for r in decisions),
                v2_wins=sum(r['baseline']=='win' for r in decisions),router_wins=sum(r['outcome']=='win' for r in decisions),
                gained=sum(r['baseline']!='win' and r['outcome']=='win' for r in decisions),
                lost=sum(r['baseline']=='win' and r['outcome']!='win' for r in decisions))
    args.out.mkdir(parents=True,exist_ok=False)
    for name,obj in [('gate',dict(kind='archived_v2_swarm_gate',gate=gate)),('fit_report',diagnostics),('replay',report),('decisions',decisions)]:
        ev.write_json(args.out/(name+'.json'),obj)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
