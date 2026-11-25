"""New paired v2/alternative trials, or sticky first-divergence gate regression."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from itertools import zip_longest
from pathlib import Path
import secrets
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from decision_tree.scripts.linux import eval_zerg_v2_v3 as old

ev, trees = old.ev, old.trees
ALT = 'decision_tree/zerg/policy_route_alt.py'
GATE = 'decision_tree/zerg/policy_route_gate.py'
SCHEMA = 'zerg_route_pair_v1'


def prepare(out, workers, batch, ids, dry, gate_path):
    ev.require(sys.executable == ev.RSI_PYTHON, f'use {ev.RSI_PYTHON}')
    batch = old.ALIASES.get(batch, batch)
    ev.require(not out.exists() and 1 <= workers <= 60 and (not batch or batch in old.ARCHIVES), 'invalid output, workers or batch')
    ev.require(ids is None or batch and 0 < len(ids) == len(set(ids)) <= 240 and all(1 <= i <= 240 for i in ids), 'invalid pilot ids')
    from decision_tree.zerg.policy_route_alt import choose
    from decision_tree.zerg.policy_route_gate import choose_route, anchor_features
    original = ev.reference()
    sources = ev.frozen_sources(original)
    trial = secrets.token_hex(16)
    ctrl = old.policies(sources, ev.model_contract(ev.MODEL, original), trial)['v2']
    identity = dict(model=ctrl['model_sha256'], parent=ctrl['source_hash'], v2=ctrl['selector_sha256'],
                    opening=ctrl['opening_sha256'], alternative=sources[ALT], gate_code=sources[GATE],
                    execution_context=ctrl['execution_context_hash'], bridge=sources['sc2_skill_adapter/bridge/zerg_distill.py'])
    ctrl = dict(ctrl, policy_sha256=identity)
    arm = 'alternative' if gate_path is None else 'route'
    other = dict(ctrl, kind='zerg_distill_'+arm, policy_version=arm, selector_sha256=sources[ALT])
    if gate_path is not None:
        raw = gate_path.read_bytes()
        gate = json.loads(raw)
        ev.require(gate['schema'] == 'zerg_route_gate_v1' and gate['policy_sha256'] == identity, 'gate policy fingerprint mismatch')
        other['gate_sha256'] = hashlib.sha256(raw).hexdigest()
    design = trees.freeze_panel(out,workers,None,dry,batch,None,ids,{'v2':ctrl,arm:other},sources,
            *ev.historical_config(),old.ARCHIVES,SCHEMA,schema=SCHEMA,policy_sha256=identity,trial_id=trial)
    if gate_path is not None:
        (out/'gate.json').write_bytes(raw)
    return design


def payload_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def trace(work, arm, policy, gate):
    from decision_tree.zerg.policy_route_gate import choose_route, anchor_features
    runner = work/'runner'
    rows = ev.read_json(runner/'distill/episode-0001.json')['decisions']
    events = defaultdict(dict)
    micro = defaultdict(list)
    with (runner/'events.jsonl').open() as stream:
        for line in stream:
            e = json.loads(line)
            k = e['event']
            if k.startswith('micro_'):
                micro[e['macro_step']].append(e)
            elif k in ('skill_observation','skill_step','step'):
                i = e['step'] - (k == 'step')
                ev.require(i not in events[k], 'duplicate event')
                events[k][i] = e
    ev.require(all(set(events[k]) == set(range(len(rows))) for k in ('skill_observation','skill_step','step')), 'incomplete prefix evidence')
    choice = anchor = None
    prefix = []
    with (work/'shadow/shadow.jsonl').open() as shadow:
        for i,(r,line) in enumerate(zip_longest(rows,shadow)):
            ev.require(r is not None and line is not None, 'incomplete shadow/decision join')
            cards = {c['candidate_id']:c for c in json.loads(line)['candidates']}
            payload = events['skill_observation'][i]['payload']
            native = events['skill_step'][i]
            feedback = events['step'][i]['feedback']
            parent, alt = r['parent_plan'], r['alt_plan']
            for plan in (parent,alt):
                card = cards[plan['candidate_id']]
                ev.require(card['legal'] and plan['candidate_id'] in r['scores'] and
                           card['proposed_actions'] == {f'u{j}':a for j,a in enumerate(plan['actions'])}, 'invalid candidate evidence')
            first = choice is None and parent != alt
            if first:
                ev.require(parent['actions'] != alt['actions'], 'same actions, different plans')
                features = anchor_features(payload,parent,alt)
                anchor = dict(step=i,game_loop=r['game_loop'],features=features,parent_plan=parent,alt_plan=alt,
                              payload_sha256=payload_hash(payload),policy_sha256=policy['policy_sha256'])
                choice = choose_route(gate, features) if arm == 'route' else arm
            selected = alt if choice == 'alternative' else parent
            ev.require(r['route_choice'] == choice and r['route_anchor'] == (anchor if first else None) and
                       r['selected_candidate'] == selected['candidate_id'] and r['trial_id'] == policy['trial_id'] and
                       r['opening_sha256'] == policy['opening_sha256'] and r['applied'] is r['offered'] and
                       selected['actions'] == events['step'][i]['actions'] and native['actions'] == r['final_actions'] and
                       selected['plan_ids'] == r['plan_ids'] == [p['id'] for p in native['chosen_plans']], 'wrong route/native choice')
            closes = [e for e in micro[i] if e['event']=='micro_window_closed']
            closed, = closes
            records = closed['records']
            subs = [s for s in records if s['status']=='submitted']
            ev.require(closed['authorizations'] == len(records) and closed['issued_loop']==r['game_loop'] and
                       closed['interior_checks']>=sum(e['event']=='micro_loop_check' for e in micro[i]) and
                       all(s['macro_action_id'] == r['final_actions'][s['actor']] for s in records) and
                       sorted(subs,key=lambda s:s['actor']) == sorted(feedback.get('micro_submissions',[]),key=lambda s:s['actor']),
                       'micro window mismatch')
            if choice is None:
                prefix.append(payload_hash([payload,parent,native['units'],native['chosen_plans'],micro[i],feedback]))
    return prefix,anchor,choice


def audit_panel(out):
    out = Path(out)
    d = ev.read_json(out/'design.json')
    ev.require(d['schema']==d['kind']==SCHEMA and not d['dry_run'] and d['arms'] in (['v2','alternative'],['v2','route']) and
               ev.read_json(ev.HISTORY/f'{ev.digest(d)}.json')==d and ev.sha(out/'model.json')==ev.MODEL_HASH, 'not a frozen played route panel')
    _,arm = d['arms']
    gate = None
    if arm == 'route':
        ev.require(ev.sha(out/'gate.json')==d['policies'][arm]['gate_sha256'], 'changed gate')
        gate=ev.read_json(out/'gate.json')
    ids,batch = d['pilot_ids'],d['replay_batch']
    ev.require(len(d['pairs'])==(len(ids) if ids else 240), 'incomplete panel')
    if batch:
        archived = old.archive(batch)['pairs']
        ev.require(d['pairs']==([archived[i-1] for i in ids] if ids else archived), 'changed replay panel')
    rows=[]
    original=ev.reference()
    for i,p in enumerate(d['pairs'],1):
        m=ev.read_json(out/p['manifest'])
        trees.validate_pair(p,m,ids[i-1] if ids else i,original['env_args'])
        result=dict(p)
        traces=[]
        for a in d['arms']:
            policy=d['policies'][a]
            result[a]=ev.audit_arm(out,p,a,m,original,d,policy)
            traces.append(trace(out/p['pair_id']/a,a,policy,gate))
        left,right=traces
        ev.require(result['v2']['initial']==result[arm]['initial'], 'different paired initial state')
        if batch:
            ep,=ev.read_json(ev.RUNS/batch/p['pair_id']/'v2/runner/episodes.json')
            ev.require(ep['initial_state_sha256']==result['v2']['initial'], 'different archived initial state')
        verified = left[:2] == right[:2]
        if verified and left[1] is None:
            verified = all(result['v2'][k] == result[arm][k] for k in ('outcome','steps','reward','enemy_elimination_fraction'))
        result.update(anchor=left[1] if verified else None, prefix_verified=verified,micro_verified=verified,
                      route_choice=right[2],exclusion_reason=None if verified else 'paired prefix or anchor mismatch')
        rows.append(result)
    transitions=Counter(f"{r['v2']['outcome']}_to_{r[arm]['outcome']}" for r in rows)
    summary={a:dict(outcomes=dict(Counter(r[a]['outcome'] for r in rows)),
                    severe_losses=sum(r[a]['outcome']=='loss' and r[a]['enemy_elimination_fraction']<=.5 for r in rows),
                    mean_return=sum(r[a]['reward'] for r in rows)/len(rows)) for a in d['arms']}
    return dict(schema=SCHEMA,policy_sha256=d['policy_sha256'],arms=d['arms'],pairs=len(rows),paired_outcomes=rows,
                trial_id=d['trial_id'],replay_batch=batch,policies=d['policies'],arm_summary=summary,
                transitions=dict(transitions),verified_pairs=sum(r['prefix_verified'] for r in rows),
                anchors=sum(r['anchor'] is not None for r in rows))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--out',type=Path)
    mode.add_argument('--compare',type=Path)
    ap.add_argument('--route-model',type=Path,help='gate JSON; omit to collect actual v2 vs alternative pairs')
    ap.add_argument('--workers',type=int)
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--replay-batch',choices=(*old.ALIASES,*old.ARCHIVES))
    ap.add_argument('--pilot-ids',type=int,nargs='+')
    args=ap.parse_args(argv)
    if args.compare:
        if any((args.route_model,args.workers is not None,args.dry_run,args.replay_batch,args.pilot_ids)):
            ap.error('--compare takes no launch options')
        print(json.dumps(audit_panel(args.compare.resolve()),indent=2,allow_nan=False))
        return
    out=args.out.resolve()
    d=prepare(out,60 if args.workers is None else args.workers,args.replay_batch,args.pilot_ids,args.dry_run,args.route_model)
    print(f"{out}: {len(d['pairs'])} pairs; arms={d['arms']}; dry_run={args.dry_run}",flush=True)
    if not args.dry_run:
        with ThreadPoolExecutor(max_workers=d['workers']) as pool:
            jobs=[pool.submit(ev.play,out,p,a,d['policies'][a]) for p in d['pairs'] for a in d['arms']]
            try:
                for job in as_completed(jobs):
                    job.result()
            except BaseException:
                for job in jobs:
                    job.cancel()
                raise
        report=audit_panel(out)
        ev.write_json(out/'report.json',report)
        print(json.dumps({k:v for k,v in report.items() if k!='paired_outcomes'},indent=2))


if __name__=='__main__':
    main()
