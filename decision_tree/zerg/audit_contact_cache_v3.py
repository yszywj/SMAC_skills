"""Offline prefix replay only; PYTHONPATH=.:sc2_skill_adapter python -m decision_tree.zerg.audit_contact_cache_v3."""
import argparse
import hashlib
import json
from math import dist
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from bridge.zerg_distill import branch, vector
from bridge.zerg_fork import install_pending
from bridge.zerg_whole_ac import parent_step
from .factory import load_resolved_package
from .opening_support_v2 import make_entry
from .policy_repair_v3 import volley


def vec(actions):
    return tuple(actions[f'u{i}'] for i in range(10))


def check(test, msg):
    if not test:
        raise ValueError(msg)


def replay(item):
    path, roster, span = item
    arm = path/'v2'
    receipt = json.loads((arm/'runner/distill/episode-0001.json').read_text())
    source = Path('decision_tree/zerg/resolved/tactical_v1')
    package = load_resolved_package(source)
    m = package.SkillManager(); m.reset(); m.micro_all_attackers = True
    wrapper = SimpleNamespace(manager=m, step=m.step)
    out = dict(panel=path.parent.name, pair=path.name, roster=roster, steps=[])
    first = None
    with (arm/'runner/events.jsonl').open() as events, (arm/'shadow/shadow.jsonl').open() as shadow:
        for line in events:
            ev = json.loads(line)
            if ev['event'] == 'skill_configuration':
                for f, h in ev['skill_source_hashes'].items():
                    check(hashlib.sha256(Path(f).read_bytes()).hexdigest() == h, 'source drift')
            if ev['event'] == 'micro_window_closed':
                grants = {(r['actor'],r['target'],r['macro_action_id']) for r in ev['records']}
                check(grants == {(r['actor'],r['target'],r['macro_action_id']) for r in m.last_micro_intents}, 'micro drift')
                for r in ev['records']:
                    if r['submitted_action_id'] is not None:
                        check(d['micro_receipt'] == 'accepted', 'unknown micro')
                        m.record_micro_action(r['actor'],r['submitted_action_id'])
            if ev['event'] != 'skill_observation':
                continue
            i = ev['step']
            if first is not None and i >= first + span:
                break
            p = ev['payload']; state = p['state']; d = receipt['decisions'][i]; row = json.loads(next(shadow))
            if first is None and any(r['smac_attack_available'] for rel in state['relations'].values() for r in rel.values()):
                first = i
            _, base, render = parent_step(wrapper,p,10)
            menu = [base['actions'], *(a['actions'] for a in base['top_alternates'])]
            check(menu == [c['proposed_actions'] for c in row['candidates'] if c.get('origin') != 'distill_opening_support'], 'menu drift')
            check(d['parent_actions'] == menu[0] and state['game_loop'] == d['game_loop'] == row['game_loop'] and
                  base['snapshot_id'] == d['snapshot_id'] == row['snapshot_id'] and d['committed'] and d['macro_receipt']=='accepted', 'receipt drift')
            ref = d['final_actions']; refv = vec(ref)
            masks = [[str(a) in p['questions'][u]['criteria'] if state['allies'][u]['alive'] else a==0 for a in range(16)] for u in (f'u{j}' for j in range(10))]
            if first is not None:
                reported = set(map(vec,menu)); cache = {}
                table = volley(state)
                pressure = len({e for rel in state['relations'].values() for e,r in rel.items() if r['in_enemy_weapon_range'] is True})>len(table)
                ctx=m._pending_commit[4]
                def reach(v):
                    if any(a>=6 and a!=b for a,b in zip(refv,v)):
                        return False
                    if any(v[j] in shots and refv[j] not in shots for j,shots in table.items()):
                        return True
                    for j,b in enumerate(v):
                        u=f'u{j}'; ally=ctx.snapshot.allies[u]
                        if b==refv[j] or not (2<=b<=5 and ally.alive and state['allies'][u]['weapon_ready']):continue
                        pos=ctx.predict_position(u,b).position; old=ctx.predict_position(u,refv[j]).position
                        for k,en in ctx.snapshot.enemies.items():
                            rng=ctx.effective_attack_reach(u,k)
                            if rng and en.effective_health>0 and dist(old,en.position)>rng>=dist(pos,en.position):return True
                    return False
                for e in sorted(m.coordinator._cache.values(),key=lambda e:m.coordinator._rank(e,m.memory)):
                    cache.setdefault(vec(e.actions),e)
                extras = []
                for v,e in cache.items():
                    if v in reported:
                        continue
                    plans = [dict(id=q.proposal_id,skill=q.primary_skill,actors=list(q.controlled_actors),goal=q.goal) for q in e.plans]
                    alt = dict(actions=e.actions,plans=plans)
                    valid,reason = branch(m,base,alt,render) if vector(e.actions,masks) else (None,'mask')
                    extras.append(dict(actions=v,plans=[q['id'] for q in plans],reject=reason,reach_gain=reach(v),existing_attacks_preserved=all(a<6 or a==b for a,b in zip(refv,v))))
                out['steps'].append(dict(step=i,loop=d['game_loop'],pressure=pressure,top_reach=any(reach(v) for v in reported),cache_entries=len(m.coordinator._cache),cache_vectors=len(cache),reported=len(reported),v2=refv,extras=extras))
            if ref != menu[0]:
                alt = next((a for a in base['top_alternates'] if a['actions']==ref),None)
                if alt:
                    e = branch(m,base,alt,render)[0]
                else:
                    before = tuple(d['selector_audit']['before_opening_actions'])
                    prior = next((a for a in base['top_alternates'] if vec(a['actions'])==before),None)
                    ref_entry = branch(m,base,prior,render)[0] if prior else None
                    check(before==vec(menu[0]) or ref_entry is not None,'opening reference drift')
                    e = make_entry(m,base,p,reference=ref_entry)
                check(e and e.actions == ref, 'native reference differs')
                summary,t = render(e); summary['local_coord'] = base['local_coord']; install_pending(m,base,summary,t,list(e.plans),list(refv))
            check([q.proposal_id for q in m._pending_commit[1]] == d['plan_ids'], 'commit plan drift')
            m.commit_final_actions(list(refv))
    out['first_legal_contact'] = first
    return out


def checked(item):
    try:
        row = replay(item)
        arm=item[0]/'v2/runner'; steps={s['step']:s for s in row['steps']}
        decisions=json.loads((arm/'distill/episode-0001.json').read_text())['decisions']
        for line in (arm/'events.jsonl').open():
            ev=json.loads(line); i=ev.get('step',ev.get('macro_step')); s=steps.get(i)
            if s is None:continue
            if ev['event']=='skill_step':
                check(vec(ev['actions'])==tuple(s['v2']), 'final summary drift')
                s['protected_actors']=sorted(set(ev['output_duties'])|set(ev['contact_recovery'])|set(ev['joint_rescue_substitutions'])|
                    {u for u,d in ev['execution_duties']['actors'].items() if d['required']})
            elif ev['event']=='micro_window_closed':
                s['micro_actors']=sorted({r['actor'] for r in ev['records']})
        for i,s in steps.items():
            # Necessary conditions only, never grant a new micro authorization.
            protected=set(s['protected_actors']+s['micro_actors']); survivors=[]
            for j,e in enumerate(s['extras']):
                changes=[f'u{k}' for k,(a,b) in enumerate(zip(s['v2'],e['actions'])) if a!=b]
                if (s['pressure'] and not s['top_reach'] and e['reject'] is None and e['reach_gain'] and
                    len(changes)==1 and protected.isdisjoint(changes) and i>0 and
                    decisions[i]['selection_reason'] not in ('protect_hydra_support_gap','no_existing_formation_fix','generated_opening_support')):
                    survivors.append(j)
            s['guarded_upper_bound_indices']=survivors
        return row
    except ValueError as e:
        p,r,_=item
        return dict(panel=p.parent.name,pair=p.name,roster=r,steps=[],excluded=str(e))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--per-roster',type=int,default=1); ap.add_argument('--beats',type=int,default=4); ap.add_argument('--workers',type=int,default=8); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    items=[]
    for panel in ('zerg_v1_v2_regression_20261124_185519','zerg_v1_v2_fresh_20261124_190338'):
        root=Path('runs')/panel; counts=Counter()
        for r in json.loads((root/'report.json').read_text())['paired_outcomes']:
            name=r['roster']
            if r['v1']['outcome']==r['v2']['outcome']=='loss' and counts[name]<args.per_roster:
                items.append((root/r['pair_id'],name,args.beats)); counts[name]+=1
    with ProcessPoolExecutor(args.workers) as pool:
        rows=list(pool.map(checked,items))
    args.out.write_text(json.dumps(rows,indent=2)+'\n')
    for panel in sorted({r['panel'] for r in rows}):
        selected=[r for r in rows if r['panel']==panel]; beats=[s for r in selected for s in r['steps']]; print(panel,len(selected),len(beats),sum(bool(s['extras']) for s in beats),sum(any(e['reject'] is None for e in s['extras']) for s in beats))


if __name__=='__main__':main()
