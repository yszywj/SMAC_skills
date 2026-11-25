"""Recompute development-panel diagnostics from runner and shadow logs, no engine.

From repository root: python -m decision_tree.zerg.analyze_severe --out PATH
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from math import dist
from functools import lru_cache
from .policy_repair_v2 import formation, support_gap, endpoint_gap, v2_choose
from .policy_distill import vector

PANELS = ('20261124_162949', '6x40x2_20261124_164422', 'random_20261124_165149')


@lru_cache(maxsize=3)
def archived_model(panel_dir):
    return json.loads((panel_dir/'model.json').read_text())


def volley_targets(s):
    if sum(u['type'].lower()=='hydralisk' for u in s['allies'].values())<6: return {}
    return {int(k[1:]): valid for k,u in s['allies'].items() if u['type'].lower()=='hydralisk' and u['alive'] and u['weapon_ready'] and (valid:={int(e[1:])+6 for e,r in s['relations'][k].items() if r['in_weapon_range'] is True and r['smac_attack_available']})}


def shooters(a,targets): return {i for i,valid in targets.items() if a[i] in valid}


def pressure(s):
    out = {}
    for k, u in s['allies'].items():
        if not u['alive']:
            continue
        enemies = [r['distance'] for r in s['relations'][k].values() if r['distance'] <= 7]
        friends = sum(v['alive'] and dist(u['position'], v['position']) <= 5 for v in s['allies'].values())
        if len(enemies) >= friends + 1:
            out[k] = min(enemies)
    return out


def advancing(s, a, exposed):
    return sum(2 <= a[k] <= 5 and min(s['movement_candidates'][k][str(a[k])]['distances_to_visible_enemies'].values(), default=d) < d - .25 for k, d in exposed.items())


def examine(job):
    panel, p = job
    terminals = [json.loads((p / arm / 'runner/episodes.json').read_text())[0] for arm in ('baseline','tree')]
    assert len({(t['initial_state_sha256'],t['scenario_sha256']) for t in terminals}) == 1
    outcomes = [t['outcome'] for t in terminals]
    kills = [t['info']['dead_enemies'] for t in terminals]
    cohort = ('severe' if max(kills) <= 5 else 'other_LL') if outcomes == ['loss']*2 else ('WW' if outcomes == ['win']*2 else 'other')
    manifest = json.loads((p/'manifest.json').read_text())
    config = manifest['episodes'][0]['episode_config']
    ally, enemy = [config[k+'_start_positions']['item'] for k in ('ally','enemy')]
    reflected = all(abs(a[0]+b[0]-32)<1e-6 and abs(a[1]-b[1])<1e-6 for a,b in zip(ally,enemy))
    row = dict(panel=panel, pair=p.name, outcomes=outcomes,kills=kills,cohort=cohort,
               seed=manifest['seed'],layout='x_reflection' if reflected else 'other',
               initial_state_sha256=terminals[0]['initial_state_sha256'])
    for arm in ('baseline','tree'):
        root = p / arm
        receipts = json.loads((root/'runner/distill/episode-0001.json').read_text())['decisions'] if arm=='tree' else None
        stats = Counter(); pristine=True
        with (root/'runner/events.jsonl').open() as logs, (root/'shadow/shadow.jsonl').open() as shadows:
            for line in logs:
                if '"skill_observation"' not in line[:150]:
                    continue
                ev=json.loads(line)
                s=ev['payload']['state']; sh=json.loads(next(shadows))
                assert s['game_loop']==sh['game_loop']
                if s['game_loop'] > 82: break
                if ev['step']==0:
                    stats['heavy_hydras']=int(sum(u['type'].lower()=='hydralisk' for u in s['allies'].values())>=6)
                d=receipts[ev['step']] if receipts else None
                if d: assert d['game_loop']==s['game_loop'] and d['snapshot_id']==int(sh['snapshot_id'])
                actual=vector(s,d['final_actions'] if d else sh['final_actions'])
                choices=[c['proposed_actions'] for c in sh['candidates'] if d and c['candidate_id'] in (d['scores'] or {})]
                if d and d['mode']!='fallback':
                    assert set(d['scores'])=={c['candidate_id'] for c in sh['candidates'] if c['candidate_id'] in d['scores']}
                hydras=formation(s)
                if hydras:
                    now=support_gap(hydras); nxt=endpoint_gap(s,hydras,actual)
                    if now>=5: stats['isolated_backline']+=1
                    if nxt>=5 and nxt>=now+.5:
                        stats['widening_backline']+=1
                    if now>=5 and d and d['mode']!='fallback':
                        picked=v2_choose(archived_model(p.parent),s,choices)
                        assert tuple(picked['audit']['v1_actions'])==actual
                        if picked['reason']=='protect_hydra_support_gap':
                            stats['formation_fix']+=1
                            row.setdefault('changed_steps',[]).append(ev['step'])
                            row.setdefault('changes',[]).append(dict(loop=s['game_loop'],v2=picked['actions'],
                                candidates=choices,reason=picked['reason'],audit=picked['audit']))
                targets=volley_targets(s)
                if targets:
                    d=receipts[ev['step']] if receipts else None
                    actual=vector(s,d['final_actions'] if d else sh['final_actions'])
                    firing=shooters(actual,targets)
                    stats['early_volley_opportunity']+=1
                    if len(firing)<len(targets):
                        stats['incomplete_volley']+=1
                        if d and d['mode']!='fallback':
                            if any(firing < shooters(vector(s,a),targets) for a in choices): stats['volley_upgrade']+=1
                if any(not u['alive'] or u['health'] < u['max_health'] for u in s['allies'].values()):
                    pristine=False
                if not pristine: continue
                stats['pristine_beats']+=1
                exposed=pressure(s)
                if not exposed:
                    continue
                stats['exposed_beats']+=1
                if receipts is None:
                    chosen = sh['final_actions']
                    if advancing(s,chosen,exposed): stats['advance_beats']+=1
                    continue
                d=receipts[ev['step']]; assert d['game_loop']==s['game_loop']
                chosen=d['final_actions']; n=advancing(s,chosen,exposed)
                if n:
                    stats['advance_beats']+=1
                    candidates=[c['proposed_actions'] for c in sh['candidates'] if c['candidate_id'] in (d.get('scores') or {})]
                    if any(advancing(s,a,exposed)<n for a in candidates): stats['alternative_beats']+=1
        row[arm]=dict(stats)
    return row


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--runs',type=Path,default=Path('runs'));ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    jobs=[]
    for panel in PANELS:
        pairs=sorted((args.runs/('zerg_policy_eval_'+panel)).glob('pair-*'));assert len(pairs)==240
        jobs.extend((panel,p) for p in pairs)
    with ProcessPoolExecutor(args.workers) as pool: rows=list(pool.map(examine,jobs,chunksize=4))
    summary={}
    for panel in PANELS:
        summary[panel]={}
        for cohort in ('severe','WW','other_LL','other'):
            members=[r for r in rows if r['panel']==panel and r['cohort']==cohort]; group={'pairs':len(members)}
            for arm in ('baseline','tree'):
                group[arm]={k:{'beats':sum(r[arm].get(k,0) for r in members),'episodes':sum(bool(r[arm].get(k)) for r in members)} for k in ('heavy_hydras','isolated_backline','widening_backline','formation_fix','early_volley_opportunity','incomplete_volley','volley_upgrade','pristine_beats','exposed_beats','advance_beats','alternative_beats') if arm=='tree' or k not in ('formation_fix','volley_upgrade','alternative_beats')}
            summary[panel][cohort]=group
    strata={}
    for layout in ('x_reflection','other'):
        for heavy in (0,1):
            subset=[r for r in rows if r['layout']==layout and r['tree']['heavy_hydras']==heavy]
            strata[f'{layout}/heavy={heavy}']={}
            for cohort in ('severe','WW','other_LL','other'):
                group=[r for r in subset if r['cohort']==cohort]
                strata[f'{layout}/heavy={heavy}'][cohort]=dict(pairs=len(group),
                    isolated=sum(bool(r['tree'].get('isolated_backline')) for r in group),
                    changed=sum(bool(r.get('changes')) for r in group))
    report=dict(summary=summary,layout_and_roster=strata,pairs=rows,
                cohort_definition='severe: both literal loss and each killed <=5/10; WW: two wins',
                limitation='Development only. Endpoint distances are intended geometry, not realized travel, damage or win benefit. Repeated states follow v1, not v2.')
    args.out.write_text(json.dumps(report,sort_keys=True,separators=(',',':'))+'\n');print(args.out)


if __name__=='__main__': main()
