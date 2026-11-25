"""Offline opening-only coverage. Run from repo root with PYTHONPATH=.:sc2_skill_adapter."""
import argparse
import json
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from bridge import runner, zerg_shadow
from bridge.skill_runtime import manager_configuration
from bridge.zerg_distill import branch
from bridge.zerg_fork import install_pending
from bridge.zerg_whole_ac import action_map, certify, parent_step
from .opening_support_v2 import RULE_ID, make_entry, ranked_moves, rule_hash


def examine(r):
    p = Path('runs/zerg_policy_eval_'+r['panel']) / r['pair']
    root = p/'tree/runner'
    with (root/'events.jsonl').open() as f:
        payload = next(json.loads(l)['payload'] for l in f if '"skill_observation"' in l[:150])
    d = json.loads((root/'distill/episode-0001.json').read_text())['decisions'][0]
    assert d['game_loop']==payload['state']['game_loop'] and d['step']==0 and d['snapshot_id']==1
    out = dict(panel=r['panel'],pair=r['pair'],cohort=r['cohort'],prior=bool(r.get('changes')))
    if 0 in r.get('changed_steps',[]):
        return dict(out,status='existing_fix_first')
    if not ranked_moves(payload,d['final_actions']):
        return dict(out,status='no_opening_move')
    manifest=json.loads((p/'manifest.json').read_text()); run=json.loads((root/'run.json').read_text())
    args=runner.parser().parse_args(['run','--manifest','unused','--output-dir','unused','--method','adaptive-skill',
        '--skill-version','tactical_v1','--execution-mode','micro8-all'])
    with tempfile.TemporaryDirectory() as tmp:
        rt,ctrl=zerg_shadow.start_shadow(str(Path(tmp)/'shadow'),run['zerg_shadow']['scene_config_id'],manifest,
            'adaptive-skill',args,manager_configuration(args)['manager_kwargs'])
        rt.episode_seed=manifest['episodes'][0]['sc2_seed']; ctrl.reset(); m=ctrl.manager
        m.micro_all_attackers=True;ctrl.select('adaptive_focus_fire',posture='balanced')
        _,base,render=parent_step(ctrl,payload,10)
        if base['actions']!=d['parent_actions']:
            return dict(out,status='parent_mismatch')
        row=next(iter(rt._buffer.values()))
        ref=None
        if d['final_actions']!=base['actions']:
            alt=next((a for a in base['top_alternates'] if a['actions']==d['final_actions']),None)
            if alt is None: return dict(out,status='reference_mismatch')
            ref,_=branch(m,base,alt,render)
            if ref is None: return dict(out,status='reference_invalid')
        if [p.proposal_id for p in (ref.plans if ref else m._pending_commit[1])] != d['plan_ids']:
            return dict(out,status='reference_plan_mismatch')
        env=SimpleNamespace(n_agents=10,get_avail_agent_actions=lambda i:[str(a) in payload['questions'][f'u{i}']['criteria'] for a in range(16)])
        entry=certify(m,make_entry(m,base,payload,reference=ref),base,row,env)
        if entry is None: return dict(out,status='native_abstain')
        summary,telemetry=render(entry); summary['local_coord']=base['local_coord']
        v=[entry.actions[f'u{i}'] for i in range(10)]; plans=list(entry.plans)
        assert summary['actions']==entry.actions=={u:b['action'] for u,b in summary['units'].items()}
        changed,=[u for u in entry.actions if entry.actions[u]!=d['final_actions'][u]]
        new,=[q for q in plans if q.primary_skill==RULE_ID];assert new.controlled_actors==(changed,)
        install_pending(m,base,summary,telemetry,plans,v);ctrl.commit_final_actions(v)
        assert m.memory.committed and m.memory.last_committed_actions==action_map(v) and all(g['actor']!=changed for g in m.last_micro_intents)
        return dict(out,status='certified',actor=changed,prior_actions=d['final_actions'],actions=entry.actions,
            plan_id=new.proposal_id,metrics=new.goal_data,engine_receipt='not_run')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    data=json.loads(Path('runs/zerg_policy_repair_v2_analysis.json').read_text())['pairs']
    with ProcessPoolExecutor(args.workers) as pool: rows=list(pool.map(examine,data,chunksize=2))
    groups={}
    for r in rows:
        g=groups.setdefault(f"{r['panel']}/{r['cohort']}",Counter());g['pairs']+=1;g[r['status']]+=1
        g['prior']+=r['prior'];g['union']+=r['prior'] or r['status']=='certified'
    args.out.write_text(json.dumps(dict(rule_sha256=rule_hash(),groups=groups,rows=rows),indent=2)+'\n');print(groups)


if __name__=='__main__':main()
