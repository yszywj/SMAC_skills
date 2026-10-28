import json,collections
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'runs/native-10v10-v2-skill-n20-seed110'
out=root/'trajectory_analysis'
es=[json.loads(l) for l in (root/'events.jsonl').open(encoding='utf-8')]
obs={(e['episode'],e['step']):e['payload']['state'] for e in es if e['event']=='skill_observation'}
dec={(e['episode'],e['step']):e for e in es if e['event']=='skill_step'}
acts={(e['episode'],e['step']-1):e['actions'] for e in es if e['event']=='step'}
evidence={'scope':'Recorded partial observations and actions; step_index is zero-based; no replay-level damage attribution or counterfactual proof.','episode14_frontline':{},'colossus_access':{},'episode12_boundary_case':[]}
for actor in ('u1','u2'):
 attack_steps=[t for (ep,t),aa in acts.items() if ep==14 and aa[int(actor[1:])]>=6]
 evidence['episode14_frontline'][actor]={'type':obs[14,0]['allies'][actor]['type'],'hp_plus_shield_at_step_index20':obs[14,20]['allies'][actor]['effective_health'],'first_attack_step_index':min(attack_steps)}
evidence['episode14_alive_at_step_index20']={a:u['type'] for a,u in obs[14,20]['allies'].items() if u['alive']}
for ep,enemy in [(12,'e0'),(14,'e2'),(19,'e2')]:
 rows=[(t,s) for (e,t),s in obs.items() if e==ep and enemy in s['visible_enemies']]
 firsthurt=next((t for t,s in rows if s['visible_enemies'][enemy]['effective_health']<350),None)
 early=[(t,s) for t,s in rows if t<=30]
 evidence['colossus_access'][str(ep)]={'enemy':enemy,'visible_snapshots_first31':len(early),'snapshots_any_legal_attack_first31':sum(any(rs.get(enemy,{}).get('smac_attack_available') for rs in s['relations'].values()) for t,s in early),'first_observed_damage_step_index':firsthurt}
for t in (90,91,92):
 s=obs[12,t];u=s['allies']['u7'];evidence['episode12_boundary_case'].append({'step_index':t,'hp_plus_shield':u['effective_health'],'cooldown':u['weapon_cooldown_loops'],'action':acts[12,t][7],'role':dec[12,t]['roles']['u7'],'enemy_relation':s['relations']['u7']['e0']})
(root/'trajectory_analysis/case_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(evidence,ensure_ascii=False,indent=2))
