import json,collections
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'runs/native-10v10-v2-jev-astra-skill-n10-seed100'
result={}
for name in ['jev','astra-jev']:
 requests={};responses={};steps={};group=collections.defaultdict(list)
 for line in (root/name/'events.jsonl').open(encoding='utf-8'):
  e=json.loads(line)
  if e['event']=='jev_request':
   requests[e['request_id']]=e;group[e['episode'],e['payload']['state']['game_loop']].append(e)
  elif e['event']=='jev_response':responses[e['request_id']]=e
  elif e['event']=='step':steps[e['episode'],e['game_loop']]=e
 assert requests.keys()==responses.keys()
 assert group.keys()==steps.keys()
 for key,rs in group.items():
  answers={}
  for r in rs:
   a=responses[r['request_id']]['answers'];q=r['payload']['questions']
   assert set(a)==set(q) and not (set(a)&set(answers))
   assert all(a[k]['choice'] in q[k]['criteria'] for k in a)
   answers.update(a)
  assert [int(answers[f'u{i}']['choice']) if f'u{i}' in answers else 0 for i in range(10)]==steps[key]['actions']
 result[name]={'requests':len(requests),'responses':len(responses),'steps':len(steps),'compacted_requests':sum('team_core' in e['payload']['state'] for e in requests.values()),'multi_batch_steps':sum(len(rs)>1 for rs in group.values()),'response_models':dict(collections.Counter(e['model'] for e in responses.values())),'all_response_choices_match_executed_actions':True}
(root/'request_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
