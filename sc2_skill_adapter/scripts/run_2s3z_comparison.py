import subprocess,os,json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
repo=root/'game_skill'
out=root/'smacv2_jev_adapter/runs/smac-hard-2s3z-n10-seed100'
common=['--map','2s3z','--episodes','10','--seed','100','--step-mul','8','--opponent','base','--sc2-path',r'C:\Program Files (x86)\StarCraft II','--request-timeout','15','--max-requests','5000']
methods={'jev':['--policy','jev','--planner','none','--skill-mode','off'],'astra-jev':['--policy','jev','--planner','codex','--planner-model','gpt-6-astra','--planner-effort','medium','--planner-timeout','180','--skill-mode','off'],'skill':['--policy','nearest','--planner','none','--skill-mode','always','--skill-name','adaptive_focus_fire','--adaptive-posture','balanced','--adaptive-window-fire','--adaptive-range-preserving-move','--adaptive-target-scoring']}
for method,extra in methods.items():
 if (out/method/'summary.json').exists():
  done=json.loads((out/method/'summary.json').read_text(encoding='utf-8'))
  if done['completed_episodes']==10 and not done['failure']:
   print('ALREADY COMPLETE',method,flush=True)
   continue
 print('START',method,flush=True)
 cmd=[str(repo/'.venvs/micro/Scripts/python.exe'),str(repo/'jev_star.py'),'micro',*common,*extra,'--output-dir',str(out/method)]
 with (out/f'{method}-console.txt').open('w',encoding='utf-8') as log:
  p=subprocess.run(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
 print('FINISH',method,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
