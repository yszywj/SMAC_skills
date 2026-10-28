import json,statistics,collections
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'runs/smac-hard-2s3z-n10-seed100'
rows=[]; summaries=[]
for method,label in [('jev','JEV（论文复现）'),('astra-jev','ASTRA＋JEV（论文复现）'),('skill','Skill（自主设计）')]:
 s=json.loads((root/method/'summary.json').read_text(encoding='utf-8'));summaries.append(s)
 assert s['completed_episodes']==10 and not s['failure'],(method,s['failure'])
 assert s['sc2_action_errors']==0
 for e in s['episodes']:
  assert Path(e['replay']).is_file()
  assert e['final_game_loop']-e['initial_game_loop']==8*e['steps']
 rows.append({'method':method,'label':label,'wins':s['wins'],'total':10,'enemy_elimination':statistics.mean(1-e['final']['enemies_alive']/e['initial']['enemies_alive'] for e in s['episodes']),'mean_steps':statistics.mean(e['steps'] for e in s['episodes']),'outcomes':dict(collections.Counter(e['status'] for e in s['episodes'])),'requests':s['requests'],'api_errors':s['api_errors']})
for key in ('map_sha256','source_sha256','seed','step_mul','opponent','use_ability'):
 assert all(s[key]==summaries[0][key] for s in summaries),key
(root/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# SMAC-Hard 2s3z 三种方法各 10 局测试','', '使用原版 SMAC-Hard 固定 2s3z 地图，双方各 2 名追猎者、3 名狂热者。三种方法均设置 seed=100，按原版运行器连续执行 10 局；默认 base 脚本对手，固定步长 8 loops，关闭特殊技能。沿用原版地图单位属性，未套用此前 SMACv2 的同兵种属性统一改造。','', 'JEV 每步直接决策；ASTRA＋JEV 每图生成一份计划并在 10 局复用；Skill 使用 adaptive_focus_fire，balanced 姿态、4 步技能窗口，启用窗口火力、保持攻击机会的移动评分及目标评分。Skill 不调用模型。','', '| 使用方法 | 敌方平均消灭比例 | 胜率（胜场／总局数） | 平均步数 |','|---|---:|---:|---:|']
for r in rows:lines.append(f"| {r['label']} | {r['enemy_elimination']:.0%} | {r['wins']/10:.0%}（{r['wins']}／10） | {r['mean_steps']:.1f} |")
lines+=['','敌方平均消灭比例和平均步数均按全部 10 局计算。10 局样本仅用于阶段比较，不代表稳定排名；本次与此前 SMACv2 的环境及对手协议不同，不宜直接归因比较。','','## 逐局结果','','| 局数 | JEV | ASTRA＋JEV | Skill |','|---|---|---|---|']
for i in range(10):lines.append('| '+str(i+1)+' | '+' | '.join(f"{s['episodes'][i]['status']}（{s['episodes'][i]['steps']} 步）" for s in summaries)+' |')
lines+=['','验证：30 局均有回放，每步时间推进一致，三组地图和源码哈希一致，SC2 动作错误为零。请求计数及 API 错误计数见 results.json；完整配置和轨迹在各方法子目录。']
lines+=['', '技术记录：ASTRA 首次规划因 map 字段包含额外描述而未通过精确地图名校验，发生于 0 步、正式对局开始前。失败记录保存在 astra-jev-planner-failed-attempt1；使用相同配置重新请求后通过，后续固定该计划完成全部 10 局。该技术失败不计入游戏胜负，未替换任何已进行的正式对局。']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(rows,ensure_ascii=False,indent=2))
