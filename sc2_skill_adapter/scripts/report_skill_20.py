import json,statistics
from pathlib import Path
from report_corrected_pair import behavior
root=Path(__file__).resolve().parents[1]/'runs/native-10v10-v2-skill-n20-seed110'
old=root.parent/'native-10v10-v2-jev-astra-skill-n10-seed100/adaptive-skill'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
s=load(root/'summary.json'); meta=load(root/'run.json'); manifest=load(root/'manifest.json'); eps=load(root/'episodes.json')
assert s['status']=='complete' and s['completed_episodes']==20==len(eps)
assert s['jev_requests']==0 and s['planner'] is None
assert [e['sc2_seed'] for e in manifest['episodes']]==list(range(110,130))
checks=behavior(root)
for i,e in enumerate(eps,1):
 assert load(root/f'balance/episode-{i:04d}.json')['passed']
 assert (root/f'replays/episode-{i:04d}.SC2Replay').stat().st_size>0
 assert e['scenario_sha256']==manifest['episodes'][i-1]['sha256']
previous=load(old/'run.json');oe=load(old/'episodes.json')
comparison={k:meta[k]==previous[k] for k in ['source_hashes','versions','env_args','skill_parameters','observation_protocol']}
result={'outcomes':s['outcomes'],'win_rate':s['win_rate'],'mean_enemy_elimination':s['mean_enemy_elimination'],'mean_steps':statistics.mean(e['steps'] for e in eps),'verified_replays_and_attribute_audits':20,'behavior_checks':checks,'previous_protocol_comparison':comparison}
lines=['# SMACv2 Protoss 10v10：纯 Skill 新增 20 局','', '新增种子 110–129；同兵种原生属性统一；敌方难度 7；固定步长 8 loops。使用 AdaptiveFocusFire，balanced 姿态、4 步窗口，窗口火力、保持攻击机会的移动评分及目标评分均开启；不调用模型。','', '| 方法 | 敌方平均消灭比例 | 胜率（胜场／总局数） | 平均步数 |','|---|---:|---:|---:|',f"| 纯 Skill（本轮新增） | {s['mean_enemy_elimination']:.1%} | {s['win_rate']:.1%}（{s['outcomes']['win']}／20） | {result['mean_steps']:.1f} |",'',f"胜负分布：{s['outcomes']}。",'', '与上一轮协议核对：'+str(comparison),'']
if all(comparison.values()):
 all_e=oe+eps; wins=sum(e['outcome']=='win' for e in all_e)
 result['combined_30']={'wins':wins,'total':30,'win_rate':wins/30,'mean_enemy_elimination':statistics.mean(e['enemy_elimination_fraction'] for e in all_e),'mean_steps':statistics.mean(e['steps'] for e in all_e)}
 a=result['combined_30']
 lines +=[f"与此前种子 100–109 的 10 局合计：{wins}/30 胜（{wins/30:.1%}），敌方平均消灭比例 {a['mean_enemy_elimination']:.1%}，平均步数 {a['mean_steps']:.1f}。这是纯 Skill 的累计成绩，不与模型方法的十局样本混作同规模配对比较。",'']
lines +=['## 逐局结果','','| 局数 | 种子 | 结果 | 消灭比例 | 步数 |','|---:|---:|---|---:|---:|']
for i,e in enumerate(eps):lines.append(f"| {i+1} | {110+i} | {e['outcome']} | {e['enemy_elimination_fraction']:.0%} | {e['steps']} |")
lines +=['','20 份回放与属性审计完整，逐步动作合法性及 8 loops 时间推进通过核对。','']
(root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');(root/'verified_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
