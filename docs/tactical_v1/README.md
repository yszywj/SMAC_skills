# tactical_v1 运行手册

新控制器：7 个通用 skill + 3 个机制特化 + Adaptive 联合选择，一次提交。
入口保持 `--method adaptive-skill`，用 `--skill-version tactical_v1` 显式选择；所有旧入口与历史基线不变。

以下命令在仓库根 `C:\Users\ry\skill\SMAC_agent` 的 PowerShell 中执行。
解释器：`.\sc2_skill_adapter\.venv\Scripts\python.exe`（Python 3.10）。

## 0. 离线检查（不需要 SC2）

```powershell
# tactical_v1 全部离线测试（58 个）
& .\sc2_skill_adapter\.venv\Scripts\python.exe -B -m unittest discover -s skills/tactical_v1/tests -t .

# 旧技能包回归（包模式；172 项中包内 13+159，见 audit 第 5 节）
& .\sc2_skill_adapter\.venv\Scripts\python.exe -B -m unittest discover -s skills/current/tests
& .\sc2_skill_adapter\.venv\Scripts\python.exe -B -m unittest skills.current.tests.test_race_movement_guard skills.current.tests.test_spatial_review_regressions

# 适配器集成测试（94 个，含 tactical_v1 的真实 runner 集成；不起 SC2 对局）
Push-Location sc2_skill_adapter
& .\.venv\Scripts\python.exe -m unittest discover -s tests
Pop-Location
```

## 1. 生成/复用配对清单

已有清单在 `sc2_skill_adapter/configs/`（如 `protoss_10_vs_10_smoke.json`、
`{terran,zerg}_10_vs_10_native_n1_seed1600.json`、`*_n40_seed170.json`）。生成新的：

```powershell
& .\sc2_skill_adapter\.venv\Scripts\python.exe sc2_skill_adapter\run.py scenarios `
  --size 10 --race terran --episodes 3 --seed 170 --unit-balance native `
  --output sc2_skill_adapter\configs\terran_10v10_tactical_pair.json
```

## 2. 冒烟（真实 SC2，每族 1–3 局）

```powershell
$py = '.\sc2_skill_adapter\.venv\Scripts\python.exe'
& $py sc2_skill_adapter\run.py run --manifest sc2_skill_adapter\configs\protoss_10_vs_10_smoke.json `
  --output-dir runs\tactical_v1\<日期>\protoss --method adaptive-skill --skill-version tactical_v1
& $py sc2_skill_adapter\run.py run --manifest sc2_skill_adapter\configs\terran_10_vs_10_native_n1_seed1600.json `
  --output-dir runs\tactical_v1\<日期>\terran --method adaptive-skill --skill-version tactical_v1 `
  --terran-keep-heal-order   # 可选：显式医疗保持扩展（terran 地图 + native 专用）
& $py sc2_skill_adapter\run.py run --manifest sc2_skill_adapter\configs\zerg_10_vs_10_native_n1_seed1600.json `
  --output-dir runs\tactical_v1\<日期>\zerg --method adaptive-skill --skill-version tactical_v1
```

不通过改步长/对手难度来让冒烟通过；失败即失败。

## 3. 配对回归

```powershell
# 旧基线（current）与新架构，同清单同 seed
& $py sc2_skill_adapter\run.py run --manifest sc2_skill_adapter\configs\terran_10_vs_10_native_n40_seed170.json `
  --output-dir runs\tactical_v1\<日期>\terran_n40 --method adaptive-skill --skill-version current --race-profile terran
& $py sc2_skill_adapter\run.py run --manifest sc2_skill_adapter\configs\terran_10_vs_10_native_n40_seed170.json `
  --output-dir runs\tactical_v1\<日期>\terran_n40 --method adaptive-skill --skill-version tactical_v1
```

native/original、医疗保持扩展开/关必须分成不同输出目录（动作空间不同，不可混为同一基线）。

## 4. 配置与消融

`--tactical-config <file>` 覆盖 `skills/tactical_v1/defaults.json`（严格 JSON，深合并）。示例消融：

```json
{ "enabled": { "splash_fire": false, "heal_support": false, "burst_engage": false } }
```

```powershell
# 通用框架消融：三类特化关闭（对应单位回落到安全合法的通用行为，不是停摆）
& $py sc2_skill_adapter\run.py run --manifest ... --output-dir runs\...\abl_generic `
  --method adaptive-skill --skill-version tactical_v1 --tactical-config ablations\generic_only.json
# 关闭多单位原子联合候选（beam 退化为单单位选择仍走同一协调器）
{ "focus_fire": { "joint_kill_max_attackers": 0 } }
```

逐项消融建议顺序（P7.4）：通用框架 → +S1 → +S2 → +S3 → 完整 → 关联合候选 → 定位回退场景后单独关 G2/G3/G5/G6。

## 5. 旧参数组合

race-profile、v8 空间开关（--enable-regroup 等）、--adaptive-* 战斗开关、--unit-value-mode、
--pair-mode、--medivac-policy/--medivac-tree、--skill-tree、--native-teacher-control 对 tactical_v1
一律在配置解析阶段报错（不静默忽略、不让默认值意外关功能）。--skill-max-steps/--adaptive-posture
被接受并记录（不影响联合调度语义）。

## 6. 遥测

每步 `events.jsonl` 的 `skill_step` 事件含：snapshot/game_loop、配置指纹、每单位 skill/提供者/目标/动作、
选中方案与 Top-K 备选、修复记录、搜索预算降级事件、每技能适用/提案/选中统计、三类特化实际参与统计、
决策耗时。`run.json` 含递归源码指纹（包内全部运行 .py + 复用的 damage_model + 生效配置）。
