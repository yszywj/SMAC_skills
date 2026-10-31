# G4_COMMANDS：教师筛选与资格验证的可复制命令（PowerShell）

在 `C:/Users/ry/skill/SMAC_agent/sc2_skill_adapter` 目录下执行。所有参数与冻结基线（docs/BASELINE_AUDIT.md 的 v7 run）一致；每个方法独立输出目录，绝不覆盖。预算上限：单 run 40 局（configs/default_budget.json）。

教师候选（G2 已验证可运行，详见 TEACHER_MANIFEST.json 的 last_token_convention）：
- **B-main**：`marl-gpt-main.pt` + `--pair-last-token on`（对齐官方 terran 评测 last_token=True；受 block_size=700 约束取 H=4）
- **B-alt**：`marl-gpt-wo-last-token.pt` + `--pair-last-token off`
- 容量计算：10v10 每帧 162 token + 1 race token；H×162+1 ≤ 700 → H ≤ 4（官方 eval 配置 H=6 需 973 > 700，不可用，已记录）

## 1. 生成筛选清单（种子 400–439，避开 170–209 / 300–313）

```powershell
cd C:/Users/ry/skill/SMAC_agent/sc2_skill_adapter
. ./.venv/Scripts/Activate.ps1
python run.py scenarios --race terran --size 10 --episodes 40 --seed 400 --unit-balance native --output configs/terran_g4_screen_n40_seed400.json
```

## 2. 公共参数（A、B 完全一致，仅评分来源不同）

```powershell
$Common = @("--method","adaptive-skill",
  "--manifest","configs/terran_g4_screen_n40_seed400.json",
  "--sc2-path","C:\Program Files (x86)\StarCraft II",
  "--race-profile","terran","--terran-keep-heal-order",
  "--unit-value-mode","contextual",
  "--adaptive-window-fire","--adaptive-range-preserving-move",
  "--adaptive-target-scoring","--adaptive-coordinated-approach",
  "--adaptive-safe-reentry","--adaptive-accurate-damage",
  "--adaptive-critical-escape","--adaptive-planned-reentry",
  "--adaptive-engagement-commitment","--adaptive-approach-progress-guard")
```

## 3. 运行（每个输出目录必须全新）

```powershell
# A：冻结基线
python run.py run @Common --output-dir runs/terran-g4-A-n40-seed400

# B-main：真实教师评分（strict；显式 checkpoint 与 last_token，禁用自动选择）
python run.py run @Common --output-dir runs/terran-g4-Bmain-n40-seed400 `
  --pair-mode teacher-score --pair-failure strict `
  --pair-teacher-checkpoint ../tree_skills/teacher/checkpoints/marl-gpt-main.pt `
  --pair-last-token on `
  --pair-collect ../tree_skills/artifacts/g4_screen_collect

# B-alt（仅在 B-main 需要对照时运行；候选总数 ≤ 3）
python run.py run @Common --output-dir runs/terran-g4-Balt-n40-seed400 `
  --pair-mode teacher-score --pair-failure strict `
  --pair-teacher-checkpoint ../tree_skills/teacher/checkpoints/marl-gpt-wo-last-token.pt `
  --pair-last-token off
```

## 4. 配对分析与资格判定

```powershell
# 三层干预指标 + 训练 shard（教师标签）
python -m tree_skills.data.build_shard --collect ../tree_skills/artifacts/g4_screen_collect --shard ../tree_skills/artifacts/g4_screen_shard.jsonl
# 配对胜率与区间（A vs B，逐局配对，含无医疗机局）
python ../tree_skills/evaluation/paired_report.py runs/terran-g4-A-n40-seed400 runs/terran-g4-Bmain-n40-seed400
```

预注册门槛（docs/EXPERIMENT_PROTOCOL.md，运行前冻结，不事后调整）：
- 40 局筛选只决定是否继续，不宣称提升；
- 资格集（另一独立 200 局清单）：B−A 配对胜率差点为正且 95% bootstrap 下界 > −5pp → 暂可蒸馏（工程门槛）；下界 > 0 → 报告提升证据；跨 0 → 证据不足。
- B 无效时的诊断顺序：输入可靠性 → 动作映射 → 决策机会数量（多候选步占比）→ 教师与 A 一致率 → 订单覆盖 → 地面协作失配；不因输局直接判教师标签错误。

## 5. 资格集清单（与筛选集不重叠）

```powershell
python run.py scenarios --race terran --size 10 --episodes 200 --seed 600 --unit-balance native --output configs/terran_g4_qual_n200_seed600.json
# 预算拆分：200 局按 default_budget 单 run 40 局上限拆 5 段（--episodes 40 --seed 600/640/680/720/760）分别运行后合并分析
```
