# tactical_v1 运行手册

新控制器：9 个通用 skill + 3 个机制特化 + Adaptive 联合选择，一次提交。
入口保持 `--method adaptive-skill`，用 `--skill-version tactical_v1` 显式选择；所有旧入口与历史基线不变。

持续作战由 `battle_intent` 记录实际血盾进展、共同目标和跨技能撤退反馈；
`weights.sustained_contact` 评价进入有效射程的机会，G2/G4/G7 分别处理短暂同步、回场和功能性会合。
相关参数见 `skills/tactical_v1/defaults.json`。跨阵容验证应同时报告最低胜率、阵容极差、
早期有效接触和实际回场结果；开火周期代理不能当作命中，观察不到敌人不能当作击杀。

以下命令在仓库根 `C:\Users\ry\skill\SMAC_agent` 的 PowerShell 中执行。
解释器：`.\sc2_skill_adapter\.venv\Scripts\python.exe`（Python 3.10）。

## 0. 离线检查（不需要 SC2）

```powershell
# tactical_v1 全部离线测试（数量以当前运行输出为准）
& .\sc2_skill_adapter\.venv\Scripts\python.exe -B -m unittest discover -s skills/tactical_v1/tests -t .

# 旧技能包回归（历史发现模式的相对导入限制见 audit 第 5 节）
& .\sc2_skill_adapter\.venv\Scripts\python.exe -B -m unittest discover -s skills/current/tests
& .\sc2_skill_adapter\.venv\Scripts\python.exe -B -m unittest skills.current.tests.test_race_movement_guard skills.current.tests.test_spatial_review_regressions

# 适配器集成测试（实际 runner/manager + 模拟环境；不起 SC2 对局）
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

也可使用版本包装器运行默认配置的 Protoss 10v10。它会生成或复用同参数的 native 清单，
并为 `tactical_v1` 跳过旧战斗开关；`OutputName` 必须是新的结果目录名：

```powershell
.\sc2_skill_adapter\scripts\run_skill_version.ps1 -Size 10 -SkillVersion tactical_v1 `
  -Episodes 3 -Seed 170 -OutputName tactical-v1-protoss-10v10
```

Terran/Zerg、医疗保持扩展或 `--tactical-config` 覆盖配置请使用上面的 `run.py` 入口。
只指定 `--method adaptive-skill` 时默认仍加载 `current`，测试新控制器必须显式选 `tactical_v1`。

## 3. 配对回归

```powershell
# 旧基线（current）与新架构，同清单同 seed
& $py sc2_skill_adapter\run.py run --manifest sc2_skill_adapter\configs\terran_10_vs_10_native_n40_seed170.json `
  --output-dir runs\tactical_v1\<日期>\terran_n40_current --method adaptive-skill --skill-version current --race-profile terran
& $py sc2_skill_adapter\run.py run --manifest sc2_skill_adapter\configs\terran_10_vs_10_native_n40_seed170.json `
  --output-dir runs\tactical_v1\<日期>\terran_n40_tactical --method adaptive-skill --skill-version tactical_v1
```

每次运行必须使用尚不存在的输出目录。native/original、医疗保持扩展开/关必须分成不同输出目录
（动作空间不同，不可混为同一基线）。

## 4. 配置与消融

`--tactical-config <file>` 覆盖 `skills/tactical_v1/defaults.json`（严格 JSON，深合并）。示例消融：

```json
{ "enabled": { "splash_fire": false, "heal_support": false, "burst_engage": false } }
```

```powershell
# 通用框架消融：三类特化关闭（对应单位回落到安全合法的通用行为，不是停摆）
& $py sc2_skill_adapter\run.py run --manifest ... --output-dir runs\...\abl_generic `
  --method adaptive-skill --skill-version tactical_v1 --tactical-config ablations\generic_only.json
```

关闭 G1 原子截杀候选的覆盖配置（其余技能仍走同一协调器）：

```json
{ "focus_fire": { "joint_kill_max_members": 0 } }
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
`candidate_search` 记录去重前后数量、每技能/单位获得首轮完整比较的候选数，以及未比较数。
这里的 `screened` 表示尝试将候选放入完整基线进行比较；冲突或非法分支会被拒绝，
不表示该候选已胜出或其所有组合都已搜索。

## 7. 战前集结与战斗稳定性

`precombat_rally` 默认启用。可见敌人接近、我方地面战斗单位分散且尚未交火时，它提交一个
多单位原子方案，与接敌、攻击等方案共同评分。Adaptive 决定是否选择，已就位的单位等待，
其余单位向安全的友军位置靠拢。它排除治疗者、空中单位和自爆单位，不要求所有战斗都集结。

同一次任务固定成员、集结点和接战编号，通过 `continue` 续接；默认至少 3 个成员，
80% 成员进入集结点 4 距离以内即可完成。默认总期限为 8H，连续 3H 无进展结束，
短承诺为 2H。H 是当前控制窗口的 game loops，`step_mul=8` 时 8H 等于 64 loops。
成员受伤、进入实际攻击接触或受敌方威胁时中断集结。一次接战的集结结束后，不会立即重建；
必须发生真实接触，再持续安静 4H 才开启下一次接战，单帧丢失视野不会重置。

配置覆盖示例：

```json
{
  "enabled": {"precombat_rally": true},
  "precombat_rally": {
    "ready_fraction": 0.8,
    "max_h": 8.0,
    "no_progress_h": 3.0,
    "rearm_quiet_h": 4.0
  }
}
```

完整方案评分增加脱离既有队友支援、打断已建立的就绪攻击、离开接触后的重新接敌成本，
并取消逃逸收益与风险降低的重复计分。集结按队伍就绪进展计分，同时扣除未就绪等待成本；
短承诺仅增加切换成本，紧急危险变化可以打断，不会锁死控制。集结原子方案在预算允许时
优先获得完整评分机会，Top-K 记录实际评分过的完整备选及分项得分。

基础修复包括：移动距离按观测原生速度的 16 loops 换算；近战接敌点使用实际射程和碰撞半径；
G4 限制撤退/回场阶段并保护健康单位的就绪攻击；G5 使用有直径约束的支援分组，并允许
正在受威胁的孤立单位向友军靠拢；G3 换目标遵守冷却并保留总任务期限。
`lifecycle.retry_cooldown_h` 和 `no_progress_window_h` 作用于真实任务，
实际提交与退出累计计数用于区分创建、续接和退出。决策时遥测统计截至本次观察和上一次提交，
以 `lifecycle_last_committed_loop` 标识边界。

## 8. 战中会合、进退连续性与搜索

`combat_rejoin` 默认启用，用于已有队伍交火、其他队伍仍分散的情况。它与战前的
`precombat_rally` 分工：前队有可用射击时保持攻击，后队一起向接应队会合；受压的小队
也可以整组向较强的友军靠拢。判断包含当前接触、预测威胁、有效血量和窗口内可用火力，
不会仅凭“小队有两人且能攻击”认定支援充足。它覆盖近战、远程和巨像等地面战斗单位；
治疗者、飞行单位、自爆单位继续使用各自机制。

会合方案包含双方，作为一个原子任务交给 Adaptive 比较，成员、接应队、锚点和职责固定。
默认支援半径 6、成员数 2–10、总期限 8H、短承诺 2H、无进展期限 3H。
续接不重置这些预算；完成以实际观察到的支援连接为准。成员丢失、接应队濒危、明显增压或
爆炸威胁可以退出。单个成员的动作被外部改写时，不会部分提交整个任务。

```json
{
  "enabled": {"combat_rejoin": true},
  "combat_rejoin": {
    "support_radius": 6.0,
    "max_h": 8.0,
    "no_progress_h": 3.0,
    "min_support_ratio": 1.2
  }
}
```

其他配套修复：

- G2 当前合法且实际在射程内的目标优先；堵塞检查只计算同一接敌任务已提交的完整历史窗口。
- G2/G4 共用短期进退记录。射程与动作掩码边界上的短承诺不会因技能替换而消失或续期；
  严重受伤等紧急变化仍可打断。默认 `combat_cycle_commit_h=2`、`combat_cycle_memory_h=6`。
- 风险评分加入可见敌人的一个控制窗口追击范围，并保留不确定移动的部分原地风险；
  威胁超过三个后仍继续增加，敌方攻击能力与我方剩余有效血量也参与计算。
  `risk_at` 仍表示当前物理接触，预测由独立 `forecast_risk` 提供。
- G3 沿用近期真实攻击目标，G1 在走位后继续保留该目标候选和软切换成本。
  下一窗口的射击/巨像覆盖机会单独计分，不计入当前伤害或击杀；
  `focus_fire.urgency_value` 通过最终实际攻击方案的敌方威胁削减生效。
- Adaptive 对完全相同的候选契约去重，优先比较持续任务和整组方案，然后交错比较各单位、
  各技能的候选，再深入组合。默认完整评分预算仍为 512，预算不足会报告未比较候选。

追击包络、支援比例和下一发覆盖都是启发式估计，不是敌方行动模拟或已校准的群伤伤害。
历史状态回放只验证在记录位置上的决策；新动作提交后，真实后续轨迹需要另行实战验证。

这些权重和阈值是待验证的初始参数。离线测试与历史状态重放验证逻辑和合法动作，不能代表新胜率；
胜率需要使用同阵容、同位置和同 seed 的实战配对测试另行测量。

## 9. 有效参战与移动反馈修订

战中会合现在同时检查支援距离与实际武器/动作掩码对应的参战时间。健康成员进入队友附近后，
没有攻击接触时仍可继续接敌；接应者濒危且受威胁时，即使本帧没有新增掉血也会退出。
接应队的血盾与可见威胁用于估计它能否等到会合，属于启发式折扣，不是未来存活保证。

会合收益按实际归队者的边际进展计算，取消按全部接应人数放大收益。最终评分增加健康、
武器就绪且有一步有效接敌机会的等待成本；STOP已可自动开火或没有接敌机会时不收取该项。
前排无有效掩护的暴露、对高威胁敌人的压力削减也按最终全队动作计算，不按兵种名称强制选敌，
不把未来支援当作当前伤害或击杀。

所有最终移动动作（包括baseline）会检查短期反向与实际受阻记录。G6在生成阶段避开刚提交的
接敌反方向；G4按追击风险保留最多三个撤退方向，再由Adaptive比较。实际位移显著低于预测时，
该方向在局部位置短期降分，死亡、换身份、离开区域和超时都会清理记录。

新增权重默认值：`combat_idle=0.35`、`movement_continuity=0.8`、`blocked_move=0.6`、
`unsupported_frontline=0.5`。`target_threat`及`urgency_value`沿用原配置，但使用新的客观压力项。
搜索预算仍为512。跨局统计按增量累计，`summary.skill.search_budget_events`不再只保留最后一局。

本轮40×3配对测试使用原始三个40局清单，seed170–209，运行脚本为
`runs/run_tactical_3x40_support_20261113.py`。先以`--prepare-only`校验清单并保存源码快照，
再以`--run`顺序执行。脚本拒绝覆盖已有结果或混用准备后变更的策略源码。

## 10. 支援评分、再入战与联合让路修订

支援人数或非致命射击不再直接抵消敌方来袭威胁。最终位置的已知暴露与本窗预计击杀的
有界威胁解除分开计分；潜在射击不计作当前伤害，也不同时从暴露项重复减免。
G4 再入战的中心距离目标取实际原生射程加双方半径，并受合法攻击指令距离约束，
近战不再统一使用距离5作为接近目标。

会合追踪正在移动的接应者，同时保留任务锚点身份和原始期限。原始短承诺结束后，
仅对近期确实推进、仍能在剩余预算内完成的归队者停止推进施加有界切换成本。
成本按最终动作判断，标签改变但继续接近、真实接触攻击或紧急脱险均可免除；
不会因续接重置期限，也不会按接应人数放大奖励。

实际受阻记录保持原有寿命。完整联合方案显示堵路友军本轮确实让出通路时，可以
部分降低方向惩罚；不据此删除记录或宣称地形必然可通行。

本轮用 `runs/run_tactical_3x40_coordination_20261114.py` 运行同样三个40局清单，
与 `tactical_protoss_10v10_3x40_support_20261113` 的完整120局逐一配对。
运行前保存源码快照；测试过程中固定代码和配置。

## 11. 连续输出、撤退交换与共享射击目标

`combat_progress` 保存最近4H真实位置、暴露和武器冷却重启。没有产生开火、净减压或有效绕障的
回访会产生有界循环成本；重复推进不再次获得全额参战收益。恢复射程后的短职责最多3H，
由当前冷却确定截止时间；攻击命令本身不等于开火，紧急危险和有效脱险可立即打断。

`combat_exchange` 用可见单次火力与最近实际掉血趋势比较短期暴露，不读取敌方隐藏冷却，
不把等待时间直接乘成确定敌方射击次数。G4在撤退失败后比较改善路线和重新接敌；G7的接应
可行性同时受每个暴露成员自己的血盾限制。会合完成必须观察到实际合法接触，不能仅由
预计2H内可达宣布完成。后续第一轮输出由共享回场职责继续保护。

G1和G3共享两窗口目标分配，按可兑现主目标伤害、威胁、原目标和确定击杀保留候选。
G3最终动作被接受后才保存下一发目标；连续选择不能延长期限。目标分配收益从最终实际动作
重算，非致死攻击不取消敌方暴露。`focus_fire.target_planning_enabled` 可关闭这一机制。

S1的 `splash_fire.coverage_reliability` 默认0.5，是工程折扣，**不是实测命中概率**。
0/0.5/1可独立消融；主目标伤害不变，副目标仍是未知伤害的覆盖机会。遥测分别记录原始和
折扣后覆盖分及未校准状态。不能将该项描述为已完成原生溅射机制校准。

本轮冻结与运行入口：`runs/tactical_protoss_10v10_3x40_continuity_20261114_r2/supervisor.py`。
只运行新版三阵容各40局，复用原manifest及seed170–209；对照是上一轮已完成的sustained/current
120局。测试中禁止改参数或源码，分析同时报告无收益往返、回场后输出、早期减员和后排压制。
初次启动因离线发现当前射击预算被未来移动来源抢占而停止，7局诊断结果单独保留、不计正式
胜率。修复两阶段分配并补充回归后，r2重新冻结，从头执行全部120局。

正式结果为6z4s 18/40、4z5s1c 31/40、2z6s2c 28/40，合计77/120；修改前对照为
26/40、27/40、26/40，合计79/120。可观察的无收益往返减少，但开火周期密度和接触率三组
均下降，阵容胜率极差由2.5扩大到32.5个百分点，尚不构成稳定升级。
完整解释见 `runs/tactical_protoss_10v10_3x40_continuity_20261114_r2/RESULTS_ANALYSIS.md`。
该批次一次严格步进异常已保留取证并按原seed补齐；后续复用120局应读取该目录的
`completed_run_index.json`，其中双巨像完整数据指向assembled，不能把原失败目录当完整对照。

## 12. 通用射击周期、受压记忆与局部接应

`common_combat` 的四个独立布尔开关默认开启，供分项回归和消融使用：

- `next_fire_budget_enabled`：用原生射程和合法命令边界共同约束下一发的ETA。
  G3/G4等最终移动从离场时就比较移动窗口、返程和剩余冷却；同一次延迟不再叠加
  `return_fire_continuity` 与 `contact_reentry`。真实减压、解除堵路及紧急中断可豁免。
  G2的近战接触位、远程射程带也使用该合法边界，未知射程不制造可开火位置。
- `pressure_memory_enabled`：最多保留2H、逐步衰减的实际血盾损失及同一可见威胁身份。
  一帧没有新增损伤不会立即清除危险证据；真实减压、威胁/自身身份变化、超时及新局清除。
  该证据贯通G4、评分和跨skill中断，不读取隐藏敌方冷却。治疗者和自爆者不继承普通战斗职责。
- `shared_front_enabled`：G2和G7复用G1/G3的短期目标计划，接应ETA、推进与完成指向同一
  局部可见战线。G2接触位满时可选择同战线另一目标；局部目标处于重试冷却时不会因此改追
  远端敌军。原任务仍可执行时继续原期限。
- `local_cover_enabled`：G7可释放临危、死亡或身份变更成员，保留仍可兑现的子组。
  裁剪只在最终提交时改变任务成员，不吸纳新人、不重置截止时间、不把人数减少算作进展或完成。
  加入战斗和逃向支援位置按职责分别估计掩护等待窗口；每名接应者仍受自身血盾约束。

这些是基于可观察状态的短期机会/暴露估计，不是未来伤害或命中承诺。
四开关全部关闭时保留上一版通用行为；固定8个回归样本的77帧动作与分数已逐帧复现。
S1源码、effects实现、覆盖参数及评分中的S1分支保持本轮修改前版本不变。

三族测试入口为 `runs/tactical_all_races_10v10_9x20_firecycle_20261114/supervisor.py`。
P/T/Z各三个镜像10v10阵容、各20局，共180局，seed170–189，native、difficulty7、step_mul8、
balanced和skill-max-steps4。神族复用上一轮同seed的60局对照；人族、虫族没有该版本相同场景
的历史对照，因此仅报告新场景覆盖结果。医疗运输机和毒爆虫的治疗/自爆行为单列分析。
实际结果、源码冻结与异常处理以该目录中的报告和审计记录为准。

正式180局已完成：神族三组10/20、12/20、14/20（36/60；旧版同seed为41/60），
人族11/20、9/20、6/20（26/60），虫族7/20、3/20、1/20（11/60）。总计73胜、106负、
1平；平局计入分母。一次非终局原生短步的未完成尝试被排除，前12局与原seed续跑8局
显式组装；完整180局的9567个原生步均通过严格8-loop校验。策略314项、adapter96项、
分析与恢复43项测试通过。后续读取该批`completed_run_index.json`，首组有效数据位于assembled。

本轮不构成性能升级：神族总胜率下降，且跨种族表现仍不均衡。已定位G2使用历史己方位置
与当前敌位置估计接近进展，同速追逐会被误判为有效推进，保留approach目标可能放大该缺口。
G7接管也不保证实际恢复连续输出。无收益往返的部分代理下降不能代替参战和生存验证；
虫族早期低接触、大量普通单位死亡，以及自爆指令是否兑现有效交换仍需进一步处理。
这些问题仅记录在结果分析中，未在本批中途修改冻结策略或额外补选胜局。
完整结果及病例见`runs/tactical_all_races_10v10_9x20_firecycle_20261114/REPORT.md`。

## 13. 三族机制修正与配对回归

本轮保留 S1 巨像实现、参数及其评分分支。G2 进度使用同一时刻的敌我位置和目标身份；
同速追逐不会再凭敌方移动获得推进信用。新增 `arrival_exposure_enabled` 检查 G2/G7
显式移动后的自身到场暴露，`reentry_window_enabled` 在实测移动减压后建立最多 3H 的
回场保护。窗口不会由查询或重复提交续期，实际开火、身份变化和到期会清除。
原生攻击命令隐含的自动接近仍没有完整路径模拟。

S3 使用本机隔离校准的地面爆炸模型，分别处理主动接触和死亡触发；空武器目录不再
意味着无爆炸收益，也不再让远处毒爆成为全图即时危险。伤害机会按敌方剩余血量封顶，
多来源共享预算；计划消耗、当前风险与前排暴露按同一单位取较大代价，避免重复扣除。
推进收益取最终动作、预计爆点、区域剩余收益与暴露折扣，不能冒充实测命中或击杀。
接近、触发和重选目标共用有界任务及重试记录。校准证据和数值适用范围保存在
`burst_native_profile.json` 与本轮 `calibration` 目录。

S2 要求已知且足够的能量、可治疗的受伤生物患者、可兑现的到达窗口以及稳定归属。
已处于合法治疗范围的患者不再为会面而被移动；只有能明显缩短到达时间、不打断就绪
射击且不增加暴露时才允许患者参与会面。救援价值包含当前可见攻击域的稀缺性，
不固定偏爱某个兵种。观测到的治疗命令只证明归属，不能用 STOP 或 AlreadyTargeted
推断治疗中断；不假设未标定的治疗速率。

本轮入口为 `runs/tactical_all_races_10v10_9x20_mechanisms_20261115/supervisor.py`。
复用上一轮九个完整清单，各 20 局、seed170–189，共 180 局，逐局比较初始状态与原生
8-loop 步长。保持 native、difficulty7、balanced、skill-max-steps4 以及原动作协议；
不开启 KeepHealOrder 或额外原生技能。正式结果按九组等权，另以 P/T/Z=0.2/0.4/0.4
辅助观察跨种族改进；测试期间禁止根据中间胜率调参。完成状态以该目录的
`completed_run_index.json` 和 `REPORT.md` 为准，校准微场景不计入胜率。

本批正式180局已完成：神族11/20、14/20、15/20，人族7/20、11/20、8/20，
虫族4/20、3/20、4/20。总计77胜103负，上一轮同场景73胜106负1平；神族净增4胜，
人族、虫族各自持平。9856个原生步均为8-loop，180局初态与旧版逐一配对，
本批无不完整尝试需要排除。策略372项、adapter96项、分析56项测试通过。

这批仍不构成跨种族的稳定升级：无医疗艇、无毒爆两组分别少4胜和3胜，
特殊技能修复的收益被通用层回退抵消。已发现局部候选过滤重排其他队友的接战路径，
以及G2拒绝的危险移动被G6重新选择；保命后接触和第二周期输出仍不足。
该批冻结代码保存在其 `source_snapshot`；完整解释见本批 `RESULTS_ANALYSIS.md`。

## 14. 按机制选择动作的联合评估与 20×3×3 入口

本次修改保留原动作集合、原生地图、8-loop 决策步长和 S1 巨像技能实现。
普通兵继续共用技能，但风险、射击节奏、可兑现支援与恢复机会按实际能力和状态计算，
不按种族名称套一组固定撤退动作。

- `common_combat.joint_assessment_enabled` 默认开启。所有完整方案（含 baseline 补全
  和最终修复）使用 `joint_combat.py` 的动作事实：显式移动、攻击隐含接近、路线暴露、
  下一次攻击 ETA、实际共同目标与接应者动作。友军攻击不会直接抹掉敌方伤害；同一暴露
  只取较大代价。STOP 不能凭空承诺攻击任意目标，治疗者和毒爆不算普通火力掩护。
- G2/G4/战中会合将边缘推进交完整方案比较，G6 不再为毒爆提出普通散阵移动。
  G3 使用实际合法射程与移动速度；G5 优先能提供有效火力的接应位置。
  返场职责从实际离场起算，跨 skill 保留原截止，观察到首次及第二次攻击周期后完成；
  紧急避险、真实减压、阻塞和目标消失仍可中断，超时不能靠换标签续期。
- 适配器附带原始精度能量和武器目录周期（正常速度秒×16 simulation loops）。
  周期用于机会窗口和职责期限，不冒充前摇、弹道或实际命中；缺数据仍为未知。
- `common_combat.mechanism_recovery_enabled` 默认开启。生命、护盾分层保留，G4 等待
  需要最终分配的治疗服务或有证据的被动恢复。自然恢复不要求 STOP；位置相同的安全动作
  获得相同恢复机会。恢复参数只接受验证过的 profile 或本局干净观测，未实测的速率和延迟
  不按种族猜测。前后任一观察存在医疗艇的区间不用于学习自然生命恢复。
- S2 区分当前首订单集合、排队订单、候选服务和最终服务，支持稳定续疗与有限替代配对；稀缺攻击域的救援收益按患者真实射程、已知速度和冷却推算的 2H 输出机会折扣，遥远或未知机会不获满额收益；
  救援期限不会在继续任务时重置。记录能量恰好 1.0 而缺少原始精度时保留边界未知。
- S3 的候选不预占另一只毒爆的残血预算。主动、途中死亡和原地死亡机会互斥，
  S3、推进评分和 baseline 共用爆点语义。增加侧入与有界等待，动作仍推进原目标时可跨
  provider 续接原任务；实际死亡/消失不自动认定爆炸成功，预测机会不计为观测进展。

通用注册项共九个：`focus_fire`、`coordinated_engage`、`combat_reposition`、
`disengage_reenter`、`regroup`、`local_formation`、`cohesive_search`、
`precombat_rally`、`combat_rejoin`。历史文档的 G7 编号曾混指搜索与战中会合，
后续分析应使用真实模块名。

在仓库根目录的 PowerShell 执行：

```powershell
.\sc2_skill_adapter\scripts\run_tactical_matrix.ps1 -DryRun
.\sc2_skill_adapter\scripts\run_tactical_matrix.ps1
```

也可指定独立输出名：

```powershell
.\sc2_skill_adapter\scripts\run_tactical_matrix.ps1 -OutputName "tactical_joint_20x3x3"
```

入口复用 `runs/tactical_all_races_10v10_9x20_mechanisms_20261115/manifests` 九份清单，
三族各3组固定阵容，每组镜像10v10、20局，seed170–189，共180局；native、difficulty7、balanced、
skill-max-steps4，不启用额外原生技能或 KeepHealOrder。三族阵容分别为：

| 种族 | 阵容1 | 阵容2 | 阵容3 |
|---|---|---|---|
| 神族 | 6狂热者4追猎者 | 4狂热者5追猎者1巨像 | 2狂热者6追猎者2巨像 |
| 人族 | 6机枪兵4掠夺者 | 4机枪兵5掠夺者1医疗艇 | 2机枪兵6掠夺者2医疗艇 |
| 虫族 | 6跳虫4刺蛇 | 4跳虫5刺蛇1毒爆 | 2跳虫6刺蛇2毒爆 |

脚本依次运行九组，冻结运行源码与依赖、校验地图和配对清单，逐组保留日志及原始事件。
输出目录已存在时拒绝覆盖，失败停止并保留现场，不自动重抽 seed。
`matrix_summary.json` 记录完成状态及与上批同 seed 的胜场比较；日志在 `logs`，
逐局轨迹在 `current/<group>`。默认目录名带时间戳。
这些修复的单元验证和离线轨迹检查不能代替这180局的胜率评估。

## 15. 通用接敌、返射与共同火力窗口修复

本轮针对 `tactical_20x3x3_20261115_173227` 已完成轨迹中的通用控制问题。
S1/S2/S3 专用技能源码保留，共同评分对最终动作的改进仍会影响组合方案。

- 所有普通单位最终动作都使用联合暴露风险下限，包含原地 ATTACK、STOP 和 MOVE。
  与旧风险取较大值，不重复累计；不会因为保持攻击就绕开更强的承伤估计。
- 支援前后都比较同一 3H 窗口内的一次火力机会，并使用相同 ETA 折扣和生存窗口。
  在射程内等待冷却的友军可贡献近期支援；其撤离会失去对应机会。移动/STOP 的未来
  火力保持折扣，排队攻击和无订单 STOP 不会虚构任意目标掩护。
- G4 在联合模式按实际承伤、储备和威胁方向判断撤离，不再仅由两名敌人触发夹击。
  单纯降低预测暴露不会完全免除下一枪延迟成本；真正危急、效果危险和观测阻塞保留
  避险通路，任务身份、截止时间及重试冷却继续沿用原生命周期。
- G2 近战接触环按单位实际直径和 `contact_slot_margin`（默认 0.05）生成，近端路线
  优先，每单位保留少量替代槽供完整方案排他选择。`slot_spacing` 仅保留给关闭
  `joint_assessment_enabled` 的旧固定间距模式。有限绕行需要可见障碍/路线进展，
  不把反复小幅侧移当作有效接敌，也不强迫单位沿致死直线推进。

同样的九组阵容改为每组10局（共90局），可从任意 PowerShell 工作目录执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\ry\skill\SMAC_agent\sc2_skill_adapter\scripts\run_tactical_matrix.ps1" -GamesPerGroup 10
```

末尾加 `-DryRun` 仅检查配置、依赖和配对输入。默认仍为每组20局；可用
`-OutputName` 指定新的独立目录名，已有目录拒绝覆盖。

10局模式取原九份清单各自的前10个场景，seed170–179，保留每局场景签名，重算裁剪后
清单的整体签名。仍为 native、difficulty7、step_mul8、balanced、skill-max-steps4。
主对照是刚中断的 `tactical_20x3x3_20261115_173227`，不是更早版本：完整90局中有89局
可与该批次配对；双毒爆 seed179 的旧局无效，明确列为 unpaired，不补造旧结果。

输出到 `runs/tactical_10x3x3_<timestamp>`，包含冻结源码、清单、日志、逐局轨迹及
`matrix_summary.json`。传输异常时保留已完成且通过验证的局数，当前无效局不计输，
不自动重抽 seed。DryRun、离线回归和历史快照检查均不能代替这90局的胜率评估。

### 完整分组在汇总阶段中断后的续跑

回放头检查兼容直接 `MPQ\x1a` 归档以及 SC2 常见的 `MPQ\x1b` 用户数据头：
后者按声明偏移检查内部归档头，并校验长度边界。只检查文件开头是否为 `MPQ\x1a`
会误拒正常回放；模拟测试同时覆盖两种布局和截断、越界文件。

若分组已经完整结束、仅批次汇总失败，可在策略和依赖保持一致时续跑：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\ry\skill\SMAC_agent\sc2_skill_adapter\scripts\run_tactical_matrix.ps1" -GamesPerGroup 10 -OutputName "tactical_10x3x3_20261115_212459" -Resume
```

末尾加 `-DryRun` 只验证续跑条件，不启动游戏。续跑重新审核已完成的连续分组，
跳过完整组并从下一组开始；保留原源码冻结及原始记录，记录续跑使用的 launcher 版本。
只有 launcher 自身修复可与原冻结不同，策略、配置、依赖变化均拒绝混入。
存在已启动但未完整完成的分组时停止，不覆盖它，也不自动将异常局补作失败。

## 16. 共同目标预算、持续输出职责与局部接敌修复

本轮依据 `tactical_10x3x3_20261115_212459` 的人族、虫族失败轨迹，先修复共同评分
与通用技能；S1/S2/S3 专用技能源码保持不变。共同评分仍会影响特殊技能与普通动作的
组合选择，不承诺仅凭离线验证即可提高胜率。

- `fire_opportunities.py` 从完整方案中的合法显式攻击、已知伤害与冷却建立共同目标
  预算。其他单位更早的火力已预计耗尽目标时，该目标只能提供折扣备用机会，不能再
  承诺完整未来支援或保证返射时间。保留当前可用攻击、敌方真实存活和承伤风险；
  预计耗尽不等同于已经击杀，未知伤害/弹道也不会被补造为确定命中。
- 高暴露风险不再在储备比 2 处变成完全相同的封顶分数：普通区间保持原刻度，
  超出后继续单调增加并有界。攻击、停止、移动使用相同规则；较低预测承伤不等同
  于该路线必定存活。
- G6 普通避挤使用同一运动层的实际体型与接触余量，空中医疗艇不会被当作地面
  路径拥堵；真实身体阻塞和范围伤害散阵仍单独处理。
- 输出职责从实际接触建立。下一枪延迟在移动前后使用同一份预算后的有效目标集合，
  已知原生周期用于计算固定窗口内损失的射击机会，并保留原延迟成本下限；未知周期
  不猜测射速。职责跨 skill 保留固定截止时间，只有观测到的攻击周期推进输出进度，
  几何靠近、切换 provider 和更换局部目标均不能无限续期。即时危险和真实阻塞保留
  避险豁免，治疗者与自爆单位不套用普通射击义务。
- `local_entry.py` 对完整动作计算同一局部战线的暴露开始与接应 ETA，包含此前支援
  为零的新接敌情况。G2 只在安全位置、无当前攻击机会且队友能够及时接应时提出短等；
  最终选择还须验证队友确实推进/攻击、目标仍有预算、路线未阻塞且有生存机会。
  连续等待最多 2H，换 provider 或新建任务不能刷新；有效短等免空闲成本，无效等待
  按单位收取一次原空闲成本，时间差成本按局部战线有界计算，不重复计算敌方伤害。

当前的 20×3×3 测试命令可从任意 PowerShell 工作目录执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\ry\skill\SMAC_agent\sc2_skill_adapter\scripts\run_tactical_matrix.ps1" -GamesPerGroup 20
```

仍为第 14 节九组原阵容，每组 20 局，seed170–189，共 180 局；native、difficulty7、
step_mul8、balanced、skill-max-steps4 均不变。末尾添加 `-DryRun` 仅检查配置。
默认建立新的 `runs/tactical_20x3x3_<timestamp>`，不应续跑旧策略的目录。

主对照已更新为最近完整 90 局的 `tactical_10x3x3_20261115_212459`，校验其各组
前 10 场清单及签名。新批次 seed170–179 的 90 局可逐局配对，seed180–189 的
另外 90 局明确列为 unpaired；不会拼入更早策略来伪造完整 180 局对照。
汇总仍写入 `matrix_summary.json`，保留逐局轨迹与源码冻结。逐步遥测增加
`local_entry`（战线时间差、等待依据）及对应评分项，便于区分无效等待和有效集结。

## 17. 通用战线、即时攻击优先与攻击追击成本

技能编号对应：G1 为 `focus_fire`（集火），G2 为 `coordinated_engage`（协同接敌），
G3 为 `combat_reposition`（战斗移位），G6 为 `local_formation`（局部阵型）。
adaptive 在当前实现中通过共同评分选择这些技能的完整动作组合。

本轮升级 G6，并让 G2、`combat_rejoin` 和共同评估复用 `battle_line.py` 的局部战线。
按各单位真实武器范围、合法攻击距离、体型、速度及冷却建立接触位置或可射击位置；
使用剩余储备和敌方实际伤害估计判断到场风险。普通近战、远程分别接近适合自己的
位置，不按种族名称指定肉盾。机枪兵与掠夺者均为远程，不强制将掠夺者排在最前。
同战线的火力必须由最终动作、目标预算和可存活的接应窗口支持，前排站位不抹掉
敌方远程伤害，也不假设嘲讽或额外原生技能。

- G6 保留避挤、范围伤害散开和巨像专用站位，增加至多两个局部两至三人的联合方案。
  已有攻击机会者尽量保持射击，离线者补进射程；危险承压者仍可减压。联合任务沿用
  固定总期限、成员和重试冷却，换目标或切换调整方案不刷新同一任务的期限。
- G2 使用共同局部战线选择接敌目标，保留近战排他接触槽和远程原射程带备选。
  同战线重选目标续接原任务。战中会合也覆盖近端尚未参战者，可接入侧向射击位置，
  不要求所有人朝接应者身体聚拢；当前可攻击目标不会被过时的战线目标列表排除。
- 联合模式下，当前窗口已有合法原生攻击机会的单位，不再为正向目标接近、射程带、
  会合或保护位几何变化获取额外奖励。真实伤害、风险与返射成本仍参与比较，紧急
  撤离不因这条规则被强制禁止。共同战线的正向参战价值替换原独立推进收益，
  不叠加两份接近奖励。遥测增加 `battle_line` 的局部成员、接应者和到达依据。
- 射程外 `ATTACK` 所引发的追击纳入返射与持续输出职责。该命令只能承诺选定目标
  的出手时间；不能拿另一个近目标充当追击后的输出。原生攻击可在抵达后开火，
  不机械增加显式 MOVE 的整步等待；目标预算、隐含位置和共同支援 ETA 使用同一口径。
  身处危险不能让向敌追击自动免费，追击只有确实减压时才继承紧急避险豁免。

保留三族原九组阵容、20 局/组及 seed170–189，native、difficulty7、step_mul8、
balanced、skill-max-steps4 和搜索配置不变。S1/S2/S3 专用技能源码未修改。
按用户要求，本次不运行定向验证、全套回归或游戏，尚无本轮修改后的胜率结论。

在任意 PowerShell 工作目录执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\ry\skill\SMAC_agent\sc2_skill_adapter\scripts\run_tactical_matrix.ps1" -GamesPerGroup 20
```

入口自动建立新的 `runs/tactical_20x3x3_<timestamp>`，不混入已有策略的部分批次。
主对照仍为完整 90 局的 `tactical_10x3x3_20261115_212459`：前 10 个 seed/组共 90 局
逐局配对，另外 90 局标为 unpaired；汇总保存在新目录的 `matrix_summary.json`。

## 18. 接触任务、爆虫推进与人虫 120 局对照

本轮保留即时攻击和射程外 ATTACK 约束，调整共同战线为以已有接触火力为锚的
局部小组任务。同窗支援按可用伤害、剩余目标血量、到达时间及生存机会计量，
不再仅凭存在一个同行单位给足收益。G2 与战中会合共用接触方向，已有射手保留
攻击机会，未参战者朝敌方接触区域移动，而不是只缩短队友距离。

`ContactProgress` 在观察阶段记录跨 G2/G6/会合/撤退的未接触时间；3H 无净进展或
6H 未接触时重新优先比较可行接触动作。换任务不刷新预算；实际接触、死亡及真正
离开局部战斗才清理记录。G4 的超期非紧急脱离须有明显减压和短期回战机会，真实
致命危险仍保留避险。遥测新增 `contact_recovery` 和 `contact_debt`。

S3 使用独立 `BurstProgress`，记录已提交区域、阻塞、反向移动及累计等待，避免
新任务重复获得 staging 时间。等待要求可观察的即将到来的掩护机会；接近阶段
比较短窗口路线暴露与有效爆炸区域，保留原生死亡爆炸校准，不把命令当成命中。
普通火力与爆炸按时间桶共享目标血量；未知或同窗口到达按比例分配，绝不重复
增加伤害/击杀。遥测新增 `burst_progress`、`burst_motion`。

在 VS Code PowerShell 中，从仓库根目录运行：

```powershell
& .\sc2_skill_adapter\scripts\run_tactical_matrix.ps1 `
    -GamesPerGroup 20 `
    -Races terran,zerg `
    -BaselinePath 'C:\Users\ry\skill\SMAC_agent\runs\tactical_20x3x3_20261116_142630'
```

仅人族、虫族，各三个原阵容，每组20局，共120局；seed170–189、原manifest、native、
difficulty7、step_mul8、balanced、skill-max-steps4及搜索预算不变。与上批对应120局
逐局配对。新输出为 `runs/tactical_20x2x3_<timestamp>`。

Python stdout/stderr直接写 `logs/<group>.log` 和 `<group>.stderr.log`，组间活动写
`matrix_activity.log`，不再依赖前台 Tee 管道。主终端运行期间不逐局输出；在另一个
终端读取日志查看进度。所有原清单、源码、回放及部分结果审计保留。

按用户要求未运行定向验证、回归、DryRun或游戏；本轮仅静态审阅，尚无新胜率结论。
每步 `plan_evaluations_used` 指 coordinator 对完整联合动作方案的实际评分次数，
不是旧决策树节点数，也不是训练迭代次数。

## 19. 跨技能参战责任与有限敌方火力预算

本轮针对 `tactical_20x2x3_20261116_213149` 的反复失效路径修改共同评分，
不新增独立技能，也不靠种族名称降低风险权重。

- `ContactProgress` 不再因没有推荐动作、短暂进入射程或小幅接近而清除责任。
  当前射程内的 ATTACK 可正常等待冷却；只有观察到武器冷却重置才记录开火周期完成。
  最终提交动作才固定路线，无进展路线短期避开；重选目标或切换技能不续接触期限。
  这是开火周期代理，并非命中证明；8-loop 采样可能漏掉快攻速单位的完整周期。
- `danger_evidence` 区分可能的全敌集火上界与近期实际损血、可见攻击目标、效果危险。
  G4 在紧急证据缓解后返场，超期未参战本身不会触发新撤退。G2/G6/会合共用的
  接近候选和空闲判断不再仅因保守生存值为零而全部消失；即时攻击保护仍保留。
- `team_exchange` 按最终整队位置为每个已知单体敌人分配一份普通攻击预算，
  不把同一枪全额重复分配给所有我方单位。范围伤害、爆虫和未知机制保持逐目标风险，
  实际掉血保留为下限，并考虑路线中点及位置不确定性。分配是规划估计，不是已知仇恨
  或安全保证；我方开火不会直接抵消敌方伤害。原上界继续留作诊断，不再覆盖共同风险。
- 附近队友实际掉血时，比较最终命令与合法最快参战动作的时间差。
  该延迟成本与接触超期成本按单位取较大值，避免两者重复计费；目标血量预算仍共享。
  新遥测包括 `contact_outcomes`、`contact_decisions`、`team_exchange`、`team_delay`
  和 `conservative_risk_diagnostic`，用于分辨责任完成、合理避险、继续拖延及模型上界。

从 VS Code 的仓库根目录 PowerShell 运行新批次：

```powershell
$runName = 'tactical_20x2x3_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& .\sc2_skill_adapter\scripts\run_tactical_matrix.ps1 `
    -GamesPerGroup 20 `
    -Races terran,zerg `
    -BaselinePath 'C:\Users\ry\skill\SMAC_agent\runs\tactical_20x2x3_20261116_213149' `
    -OutputName $runName
```

仍为人族、虫族各三个原阵容，每组20局，共120局，seed170–189；原manifest、native、
difficulty7、step_mul8、balanced、skill-max-steps4及搜索预算不变。
主对照使用最近中断批次的89局有效结果，另外31局明确列为无配对；不混入更旧策略。
启动器支持该批次尚未启动的分组，同时拒绝把已记录完成但文件缺失的分组当作无配对。
日志仍直接写入输出目录，主终端不逐局输出。

按用户要求，本轮仅人工静态审阅，未运行测试、DryRun或游戏，尚无修改后的胜率结论。

## 20. 执行优先级、直接接敌与爆虫区域承诺

本轮以完整的 `tactical_20x2x3_20261116_234210` 为对照，改变动作执行规则。
规则依据实际单位能力、射程、速度、体型、受伤和局部战线，不读取阵容名、seed或固定坐标。

- `action_duties` 为完整最终动作建立逐单位职责。排序先比较必要救命、爆虫执行、当前
  开火和接触执行的违反数，再比较原有收益分。只有存在合法替代动作才激活要求；
  所有组合都不理想时保留合法的最少违反方案，不把候选筛空。P0、搜索、最终修复
  与替代方案日志采用相同顺序，接近收益不能覆盖更高优先级的职责。
- coordinator在原评估预算内构造实际整队职责方案，避免仅惩罚旧动作却没有替代。
  G6额外提交完整局部战线原子组合，保留已有射击、补入离线成员并撤出危险者，
  避开新增身体重叠。组合本身不创建新的集结期限；跨技能记录仍追踪实际参战。
- G2普通近战在合法攻击命令范围内，按接触容量分配短期可达目标，用原生ATTACK
  完成最后追近。约2H无实际净进展或出现离线折返时，禁用失败路线2H并重选；
  换skill/目标不刷新参战总期限。同战线队友已经接触或掉血、且能短期参战时，
  不再等待长期超时才产生接近职责。会合共享同一目标、路线及原生接近动作。
- G4救命候选不再被射击锁、失败重试或旧接敌承诺阻止。只有有观察证据支持的迫近
  危险且存在明显减压路线时，撤离优先于站打。危险缓解后返场；STABILIZE必须由
  最终动作中的实际治疗服务支持，微小被动回血不再成为停留理由。
- S3先选择短期固定区域/入口，再比较动作，不能让每个MOVE各换目标解释进展。
  使用2H区域承诺、2H无进展/折返处理和局部冲击的4H期限，重选不续总期。
  潜在全敌火力不再令绕行成本直接归零；有效爆炸机会按剩余目标收益处理。
  爆虫执行要求同样约束其他skill及baseline提供的最终动作。

遥测新增 `execution_duties`（逐单位触发原因、允许动作、实际动作、满足状态和
优先级违反数）、`execution_seed_candidates`，替代方案也记录执行优先级。
数值 `joint_score` 只用于同一执行优先级内比较，不再单独表示最终方案排序。
H为当前决策窗口；阈值是本轮工程起点，不代表已通过新一轮对局验证。

在 VS Code 的仓库根目录 PowerShell 中运行：

```powershell
$runName = 'tactical_20x2x3_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& .\sc2_skill_adapter\scripts\run_tactical_matrix.ps1 `
    -GamesPerGroup 20 `
    -Races terran,zerg `
    -BaselinePath 'C:\Users\ry\skill\SMAC_agent\runs\tactical_20x2x3_20261116_234210' `
    -OutputName $runName
```

人族、虫族各3组原阵容，每组20局，共120局，全部可与上述完整批次逐局配对。
原manifest、seed170–189、native、difficulty7、step_mul8、balanced、skill-max-steps4
及搜索预算不变。新建输出目录，不续跑或混写旧版本；进度在`matrix_activity.log`，
逐局输出在`logs`，主终端不流式显示。

按用户要求仅人工静态审阅，未运行定向验证、回归、编译、DryRun或游戏。

## 21. 协同到场、可生还撤离、连续输出和爆虫净交换

本轮根据完整批次 `tactical_20x2x3_20261117_105817` 的虫族配对失败轨迹修改。
不读取种族阵容名称、seed或固定地图坐标；神族/人族继续按实际能力参与共同规则。

- `ContactProgress.coordination` 比较近战与同一战线远程首轮火力的到场时间，
  结合单位剩余血量和敌方武器周期限制危险的单兵先到。近战向可支援后排靠拢时，
  后排同时补入共同目标；已有接触单位继续交战。原地同步累计最多1H，不创建新等待任务。
  G2、战中会合、G6和最终职责共享安排；G6不再因前后排友军距离超过10而拆开同一敌方战线。
- 强救援要求路径中点和终点预测仍有生还余量；未知风险不构成强制救命证明。
  最终完整组合再次检查撤离是否仍致死。当前能够完成的威胁目标击杀可与撤离共同比较，
  不假定攻击必然先消灭敌人或取消其死亡爆炸。G4旧分支不能重新提交被排除的逃离动作。
- 普通单位保留整个冷却期间的原生攻击接触，而不是只保护当前一枪。
  冷却超过1H时，仅允许保持有效射程、不推迟下一次开火且改善安全/支援的移动。
  目标消失后重新计算接触，不永久锁定目标；医疗和一次性爆虫不使用普通开火职责。
- 效果武器进入可能范围不再自动等于致命危险或把生存/支援价值清零。
  按已知范围伤害、实际损血和目标攻击证据评估；范围伤害仍按各受害者计入风险。
- S3强触发使用共享目标剩余HP预算，比较有效伤害、完整来源消耗和敌爆虫死亡反爆炸的
  条件代价；低收益/未知交换保留为软候选供最终评分，不再因正收益就必须爆炸。
  爆虫4H接近超期实际释放旧区域、短禁旧入口并结束旧任务；只有1H内可兑现的替代入口
  可继续强接近，长路线回到联合收益比较，不续原期限。
- 最终动作的爆虫原生ATTACK位置计入隐含接近；潜在联合击杀敌爆虫的反爆炸作为风险下限，
  与原风险取大避免重复收费，不计作已知伤害或确认击杀。

新增/更新遥测包括 `arrival_coordination`、`preserve_contact_through_cooldown`、
`joint_route_still_lethal_compare_exchange`、爆虫 `exchange` 与 `deadline_expired`。
这些是执行规则和预测，不是胜率、真实伤害或成功协同的保证；已孤立接触的近战并不被
普遍强制撤出，仍由持续攻击与可生还救援共同处理。

在 VS Code 仓库根目录 PowerShell 运行新批次：

```powershell
$runName = 'tactical_20x2x3_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& .\sc2_skill_adapter\scripts\run_tactical_matrix.ps1 `
    -GamesPerGroup 20 `
    -Races terran,zerg `
    -BaselinePath 'C:\Users\ry\skill\SMAC_agent\runs\tactical_20x2x3_20261117_105817' `
    -OutputName $runName
```

仍为人族/虫族各3阵容、每组20局，共120局。原manifest、seed170–189、native、
difficulty7、step_mul8、balanced、skill-max-steps4和搜索预算保持不变。
对照上述完整批次的120局；日志写入新目录，不混写或续跑旧策略。
本轮只做人工静态审阅，未运行定向验证、回归、编译、DryRun或游戏。
