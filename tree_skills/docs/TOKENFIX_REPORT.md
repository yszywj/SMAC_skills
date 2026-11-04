# 教师末 token 修复、重标与对照实验（2026-11-01）

已完成修复、回归测试、旧轨迹重标和A/B各40局复测。修正后的教师评分方案18/40胜，原规则17/40胜，增加1局（+2.5个百分点）；这是初步正向信号，尚无稳定胜率提升或迁移效果的证据。

## 实验定义与边界

- 使用既有 `terran_g4_screen_n40_seed400.json`，种子400–439，每臂40局，单SC2实例；预算依据 `configs/default_budget.json`。
- A：原规则，新输出 `sc2_skill_adapter/runs/terran-tokenfix-A2-n40-seed400`。
- B：同一规则执行框架，仅教师使用修正 `env_indicator=0`；新输出 `sc2_skill_adapter/runs/terran-tokenfix-B0-n40-seed400`，strict失败策略。
- checkpoint仍为 `marl-gpt-main.pt`，SHA256 `c3deaeb67f679657b27e9d3373e42e4104cc9370be6dba60ab5fd0efe7b1ce5a`；last_token=on，H=4。
- 原教师环境 Python3.13.14 / PyTorch2.15.0+cpu；正式对局沿用默认10线程。未更换模型、候选过滤、keep规则或地面策略。
- 这是已检查场景上的修复诊断复测，不是新的独立测试集，不构成zero-shot transfer证据。
- 不训练学生树。本轮修复评估/选模指标，不改变训练样本权重。
- 首次沙箱内A启动失败，完成0局，保留失败目录 `terran-tokenfix-A-n40-seed400`；后续A2成功，不把失败启动计入成绩。

## 修复与测试

1. `teacher/obs_builder.py` 和 `service.py` 对齐官方默认 `env_specific=False`，末token值为0。有效历史长度4与嵌入容量10分别报告。
2. `reference_check.py` 直接使用官方 `MARLGPTInference` 构造器与 `add_last_token()`；旧值1的回归测试会明确失败，不再复制同一个硬编码。
3. 选模以 `multi_candidate_top1_agreement` 为主；保留明确命名的全体指标、零评分基线、教师分数regret及覆盖局数。无有效多候选验证组时拒绝选模。
4. 联合指标按至少一机有多候选的完整步骤筛选，保留该步所有医疗机。它仍可能包含只有一个可行联合匹配的步骤。
5. 标签意图诊断应用原生合法动作mask；修正原分配/建议分配字段混淆和keep遥测路径。
6. 配对报告严格检查manifest、场景和初始状态哈希，报告胜负四格、bootstrap退化警告及保守不确定性界。

验证：54项特征/通道/历史/标签/评估测试、3项原教师环境token测试、6项实验分析测试、183项技能测试、65项适配器测试通过，共311项。真实观察官方参考24/24一致，见 `artifacts/reference_check_token0_native_20261101.json`。测试发现的旧测试入口路径问题通过正确工作目录/包名执行解决，未修改产品代码迁就测试。

## 同观察重标结果

旧40局轨迹共1850步、962评分组，修正后全部重新标注成功，零丢弃。原观察、mask、执行动作和胜负保留；新标签不是新策略对局结果。

| 指标 | 旧token=1 | 修正token=0 |
|---|---:|---:|
| 评分组中合法argmax为Stop | 962 | 0 |
| 合法argmax为治疗 | 0 | 552 |
| 合法argmax为移动 | 0 | 410 |
| 62个多候选组中的治疗目标Top1改变 | — | 28（45.16%） |

修正教师偏好治疗的552组中，302组首选目标被原规则候选过滤；62个多候选组中，教师偏好移动30组、治疗32组，其中12组首选治疗目标被过滤。这是全动作教师偏好与局部治疗排序接口的差异，不能直接推断放宽规则会提高胜率。

使用已对齐部署tie-break的常数零评分器：旧标签总体Top1=97.09%、多候选Top1=54.84%；重标后分别为97.51%、61.29%。零模型仍可获得很高的总体分，说明全体指标不适合独立验收。上轮对话的95.74%/33.87%来自旧的反向tie-break，不能与新指标直接比较。

产物：`artifacts/g4_token0_relabel_old_trajectory/`，含962组 `corrected_labels_shard.jsonl`、完整collection、重标报告及哈希provenance。不同轨迹的episode编号从头开始；未经命名空间处理不要直接拼接用于联合匹配。

## 规则基线复测

A2为17/40胜（42.5%）。与旧A相比，manifest相同，初始状态与1856步最终动作及已检查决策字段完全一致，`compare_g3_runs.py` PASS。另行核对1856个完整观察payload与真实协调层keep决策，均零差异，见 `artifacts/tokenfix_baseline_full_observations_check.json`。

## 修正教师实时对照

两臂均完整完成40局，场景hash、manifest和初始状态hash全部匹配。B为strict，1863步正常结束，评分失败/回退为0；采集最终动作与B事件日志逐步一致。正式B运行耗时约38.1分钟。

| 分组 | 局数 | A原规则胜局 | B修正教师胜局 | 胜率差B−A |
|---|---:|---:|---:|---:|
| 全体 | 40 | 17（42.5%） | 18（45.0%） | +2.5个百分点 |
| 有医疗机 | 21 | 6（28.57%） | 7（33.33%） | +4.76个百分点 |
| 无医疗机 | 19 | 11 | 11 | 0 |
| 单医疗机 | 15 | 4 | 5 | +6.67个百分点 |
| 两机及以上 | 6 | 2 | 2 | 0 |

配对胜负四格：共同赢17、共同输22、A赢B输0、A输B赢1。唯一翻转是第37局（seed436）。配对bootstrap描述区间为[0,+7.5]个百分点，包含零；只有1个胜负分歧，exact McNemar双侧p=1.0。小样本bootstrap不会从未观察到的负向翻转中抽样，不能据下界为0断言无伤害；报告同时保留非常宽的保守界。这里的40局又是已看过的筛选场景，不能据此声称稳定提升。

| 次要指标（40局均值） | A | B | B−A |
|---|---:|---:|---:|
| 敌军消灭比例 | 74.0% | 74.0% | 0 |
| 己方存活比例 | 26.5% | 26.5% | 0 |
| 对局决策步数 | 46.400 | 46.575 | +0.175 |

消灭比例差的描述性95% bootstrap区间为[-2.5,+2.75]个百分点，存活比例为[-1.5,+1.5]个百分点。局部改善与恶化并存：例如第37局消灭比例0.5→0.9，但第17局0.8→0.5；不能只挑翻盘局解释全部效果。

### 新轨迹的作用范围与标签

- 1005个评分组，其中920个单候选（91.54%），85个多候选（8.46%），多候选来自14局。
- 43次推荐改配全部进入最终提交动作；keep覆盖0，其他覆盖0，未知动作0。发生在11局，其中第21/37局共25次，时间相关性很强。引擎反馈为28次Success/accepted、15次AlreadyTargeted/rejected，其他拒绝、未知、错误、目标不匹配、观察错误均0。AlreadyTargeted可能维持已有治疗订单，不等于完全无效果；Success也不等于实际产生治疗收益，因此不能写成43次成功切换目标。
- 11局发生A/B最终动作分歧；1835个共同loop中320步动作不同。后者包含轨迹分叉后的级联，不能称为320次独立模型干预。
- 评分组中的完整合法动作argmax：治疗610、移动390、Stop5。旧的962/962全Stop异常已消除；剩余5次Stop不等同于旧异常。
- 610个首选治疗组中308个首选目标在原规则候选集外；85个多候选组中34个首选移动、5个首选Stop，只有40个完整首选动作恰好落在候选内。因此当前接口仍经常把教师完整动作意图裁剪成不同的治疗选择。
- 新live数据的常数零评分器总体Top1=97.11%、多候选Top1=65.88%、至少一机多候选步骤的联合一致率=60.56%。高总体一致率仍不是学到策略的证据。

新标签1005组全部成功构建，零非法mask/非法候选/非有限值丢弃。保存于 `artifacts/tokenfix_analysis_final/live_corrected_teacher_shard.jsonl`，每组带本次run、教师配置与文件哈希provenance；与旧轨迹重标的962组分开保存，禁止当作独立样本简单拼接。

### 第37局的正向案例

初始状态相同、前15步动作完全相同；loop122当前完整观察也一致，唯一首次动作差异为u9医疗机从治疗u2切到u5。u2为5/45HP的Marine，u5为83.74/125HP的Marauder；教师分数分别为1.6838和8.0013，完整合法动作首选也是治疗u5，建议实际提交。此首次请求的引擎反馈是AlreadyTargeted，已有u5治疗订单继续；与A切换到u2的动作不同。

本局A失败（60步、消灭5、存活1），B获胜（68步、消灭9、存活3）。本局共有13次教师改配，所以不能把胜负全部归因于首次选择；可以确认这条不同治疗轨迹从医疗机评分干预开始。优先保住Marauder可能是机制解释之一，需要后续干预实验验证。完整证据单列 `artifacts/tokenfix_episode37_case.json`。

### 判断与后续顺序

1. 输入修复成立：同观察回放、官方API参考和实战行为均支持这一点。
2. 混合教师评分出现小幅正向信号，满足继续做独立资格实验的理由；尚未完成预定200局资格验证，`qualified=false`，不能宣称强教师或蒸馏/迁移成功。
3. 本轮没有后置keep吞掉建议的问题；主要剩余限制是候选机会少、完整教师动作与受限治疗任务不一致，以及教师本身在该混合体系下收益尚弱。扩大树数不能解决这些问题。
4. 下一步优先在独立场景验证教师收益，再考虑增加可学习的医疗机动作范围、采集更多有竞争的决策样本并训练真实树；不应直接用本批筛选结果充当独立最终测试。

完整机器可读结果、逐局CSV、配对审计、三套独立零模型基线见 `artifacts/tokenfix_analysis_final/`。

## 两个首次干预案例

以下两例分歧前A/B动作一致，分歧时完整观察payload的SHA256相同，因而确实展示评分替换造成的首次干预，而非已分叉轨迹的后果。完整证据见 `artifacts/tokenfix_case_studies.json`。

| 场景 | 原规则分配 | 教师受限排序实际提交 | 教师完整合法动作首选 | 胜负 |
|---|---|---|---|---|
| 第6局，loop314 | u9救u0，15/45HP | u9继续治疗u3，43.45/45HP | 移动；次选治疗u5被原规则过滤 | A/B均赢 |
| 第7局，loop130 | u9救u6，30/125HP | u9继续治疗u5，123.37/125HP | 治疗满血u1，被伤员候选过滤 | A/B均输 |

两次建议均真正提交，未被keep覆盖。它们表明修复后的教师会改变实际决策，但“完整策略的actor偏好”裁剪成“必须治疗时的目标评分”可能丢失意图，且可能替换掉原规则的危急救援。这不是教师优劣或胜率收益的独立因果证据；第6局84→80步、第7局36→37步也不能单独当成性能结论。

## 复现

在 `SMAC_agent` 下运行，输出目录必须全新。SC2和Store版教师Python需要可运行它们的本机用户环境。

```powershell
./tree_skills/scripts/run-tokenfix-comparison.ps1 -Arm A -OutputDir runs/<new-A>
./tree_skills/scripts/run-tokenfix-comparison.ps1 -Arm B -OutputDir runs/<new-B> -CollectDir ../tree_skills/artifacts/<new-collection>
python tree_skills/scripts/relabel_collection.py --collect tree_skills/artifacts/g4_screen_collect/collect-terran-g4-Bmain-n40-seed400.jsonl --checkpoint tree_skills/teacher/checkpoints/marl-gpt-main.pt --output-dir tree_skills/artifacts/<new-relabel>
python -m tree_skills.evaluation.analyze_tokenfix --run-a sc2_skill_adapter/runs/<new-A> --run-b sc2_skill_adapter/runs/<new-B> --old-collect tree_skills/artifacts/g4_screen_collect/collect-terran-g4-Bmain-n40-seed400.jsonl --relabel-dir tree_skills/artifacts/<new-relabel> --new-collect tree_skills/artifacts/<new-collection> --output-dir tree_skills/artifacts/<new-analysis>
```
