# tree_skills：医疗机治疗目标偏好评分器（MARL-GPT 蒸馏到树）

第一版只学习"已决定治疗时治疗谁"：在原人族 adaptive 体系内，由冻结 MARL-GPT 教师监督训练一个所有医疗机共享的树评分器，只替换"医疗机—伤员"合法配对的偏好；订单、合法性、联合分配框架、撤退、keep、执行与状态全部沿用原 skill。当前状态总览见 `docs/RESULTS.md`（已实现/已测/被阻塞边界如实分列），阻塞项见 `docs/BLOCKERS.md`。

## 快速开始

```powershell
# SMACv2 适配器环境（运行/部署路径，无 torch/sklearn）
$venv = "..\sc2_skill_adapter\.venv\Scripts\python.exe"
& $venv -m unittest tree_skills.tests.test_pair_scorer_channel tree_skills.tests.test_features   # 通道+特征
# 训练环境（系统 python，含 scikit-learn）
python -m unittest tree_skills.tests.test_rule_model        # 导出数值等价
python tree_skills/scripts/demo_synthetic_loop.py --output-dir tree_skills/artifacts/demo
```

## 目录

| 目录 | 内容 |
|---|---|
| `medivac/` | 特征 schema（features.py）、配对评分器（pair_scorer.py：original/tree/teacher） |
| `data/` | 教师标签构造与样本组权重（labeling.py） |
| `training/` | 组权重归一的 GBR 蒸馏训练与按局验证（train_tree.py） |
| `export/` | sklearn → JSON 规则导出（exporter.py）与纯标准库运行时（rule_model.py） |
| `integration/` | off/shadow/teacher-score/tree-score 模式入口（runtime.py） |
| `teacher/` | 检查点 preflight、TEACHER_MANIFEST.json 状态机 |
| `models/` | 部署模型目录（当前为空——真实模型待真实教师标签训练后放入） |
| `scripts/` | 合成闭环冒烟、通道检查；PowerShell 入口见 scripts/*.ps1 |
| `docs/` | BASELINE_AUDIT / TEACHER_CARD / DATA_PROTOCOL / EXPERIMENT_PROTOCOL / RESULTS / BLOCKERS |
| `artifacts/`, `external/` | 大文件与外部源码归档（已忽略，不入库） |

## 关键契约

1. **注入点唯一且默认关闭**：`skills/current/skills/heal_assign.py: assign_healing(..., pair_scorer=None, pair_scoring="off")`；`terran_combat.allocate_terran_combat` 透传。默认路径逐位等价原实现（183 个既有测试 + A-plug 60 种子等价测试）。
2. **固定参与集合**：只有原逻辑本步已分配目标的医疗机参与重评分；keep/自保/救援/订单锁定一律不交给模型；候选边即原合法边，不重开被过滤目标。
3. **允许负分配**：参与集合固定后每机必选一目标，负分有意义；禁止常数平移/取绝对值/截断（逐行减常数不改变匹配最优解，任务书 §8）。
4. **部署零训练依赖**：导出规则 JSON + 标准库即可运行；schema 版本不符拒绝加载。
5. **诚实状态机**：教师未就绪时 `teacher-score` 显式抛错，绝不静默回退冒充学生成绩。

## 新虚拟环境边界

- 运行/部署：沿用 `sc2_skill_adapter/.venv`（不装 torch/sklearn）。
- 训练：独立环境（当前用系统 Python 3.13 + scikit-learn 1.7.0；如建 venv 放 `tree_skills/.venv-training/`，已忽略）。
- 教师：另行独立环境（torch + 官方 marl-gpt 归档），常驻进程批量推理，禁止每步重载权重。
