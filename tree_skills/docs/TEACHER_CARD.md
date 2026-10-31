# TEACHER_CARD：marlgpt_terran10_native_teacher_v1

状态：**audited**（2026-10-31：两检查点通过 G1 结构审计；`ready` 待 G2 真实快照核对 + G4 资格验证；`qualified` 未授予）。

权重已下载并核验：`tree_skills/teacher/checkpoints/`（86,860,385 B ×2，SHA-256 与登记一致）。结构：7,232,256 参数、8 层/8 头/256、block_size 700、动作头 20、n_attr 129（多环境基础模型；SMAC 是否在训练分布内无法从权重确认，资格完全依据本项目 G4 实测）。详见 `teacher/CHECKPOINT_AUDIT.md`。

## 定义

- 角色：对每单位原始局部观察历史输出合法基础动作 logits 的冻结教师；本轮仅取医疗机治疗槽分数做条件排序蒸馏。教师不更新，学生训练不影响教师。
- 输入权限：教师使用各单位自身局部观察历史（官方 MARL-GPT 编码）；学生使用项目共享可见状态。两者分别记录，不宣称同条件比较。
- 输出语义：教师基础动作头不含本项目 keep-heal-order（ID 16）语义；该槽永远不进入标签。

## 权重来源（2026-10-30 核验）

- 入口 `https://huggingface.co/nortem/marl-gpt-model`（MIT）经 HF API 核验存在两个文件：`marl-gpt-main.pt`、`marl-gpt-wo-last-token.pt`，仓库约 331 MB。
- **文件存在 ≠ 可用教师**：是否包含人族 10v10 训练经历未知；未下载、未加载、未审计。论文/README 胜率不作为证据。
- 官方实现：`Cognitive-AI-Systems/marl-gpt`（未克隆；审计时下载归档至 external/ 并记录哈希与许可证）。

## 资格流程（未执行，入口已备）

1. 手动/显式预算下载 → `python tree_skills/teacher/preflight.py --checkpoint-dir <dir>` 记录尺寸与 SHA-256（状态 downloaded）。
2. 在独立教师环境（torch + 官方实现）中 strict 加载：参数量、层数、历史长度、上下文容量、动作头维度、token/属性编码、训练任务全部读出并填入 TEACHER_MANIFEST.json audit_checklist（状态 audited）。缺失/多余键、actor 头不完整、输入越界、非有限输出 → rejected。
3. 冒烟集 5–10 局接口验证 → 教师筛选集 40 局 → 教师资格验证集 200 局（B vs A 配对胜率差下界 > -5pp 才可标"暂可用于实验蒸馏"；>0 才报告提升证据）。至多评估 3 个候选，选定后冻结。

## 备选路径

本地建教师：官方格式人族 10v10 专家小样本审核 → 单域离线预训练 → 必要时训练集上 PPO 适配。从随机初始化训练的只能称"本地训练的 MARL-GPT 架构教师"。教师训练与验证/测试集隔离，默认不自动启动，预算上限见 configs/。
