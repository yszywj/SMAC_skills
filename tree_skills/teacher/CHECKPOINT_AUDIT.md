# CHECKPOINT_AUDIT（P0-1 / G1，2026-10-31）

审计脚本：`tree_skills/teacher/audit_checkpoints.py`（教师环境 `.venv-teacher`，torch 2.15.0+cpu）。
逐文件审计 JSON：`AUDIT_marl-gpt-main.json`、`AUDIT_marl-gpt-wo-last-token.json`。
Manifest：`TEACHER_MANIFEST.json`（两文件 status=audited，全局 state=audited；ready 需 G2 通过）。

## 结论

**G1 通过**：两个检查点均可信加载并输出合法维度的完整 Actor logits。

| 项 | marl-gpt-main.pt | marl-gpt-wo-last-token.pt |
|---|---|---|
| SHA-256 | c3deaeb…1ce5a（与下载登记一致） | a1864b5…deaf4 |
| 大小 | 86,860,385 B | 86,860,385 B |
| weights_only 加载 | ✅（未执行不受信 pickle） | ✅ |
| strict 加载（非官方 strict=False） | ✅ 0 missing / 0 unexpected | ✅ |
| `_orig_mod.` 前缀 | 存在，按官方 loader 剥离 | 存在 |
| 参数量 | 7,232,256 | 7,232,256 |
| 结构 | 8 层 / 8 头 / 256 embd / block_size 700 / vocab 400 / n_attr 129 / n_agents 20 | 同左 |
| **动作头维度** | **20** | **20** |
| Actor 全 logits 形状 | (B, 20)，全部有限 | (B, 20)，全部有限 |
| 固定输入确定性前向 | ✅ | ✅ |
| 训练任务来源 | **未能从文件确认**（model_args 无任务元数据；多环境词表 400/attr129/action20 表明是多任务基础模型，SMAC 是其中之一） | 同左 |

关键语义：`forward` 返回 `(act_logits, val_logits)`；Actor 头 `lm_head_actor` 输出动作头全宽度 logits，掩码在采样层（`act_actor`）而非模型内——标签取未掩码原始 logits。

## last-token 事实

- `history_len=10` 是 time 嵌入表容量；terran 10v10 token 布局（163 token/帧，含 race token）× 4 帧 = 652 ≤ block_size 700，×10 帧 = 1630 > 700。官方 SMAC 配置 history_len=4。推理用 **H=4**（两种检查点均按此配置验证通过）。
- `main` / `wo-last-token` 两文件的输入约定差异在末尾 env-indicator token；两种模式均已在 G2 管道验证中跑通（`INFERENCE_VALIDATION_run.json`）。选哪个做主教师由 G4 的 B 对照决定，不预设 main 更好。

## 局限（如实）

- 审计在合成输入上验证结构与数值有限性；**未在真实对局观察上验证**（那是 G2/G4）。
- 训练任务来源无法从权重文件确认；是否含人族 10v10 训练经历未知——资格判定完全依赖本项目 G4 实测，不引用论文胜率。
