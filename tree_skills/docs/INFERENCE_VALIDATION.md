# INFERENCE_VALIDATION（P0-3 管道证据，2026-10-31）

运行：`tree_skills/.venv-teacher/Scripts/python.exe tree_skills/teacher/validate_inference.py --last-token {off,on}`
结果文件：`teacher/INFERENCE_VALIDATION_run.json`（含 `synthetic_only: true`）。

常驻服务：`tree_skills/teacher/service.py`（权重启动加载一次、JSONL stdin/stdout、批量同帧全部单位一次前向、结构化错误、CPU）；客户端：`tree_skills/teacher/inference.py`（`TeacherServiceClient` + `TeacherPairScorer`，适配器 venv 可用、无 torch 依赖）。

## 验证结果（两种 last-token 变体均通过）

| 检查 | last_token=off | last_token=on |
|---|---|---|
| 服务握手 + model_sha256 | ✅ c3deaeb… | ✅ c3deaeb… |
| init（真实 env 维度） | ✅ 163 token/帧 | ✅ |
| 3 步 × 10 单位批量 logits | ✅ 长度 20，全部有限 | ✅ |
| 跨服务重启确定性（逐值相等） | ✅ | ✅ |
| 重复 (episode, loop) 双推进防护 | ✅ TeacherServiceError | ✅ |
| heal 槽位评分（u4→u0 = 槽 6） | ✅ 有限数值 | ✅ |
| 越界 ally / keep / padding 槽防护 | ✅ ActionMappingError | ✅ |

教师每次真实决策时刻只调用一次（全队一批），同一医疗机的多候选在本地切分缓存的完整 logits 向量；不为候选改变教师历史。

## 运行链接线（默认关闭）

- `SkillManager(..., pair_scorer/pair_scoring/pair_histories/pair_strict)`（实例级，仅 current+terran adaptive；off 分支零开销）
- runner 新 CLI：`--pair-mode {off,a-plug,shadow,teacher-score,tree-score}`、`--pair-model-dir`、`--pair-teacher-checkpoint`、`--pair-last-token`、`--pair-failure {strict,tolerant}`、`--pair-shadow-of`
- `PairRuntime.begin_step` 在每步 `skill_manager.step` 之前推进教师历史（原始 `get_obs_agent` + 原生 avail）与健康率历史（学生历史特征的唯一生成器，训练/部署同源）
- 失败策略：strict 抛错→该局 invalid（正式实验默认）；tolerant 回退原匹配但逐机记录（诊断用）
- 资源：`--pair-mode off`（默认）时不导入 tree_skills、不启动教师进程、无网络

## 未验证 / 下一步（G3→G4）

- G3 真实对局等价（A vs A-plug 同初始状态逐步动作一致）：冒烟 run 进行中，结果记入 RESULTS.md
- G4 教师 B 对照（40 局筛选→200 局资格）与真实快照 10 条人工核对表——`ready/qualified` 状态在此之前不授予
