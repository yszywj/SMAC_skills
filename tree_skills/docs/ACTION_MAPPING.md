# ACTION_MAPPING（P0-2 / G2）

权威实现：`tree_skills/teacher/action_mapping.py`（+ `tests/test_action_mapping.py`）。
观察构建：`tree_skills/teacher/obs_builder.py`。验证运行：`teacher/INFERENCE_VALIDATION_run.json`（synthetic_only 管道证据）。

## 动作映射（以本地 SMACv2 源码为准）

依据 `C:\Users\ry\skill\smacv2\smacv2\env\starcraft2\starcraft2.py::get_avail_agent_actions`：

- 0 = no-op（仅死亡单位），1 = stop，2–5 = 移动 N/S/E/W，**6+t = 目标 t**
- terran_gen/MMM 图上医疗机的 target_items 换成**友军列表（排除医疗机）**：对医疗机 6+j = 治疗友军 j（射程/掉血/存活由环境掩码保证）；对地面单位 6+t = 攻击敌人 t。**同一整数、两种语义，按单位类型选择，绝不复用攻击映射**
- n_actions = 6 + max(n_agents, n_enemies) = **16**（10v10）
- 教师头 20 维：槽 16–19 是跨环境 padding，掩码屏蔽，永不进入标签
- 本项目 keep-heal-order 扩展（动作 ID 16 = native_n_actions）**不是教师动作**：`validate_teacher_slots` 拒绝；不进 logits 均值、不进损失
- 官方实现无任何动作重排/offset（`smac_v2.py` 直接透传 Discrete(n_actions)），教师槽位 = 原生整数

## 场景对照

- 官方 `terran_10_vs_10`（`envs/smac_env/config.py:133`）→ 地图 `10gen_terran`，与本项目一致；单位权重 0.45/0.45/0.10、step_mul 8、难度 7 一致
- 差异：官方 base_config 未显式设 unit_balance；本项目强制 `native`（A/B/C/D 统一用本项目协议，不因此宣称与官方训练分布完全一致）
- 医疗机数量是随机的（权重 0.10/单位）：无医疗机对局是正常覆盖，不剔除

## 观察编码（官方 wrapper 流程的等价复现）

原始 `env.get_obs_agent(a)`（162 维 = move 4 + enemy 10×8 + ally 9×8 + own 6）：

1. 切分 own/ally/enemy token（官方 decompose_obs 切片逻辑）
2. 每 (episode, agent) 独立历史：H=4 帧，最新帧在前，开局零前向填充（官方 stack_dict_tensor 语义）
3. 官方 `smacTokenizer`（terran attr 表：own [0,1,2,3,11,9,10,13,14,15]，ally [5,6,7,8,11,13,14,15]，enemy [4,6,7,8,11,13,14,15]；每标量一 token；group 0/1，agent own=0/ally=1..9/enemy=0..9，time 最新=0 递增）
4. race token 追加（terran=0，attr 16）——官方 `obs_with_race` 默认开
5. `reshape_obs` 补零至 block_size 700（pad 的位置 id 全 0，与官方一致）
6. 可选 last_token（attr 128，obs=env_indicator=1）

容量守卫：token 数超 block_size 显式报错（H=10 需 1630 > 700，被拒绝而非截断）。

## 幂等与隔离

- 同一 (episode, agent, game_loop) 重复请求不重复推进历史（服务端 `observe` 幂等；客户端 `begin_step` 重复调用直接抛错）
- 每局 reset 清空该局全部 agent 历史；单位死亡不串历史（死亡单位 obs 全零照常进历史，与官方一致）
- 一次批量请求包含同帧全部单位，每个 agent 的历史与掩码一一对应

## 已验证 / 未验证

- ✅ 管道级（synthetic_only）：形状、索引范围、有限 logits、跨服务重启确定性、重复步防护、越界槽位防护、last-token 两变体
- ❌ 未做：10 个真实医疗机决策快照的人工核对表（需真实轨迹采集时逐条产出，`DATA_PROTOCOL` 采集器已含字段）；与官方参考 wrapper 在真实环境上的逐字段对照（官方 wrapper 需要其自有 SC2 安装方式，暂以源码级对照替代）——`ready` 前必须补上真实快照核对
