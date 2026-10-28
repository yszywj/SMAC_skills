# v8 群体空间技能验收报告（2026-10-25）

## 完成层级声明

1. **独立技能实现 + 离线测试**：已完成并通过（见下）。
2. **适配器/运行入口接入**：CLI 开关、候选协议与 always 模式合并路径已接入 `game_skill/micro/jev_micro/run.py` 并通过解析器冒烟；未启动真实 SC2 进程。
3. **实际对局评估（任务10 A0–A3 配对测试）**：**未执行**。本机本轮未启动 SC2 对局；离线结果不能也不被解释为胜率变化。需要时可用：
   `python -m jev_micro.run --map 3m --policy nearest --skill-mode always --skill-name adaptive_focus_fire --adaptive-coordinated-approach --enable-regroup --enable-cohesive-search --enable-combat-positioning --auto-spatial-dispatch ...`（A0 为去掉三个空间开关的同一命令）。

## 真实执行的命令与结果

解释器：`C:\Users\ry\skill\smacv2_jev_adapter\.venv\Scripts\python.exe`（工作区唯一含 httpx 等依赖的环境；系统 Python 缺依赖且 pip 无索引）。

| 命令 | 结果 |
| --- | --- |
| `-m unittest skills.current.tests.test_group_skills`（cwd=C:\Users\ry\skill） | 25/25 通过 |
| `-m unittest skills.current.tests.test_approach_progress` | 11/11 通过 |
| `-m unittest discover -s tests`（cwd=game_skill\micro） | 407/407 通过 |
| `-m unittest discover -s tests`（cwd=smacv2_jev_adapter） | 36/36 通过 |
| `build_parser().parse_args(['--enable-regroup','--auto-spatial-dispatch',...])` 冒烟 | 开关解析正确 |

跳过：无（以上套件全部执行，0 跳过）。真实对局：0 局（未执行，见上）。

## 性能（离线基准，单进程，10 单位快照）

| 步骤 | mean | p95 | max (ms) |
| --- | --- | --- | --- |
| allocate_regroup | 0.447 | 0.603 | 1.264 |
| allocate_cohesive_search | 0.410 | 0.498 | 1.674 |
| allocate_combat_positioning | 0.399 | 0.519 | 42.957（首次含导入的一次离群） |

复杂度为成员数×合法候选的常数倍（无寻路搜索），随人数线性增长，无界搜索不存在。

## 覆盖情况对照（任务9测试组）

- 旧行为回归：候选表全等断言、默认全关、禁用时 select 拒绝、combat_params 无 positioning 键。
- 规模与控制范围：0/1/3/4/7/10 人场景；死亡单位（alive=False）分组测试；部分控制（站位测试仅 u3）。
- 无敌人观察：集结与搜索测试均空 `visible_enemies`、空 `relations` 敌项。
- 集结：7+3 收敛且主体不被拖动、不可达掉队者超时/受阻、接触中断≠成功、单体/空集快速结束。
- 搜索：航点到达消耗、接触中断、无航点退化输出合法动作、严重脱节 needs_regroup、12 步不长期分裂。
- 站位：合法攻击与固定动作绝不被覆盖、间距配置生效、多方向敌不冲中心、无敌退回协同接敌。
- 生命周期：预算跨窗口累计、reset 清空、同一快照重复调用不重复扣预算、时间回退不爆预算、输入顺序无关。

尚未单测覆盖（诚实列出）：移动碰撞子项（出生重叠、单方向、A-B-A 在空间技能下的专门夹具——部分由复用的 best_move_toward/换向惩罚间接覆盖）；协议组中 `max_explicit_targets` 变值与旧请求解码的专门断言（候选表生成逻辑未改动，风险低）。

## 修改文件清单

skills/ 内：`skills/current/skills/group_geometry.py`（新增）、`regroup.py`（新增）、`cohesive_search.py`（新增）、`combat_positioning.py`（新增）、`tests/test_group_skills.py`（新增）、`docs/GROUP_SKILLS.md`（新增）、`docs/GROUP_SKILLS_ACCEPTANCE.md`（新增）；修改 `skills/current/skill_manager.py`、`skill_policy.py`、`skills/adaptive_focus_fire.py`、`README.md`。
skills/ 外（已按要求申报）：`game_skill/micro/jev_micro/run.py`（CLI 开关、spatial_modes 请求/解码、always 模式部分动作合并）。

未改动：`baseline_v2_70pct`、`baseline_v6_20261024`、历史报告与迁移校验记录。
