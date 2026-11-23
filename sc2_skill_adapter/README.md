# SC2 环境适配器

统一维护 SMACv1（SMAC-Hard）与 SMACv2 的观察、合法动作和运行入口。技能源码在 [skills](../skills/README.md)，可学习选择层在 [decision_tree](../decision_tree/README.md)。

| 路径 | 职责 |
| --- | --- |
| `adapter.py` | 为不同后端启动独立 Python 子进程 |
| `bridge/`、`run.py` | SMACv2 观察转换、执行、场景与审计 |
| `smacv1/jev_micro/` | SMACv1/SMAC-Hard 封装与运行代码 |
| `configs/`、`scripts/`、`tests/` | 基础配置、命令包装与离线测试 |

## 环境

SMACv2 使用独立 Python 3.10 环境，依赖见 `requirements.txt` 和 `requirements-lock.txt`。StarCraft II 需另外安装；`smacv2` 外部源码及地图可位于仓库内或仓库同级。SMACv1 和旧 JEV/ASTRA 路径还依赖同位置的 `game_skill` 工程及其独立环境。

从仓库根目录在 PowerShell 执行：

```powershell
$python310 = py -3.10 -c "import sys; print(sys.executable)"
.\sc2_skill_adapter\scripts\setup.ps1 -PythonExe $python310
$py = '.\sc2_skill_adapter\.venv\Scripts\python.exe'
& $py sc2_skill_adapter/run.py check
.\sc2_skill_adapter\scripts\test.ps1
```

用 `SC2PATH` 或 `--sc2-path` 指定游戏路径。统一入口的后端参数分别见 `adapter.py smacv1 --help`、`adapter.py smacv2 --help`，后端环境需先就绪。

## SMACv2 使用

`scenarios` 支持 `protoss|terran|zerg` 和 5v5/10v10。以下第一条命令生成清单，第二条启动真实 Adaptive 对局；清单和结果目录必须使用新名称。

```powershell
& $py sc2_skill_adapter/run.py scenarios --race protoss --size 10 --episodes 2 --seed 100 --unit-balance native --output sc2_skill_adapter/configs/example.json
& $py sc2_skill_adapter/run.py run --method adaptive-skill --manifest sc2_skill_adapter/configs/example.json --output-dir sc2_skill_adapter/runs/example
& $py sc2_skill_adapter/run.py run --help
```

`--unit-balance native` 使用双方同兵种的原生属性；`original` 保留 SMACv2 原始单位协议。二者属于不同实验配置，结果应分别比较。对照实验复用同一场景清单，并核验初始化状态、控制参数和源码版本。

`--skill-version` 选择技能源码，`--local-policy` 接入单次局部策略。条件触发连续干预通过决策树兼容入口运行。旧 `jev`、`astra-jev` 方法还需配置相应外部模型服务；纯 Adaptive 不调用这些服务。

运行日志、回放和分析输出留在本地，不包含在代码发布中。

### 执行模式：8-loop 战术决策 + 1-loop 刺蛇攻击触发器

`run`/`suite` 接受 `--execution-mode`：

- `legacy8`（默认）：每拍一次 `env.step(actions)`，行为与旧命令完全一致。
- `micro8-off` / `micro8-hydra`：共用同一段微步循环。仍保留 `step_mul=8`：宏步在 L 拍提交全队最终动作，L+1..L+7 每 1 个游戏循环推进并读观测，L+8 结算该宏步（奖励、减员、`episode_steps` 各只计一次）。两种模式由同一授权状态机逐循环检查相同的 `micro_attack_intents`；唯一差异是 `off` 只记录，`hydra` 才发送授权的定向 `Attack(A)`。仅支持 `step_mul=8` 的清单，且要求 `--method adaptive-skill --skill-version tactical_v1`。

授权只由 `skills/tactical_v1` 的 `approach_hydralisk` 分支在最终动作仲裁之后产生（存活刺蛇的最终动作未被覆盖、目标是决策时可见的特定敌人、且该移动确实缩短与目标的距离）。执行层不选敌、不补 STOP/no-op/移动，每刺蛇每窗口至多补发一次非排队攻击；目标死亡或丢失视野即撤销。

日志：`run.json` 记 `execution_mode`/`macro_loops`/`engine_step_loops`；`events.jsonl` 记授权注册、每循环检查、状态迁移与窗口关闭；`micro_triggers/episode-*.json` 保存全部授权记录（宏步编号、双方 tag、授权/首次合法/实际提交 loop、最终状态：`submitted`/`already_ordered`/`target_lost`/`never_legal`/`legal_control_off`/`actor_dead`；`submitted` 仅表示命令已提交，不是开火或命中）。配对入口：`scripts/run_contact_zerg_ab.py --micro-trigger`，两组子进程分别以 `--execution-mode micro8-off`/`micro8-hydra` 启动，共用同一批清单与同一份当前策略代码。

### 冻结版刺蛇开关实验（isolated_hydra_escape / protect_hydra_next_shot）

在冻结快照 `runs/contact_response_micro_20261121_153552_909880/source_snapshot/` 的**副本**上注入两个默认关闭的刺蛇开关（冻结源、历史结果与工作区默认策略都不改动）。`scripts/hydra_control_switches.py::apply_variant` 把 `hydra_switch_module_template.py` 写入副本的 `skills/tactical_v1/hydra_switches.py`，按组写入 `hydra_switches.json` 开关标志，并在副本的 `skill_manager.py`（coordinator 最终仲裁之后）和 runner（决策行收集与每局 `hydra_switches/episode-*.json`）打入锚点补丁。

- A `isolated_escape`：仅当最终动作是攻击、且同时满足孤立（当前受威胁且没有第二条存活刺蛇能对同一威胁立即还击）、交换不利（敌狗+敌蛇合围，或连续掉血且对射血量明显落后）、存在值得移动的一步落点（甩掉威胁且不引入新威胁，或把追击引入友蛇射界，不占用友军保留落点）时，把这一条蛇的这一次攻击换成一个合法移动。立即击杀、或武器就绪且非危急时保留攻击。每窗口至多一条，下一拍重新判断。
- B `protect_next_shot`：只检查最终动作为后撤类移动的刺蛇（disengage ESCAPE 非救援、combat_reposition retreat、或共享几何分类为后撤），当存在合法保留目标、武器已就绪或下一宏步前就绪、且该移动既不减威胁也不改善支援还丢失现有射程时，改发保留目标的攻击。紧急救援、真实脱险、改善支援的移动不拦。A 先行；B 不会改回 A 的有效撤离。

判定与跟随观察记录在每局 `hydra_switches/episode-*.json`（原技能/动作、候选、实发、目标、CD、血量史、友蛇支援、估计标志；followup 行含 16/24 loops 存活、冷却重启开火代理、威胁进入友蛇射界）。离线检查：`python scripts/test_hydra_switches.py`（29 项断言，无 SC2）。四组实验：`python scripts/run_hydra_switches_ab.py [--dry-run|--retry <batch>|--report-only]`。

### 双边伤害遥测地图

从本目录运行 `export SMAC_DAMAGE_MAP_DIR="$PWD/damage_maps"` 可选用已通过 runner 真机验证的三张副本，无须 `smpq`。`10gen_{protoss,terran,zerg}.SC2Map` 是逻辑场景文件名；三者的源底图均为历史基线 `32x32_flat.SC2Map`，不是带预置部队的 `10gen_*` 原图。部队由 runner 生成。

如需在别处生成副本，使用 `python scripts/make_damage_map.py --resources --race zerg /path/to/32x32_flat.SC2Map /new/path/10gen_zerg.SC2Map`。种族必须显式指定，输出目录须存在，已有文件不会被覆盖。生成器验证源底图 SHA-256，找不到 `smpq` 时使用校验过的对应正式件。三个种族的遥测读取 ABI 不变。
