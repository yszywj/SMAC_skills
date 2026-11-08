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
