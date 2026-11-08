# SMAC Skills

基于 SMAC/SMACv2 的规则技能、环境适配和可学习技能选择。基础 Adaptive 控制器生成全队动作，局部决策层在合法动作范围内选择或撤销单单位修改，最后由同一管理器提交动作和更新记忆。

| 模块 | 用途 |
| --- | --- |
| [skills](skills/README.md) | 当前 Adaptive、历史基线和局部技能 |
| [sc2_skill_adapter](sc2_skill_adapter/README.md) | SMACv1/SMACv2 环境、观察、动作与运行入口 |
| [decision_tree](decision_tree/README.md) | 条件触发、反事实采集、收益/撤销模型及整局验证 |
| [tree_skills](tree_skills/README.md) | 可选的医疗目标偏好评分与树蒸馏模块 |

## 环境与安装

SMACv2 运行环境使用 Python 3.10。需另外安装 StarCraft II，并准备 SMACv2 源码及地图：将 `smacv2` 放在本仓库内或与本仓库同级。SMACv1 和旧 JEV/ASTRA 路径还依赖相同位置的 `game_skill` 外部工程；这些依赖、游戏和模型权重不包含在本仓库中。

从仓库根目录在 PowerShell 执行：

```powershell
$python310 = py -3.10 -c "import sys; print(sys.executable)"
.\sc2_skill_adapter\scripts\setup.ps1 -PythonExe $python310
$py = '.\sc2_skill_adapter\.venv\Scripts\python.exe'
& $py sc2_skill_adapter/run.py check
```

可通过 `SC2PATH` 或运行命令的 `--sc2-path` 指定游戏安装目录。具体对局入口见适配器说明；局部连续干预和训练入口见决策树说明。

## 离线检查

```powershell
& $py -B -m unittest discover -s skills/local_v3/tests
& $py -B -m unittest decision_tree.conditional.test_objective
```

训练与对局输出保存在本地结果目录，不随源码发布。示例策略、部署策略和必要配置随源码保留；它们的存在不代表已验证优于 Adaptive。第三方来源及许可证见 [技能库声明](skills/THIRD_PARTY_NOTICES.md) 和 [许可证目录](skills/licenses)。

当前冻结策略带有源码与环境指纹门禁，要求匹配原 Windows 执行环境：SMACv2 位于仓库同级目录、SC2 5.0.17.98310，以及锁定的依赖和源码。更换平台、布局或版本后需重新采集/拟合并验证，不能绕过上下文检查。
