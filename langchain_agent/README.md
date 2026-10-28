# smac-langchain：LangChain + Codex CLI 高层战术 Agent

在既有 `skills/current` + `sc2_skill_adapter` 之上叠加一个 LangChain Agent：
`create_agent` 负责工具循环，本机 Codex CLI（`codex exec` 非交互）作为受限结构化
决策后端，`BattleCoordinator` 单点串行写入，真实技能栈（adaptive focus fire 等）
继续执行每一步微操。原 pure-skill 入口完全不受影响。

## 安装（Windows）

```bat
cd C:\Users\ry\skill\SMAC_agent\langchain_agent
python -m venv .venv
.venv\Scripts\python -m pip install -e .[test] -i https://pypi.org/simple
```

实测版本：langchain 1.4.7 / langchain-core 1.7.0 / pydantic 2.14.1 / Python 3.13。
SMACv2 后端 worker 需要原来的后端解释器（含 pysc2/smacv2/numpy/yaml），
在 `configs/local.toml` 的 `[paths] backend_python` 指定；不设置则用当前 Python。

## 配置

`configs/default.toml` 是默认（学习配置，非调优）。可复制 `configs/example.local.toml`
为 `configs/local.toml` 覆盖。相对路径一律以 `SMAC_agent/` 为基准。
Codex 模型默认 `"default"`（使用 CLI 账户默认模型，不硬编码账户相关模型名）。
不读取、不复制任何登录令牌；CLI 使用它自己的既有登录。

## 命令

| 命令 | 说明 | 真实模型 | 启动 SC2 |
|---|---|---|---|
| `.venv\Scripts\python -m smac_langchain doctor` | 本地能力/配置/路径检查 | 否 | 否 |
| `... demo` | 离线完整 Agent 演示（mock 模型 + mock 后端） | 否 | 否 |
| `... codex-smoke --live-llm` | 真实 CLI 结构化输出冒烟（约 1 次请求） | 是 | 否 |
| `... agent-smoke --backend mock --live-llm` | 真实 Codex + 完整工具循环 | 是 | 否 |
| `... run --backend mock --policy pure-skill` | 无模型技能对照 | 否 | 否 |
| `... run --backend smacv2 --policy pure-skill` | 真环境技能对照（worker） | 否 | 是 |
| `... run --backend smacv2 --policy agent --live-llm` | 真实 Agent 入口 | 是 | 是 |
| `.venv\Scripts\python -m pytest tests` | 离线测试（默认跳过 live 标记） | 否 | 否 |

SMACv1 本期未接入：传入 smacv1 后端会明确报错，不用 SMACv2 冒充。

## 测试边界

默认测试完全离线：mock 模型（`MockTacticalChatModel`）、mock 后端（`MockBackend`，
合成战斗 + **真实 SkillManager**），以及用 Python 替身进程测试 Codex CLI 协议
（超时/退出码/JSONL/Schema/禁用内置工具事件/重试预算等）。真实 CLI 测试只经
`--live-llm` 显式触发；真实游戏测试需要 SC2 与后端解释器。

## 常见故障

- `codex_native_not_found`：npm shim 未解析到 codex.exe，在 config 设 `codex_path`。
- `model_not_found`（CLI 侧）：账户不支持该模型 → 用 `model = "default"`。
- `rpc_worker_not_ready` / worker stderr 报缺依赖：backend_python 指向的解释器
  需已安装 pysc2/smacv2/numpy/yaml，且 `smac_root` 路径正确。
- 地图缺失：worker 使用 `smacv2/.../SMAC_Maps/10gen_<race>.SC2Map`。
