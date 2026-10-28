"""最小完整闭环示例：读取冻结战况 → 列候选 → 一次受限执行 → 读报告。

离线运行（mock 后端，真实 SkillManager，无模型、无 SC2）：
    .venv/Scripts/python examples/full_loop_example.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smac_langchain.config import load_config, resolve_path          # noqa: E402
from smac_langchain.backends.mock import MockBackend                 # noqa: E402
from smac_langchain.runtime.coordinator import BattleCoordinator, new_run_id  # noqa: E402
from smac_langchain.tools import build_battle_tools                  # noqa: E402


def main() -> None:
    config = load_config()
    backend = MockBackend(resolve_path("skills/current"))
    coordinator = BattleCoordinator(new_run_id(), 1, backend, config)
    backend.start(1, seed=1, manager_kwargs=coordinator.manager_kwargs)

    tools = {tool.name: tool for tool in build_battle_tools(coordinator)}

    state = json.loads(tools["get_battle_state"].invoke({}))
    print("battle state:", json.dumps(state["battle"], ensure_ascii=False)[:120], "...")
    skills = json.loads(tools["list_available_skills"].invoke(
        {"snapshot_id": coordinator.snapshot.snapshot_id}))
    choice = next(c for c in skills["candidates"]
                  if c["skill_name"].startswith("adaptive_focus_fire"))
    report = json.loads(tools["execute_skill_window"].invoke({
        "choice_id": choice["choice_id"],
        "expected_snapshot_id": coordinator.snapshot.snapshot_id,
        "decision_steps": 6}))
    print("execution:", {k: report[k] for k in
                         ("request_accepted", "outcome", "steps_executed",
                          "stop_reason", "simulated")})
    print("last report:", json.loads(
        tools["get_last_execution_report"].invoke({})).get("outcome"), "...")
    backend.close()


if __name__ == "__main__":
    main()
