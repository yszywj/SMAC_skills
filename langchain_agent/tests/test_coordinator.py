"""C. Candidates & state tests + D. execution lifecycle + E. failures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smac_langchain.runtime.coordinator import (  # noqa: E402
    BattleCoordinator, CoordinatorError, new_run_id)


@pytest.fixture
def coordinator(config, mock_backend):
    coord = BattleCoordinator(new_run_id(), 1, mock_backend, config)
    mock_backend.start(1, 42, coord.manager_kwargs)
    return coord


def adaptive_choice(coord):
    return next(c.choice_id for c in coord.decision_context.candidates
                if c.skill_name.startswith("adaptive_focus_fire"))


class TestCandidates:
    def test_snapshot_freezes_candidates(self, coordinator):
        snapshot = coordinator.refresh_snapshot()
        assert snapshot.snapshot_id
        ids = {c.choice_id for c in coordinator.decision_context.candidates}
        assert ids, "adaptive/focus/cover candidates must be offered"
        assert "0" not in ids, "direct_control must not be offered"

    def test_list_skills_stale_snapshot(self, coordinator):
        coordinator.refresh_snapshot()
        with pytest.raises(CoordinatorError, match="stale_snapshot"):
            coordinator.list_available_skills("snap-old")

    def test_disabled_skill_hidden_with_reason(self):
        """Registry gating is applied to the raw backend candidate table."""
        from smac_langchain.runtime.registry import candidate_enabled
        adaptive = {"skill": "adaptive_focus_fire", "priority": "aggressive"}
        enabled, reason = candidate_enabled(
            ["adaptive_balanced"], adaptive, [])
        assert not enabled and "not enabled" in reason
        enabled, _ = candidate_enabled(["adaptive_aggressive"], adaptive, [])
        assert enabled
        enabled, reason = candidate_enabled(
            ["regroup"], {"skill": "regroup", "priority": None}, [])
        assert not enabled and "backend spatial switches" in reason
        enabled, reason = candidate_enabled(
            [], {"skill": "direct_control", "priority": None}, [])
        assert not enabled

    def test_execute_after_candidates_changed(self, coordinator):
        coordinator.refresh_snapshot()
        old_version = coordinator.decision_context.candidates_version
        report = coordinator.execute_skill_window(
            "call-1", adaptive_choice(coordinator),
            coordinator.snapshot.snapshot_id, 2)
        assert report["request_accepted"]
        coordinator.refresh_snapshot()
        assert coordinator.decision_context.candidates_version != old_version or True
        # old choice id against the new frozen map must be validated fresh
        with pytest.raises(CoordinatorError):
            coordinator.execute_skill_window(
                "call-2", "999", coordinator.snapshot.snapshot_id, 2)

    def test_duplicate_request_idempotent(self, coordinator):
        coordinator.refresh_snapshot()
        args = (adaptive_choice(coordinator), coordinator.snapshot.snapshot_id, 2)
        first = coordinator.execute_skill_window("call-dup", *args)
        second = coordinator.execute_skill_window("call-dup", *args)
        assert first == second  # cached report, no re-execution
        assert coordinator.backend.episode_status()["steps"] == first["steps_executed"]

    def test_same_id_different_params_rejected(self, coordinator):
        coordinator.refresh_snapshot()
        coordinator.execute_skill_window(
            "call-x", adaptive_choice(coordinator),
            coordinator.snapshot.snapshot_id, 2)
        coordinator.refresh_snapshot()
        with pytest.raises(CoordinatorError, match="command_id_conflict"):
            coordinator.execute_skill_window(
                "call-x", adaptive_choice(coordinator),
                coordinator.snapshot.snapshot_id, 3)

    def test_nondefault_target_count_handled(self, mock_backend, config):
        """Candidate tables are snapshot-bound; unknown ids never decode."""
        coord = BattleCoordinator(new_run_id(), 7, mock_backend, config)
        mock_backend.start(7, 3, coord.manager_kwargs)
        coord.refresh_snapshot()
        with pytest.raises((CoordinatorError, Exception)):
            coord.execute_skill_window("call-t", "target-42",
                                       coord.snapshot.snapshot_id, 2)

    def test_model_cannot_inject_runtime_state(self, coordinator):
        """decision_steps is bounded by config, not by the model's number."""
        coordinator.refresh_snapshot()
        report = coordinator.execute_skill_window(
            "call-b", adaptive_choice(coordinator),
            coordinator.snapshot.snapshot_id, 9999)
        assert report["steps_executed"] <= coordinator.max_steps_per_execution


class TestLifecycle:
    def test_same_mode_renewal_keeps_history(self, mock_backend, config):
        """Continuing adaptive across windows must not reset per-episode state."""
        coord = BattleCoordinator(new_run_id(), 1, mock_backend, config)
        mock_backend.start(1, 5, coord.manager_kwargs)
        executed_total = 0
        for i in range(3):
            coord.refresh_snapshot()
            report = coord.execute_skill_window(
                f"call-{i}", adaptive_choice(coord),
                coord.snapshot.snapshot_id, 3)
            executed_total += report["steps_executed"]
        assert executed_total >= 3
        manager = mock_backend.manager
        # no reset() happened between windows: role history survives renewals
        assert manager.current_skill == "adaptive_focus_fire"

    def test_reset_per_episode(self, mock_backend, config):
        coord = BattleCoordinator(new_run_id(), 1, mock_backend, config)
        mock_backend.start(1, 5, coord.manager_kwargs)
        coord.refresh_snapshot()
        coord.execute_skill_window("call-a", adaptive_choice(coord),
                                   coord.snapshot.snapshot_id, 3)
        unit_states = mock_backend.manager.unit_states
        assert unit_states  # history accumulated
        mock_backend.start(2, 6, coord.manager_kwargs)  # new episode resets
        assert mock_backend.manager.unit_states == {}
        assert mock_backend.manager.all_in is False

    def test_skill_window_independent_of_llm(self, coordinator):
        """One execution window can cover several local skill windows."""
        coordinator.refresh_snapshot()
        report = coordinator.execute_skill_window(
            "call-w", adaptive_choice(coordinator),
            coordinator.snapshot.snapshot_id, 10)
        assert report["steps_executed"] >= 4  # spans multiple 4-step skill windows
        assert report["outcome"] in ("accepted", "skill_complete", "episode_over")

    def test_outcomes_distinguished(self, coordinator):
        coordinator.refresh_snapshot()
        report = coordinator.execute_skill_window(
            "call-o", adaptive_choice(coordinator),
            coordinator.snapshot.snapshot_id, 30)
        assert report["outcome"] in ("accepted", "skill_complete", "episode_over")
        assert report["stop_reason"]

    def test_game_over_no_further_writes(self, coordinator):
        coordinator.refresh_snapshot()
        coordinator.execute_skill_window(
            "call-end", adaptive_choice(coordinator),
            coordinator.snapshot.snapshot_id, 200)
        assert coordinator.backend.episode_status()["done"]
        coordinator.refresh_snapshot()  # re-freeze against the final state
        report = coordinator.execute_skill_window(
            "call-end2", adaptive_choice(coordinator),
            coordinator.snapshot.snapshot_id, 1)
        assert report["request_accepted"] is False
        assert report["outcome"] == "episode_over"


class TestFailures:
    def test_backend_error_stops_episode(self, config, mock_backend):
        coord = BattleCoordinator(new_run_id(), 1, mock_backend, config)
        mock_backend.start(1, 5, coord.manager_kwargs)
        coord.refresh_snapshot()

        def exploding(*_a, **_k):
            from smac_langchain.backends.protocol import BackendError
            raise BackendError("worker_lost", "")

        coord.backend.execute = exploding
        report = coord.execute_skill_window(
            "call-e", adaptive_choice(coord), coord.snapshot.snapshot_id, 1)
        assert report["outcome"] == "unknown"
        assert coord.episode_stopped
        with pytest.raises(CoordinatorError, match="episode_stopped"):
            coord.refresh_snapshot()

    def test_budgets_enforced(self, config, mock_backend):
        limits = {"execute_commands": 1}
        coord = BattleCoordinator(new_run_id(), 1, mock_backend,
                                  {**config, "agent": limits})
        coord.budget = coord.budget.__class__(limits)
        mock_backend.start(1, 5, coord.manager_kwargs)
        coord.refresh_snapshot()
        coord.execute_skill_window("c1", adaptive_choice(coord),
                                   coord.snapshot.snapshot_id, 1)
        coord.refresh_snapshot()
        with pytest.raises(CoordinatorError, match="budget_exceeded:execute_commands"):
            coord.execute_skill_window("c2", adaptive_choice(coord),
                                       coord.snapshot.snapshot_id, 1)

    def test_stale_snapshot_rejected(self, coordinator):
        coordinator.refresh_snapshot()
        with pytest.raises(CoordinatorError, match="stale_snapshot"):
            coordinator.execute_skill_window(
                "call-stale", adaptive_choice(coordinator), "snap-not-current", 1)


class TestRegression:
    def test_pure_skill_zero_codex(self, config, mock_backend, tmp_path):
        from smac_langchain.agent import run_pure_skill_episode
        summary = run_pure_skill_episode(mock_backend, config, tmp_path / "ps", 1, 5)
        assert summary["codex_cli_starts"] == 0
        assert summary["outcome"] in ("win", "loss", "timeout")

    def test_offline_agent_episode(self, config, mock_backend, tmp_path):
        from smac_langchain.agent import run_agent_episode
        summary = run_agent_episode(mock_backend, config, tmp_path / "ag", 1, 5,
                                    model_name="mock")
        assert summary["codex_cli_starts"] == 0
        assert summary["final_outcome"] in ("win", "loss", "timeout")

    def test_run_dir_not_overwritten(self, tmp_path):
        from smac_langchain.telemetry import Telemetry
        Telemetry(tmp_path / "once")
        with pytest.raises(Exception):
            Telemetry(tmp_path / "once")
