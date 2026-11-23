"""Exploration and bounded-experiment protocol structures (Phase 8).

Data structures and gating only — nothing here runs games, chooses random
actions, or collects trajectories. The existing rule skill is always the
BASE arm; interventions are bounded, cooled-down, budgeted per family and
per unit, and any unfinished experiment stays flagged incomplete and is
excluded from gain training by the dataset layer.

``EXPERIMENT_MANIFEST`` bookkeeping lives in ``experiment_manifest.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .schema import load_json

EXPLORATION_CONFIG_KIND = "zerg_exploration_config"
DEFAULT_CONFIG_PATH = "decision_tree/zerg/configs/exploration.json"


def default_exploration_config() -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    return load_json(str(repo_root / DEFAULT_CONFIG_PATH))


def validate_exploration_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if config.get("kind") != EXPLORATION_CONFIG_KIND:
        raise ValueError("exploration config requires kind="
                         f"{EXPLORATION_CONFIG_KIND}")
    limits = config.get("limits", {})
    for field in ("max_interventions_per_episode",
                  "max_interventions_per_unit", "unit_cooldown_steps",
                  "max_counterfactual_branches_per_scene",
                  "intervention_horizon_steps"):
        if type(limits.get(field)) is not int or limits[field] < 0:
            raise ValueError(f"limits.{field} must be a non-negative integer")
    if not isinstance(limits.get("family_budgets", {}), dict):
        raise ValueError("limits.family_budgets must be an object")
    gating = config.get("gating", {})
    if gating.get("low_confidence_fallback") not in ("base", "rule", None):
        raise ValueError("gating.low_confidence_fallback must be 'base'/'rule'")
    return config


class InterventionBudget:
    """Per-episode ledger for bounded interventions."""

    def __init__(self, config: Dict[str, Any]):
        self.config = validate_exploration_config(config)
        limits = self.config["limits"]
        self.max_episode = int(limits["max_interventions_per_episode"])
        self.max_unit = int(limits["max_interventions_per_unit"])
        self.cooldown = int(limits["unit_cooldown_steps"])
        self.family_budgets = {family: int(budget) for family, budget
                               in limits.get("family_budgets", {}).items()}
        self.max_branches = int(limits["max_counterfactual_branches_per_scene"])
        self.episode_count = 0
        self.unit_counts: Dict[str, int] = {}
        self.unit_last_step: Dict[str, int] = {}
        self.family_counts: Dict[str, int] = {}
        self.scene_branches: Dict[str, int] = {}
        self.interventions: List[Dict[str, Any]] = []

    def allow(self, *, actor: Optional[str], family: Optional[str],
              scene_id: str, step: int) -> Tuple[bool, str]:
        """Deterministic budget gate. Returns (allowed, reason)."""
        if self.episode_count >= self.max_episode:
            return False, "episode_budget_exhausted"
        if actor is not None:
            if self.unit_counts.get(actor, 0) >= self.max_unit:
                return False, "unit_budget_exhausted"
            last = self.unit_last_step.get(actor)
            if last is not None and step - last < self.cooldown:
                return False, "unit_cooldown_active"
        if family is not None and family in self.family_budgets:
            if self.family_counts.get(family, 0) >= self.family_budgets[family]:
                return False, "family_budget_exhausted"
        if self.scene_branches.get(scene_id, 0) >= self.max_branches:
            return False, "scene_branch_budget_exhausted"
        return True, "allowed"

    def record(self, *, actor: Optional[str], family: Optional[str],
               scene_id: str, step: int, candidate_id: str,
               intervention_type: str, horizon: int) -> Dict[str, Any]:
        allowed, reason = self.allow(actor=actor, family=family,
                                     scene_id=scene_id, step=step)
        if not allowed:
            raise RuntimeError(f"budget gate refused: {reason}")
        self.episode_count += 1
        if actor is not None:
            self.unit_counts[actor] = self.unit_counts.get(actor, 0) + 1
            self.unit_last_step[actor] = step
        if family is not None:
            self.family_counts[family] = self.family_counts.get(family, 0) + 1
        self.scene_branches[scene_id] = self.scene_branches.get(scene_id, 0) + 1
        row = {
            "actor": actor, "family": family, "scene_id": scene_id,
            "step": step, "candidate_id": candidate_id,
            "intervention_type": intervention_type,
            "intervention_horizon": int(horizon),
            "unfinished": True,          # flagged until the pair completes
        }
        self.interventions.append(row)
        return row

    def mark_finished(self, row: Dict[str, Any]) -> None:
        row["unfinished"] = False

    def summary(self) -> Dict[str, Any]:
        return {
            "interventions": len(self.interventions),
            "unfinished": sum(1 for row in self.interventions
                              if row["unfinished"]),
            "per_family": dict(self.family_counts),
            "per_scene_branches": dict(self.scene_branches),
        }
