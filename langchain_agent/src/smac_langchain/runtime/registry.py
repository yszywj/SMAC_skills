"""Candidate gating: config-decided skill availability, applied host-side.

The backend supplies the RAW candidate table (real skill_policy.skill_choices
output bound to its snapshot). This registry decides which of those candidates
the model may see/use this run, per the trusted configuration, and derives the
posture for composed adaptive candidates.
"""
from __future__ import annotations

from typing import Any

ADAPTIVE_POSTURES = ("aggressive", "balanced", "defensive")


def candidate_enabled(config_skills: list[str], backend_candidate: dict[str, Any],
                      spatial_available: list[str]) -> tuple[bool, str | None]:
    """Return (enabled, reason_if_not)."""
    skill = backend_candidate["skill"]
    priority = backend_candidate["priority"]
    if skill == "direct_control":
        return False, "direct_control is not exposed to the agent"
    if skill == "adaptive_focus_fire":
        posture = priority if priority in ADAPTIVE_POSTURES else "balanced"
        if f"adaptive_{posture}" in config_skills:
            return True, None
        return False, f"adaptive posture '{posture}' not enabled"
    if skill == "focus_fire_budget":
        if "focus_fire_budget" in config_skills:
            return True, None  # auto priority and explicit targets both allowed
        return False, "focus_fire_budget not enabled"
    if skill == "cover_retreat":
        return ("cover_retreat" in config_skills, None if "cover_retreat" in config_skills
                else "cover_retreat not enabled")
    if skill == "heal_assign":
        return ("heal_assign" in config_skills,
                None if "heal_assign" in config_skills else "heal_assign not enabled")
    if skill in ("regroup", "cohesive_search"):
        if skill not in spatial_available:
            return False, f"{skill} requires backend spatial switches"
        if skill not in config_skills:
            return False, f"{skill} not enabled in config"
        return True, None
    return False, f"unknown skill {skill}"


def posture_for(backend_candidate: dict[str, Any]) -> str | None:
    """Posture to pass to manager.select for this candidate, else None."""
    if backend_candidate["skill"] != "adaptive_focus_fire":
        return None
    priority = backend_candidate["priority"]
    return priority if priority in ADAPTIVE_POSTURES else "balanced"


def display_name(backend_candidate: dict[str, Any]) -> str:
    skill, priority = backend_candidate["skill"], backend_candidate["priority"]
    if skill == "adaptive_focus_fire":
        return f"adaptive_focus_fire:{priority or 'balanced'}"
    if skill == "focus_fire_budget" and priority:
        return f"focus_fire_budget@{priority}"
    return skill
