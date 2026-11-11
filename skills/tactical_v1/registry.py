"""Skill registry: exactly 7 generic + 3 specialization skills.

Specializations act as REGISTRY-INJECTED providers for G1/G3 — a disabled
S1 is simply absent from the provider dict, so no generic skill can keep a
privately-constructed always-on copy (R5).
"""
from typing import Dict, List

from .skills import (burst_engage, combat_reposition, cohesive_search,
                     coordinated_engage, disengage_reenter, focus_fire,
                     heal_support, local_formation, regroup, splash_fire)

GENERIC = ["focus_fire", "coordinated_engage", "combat_reposition",
           "disengage_reenter", "regroup", "local_formation", "cohesive_search"]
SPECIALIZED = ["splash_fire", "heal_support", "burst_engage"]


def build_skills(config: Dict) -> Dict[str, object]:
    enabled = config.get("enabled", {})
    skills: Dict[str, object] = {}
    for name, factory in (
            ("focus_fire", focus_fire.FocusFireSkill),
            ("coordinated_engage", coordinated_engage.CoordinatedEngageSkill),
            ("combat_reposition", combat_reposition.CombatRepositionSkill),
            ("disengage_reenter", disengage_reenter.DisengageReenterSkill),
            ("regroup", regroup.RegroupSkill),
            ("local_formation", local_formation.LocalFormationSkill),
            ("cohesive_search", cohesive_search.CohesiveSearchSkill),
            ("splash_fire", splash_fire.SplashFireSkill),
            ("heal_support", heal_support.HealSupportSkill),
            ("burst_engage", burst_engage.BurstEngageSkill)):
        skill = factory(config)
        skill.enabled = bool(enabled.get(name, True))
        skills[name] = skill
    # provider injection: only ENABLED specializations reach the generics
    providers = {}
    if skills["splash_fire"].enabled:
        providers["splash_fire"] = skills["splash_fire"]
    skills["focus_fire"].providers = providers
    skills["combat_reposition"].providers = providers
    return skills


def skill_names() -> List[str]:
    return [*GENERIC, *SPECIALIZED]
