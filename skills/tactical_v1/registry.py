"""Skill registry: exactly 7 generic + 3 specialization skills."""
from typing import Dict, List

from .skills import (burst_engage, combat_reposition, cohesive_search,
                     coordinated_engage, disengage_reenter, focus_fire,
                     heal_support, local_formation, regroup, splash_fire)

GENERIC = ["focus_fire", "coordinated_engage", "combat_reposition",
           "disengage_reenter", "regroup", "local_formation", "cohesive_search"]
SPECIALIZED = ["splash_fire", "heal_support", "burst_engage"]

_FACTORIES = {
    "focus_fire": focus_fire.FocusFireSkill,
    "coordinated_engage": coordinated_engage.CoordinatedEngageSkill,
    "combat_reposition": combat_reposition.CombatRepositionSkill,
    "disengage_reenter": disengage_reenter.DisengageReenterSkill,
    "regroup": regroup.RegroupSkill,
    "local_formation": local_formation.LocalFormationSkill,
    "cohesive_search": cohesive_search.CohesiveSearchSkill,
    "splash_fire": splash_fire.SplashFireSkill,
    "heal_support": heal_support.HealSupportSkill,
    "burst_engage": burst_engage.BurstEngageSkill,
}


def build_skills(config: Dict) -> Dict[str, object]:
    """Instantiate every registered skill. Disabled skills (ablation switches
    config['enabled'][skill] = False) are still registered but their propose()
    is skipped by the coordinator, keeping the registry at exactly ten."""
    enabled = config.get("enabled", {})
    skills = {}
    for name, factory in _FACTORIES.items():
        skill = factory(config)
        skill.enabled = bool(enabled.get(name, True))
        skills[name] = skill
    return skills


def skill_names() -> List[str]:
    return [*GENERIC, *SPECIALIZED]
