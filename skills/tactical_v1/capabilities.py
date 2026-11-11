"""Unit capability normalization.

Two generic combat configurations (ranged, melee) cover the six normal unit
types; three mechanism specializations are registered by actual capability,
never by requiring the caller to pass a race profile. Unit counts, action
dimensions and alive sets always come from the snapshot.
"""
from dataclasses import dataclass
from typing import Dict, Optional

# Verified SC2 catalog ids used by the adapter's native balance
# (bridge/config.py RACE_UNITS) and the standard SMACv2 rosters.
TYPE_IDS = {
    48: "marine", 51: "marauder", 54: "medivac",
    73: "zealot", 74: "stalker", 4: "colossus",
    105: "zergling", 9: "baneling", 107: "hydralisk",
}
NAME_ALIASES = {name: name for name in TYPE_IDS.values()}


@dataclass(frozen=True)
class Capability:
    key: str                    # canonical lowercase name
    combat_class: str           # ranged | melee | support | burst
    specialization: Optional[str]   # splash_fire | heal_support | burst_engage | None
    flying: bool
    healer: bool
    suicide_attack: bool        # consumed on a successful detonation
    splash_attack: bool
    data_completeness: str      # complete | partial | unknown

    @property
    def uses_normal_focus_fire(self) -> bool:
        return self.combat_class in ("ranged", "melee") and self.specialization is None

    @property
    def uses_normal_kiting(self) -> bool:
        return self.combat_class == "ranged" and self.specialization is None


RANGED = ("marine", "marauder", "stalker", "hydralisk")
MELEE = ("zealot", "zergling")
SPECIALIZATIONS = {"colossus": "splash_fire", "medivac": "heal_support",
                   "baneling": "burst_engage"}

_CAPABILITY_CACHE: Dict[str, Capability] = {}

# SMACv2 "original" balance renames units (e.g. "Stalker_RL") and the
# generated Protoss config misspells the Colossus as "Colosus". These are
# verified roster facts of this environment, not name-prefix guesses: the
# suffix strip is exact ("_rl"), and unknown names stay unknown.
NAME_ALIASES.update({"colosus": "colossus"})


def canonical_type(type_name: str, unit_type_id: Optional[int]) -> str:
    """Identity by verified catalog id first, known name second.

    An unverified name prefix or a bare native id with no entry is reported
    as unknown rather than guessed (constraints section 1).
    """
    if unit_type_id is not None and int(unit_type_id) in TYPE_IDS:
        return TYPE_IDS[int(unit_type_id)]
    name = (type_name or "").strip().lower()
    if not name:
        return "unknown"
    if name in NAME_ALIASES:
        return NAME_ALIASES[name]
    if name.endswith("_rl"):
        base = name[:-3]
        if base in NAME_ALIASES:
            return NAME_ALIASES[base]
    return "unknown"


def capability_for(type_key: str, is_flying: bool = False) -> Capability:
    key = type_key
    cached = _CAPABILITY_CACHE.get(key)
    if cached is not None:
        return cached
    specialization = SPECIALIZATIONS.get(key)
    if key in RANGED:
        combat_class, complete = "ranged", "complete"
    elif key in MELEE:
        combat_class, complete = "melee", "complete"
    elif specialization == "heal_support":
        combat_class, complete = "support", "complete"
    elif specialization == "burst_engage":
        combat_class, complete = "burst", "partial"  # splash radius uncalibrated
    elif specialization == "splash_fire":
        combat_class, complete = "ranged", "partial"  # splash geometry uncalibrated
    else:
        # Unknown unit: conservative melee-ish default with unknown data.
        combat_class, complete = "melee", "unknown"
    cap = Capability(key=key, combat_class=combat_class,
                     specialization=specialization,
                     flying=is_flying or key == "medivac",
                     healer=key == "medivac",
                     suicide_attack=key == "baneling",
                     splash_attack=key in ("colossus", "baneling"),
                     data_completeness=complete)
    _CAPABILITY_CACHE[key] = cap
    return cap
