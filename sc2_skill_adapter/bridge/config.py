"""Resolve sibling sources without importing SMAC-Hard or its vendored PySC2."""
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
def _dependency_root(name, marker):
    # Standalone sibling layout and the SMAC_agent subdirectory layout are
    # both supported. Prefer the local dependency when both are present.
    for parent in (WORKSPACE, WORKSPACE.parent):
        candidate = parent / name
        if (candidate / marker).is_file():
            return candidate
    return WORKSPACE / name


SMAC_ROOT = _dependency_root("smacv2", "smacv2/env/starcraft2/wrapper.py")
GAME_ROOT = _dependency_root("game_skill", "micro/pysc2/__init__.py")
MAP_FILE = SMAC_ROOT / "smacv2/env/starcraft2/maps/SMAC_Maps/10gen_protoss.SC2Map"
RACE_UNITS = {
    "protoss": {"stalker": 74, "zealot": 73, "colossus": 4},
    "terran": {"marine": 48, "marauder": 51, "medivac": 54},
    "zerg": {"zergling": 105, "baneling": 9, "hydralisk": 107},
}
RACE_RANGES = {
    "protoss": {"zealot": 2, "stalker": 6, "colossus": 7},
    "terran": {"marine": 5, "marauder": 6, "medivac": 4},
    "zerg": {"zergling": 2, "baneling": 2, "hydralisk": 5},
}


def race_for_map(map_name):
    for race in RACE_UNITS:
        if map_name == f"10gen_{race}":
            return race
    raise ValueError(f"Unsupported map: {map_name}")


def map_file(args):
    race_for_map(args["map_name"])
    return MAP_FILE.parent / (args["map_name"] + ".SC2Map")


def source_path():
    if not (SMAC_ROOT / "smacv2/env/starcraft2/wrapper.py").is_file():
        raise FileNotFoundError(f"Missing sibling SMACv2 source: {SMAC_ROOT}")
    if str(SMAC_ROOT) not in sys.path:
        sys.path.insert(0, str(SMAC_ROOT))


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":")).encode()).hexdigest()


def load_config(size, race="protoss"):
    import yaml
    if size not in (5, 10) or race not in RACE_UNITS:
        raise ValueError("Only Protoss/Terran/Zerg 5v5 and 10v10 are supported.")
    path = SMAC_ROOT / f"smacv2/examples/configs/sc2_gen_{race}.yaml"
    args = copy.deepcopy(yaml.safe_load(path.read_text(encoding="utf-8"))["env_args"])
    args["capability_config"].update(n_units=size, n_enemies=size)
    validate_config(args)
    return args


def validate_config(args):
    if args.get("unit_balance", "original") not in ("original", "native"):
        raise ValueError("unit_balance must be original or native")
    cap = args["capability_config"]
    race = race_for_map(args["map_name"])
    if (cap["n_units"] != cap["n_enemies"] or cap["n_units"] not in (5, 10)):
        raise ValueError("Unsupported scenario")
    if not set(cap["team_gen"]["unit_types"]) <= set(RACE_UNITS[race]):
        raise ValueError("Unit types must belong to the selected race")
    if (args.get("conic_fov", False) or not args.get("action_mask", True)
            or args.get("prob_obs_enemy", 1) != 1
            or args.get("fully_observable", False) or args.get("heuristic_ai", False)):
        raise ValueError("This adapter supports masked, non-EPO, non-conic standard observations only.")
    if not args.get("use_unit_ranges") or args.get("min_attack_range") != 2:
        raise ValueError("The current action contract requires unit ranges and minimum range 2.")
    if not args.get("obs_all_health", True) or not args.get("obs_starcraft", True):
        raise ValueError("Standard observed combat features are required.")
    if set(cap) != {"n_units", "n_enemies", "team_gen", "start_positions"}:
        raise ValueError("Unexpected capability configuration")


def public_brief(args):
    """Distribution-level facts only: never episode_config or hidden positions."""
    size = args["capability_config"]["n_units"]
    race = race_for_map(args["map_name"])
    return {
        "scenario_id": f"{race}_{size}_vs_{size}",
        "map": args["map_name"], "n_agents": size, "n_enemies": size,
        "distribution": copy.deepcopy(args["capability_config"]),
        "map_size": [32, 32], "step_mul": args["step_mul"],
        "move_amount": args["move_amount"], "difficulty": args["difficulty"],
        "use_unit_ranges": args["use_unit_ranges"],
        "min_attack_range": args["min_attack_range"],
        "candidate_ranges": copy.deepcopy(RACE_RANGES[race]),
        "visibility": "centralized union of living allies' SMACv2 observations",
        "scope": "Public generation rules; no sampled enemy positions or initial snapshot.",
        "skills": "off", "realtime": False,
        "unit_balance": args.get("unit_balance", "original"),
        "unit_control_note": (
            f"Both owners use identical native {race.capitalize()} unit definitions, including native automatic acquisition. "
            "Controllers and observation restrictions still differ."
            if args.get("unit_balance") == "native" else
            "Allies use SMACv2 custom RL units; enemies use native units."),
    }
