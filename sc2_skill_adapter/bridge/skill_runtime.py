"""Load independent skill packages without importing either environment's PySC2."""
import argparse
import importlib
import sys
import types

from .config import WORKSPACE

SOURCE = WORKSPACE / "skills/current"
PACKAGE = "_smacv2_portable_jev_skills"
SOURCES = {"current": SOURCE,
           "baseline_v2_70pct": WORKSPACE / "skills/baseline_v2_70pct",
           "baseline_v6_20261024": WORKSPACE / "skills/baseline_v6_20261024",
           "baseline_v7_20261024": WORKSPACE / "sc2_skill_adapter/analysis/skill_v7_20261024/source_snapshot/skills/current"}
LEGACY_COMBAT_PARAMETERS = ("window_fire", "range_preserving_move", "target_scoring")
NEW_COMBAT_PARAMETERS = ("coordinated_approach", "safe_reentry", "accurate_damage",
                         "critical_escape", "planned_reentry", "engagement_commitment",
                         "approach_progress_guard")
# v8 spatial group skills: every switch defaults to OFF so existing suite
# commands keep byte-identical behavior. Baseline versions reject them.
SPATIAL_FLAGS = ("enable_regroup", "enable_cohesive_search", "enable_combat_positioning",
                 "auto_spatial_dispatch")


def add_spatial_arguments(parser):
    """Expose the v8 spatial switches on entries that build a SkillManager."""
    parser.add_argument("--enable-regroup", action="store_true",
                        help="Regroup skill: budgeted rally of a scattered squad (default off)")
    parser.add_argument("--enable-cohesive-search", action="store_true",
                        help="CohesiveSearch skill: shared-waypoint search without contact (default off)")
    parser.add_argument("--enable-combat-positioning", action="store_true",
                        help="CombatPositioning sub-skill inside the combat approach (default off)")
    parser.add_argument("--auto-spatial-dispatch", action="store_true",
                        help="Local rule dispatcher regroup/search/combat without model requests (default off)")
    parser.add_argument("--positioning-spacing", choices=("compact", "normal", "spread"),
                        default="normal", help="CombatPositioning spacing preset")
    parser.add_argument("--positioning-layout", choices=("free", "front_back"),
                        default="free", help="CombatPositioning layout preset")


def spatial_parameters(args):
    return {flag: bool(getattr(args, flag, False)) for flag in SPATIAL_FLAGS}


def _source(version):
    if version not in SOURCES:
        raise ValueError(f"Unknown skill version: {version}")
    source = SOURCES[version]
    if not source.is_dir():
        raise FileNotFoundError(f"Missing skill package: {source}")
    return source


def _load_module(name, version="current"):
    source = _source(version)
    package_name = PACKAGE if version == "current" else PACKAGE + "_" + version
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(source)]
        sys.modules[package_name] = package
    return importlib.import_module(package_name + "." + name)


def load_manager(version="current"):
    return _load_module("skill_manager", version=version).SkillManager


def add_unit_value_arguments(parser):
    """Expose the evaluator's shared defaults without importing SMAC-Hard's runner."""
    module = _load_module("unit_value")
    parser.add_argument("--unit-value-mode", choices=("off", "basic", "contextual"),
                        default=module.DEFAULT_PARAMETERS["mode"])

    def parameter_type(key):
        def parse(value):
            try:
                return module.validate_parameters({key: float(value)})[key]
            except (TypeError, ValueError) as exc:
                raise argparse.ArgumentTypeError(str(exc)) from exc
        return parse

    for key, default in module.DEFAULT_PARAMETERS.items():
        if key == "mode":
            continue
        flag = "--unit-value-" + key.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action="store_true", default=default)
        else:
            parser.add_argument(flag, type=parameter_type(key), default=default)


def unit_value_parameters(args):
    """Validate full effective parameters; old programmatic callers remain off."""
    module = _load_module("unit_value")
    return module.validate_parameters({
        key: getattr(args, "unit_value_" + key, default)
        for key, default in module.DEFAULT_PARAMETERS.items()
    })


def unit_value_metadata(parameters):
    module = _load_module("unit_value")
    validated = module.validate_parameters(parameters)
    return {"unit_value_mode": validated["mode"],
            "unit_value_version": module.EVALUATION_VERSION,
            "unit_value_parameters": validated,
            "unit_value_semantics": "heuristic tactical importance; not win probability or actual damage"}


def manager_configuration(args):
    """Resolve a version before creating output or starting either environment.

    The historical version is a fixed reproduction preset. Its constructor never
    receives parameters for mechanisms that did not exist in that version.
    """
    version = getattr(args, "skill_version", "current")
    source = _source(version)
    race_profile = getattr(args, "race_profile", "off")
    if race_profile not in ("off", "terran", "zerg"):
        raise ValueError("race profile must be off, terran or zerg")
    if version != "current" and race_profile != "off":
        raise ValueError("race profiles require --skill-version current")
    values = unit_value_parameters(args)
    combat = {key: bool(getattr(args, "adaptive_" + key, False))
              for key in (*LEGACY_COMBAT_PARAMETERS, *NEW_COMBAT_PARAMETERS)}
    max_steps = getattr(args, "skill_max_steps", 4)
    posture = getattr(args, "adaptive_posture", "balanced")
    spatial = spatial_parameters(args)
    if version != "current" and any(spatial.values()):
        raise ValueError(f"{version} does not support the v8 spatial skill switches")
    if version == "baseline_v2_70pct":
        if max_steps != 4 or posture != "balanced":
            raise ValueError("baseline_v2_70pct requires --skill-max-steps 4 and --adaptive-posture balanced")
        missing = ["--adaptive-" + key.replace("_", "-")
                   for key in LEGACY_COMBAT_PARAMETERS if not combat[key]]
        if missing:
            raise ValueError("baseline_v2_70pct requires its original flags: " + ", ".join(missing))
        unsupported = ["--adaptive-" + key.replace("_", "-")
                       for key in NEW_COMBAT_PARAMETERS if combat[key]]
        if unsupported:
            raise ValueError("baseline_v2_70pct does not support: " + ", ".join(unsupported))
        if values != _load_module("unit_value").validate_parameters({}):
            raise ValueError("baseline_v2_70pct has no unit-value module; use --unit-value-mode off and default value parameters")
        combat = {key: combat[key] for key in LEGACY_COMBAT_PARAMETERS}
        manager_kwargs = {"max_steps": max_steps, "combat_params": combat}
        value_metadata = {"unit_value_mode": "off", "unit_value_version": None,
                          "unit_value_parameters": None,
                          "unit_value_semantics": "unavailable in baseline_v2_70pct"}
    else:
        if version == "baseline_v6_20261024":
            if combat["approach_progress_guard"]:
                raise ValueError("baseline_v6_20261024 does not support --adaptive-approach-progress-guard")
            combat.pop("approach_progress_guard")
        if any(spatial.values()):
            spatial.update(positioning_spacing=getattr(args, "positioning_spacing", "normal"),
                           positioning_layout=getattr(args, "positioning_layout", "free"))
        manager_kwargs = {"max_steps": max_steps, "combat_params": combat,
                          "unit_value_params": values,
                          **({"spatial_params": spatial} if any(spatial.values()) else {})}
        value_metadata = unit_value_metadata(values)
    if race_profile != "off":
        manager_kwargs["race_profile"] = race_profile
    return {"version": version, "source": str(source), "posture": posture,
            "manager_kwargs": manager_kwargs, "value_metadata": value_metadata}


def source_files(version="current"):
    # Include root-level helpers (notably unit_value.py), as well as allocators.
    # Hashing them does not import their SMAC-Hard environment dependencies.
    source = _source(version)
    return [*sorted(source.glob("*.py")), *sorted((source / "skills").glob("*.py"))]
