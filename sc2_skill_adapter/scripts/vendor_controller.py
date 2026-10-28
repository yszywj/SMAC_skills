"""Snapshot only portable JEV components; fail loudly if source anchors change."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "smacv1/jev_micro"
SKILL_SOURCE = ROOT.parent / "skills/current"
DEST = ROOT / "bridge"


def replace(text, old, new):
    if old not in text:
        raise RuntimeError(f"Source anchor missing: {old[:100]}")
    return text.replace(old, new)


def main(destination=DEST, manifest_path=ROOT / "SOURCE_MANIFEST.json"):
    manifest = {}
    for name in ("client.py", "feedback.py", "policy.py", "planner.py"):
        source = (SKILL_SOURCE if name == "client.py" else SOURCE) / name
        source_name = source.relative_to(ROOT.parent).as_posix()
        raw = source.read_bytes()
        # Normalize source newlines before write_text applies Windows newlines.
        text = raw.decode("utf-8").replace("\r\n", "\n")
        manifest[name] = {"source": source_name,
                          "source_sha256": hashlib.sha256(raw).hexdigest()}
        if name == "policy.py":
            text = replace(text, "from .catalog import SUPPORTED_MAPS", "")
            text = replace(text, "from .environment import is_medivac",
                           "from .environment import is_medivac, visible_enemies")
            text = replace(text, "from .scenario import ACTION_CONTRACT, compact_brief",
                           "from .semantics import ACTION_CONTRACT, compact_brief")
            text = replace(text, 'if env.map_name not in SUPPORTED_MAPS or env.use_ability:',
                           'if env.map_name != "10gen_protoss" or env.conic_fov:')
            text = replace(text, "This adapter uses registered SMAC maps with abilities disabled.",
                           "This adapter uses standard SMACv2 Protoss with abilities disabled.")
            start = text.index("    visible = {enemy_ids[u.tag]")
            end = text.index("    references =", start)
            text = text[:start] + "    visible = visible_enemies(env)\n" + text[end:]
            text = replace(text, 'references.update({tag: f"e{i}" for tag, i in enemy_ids.items()})',
                           'references.update({u.tag: f"e{i}" for i, u in visible.items()})')
            text = replace(text, 'INTERFACE_VERSION = "p0-v1"',
                           'INTERFACE_VERSION = "smacv2-p0-v1"')
            text = replace(text, 'SMAC attack candidates have a center-distance cap of 6, ',
                           'SMACv2 attack candidates use per-unit distance thresholds in action_contract, ')
            text = replace(text, '"smac_center_distance_allowed": d <= 6,',
                           '"smac_center_distance_allowed": d <= env.unit_shoot_range(i),')
            text = replace(text, '"smac_attack_distance_limit": 6,',
                           '"smac_attack_distance_limit": env.unit_shoot_range(i),')
            text = replace(text, 'full observation are derivable from these statuses, distance and cap=6.',
                           'full observation are derivable from statuses, distance and per-type thresholds.')
        elif name == "feedback.py":
            text = replace(text, "from s2clientprotocol import error_pb2",
                           "from s2clientprotocol import error_pb2\nfrom .environment import visible_enemies")
            text = replace(text, 'references.update({u.tag: f"e{i}" for i, u in env.enemies.items()})',
                           'references.update({u.tag: f"e{i}" for i, u in visible_enemies(env).items()})')
        elif name == "planner.py":
            text = replace(text, "from .catalog import map_param_registry",
                           "from .semantics import planning_input")
            start = text.index("def planning_input(payload, env):")
            end = text.index("\ndef attach_plan", start)
            text = text[:start] + text[end:]
            start = text.index('INSTRUCTIONS = """')
            end = text.index('\n\n\nclass PlannerError', start)
            text = text[:start] + "from .semantics import PLANNER_INSTRUCTIONS as INSTRUCTIONS" + text[end:]
            text = replace(text, "one plan per map, reused across episodes",
                           "one distribution-level plan per scenario configuration, reused across episodes")
            text = replace(text, "    async def get_plan(self, payload, env):\n",
                           "    async def get_plan(self, payload, env):\n"
                           "        scenario_key = hashlib.sha256(json.dumps(planning_input(payload, env),\n"
                           "            sort_keys=True).encode('utf-8')).hexdigest()\n"
                           "        if self.active is not None and self.active['scenario_sha256'] != scenario_key:\n"
                           "            raise PlannerError('planner_wrong_scenario_configuration')\n")
            text = replace(text, 'self.active = {"model": self.model, "plan_sha256": digest, "plan": plan}',
                           'self.active = {"model": self.model, "plan_sha256": digest, "plan": plan,\n'
                           '                           "scenario_sha256": scenario_key}')
        (destination / name).write_text(
            f"# Adapted from sibling {source_name}; see SOURCE_MANIFEST.json.\n" + text,
            encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
