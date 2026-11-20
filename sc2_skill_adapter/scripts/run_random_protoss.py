"""Run three distinct, randomly drawn Protoss compositions on the same 20 scenarios."""

from collections import Counter
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import random
import subprocess
import sys

ADAPTER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ADAPTER))
from bridge.config import digest
from bridge.scenarios import validate_manifest


def prepare_groups():
    template_path = ADAPTER / "configs" / "protoss_10_vs_10_native_n20_seed170.json"
    template = json.loads(template_path.read_text(encoding="utf-8"))
    validate_manifest(template)
    cap = template["env_args"]["capability_config"]
    distribution = cap["team_gen"]
    groups = {}
    while len(groups) < 3:
        team = random.choices(distribution["unit_types"], weights=distribution["weights"],
                              k=cap["n_units"])
        counts = Counter(team)
        name = f"p_{counts['zealot']}z{counts['stalker']}s{counts['colossus']}c"
        if name in groups:
            continue
        manifest = deepcopy(template)
        for row in manifest["episodes"]:
            config = row["episode_config"]
            config["team_gen"].update(ally_team=team.copy(), enemy_team=team.copy())
            row["sha256"] = digest(config)
        del manifest["sha256"]
        manifest["sha256"] = digest(manifest)
        validate_manifest(manifest)
        groups[name] = manifest
    return groups


def main():
    groups = prepare_groups()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    batch = ADAPTER.parent / "runs" / f"protoss_random_20x3_{stamp}"
    manifests = batch / "manifests"
    manifests.mkdir(parents=True)
    print(f"{batch}\nGroups: {', '.join(groups)}", flush=True)
    for name, manifest in groups.items():
        (manifests / f"{name}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name in groups:
        subprocess.run([
            sys.executable, "-u", "-B", str(ADAPTER / "run.py"), "run",
            "--method", "adaptive-skill", "--skill-version", "tactical_v1",
            "--adaptive-posture", "balanced", "--skill-max-steps", "4",
            "--manifest", str(manifests / f"{name}.json"), "--output-dir", str(batch / name),
        ], cwd=ADAPTER.parent, check=True)


if __name__ == "__main__":
    main()
