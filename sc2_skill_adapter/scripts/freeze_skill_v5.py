"""Copy exactly the runtime files hashed by runner before the v5 games start."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge.config import MAP_FILE, SMAC_ROOT
from bridge.skill_runtime import source_files


def main():
    output = ROOT / "analysis/skill_repairs_v5_20261023/source_snapshot"
    output.mkdir(parents=True, exist_ok=False)
    sources = list((ROOT / "bridge").glob("*.py"))
    sources += [SMAC_ROOT / "smacv2/env/starcraft2" / name for name in (
        "starcraft2.py", "wrapper.py", "distributions.py")]
    sources += [MAP_FILE, *source_files()]
    hashes = {}
    for path in sources:
        relative = path.relative_to(ROOT.parent)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "snapshot.json").write_text(json.dumps({"source_hashes": hashes}, indent=2), encoding="utf-8")
    print(f"Frozen {len(hashes)} runtime files in {output}")


if __name__ == "__main__":
    main()
