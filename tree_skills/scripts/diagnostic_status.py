"""Compact, read-only progress for the local four-arm experiment."""
import argparse
import json
from pathlib import Path


def status(prefix):
    result = {}
    for arm in "ABCD":
        run = Path(str(prefix) + arm)
        if not run.exists():
            result[arm] = {"state": "pending"}
            continue
        records = []
        try:
            records = json.loads((run / "episodes.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        item = {"completed": len(records), "wins": sum(r["outcome"] == "win" for r in records), "state": "running"}
        if (run / "summary.json").exists():
            summary = json.loads((run / "summary.json").read_text())
            item.update(state=summary["status"], error=summary["error"], wall_seconds=round(summary["wall_seconds"]))
        else:
            p = run / "events.jsonl"
            if p.exists():
                with p.open("rb") as f:
                    f.seek(max(0, p.stat().st_size - 1024 * 1024))
                    lines = f.read().decode("utf-8", errors="replace").splitlines()
                for line in reversed(lines):
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("event") == "step":
                        item["current_episode"] = r["episode"]
                        item["current_step"] = r["step"]
                        break
        result[arm] = item
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-prefix", type=Path, required=True)
    print(json.dumps(status(p.parse_args().run_prefix)))
