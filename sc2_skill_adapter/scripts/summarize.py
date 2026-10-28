"""Compare completed runs on exactly one scenario manifest; reject partial runs."""
import argparse
import json
from pathlib import Path


def summarize(paths):
    rows = []
    manifest = None
    for path in paths:
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        if summary["status"] != "complete" or summary["completed_episodes"] != summary["expected_episodes"]:
            raise ValueError(f"Incomplete/failed run cannot enter comparison: {path}")
        if manifest is not None and manifest != summary["manifest_sha256"]:
            raise ValueError("Different manifests; compare 5v5 and 10v10 separately.")
        manifest = summary["manifest_sha256"]
        rows.append(dict(directory=str(path.resolve()), **summary))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = summarize(args.runs)
    print("method       episodes  win loss draw timeout  requests  median_wait_ms")
    for row in rows:
        o = row["outcomes"]
        print(f"{row['method']:12} {row['completed_episodes']:8}  "
              f"{o['win']:3} {o['loss']:4} {o['draw']:4} {o['timeout']:7}  "
              f"{row['jev_requests']:8}  {row['median_team_wait_ms']}")
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
