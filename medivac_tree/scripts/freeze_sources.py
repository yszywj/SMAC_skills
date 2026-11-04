"""Verify and preserve the exact source bytes recorded by a completed/running arm."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def freeze(run: Path, output: Path):
    run, output = run.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite source snapshot: {output}")
    metadata = read(run / "run.json")
    expected = metadata.get("source_hashes")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Run has no recorded runtime source hashes")
    mismatches, validated = [], []
    for name, recorded_hash in sorted(expected.items()):
        source = (ROOT / name).resolve()
        if not source.is_relative_to(WORKSPACE):
            raise ValueError(f"Source is outside the verified workspace: {source}")
        if not source.is_file():
            mismatches.append({"source": str(source), "expected": recorded_hash, "error": "missing"})
            continue
        data = source.read_bytes()
        actual = sha(data)
        if actual != recorded_hash:
            mismatches.append({"source": str(source), "expected": recorded_hash, "actual": actual})
            continue
        # Preserve checked bytes rather than re-reading a possibly edited file.
        relative = Path("sources") / source.relative_to(WORKSPACE)
        validated.append((relative, data, {"run_source_key": name, "original_path": str(source),
                                         "snapshot_path": relative.as_posix(), "sha256": actual,
                                         "bytes": len(data)}))
    if mismatches:
        raise ValueError("Source snapshot rejected; no output created: " + json.dumps(mismatches, ensure_ascii=False))
    tree_meta = metadata.get("medivac_tree_metadata") or {}
    if tree_meta:
        tree_path = Path(tree_meta["tree_path"]).resolve()
        if not any(row[2]["original_path"] == str(tree_path) for row in validated):
            raise ValueError("Tree file was not hash-bound in run source_hashes")
    output.mkdir(parents=True, exist_ok=False)
    for relative, data, _ in validated:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    metadata_files = {}
    for name in ("run.json", "manifest.json"):
        data = (run / name).read_bytes()
        (output / name).write_bytes(data)
        metadata_files[name] = sha(data)
    receipt = {"version": 1, "run": str(run), "verified_against_recorded_hashes": True,
               "source_count": len(validated), "files": [row[2] for row in validated],
               "metadata_sha256": metadata_files, "tree_metadata": tree_meta or None,
               "scope": "Source bytes and run metadata only; not a recoverable SC2 process snapshot"}
    (output / "snapshot.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"run": str(run), "output": str(output), "source_count": len(validated),
            "tree_preserved": bool(tree_meta), "verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.run, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
