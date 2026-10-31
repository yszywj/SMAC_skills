"""Teacher checkpoint preflight: audit before any teacher label is trusted.

State machine (TEACHER_MANIFEST.json is the durable record):
  absent      -> no local checkpoint files at all
  downloaded  -> bytes present, hash/size recorded, NOT yet structurally audited
  audited     -> loads with the official implementation, structure matches
                 metadata (param count, history length, action head), no
                 missing/unexpected keys, finite outputs on smoke input
  ready       -> action mapping validated on the 10gen_terran protocol
  rejected    -> audit failed; reasons recorded

No automatic downloads and no unbounded training are performed here. The
checkpoints are torch .pt files: loading them requires the official
marl-gpt implementation and torch, only ever inside the dedicated teacher
environment (never the SMACv2 adapter venv), and always with
weights_only-style safe loading when available.
"""

import hashlib
import json
import os

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "TEACHER_MANIFEST.json")

EXPECTED = [
    # url, filename  -- verified to exist on 2026-10-30 via HF API; NOT downloaded
    {"url": "https://huggingface.co/nortem/marl-gpt-model/resolve/main/marl-gpt-main.pt",
     "file": "marl-gpt-main.pt", "license": "MIT"},
    {"url": "https://huggingface.co/nortem/marl-gpt-model/resolve/main/marl-gpt-wo-last-token.pt",
     "file": "marl-gpt-wo-last-token.pt", "license": "MIT"},
]


def _load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def register_download(directory):
    """Record hashes/sizes of checkpoint files placed manually under
    teacher/artifacts. Downloads are never started automatically."""
    manifest = _load_manifest()
    found = {}
    for entry in EXPECTED:
        path = os.path.join(directory, entry["file"])
        if os.path.isfile(path):
            found[entry["file"]] = {"size": os.path.getsize(path),
                                    "sha256": sha256_file(path),
                                    "url": entry["url"]}
    manifest["checkpoints"] = found
    manifest["state"] = "downloaded" if found else "absent"
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def teacher_status():
    return _load_manifest()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default=None,
                        help="directory containing manually downloaded .pt files")
    args = parser.parse_args()
    if args.checkpoint_dir:
        register_download(args.checkpoint_dir)
    print(json.dumps(teacher_status(), indent=1))
