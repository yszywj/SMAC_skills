"""Teacher checkpoint preflight: audit before any teacher label is trusted.

Durable state lives in TEACHER_MANIFEST.json:

  checkpoints[<file>].status in {registered, audited, rejected}
  manifest["state"]           in {absent, downloaded, audited, ready}

Rules (P0-1):
- Idempotent registration: re-running never downgrades audited/ready or a
  per-checkpoint audited/rejected status back to registered. Hash changes
  on an already-audited file DO invalidate that checkpoint's audit
  (recorded as rejected:hash_changed) because the audited bytes are gone.
- All writes are atomic (temp file + os.replace); a crash never leaves a
  half-written manifest, and an exception never erases prior success.
- Global state only advances: absent -> downloaded -> audited -> ready.
"""

import hashlib
import json
import os
import tempfile

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "TEACHER_MANIFEST.json")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

EXPECTED = [
    {"url": "https://huggingface.co/nortem/marl-gpt-model/resolve/main/marl-gpt-main.pt",
     "file": "marl-gpt-main.pt", "license": "MIT",
     "size": 86860385,
     "sha256": "c3deaeb67f679657b27e9d3373e42e4104cc9370be6dba60ab5fd0efe7b1ce5a"},
    {"url": "https://huggingface.co/nortem/marl-gpt-model/resolve/main/marl-gpt-wo-last-token.pt",
     "file": "marl-gpt-wo-last-token.pt", "license": "MIT",
     "size": 86860385,
     "sha256": "a1864b541c51619722f514bfbd6f8572b6b17992514c285097cace3e2b8deaf4"},
]

STATE_ORDER = {"absent": 0, "downloaded": 1, "audited": 2, "ready": 3}


def _load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _atomic_write(path, payload):
    directory = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _advance_state(manifest, new_state):
    current = manifest.get("state", "absent")
    if STATE_ORDER[new_state] > STATE_ORDER.get(current, 0):
        manifest["state"] = new_state


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def register_checkpoints(directory=CHECKPOINT_DIR):
    """Idempotently (re-)register checkpoint files found under `directory`.

    Per-checkpoint status transitions:
      - new file / hash unchanged: status registered (or keep existing)
      - hash changed after a successful audit: status rejected with reason
      - already audited/rejected and hash unchanged: untouched
    """
    manifest = _load_manifest()
    checkpoints = manifest.setdefault("checkpoints", {})
    for entry in EXPECTED:
        path = os.path.join(directory, entry["file"])
        record = checkpoints.get(entry["file"], {})
        if not os.path.isfile(path):
            # missing file invalidates any prior audit of it
            if record.get("status") in ("audited",):
                checkpoints[entry["file"]] = {**record, "status": "rejected",
                                              "reject_reason": "file_missing"}
            continue
        size = os.path.getsize(path)
        digest = sha256_file(path)
        if digest != entry["sha256"] or size != entry["size"]:
            # bytes do not match the verified download manifest
            checkpoints[entry["file"]] = {
                **record, "status": "rejected",
                "reject_reason": "hash_or_size_mismatch",
                "size": size, "sha256": digest, "url": entry["url"]}
            continue
        if record.get("status") in ("audited", "rejected"):
            if record.get("sha256") == digest:
                continue  # nothing new; never downgrade
            checkpoints[entry["file"]] = {
                **record, "status": "rejected",
                "reject_reason": "hash_changed_after_audit",
                "sha256": digest, "size": size}
            continue
        checkpoints[entry["file"]] = {"status": "registered", "size": size,
                                      "sha256": digest, "url": entry["url"],
                                      "license": entry["license"]}
    if any(c.get("status") == "registered" for c in checkpoints.values()):
        _advance_state(manifest, "downloaded")
    _atomic_write(MANIFEST_PATH, manifest)
    return manifest


def record_audit(filename, audit_result):
    """Store a structural-audit outcome for one checkpoint.

    audit_result: dict with keys ok (bool), details (dict) and, on failure,
    reject_reason (str). Only a full success moves the per-checkpoint
    status to 'audited'; failures are recorded as 'rejected' without
    touching other checkpoints.
    """
    manifest = _load_manifest()
    checkpoints = manifest.setdefault("checkpoints", {})
    record = checkpoints.get(filename)
    if record is None:
        raise KeyError(f"{filename} is not registered; run register first")
    if audit_result.get("ok"):
        record.update({"status": "audited", "audit": audit_result["details"]})
        if all(c.get("status") == "audited"
               for c in checkpoints.values()) and checkpoints:
            _advance_state(manifest, "audited")
    else:
        record.update({"status": "rejected",
                       "reject_reason": audit_result.get("reject_reason", "audit_failed"),
                       "audit": audit_result.get("details", {})})
    _atomic_write(MANIFEST_PATH, manifest)
    return manifest


def teacher_status():
    return _load_manifest()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    args = parser.parse_args()
    print(json.dumps(register_checkpoints(args.checkpoint_dir), indent=1))
