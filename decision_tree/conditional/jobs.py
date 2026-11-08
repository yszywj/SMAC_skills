"""Bounded serial native jobs with durable attempts and owned-process cleanup.

The caller audits complete runs. A returned completion only means the runner
exited successfully with a complete summary. Every attempt reserves its entire
manifest's scene count, including retries, before starting any game.
"""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

from decision_tree.local import study
from decision_tree.local.runtime import digest
from decision_tree.scripts.local_campaign_recover import atomic_json, failure_kind, terminate_owned
from decision_tree.scripts.local_campaign_validate import validation_lock
from decision_tree.scripts.local_v2_collect import still_running


MIN_FREE_BYTES = 12 * 1024 ** 3
STOP_NAMES = ("STOPPED_BY_USER.json", "STOP.json", "STOP")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stop_check(root, output):
    current = output
    while current.is_relative_to(study.ROOT):
        for name in STOP_NAMES:
            marker = current/name
            if marker.exists():
                raise RuntimeError("Native campaign stopped by marker " + str(marker))
        if current == study.ROOT:
            break
        current = current.parent
    _require(output.is_relative_to(root), "Output escaped this campaign")


def _inputs(command, output, root):
    _require(isinstance(command, (list, tuple)) and command and all(isinstance(x, str) for x in command),
             "Command must be an argument list")
    root, output = Path(root).resolve(), Path(output).resolve()
    artifacts = (study.ROOT/"decision_tree/artifacts").resolve()
    _require(root.is_relative_to(artifacts) and root != artifacts and output.is_relative_to(root) and output != root,
             "Choose a separate job output inside this artifact campaign")
    _stop_check(root, output)
    protocol = _read(root/"protocol.json")
    kind = protocol.get("kind", "")
    _require(isinstance(kind, str) and ("conditional" in kind or "v3" in kind),
             "A new conditional/v3 campaign protocol is required")
    limit = protocol.get("max_native_attempts", protocol.get("budget", {}).get("max_native_attempts"))
    _require(type(limit) is int and limit > 0, "Protocol must declare max_native_attempts in episode units")
    output_flags = [flag for flag in ("--output-dir", "--output") if flag in command]
    _require(len(output_flags) == 1 and command.count(output_flags[0]) == 1,
             "Command must have exactly one native output argument")
    output_index = command.index(output_flags[0]) + 1
    _require(output_index < len(command) and Path(command[output_index]).resolve() == output,
             "Command output differs from registered output")
    _require(command.count("--manifest") == 1, "A fixed manifest is required")
    manifest_path = Path(command[command.index("--manifest") + 1]).resolve()
    manifest = _read(manifest_path)
    scenes = manifest.get("episodes", [])
    _require(bool(scenes) and len({s["sha256"] for s in scenes}) == len(scenes), "Invalid native scene list")
    bound_inputs = {str(manifest_path): _digest_file(manifest_path)}
    for flag in ("--policy", "--local-policy", "--selector"):
        if flag in command:
            _require(command.count(flag) == 1, "Duplicate policy flag")
            path = Path(command[command.index(flag) + 1]).resolve()
            bound_inputs[str(path)] = _digest_file(path)
    identity = {"command": list(command), "inputs": bound_inputs, "episode_slots": len(scenes)}
    return root, output, protocol, limit, identity, output_index


def _database(root, protocol):
    conn = sqlite3.connect(root/"native_attempts.sqlite3")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("""CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS jobs
                    (output TEXT PRIMARY KEY, identity TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS attempts
                    (id INTEGER PRIMARY KEY, output TEXT NOT NULL, attempt INTEGER NOT NULL,
                     path TEXT NOT NULL, slots INTEGER NOT NULL, state TEXT NOT NULL,
                     pid INTEGER, started REAL, returncode INTEGER, error TEXT,
                     UNIQUE(output, attempt))""")
    with conn:
        existing = conn.execute("SELECT value FROM meta WHERE key='protocol'").fetchone()
        _require(existing is None or existing["value"] == digest(protocol), "Campaign protocol changed after reservation")
        conn.execute("INSERT OR IGNORE INTO meta VALUES ('protocol', ?)", (digest(protocol),))
    return conn


def _complete(path):
    summary = path/"summary.json"
    return summary.exists() and _read(summary).get("status") == "complete" and not _read(summary).get("error")


def _progress(path):
    values = []
    for name in ("events.jsonl", "summary.json"):
        try:
            stat = (path/name).stat()
            values.append((stat.st_size, stat.st_mtime_ns))
        except FileNotFoundError:
            values.append(None)
    return tuple(values)


def _watch(process, path, root, idle_seconds):
    last_change = time.monotonic()
    marker = _progress(path)
    while True:
        _stop_check(root, path)
        if shutil.disk_usage(root).free < MIN_FREE_BYTES:
            raise RuntimeError("Native campaign reached the 12 GiB disk reserve")
        try:
            return process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            current = _progress(path)
            if current != marker:
                marker, last_change = current, time.monotonic()
            if time.monotonic() - last_change >= idle_seconds:
                raise TimeoutError(f"Native job made no event progress for {idle_seconds}s")


def _reserve(conn, output, path, attempt, slots, limit):
    with conn:
        used = conn.execute("SELECT coalesce(sum(slots), 0) FROM attempts").fetchone()[0]
        _require(used + slots <= limit,
                 f"Native attempt budget exhausted: {used} reserved + {slots} > {limit}")
        cursor = conn.execute("INSERT INTO attempts(output, attempt, path, slots, state) VALUES (?, ?, ?, ?, 'reserved')",
                              (str(output), attempt, str(path), slots))
    return cursor.lastrowid


def run_job(command, output, campaign_root, max_attempts=2, idle_seconds=120):
    """Return the completed attempt path, preserving all failed attempts.

    max_attempts includes the first launch. Known transport/watchdog failures
    can retry once. Logic/audit failures, low disk and user stops propagate.
    The campaign lock is held while the process runs, so independent callers
    cannot accidentally start overlapping native workers.
    """
    _require(type(max_attempts) is int and 1 <= max_attempts <= 2, "Native jobs allow one attempt plus at most one retry")
    _require(type(idle_seconds) in (int, float) and 30 <= idle_seconds <= 600, "Invalid native idle timeout")
    root, output, protocol, limit, identity, output_index = _inputs(command, output, campaign_root)
    state_dir = root/"launch_state"
    state_dir.mkdir(exist_ok=True)
    with validation_lock(state_dir), closing(_database(root, protocol)) as conn:
        frozen = json.dumps(identity, sort_keys=True)
        prior = conn.execute("SELECT identity FROM jobs WHERE output=?", (str(output),)).fetchone()
        _require(prior is None or prior["identity"] == frozen, "Job inputs changed on resume")
        with conn:
            conn.execute("INSERT OR IGNORE INTO jobs VALUES (?, ?)", (str(output), frozen))
        history = list(conn.execute("SELECT * FROM attempts WHERE output=? ORDER BY attempt", (str(output),)))
        for old in history:
            path = Path(old["path"])
            if old["state"] in ("running", "reserved"):
                _require(old["pid"] is None or not still_running(old["pid"]),
                         "A previous native worker may still be running; do not duplicate it")
            if old["state"] == "exited" and old["returncode"] == 0 and _complete(path):
                return {"status": "complete_needs_audit", "path": str(path), "reused": True, "attempts": old["attempt"]}
            if old["state"] in ("running", "reserved"):
                # An interrupted launch is deliberately left for diagnosis.
                raise RuntimeError("Prior native attempt was interrupted before exit was recorded: " + str(path))
            _require(old["state"] == "retryable", "Prior native failure requires diagnosis: " + str(path))
        start = len(history) + 1
        _require(start <= max_attempts, "Native job retry allowance exhausted")
        for attempt in range(start, max_attempts + 1):
            path = output if attempt == 1 else output.with_name(output.name + f".retry{attempt:03d}")
            _stop_check(root, path)
            _require(not path.exists(), "Refusing to overwrite an existing native run")
            if shutil.disk_usage(root).free < MIN_FREE_BYTES:
                raise RuntimeError("Native campaign reached the 12 GiB disk reserve")
            path.parent.mkdir(parents=True, exist_ok=True)
            attempt_id = _reserve(conn, output, path, attempt, identity["episode_slots"], limit)
            sidecar = state_dir/f"attempt_{attempt_id:05d}"
            sidecar.mkdir()
            actual_command = list(command)
            actual_command[output_index] = str(path)
            atomic_json(sidecar/"command.json", actual_command)
            launch = {"cwd": study.ROOT, "stdin": subprocess.DEVNULL, "stderr": subprocess.STDOUT,
                      "env": {**os.environ, "PYTHONUNBUFFERED": "1"}}
            if os.name == "nt":
                launch["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            else:
                launch["start_new_session"] = True
            process, code, error, retryable = None, None, None, False
            try:
                with (sidecar/"console.log").open("xb") as console:
                    process = subprocess.Popen(actual_command, stdout=console, **launch)
                    record = {"pid": process.pid, "command_sha256": digest(actual_command),
                              "started_unix": time.time(), "owned_process": True}
                    atomic_json(sidecar/"process.json", record)
                    with conn:
                        conn.execute("UPDATE attempts SET state='running', pid=?, started=? WHERE id=?",
                                     (process.pid, record["started_unix"], attempt_id))
                    code = _watch(process, path, root, idle_seconds)
                if code != 0 or not _complete(path):
                    kind = failure_kind(path)
                    retryable = kind in ("environment_transport", "watchdog_timeout")
                    raise RuntimeError(f"Native runner exit {code}, {kind}; see {sidecar/'console.log'}")
            except BaseException as exc:
                error = str(exc)
                retryable = retryable or isinstance(exc, TimeoutError)
                if process is not None and process.poll() is None:
                    terminate_owned(process, actual_command, record)
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    retryable = False
                state = "retryable" if retryable else "failed"
                with conn:
                    conn.execute("UPDATE attempts SET state=?, returncode=?, error=? WHERE id=?",
                                 (state, process.returncode if process else None, error, attempt_id))
                atomic_json(sidecar/"exited.json", {"returncode": process.returncode if process else None,
                                                   "error": error, "retryable": retryable})
                if not retryable or attempt == max_attempts:
                    raise
            else:
                with conn:
                    conn.execute("UPDATE attempts SET state='exited', returncode=0 WHERE id=?", (attempt_id,))
                atomic_json(sidecar/"exited.json", {"returncode": 0})
                return {"status": "complete_needs_audit", "path": str(path), "reused": False, "attempts": attempt}
    raise RuntimeError("Native job did not produce a complete attempt")
