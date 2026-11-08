"""JSON-representation compatibility for the frozen conditional-v3 auditor."""
import argparse
from contextlib import closing, contextmanager
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

from decision_tree.conditional import campaign, jobs
from decision_tree.conditional.policy import normalize_policy
from decision_tree.local.study import save_once
from skills.local_v2.explore import checksum, require
from skills.local_v3 import run as native


AUDIT_ERROR = "Conditional policy, actions, candidates or counters differ from replay"


def canonical(value):
    """Use exactly the JSON types emitted by the native log, without rounding."""
    return json.loads(json.dumps(value, allow_nan=False))


@contextmanager
def audit_compatibility():
    original_audit = native._audit_episode
    original_controller = native.ConditionalController

    class CanonicalReplay(original_controller):
        def _decide(self, payload, baseline):
            result = super()._decide(payload, baseline)
            self.events[-1] = canonical(self.events[-1])
            return result

    def audit_episode(*args, **kwargs):
        # This class replacement exists only during the pure audit replay.
        # Runtime control, action submission and physical games use the frozen class.
        with patch.object(native, "ConditionalController", CanonicalReplay):
            return original_audit(*args, **kwargs)

    with patch.object(native, "_audit_episode", audit_episode):
        yield


def native_command(command):
    result = list(command)
    if "-m" in result:
        index = result.index("-m") + 1
        if result[index] == "skills.local_v3.run":
            result[index:index+1] = ["decision_tree.scripts.conditional_compat", "native"]
    return result


def recovered_run(root, directory, manifest, policy, forced):
    """Fully audit completed evidence before reusing it; leave attempt exits intact."""
    root, directory = Path(root).resolve(), Path(directory).resolve()
    jobs._stop_check(root, directory)
    expected_manifest = campaign.read(manifest)
    recorded_manifest = campaign.read(directory/"manifest.json")
    require(expected_manifest == recorded_manifest, "Recovery manifest differs from requested scenes")
    metadata = campaign.read(directory/"run.json")["conditional_skills"]
    expected_forced = None if forced is None else {"step": forced["step"], "binding": forced["binding"]}
    require(metadata["policy"] == normalize_policy(campaign.read(policy))
            and metadata.get("forced") == expected_forced, "Recovery requested a different policy or override")
    database = root/"native_attempts.sqlite3"
    with closing(sqlite3.connect(database.as_uri()+"?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        rows = list(conn.execute("SELECT * FROM attempts WHERE output=? AND path=?",
                                (str(directory), str(directory))))
    require(len(rows) == 1 and rows[0]["state"] in ("failed", "exited")
            and rows[0]["returncode"] in (0, 1), "Recovery needs a finished registered native attempt")
    attempt = rows[0]
    if attempt["returncode"] == 1:
        console = root/"launch_state"/f"attempt_{attempt['id']:05d}"/"console.log"
        require(AUDIT_ERROR in console.read_text(encoding="utf-8", errors="replace"),
                "Native failure has a different cause")
    native.check_completion(directory, metadata, len(recorded_manifest["episodes"]))
    if attempt["returncode"] == 1:
        save_once(directory/"audit_recovery.json", {
            "kind": "json_representation_audit_recovery",
            "reason": "Logged JSON arrays and recomputed Python tuples compare after identical JSON serialization.",
            "native_attempt_id": attempt["id"], "original_native_returncode": 1,
            "native_ledger_modified": False,
            "auditor_sha256": checksum(Path(__file__)),
            "completed_episodes": len(recorded_manifest["episodes"]),
            "source_hashes": metadata["source_hashes"],
            "evidence_sha256": {name: checksum(directory/name) for name in
                                ("run.json", "summary.json", "manifest.json", "episodes.json", "events.jsonl")},
        })
    return directory


@contextmanager
def campaign_compatibility():
    original_job, original_launch = jobs.run_job, campaign.launch

    def run_job(command, *args, **kwargs):
        return original_job(native_command(command), *args, **kwargs)

    def launch(root, directory, manifest, policy=None, forced=None):
        summary = Path(directory)/"summary.json"
        if policy is not None and summary.is_file() and campaign.read(summary).get("status") == "complete":
            return recovered_run(root, directory, manifest, policy, forced)
        return original_launch(root, directory, manifest, policy, forced)

    with audit_compatibility(), patch.object(jobs, "run_job", run_job), patch.object(campaign, "launch", launch):
        yield


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "native":
        with audit_compatibility(), patch.object(sys, "argv", [sys.argv[0], *sys.argv[2:]]):
            native.main()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    argv = [sys.argv[0], args.stage, "--root", str(args.root), "--round", str(args.round)]
    with campaign_compatibility(), patch.object(sys, "argv", argv):
        campaign.main()


if __name__ == "__main__":
    main()
