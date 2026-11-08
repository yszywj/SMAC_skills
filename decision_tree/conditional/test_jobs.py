import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from decision_tree.conditional import jobs


class FakeProcess:
    def __init__(self, command, *, outcome="complete", exit_code=0, **kwargs):
        self.args, self.pid, self.returncode = command, 1234567, None
        self.exit_code = exit_code
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir()
        value = {"status": outcome}
        if outcome != "complete":
            value["error"] = "ConnectionResetError: test transport"
        (output/"summary.json").write_text(json.dumps(value), encoding="utf-8")

    def wait(self, timeout):
        self.returncode = self.exit_code
        return self.returncode

    def poll(self):
        return self.returncode


class NativeJobs(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.campaign = self.root/"decision_tree/artifacts/new_v3"
        self.campaign.mkdir(parents=True)
        self.write(self.campaign/"protocol.json", {"kind": "conditional_3x40", "max_native_attempts": 6})
        self.manifest = self.campaign/"manifest.json"
        self.write(self.manifest, {"episodes": [{"sha256": "one"}, {"sha256": "two"}]})
        self.policy = self.campaign/"policy.json"
        self.write(self.policy, {"rule": "one"})
        self.output = self.campaign/"round_1/base"
        self.command = ["python", "-m", "skills.local_v3.run", "--manifest", str(self.manifest),
                        "--policy", str(self.policy), "--output-dir", str(self.output)]
        root_patch = patch.object(jobs.study, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        disk = patch.object(jobs.shutil, "disk_usage", return_value=type("Disk", (), {"free": 100 * 1024**3})())
        disk.start()
        self.addCleanup(disk.stop)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def run_job(self):
        return jobs.run_job(self.command, self.output, self.campaign)

    def test_complete_resumes_without_relaunch_and_changed_inputs_are_rejected(self):
        with patch.object(jobs.subprocess, "Popen", side_effect=FakeProcess) as launch:
            result = self.run_job()
            self.assertFalse(result["reused"])
            self.assertTrue(self.run_job()["reused"])
            self.assertEqual(launch.call_count, 1)
            self.write(self.policy, {"rule": "changed"})
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                self.run_job()
            self.assertEqual(launch.call_count, 1)

    def test_retry_preserves_first_output_and_charges_manifest_slots(self):
        attempts = []
        def launch(command, **kwargs):
            attempts.append(command)
            return FakeProcess(command, outcome="failed" if len(attempts) == 1 else "complete",
                               exit_code=1 if len(attempts) == 1 else 0, **kwargs)
        with patch.object(jobs.subprocess, "Popen", side_effect=launch):
            result = self.run_job()
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(Path(result["path"]).name, "base.retry002")
        self.assertTrue((self.output/"summary.json").exists())
        with jobs._database(self.campaign, jobs._read(self.campaign/"protocol.json")) as conn:
            self.assertEqual(conn.execute("SELECT sum(slots) FROM attempts").fetchone()[0], 4)

    def test_budget_blocks_before_next_launch(self):
        self.write(self.campaign/"protocol.json", {"kind": "conditional_3x40", "max_native_attempts": 3})
        with patch.object(jobs.subprocess, "Popen", side_effect=FakeProcess) as launch:
            self.run_job()
            second = self.campaign/"round_1/other"
            command = self.command[:-1] + [str(second)]
            with self.assertRaisesRegex(ValueError, "budget exhausted"):
                jobs.run_job(command, second, self.campaign)
            self.assertEqual(launch.call_count, 1)

    def test_user_stop_prevents_launch_and_cannot_resume_old_full_plan(self):
        self.write(self.campaign/"STOPPED_BY_USER.json", {"state": "stopped"})
        with patch.object(jobs.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(RuntimeError, "stopped by marker"):
                self.run_job()
            launch.assert_not_called()
        (self.campaign/"STOPPED_BY_USER.json").unlink()
        self.write(self.campaign/"protocol.json", {"kind": "full_single_effect_v2_train40", "max_native_attempts": 20000})
        with self.assertRaisesRegex(ValueError, "new conditional"):
            self.run_job()

    def test_stop_or_timeout_terminates_only_live_owned_process(self):
        with patch.object(jobs.subprocess, "Popen", side_effect=FakeProcess), \
                patch.object(jobs, "_watch", side_effect=RuntimeError("user stop")), \
                patch.object(jobs, "terminate_owned") as terminate:
            with self.assertRaisesRegex(RuntimeError, "user stop"):
                self.run_job()
            self.assertEqual(terminate.call_count, 1)
            process, command, record = terminate.call_args.args
            self.assertEqual((process.args, process.pid), (command, record["pid"]))

    def test_auditor_exit_failure_is_not_retried_or_reused(self):
        with patch.object(jobs.subprocess, "Popen",
                          side_effect=lambda cmd, **kw: FakeProcess(cmd, exit_code=1, **kw)) as launch:
            with self.assertRaisesRegex(RuntimeError, "exit 1"):
                self.run_job()
            with self.assertRaisesRegex(ValueError, "requires diagnosis"):
                self.run_job()
            self.assertEqual(launch.call_count, 1)


if __name__ == "__main__":
    unittest.main()
