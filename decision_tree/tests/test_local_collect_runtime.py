"""Collector process contracts using mocks; no game process is started."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from decision_tree.local import study
from decision_tree.local.runtime import load_policy, digest


class LocalCollectRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root/'manifest.json'
        self.policy = self.root/'policy.json'
        self.m = {'example': 'manifest'}
        self.p = load_policy({'schema_version': 1, 'kind': 'local_patch_policy',
                              'selection': {'mode': 'base'}})
        study.save_once(self.manifest, self.m)
        study.save_once(self.policy, self.p)
        self.protocol = {'sources': {}, 'python': 'fake-python', 'flags': []}
        self.output = self.root/'arm'

    def tearDown(self):
        self.temp.cleanup()

    def log_run(self, output, *, failed=False, started=False, error=None):
        output.mkdir()
        study.save_once(output/'manifest.json', self.m)
        study.save_once(output/'run.json', {'local_policy': {'policy': self.p}})
        study.save_once(output/'summary.json', {
            'status': 'failed' if failed else 'complete', 'completed_episodes': 0 if failed else 1,
            'error': error or ('ConnectError: Failed to connect to the SC2 websocket' if failed else None)})
        (output/'events.jsonl').write_text(json.dumps({'event': 'episode_started' if started else 'run_failed'}) + '\n')

    def audit(self, output):
        if study.read(Path(output)/'summary.json')['status'] != 'complete':
            raise ValueError('incomplete_run')
        return {'manifest.json': study.read(Path(output)/'manifest.json'),
                'run.json': study.read(Path(output)/'run.json')}

    def test_startup_failure_is_archived_with_command_and_console_then_retried(self):
        calls = []
        def execute(cmd, **kwargs):
            calls.append(cmd)
            kwargs['stdout'].write(f'attempt {len(calls)}\n')
            self.log_run(self.output, failed=len(calls) == 1)
            if len(calls) == 1:
                raise subprocess.CalledProcessError(1, cmd)
        with patch.object(study.subprocess, 'run', side_effect=execute), \
                patch.object(study, 'read_run', side_effect=self.audit), redirect_stdout(io.StringIO()):
            study.run_arm(self.protocol, self.manifest, self.policy, self.output, archive_root=self.root)
        self.assertEqual(len(calls), 2)
        archived = self.root/'arm.failed-start-001'
        self.assertEqual(study.read(archived/'summary.json')['status'], 'failed')
        self.assertEqual((self.root/'arm.failed-start-001.console.log').read_text(), 'attempt 1\n')
        self.assertTrue((self.root/'arm.failed-start-001.command.json').is_file())
        self.assertEqual(study.read(self.output/'summary.json')['status'], 'complete')

    def test_two_retries_limit_preserves_third_failure(self):
        def execute(cmd, **kwargs):
            self.log_run(self.output, failed=True)
            raise subprocess.CalledProcessError(1, cmd)
        with patch.object(study.subprocess, 'run', side_effect=execute) as runner, \
                patch.object(study, 'read_run', side_effect=self.audit), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'retry limit'):
                study.run_arm(self.protocol, self.manifest, self.policy, self.output, archive_root=self.root)
        self.assertEqual(runner.call_count, 3)
        self.assertTrue((self.root/'arm.failed-start-001').is_dir())
        self.assertTrue((self.root/'arm.failed-start-002').is_dir())
        self.assertTrue((self.output/'summary.json').is_file())

    def test_started_episode_failure_never_retried(self):
        def execute(cmd, **kwargs):
            self.log_run(self.output, failed=True, started=True)
            raise subprocess.CalledProcessError(1, cmd)
        with patch.object(study.subprocess, 'run', side_effect=execute) as runner:
            with self.assertRaises(subprocess.CalledProcessError):
                study.run_arm(self.protocol, self.manifest, self.policy, self.output, archive_root=self.root)
        self.assertEqual(runner.call_count, 1)
        self.assertFalse((self.root/'arm.failed-start-001').exists())

    def test_controller_error_never_retried(self):
        def execute(cmd, **kwargs):
            self.log_run(self.output, failed=True, error='ValueError: illegal action')
            raise subprocess.CalledProcessError(1, cmd)
        with patch.object(study.subprocess, 'run', side_effect=execute) as runner:
            with self.assertRaises(subprocess.CalledProcessError):
                study.run_arm(self.protocol, self.manifest, self.policy, self.output, archive_root=self.root)
        self.assertEqual(runner.call_count, 1)

    def test_successful_run_is_reused_without_subprocess_or_changes(self):
        self.log_run(self.output)
        before = (self.output/'summary.json').read_bytes()
        with patch.object(study.subprocess, 'run') as runner, patch.object(study, 'read_run', side_effect=self.audit):
            study.run_arm(self.protocol, self.manifest, self.policy, self.output)
        runner.assert_not_called()
        self.assertEqual((self.output/'summary.json').read_bytes(), before)

    def test_archive_refuses_paths_outside_explicit_root(self):
        self.log_run(self.output, failed=True)
        with self.assertRaisesRegex(ValueError, 'outside'):
            study._archive_start_failure(self.output, self.root/'different')
        self.assertTrue(self.output.is_dir())

    def test_bounded_workers_continue_and_persist_failed_candidate(self):
        arms = [{'name': f'arm{i}', 'path': str(self.root/f'arm{i}'), 'candidate_id': f'p{i}',
                 'manifest': str(self.manifest), 'policy': str(self.policy),
                 'manifest_sha256': digest(self.m), 'policy_sha256': digest(self.p)} for i in range(6)]
        lock = threading.Lock()
        active = peak = completed = 0
        barrier = threading.Barrier(2)
        def run(protocol, manifest, policy, output):
            nonlocal active, peak, completed
            with lock:
                active += 1; peak = max(peak, active)
            try:
                barrier.wait(timeout=2)
                time.sleep(0.01)
                if Path(output).name == 'arm1':
                    raise ValueError('deliberate recorded failure')
            finally:
                with lock:
                    active -= 1; completed += 1
        with patch.object(study, 'run_arm', side_effect=run), redirect_stdout(io.StringIO()):
            succeeded, errors = study.run_candidates(self.protocol, arms, self.root, workers=2)
        self.assertEqual((peak, completed, succeeded, len(errors)), (2, 6, 5, 1))
        logged = study.read(self.root/'candidate_errors.json')['errors']
        self.assertEqual(logged[0]['candidate_id'], 'p1')
        self.assertIn('deliberate recorded failure', logged[0]['error'])
        self.assertFalse((self.root/'candidate_errors.json.tmp').exists())

    def test_invalid_worker_counts_fail_before_work(self):
        for workers in (0, 3, True):
            with self.assertRaisesRegex(ValueError, 'workers'):
                study.run_candidates(self.protocol, [], self.root, workers)


if __name__ == '__main__':
    unittest.main()
