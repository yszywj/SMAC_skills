"""Synthetic recovery/process contracts; these tests never start SC2."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from decision_tree.scripts import local_campaign_recover as recovery
from decision_tree.local import study
from decision_tree.local.runtime import digest, load_policy


class CampaignRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/'inputs').mkdir()
        self.manifest = {'test_scene': 1}
        self.policy = load_policy({'schema_version': 1, 'kind': 'local_patch_policy',
            'selection': {'mode': 'forced', 'candidate_id': 'retarget:u0:7', 'anchor_step': 1}})
        self.mp, self.pp = self.root/'inputs/manifest.json', self.root/'inputs/policy.json'
        study.save_once(self.mp, self.manifest); study.save_once(self.pp, self.policy)
        self.arm = {'name': 'arm', 'path': str(self.root/'arm'), 'manifest': str(self.mp),
                    'policy': str(self.pp), 'manifest_sha256': digest(self.manifest),
                    'policy_sha256': digest(self.policy), 'candidate_id': 'retarget:u0:7', 'anchor_step': 1}
        self.protocol = {'sources': {}, 'python': 'fake-python', 'flags': []}

    def tearDown(self):
        self.temp.cleanup()

    def test_atomic_json_retries_transient_windows_sharing_error(self):
        report = self.root/'progress.json'
        study.save_once(report, {'processed': 2})
        replace = Path.replace
        attempts = []

        def fail_once(source, destination):
            attempts.append(destination)
            if len(attempts) == 1:
                self.assertEqual(study.read(destination), {'processed': 2})
                fault = PermissionError('simulated Windows sharing violation')
                fault.winerror = 32
                raise fault
            return replace(source, destination)

        with patch.object(Path, 'replace', autospec=True, side_effect=fail_once):
            recovery.atomic_json(report, {'processed': 3})
        self.assertEqual(attempts, [report, report])
        self.assertEqual(study.read(report), {'processed': 3})

    def test_progress_write_failure_retains_last_checkpoint(self):
        report = self.root/'progress.json'
        study.save_once(report, {'processed': 2})
        fault = PermissionError('simulated persistent Windows access denial')
        fault.winerror = 5
        with patch.object(Path, 'replace', side_effect=fault), patch.object(recovery.time, 'sleep'):
            result = recovery.write_progress(report, {'processed': 3})
        self.assertIn(result, (False, None))
        self.assertEqual(study.read(report), {'processed': 2})

    def test_evidence_write_propagates_permanent_io_error(self):
        report = self.root/'timeout.json'
        study.save_once(report, {'termination': 'requested'})
        fault = OSError('simulated disk full')
        fault.winerror = 112
        with patch.object(Path, 'replace', side_effect=fault), patch.object(recovery.time, 'sleep'):
            with self.assertRaises(OSError):
                recovery.atomic_json(report, {'termination': 'owned process tree terminated'})
        self.assertEqual(study.read(report), {'termination': 'requested'})

    def record(self, path, status='failed', error='ConnectError: SC2 websocket connection failed'):
        path = Path(path)
        path.mkdir()
        study.save_once(path/'summary.json', {'status': status, 'error': error if status == 'failed' else None})
        return path

    def successful(self, path, manifest, policy):
        if study.read(Path(path)/'summary.json')['status'] != 'complete':
            raise ValueError('incomplete_run')
        return {}

    def test_environment_failure_retries_new_sibling_preserving_original(self):
        original = self.record(self.arm['path'])
        before = (original/'summary.json').read_bytes()
        def launch(protocol, manifest, policy, destination, timeout):
            self.record(destination, 'complete')
            return {'status': 'exited', 'returncode': 0}
        with patch.object(recovery, 'successful', side_effect=self.successful), \
                patch.object(recovery, 'run_attempt', side_effect=launch) as runner:
            result = recovery.recover_arm(self.arm, self.root, self.protocol)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(result['path'], str(self.root/'arm.recovery-001'))
        self.assertEqual(result['recovery_status'], 'recovered')
        self.assertEqual((original/'summary.json').read_bytes(), before)

    def test_logic_failure_and_completed_audit_failure_are_not_retried(self):
        for name, status, error in [('logic', 'failed', 'ValueError: illegal action'),
                                     ('complete', 'complete', None)]:
            arm = {**self.arm, 'name': name, 'path': str(self.root/name)}
            self.record(arm['path'], status, error)
            with patch.object(recovery, 'successful', side_effect=ValueError('audit failed')), \
                    patch.object(recovery, 'run_attempt') as runner:
                result = recovery.recover_arm(arm, self.root, self.protocol)
            self.assertEqual(result['recovery_status'], 'rejected')
            runner.assert_not_called()

    def test_explicit_manual_hang_can_retry_an_incomplete_directory(self):
        Path(self.arm['path']).mkdir()
        def launch(protocol, manifest, policy, destination, timeout):
            self.record(destination, 'complete')
            return {'status': 'exited', 'returncode': 0}
        with patch.object(recovery, 'successful', side_effect=self.successful), \
                patch.object(recovery, 'run_attempt', side_effect=launch):
            result = recovery.recover_arm(self.arm, self.root, self.protocol, hung_runs=['arm'])
        self.assertEqual(result['recovery_status'], 'recovered')
        self.assertEqual(result['attempts'][0]['kind'], 'manual_hang')

    def test_only_failed_does_not_start_unstarted_branch(self):
        with patch.object(recovery, 'run_attempt') as runner:
            result = recovery.recover_arm(self.arm, self.root, self.protocol, only_failed=True)
        self.assertEqual(result['recovery_status'], 'pending_unstarted')
        runner.assert_not_called()

    def test_max_games_budget_includes_retries_and_never_imputes_reward(self):
        self.record(self.arm['path'])
        def launch(protocol, manifest, policy, destination, timeout):
            self.record(destination)
            return {'status': 'exited', 'returncode': 1}
        gate = recovery.LaunchGate(1)
        with patch.object(recovery, 'successful', side_effect=self.successful), \
                patch.object(recovery, 'run_attempt', side_effect=launch) as runner:
            result = recovery.recover_arm(self.arm, self.root, self.protocol, gate=gate)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(result['recovery_status'], 'deferred')
        self.assertNotIn('reward', result)
        self.assertFalse((self.root/'arm.recovery-002').exists())

    def test_circuit_breaker_latches_after_three_environment_failures(self):
        gate = recovery.LaunchGate(20)
        for _ in range(3):
            self.assertTrue(gate.reserve())
            gate.outcome(environment_failure=True)
        self.assertFalse(gate.reserve())
        gate.outcome()  # An already-in-flight success cannot reopen the circuit.
        self.assertFalse(gate.reserve())
        self.assertEqual(gate.describe()['new_games_started'], 3)

    def test_watchdog_terminates_only_owned_handle_and_saves_timeout(self):
        destination = self.root/'attempt'
        expected = study.command(self.protocol, self.mp, self.pp, destination)
        child = Mock(pid=1234, args=expected)
        child.wait.side_effect = subprocess.TimeoutExpired(expected, 120)
        with patch.object(recovery.subprocess, 'Popen', return_value=child) as popen, \
                patch.object(recovery, 'terminate_owned') as terminate:
            result = recovery.run_attempt(self.protocol, self.mp, self.pp, destination)
        self.assertEqual(result['status'], 'timeout')
        terminate.assert_called_once()
        self.assertIs(terminate.call_args.args[0], child)
        self.assertEqual(popen.call_args.kwargs['env']['PYTHONUNBUFFERED'], '1')
        timeout = study.read(destination/'timeout.json')
        self.assertEqual(timeout['pid'], 1234)
        self.assertEqual(timeout['command_sha256'], digest(expected))
        self.assertEqual(recovery.failure_kind(destination), 'watchdog_timeout')

    def test_watchdog_refuses_unrelated_pid_or_command(self):
        child = Mock(pid=999, args=['another command'])
        with self.assertRaisesRegex(ValueError, 'not created'):
            recovery.terminate_owned(child, ['expected'], {'pid': 123, 'command_sha256': digest(['expected'])})
        child.poll.assert_not_called()

    def test_successful_recovery_reuse_consumes_no_launch_budget(self):
        self.record(self.arm['path'])
        self.record(self.root/'arm.recovery-001', 'complete')
        gate = recovery.LaunchGate(0)
        with patch.object(recovery, 'successful', side_effect=self.successful), \
                patch.object(recovery, 'run_attempt') as runner:
            result = recovery.recover_arm(self.arm, self.root, self.protocol, gate=gate)
        self.assertEqual(result['recovery_status'], 'reused')
        self.assertEqual(gate.started, 0)
        runner.assert_not_called()


if __name__ == '__main__':
    unittest.main()
