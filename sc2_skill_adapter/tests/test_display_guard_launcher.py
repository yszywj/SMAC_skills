import unittest
from contextlib import ExitStack, contextmanager
from unittest import mock

from display_guard import launcher

monitor_child = launcher._monitor


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.leases = []
        self.attempts = []
        self.outcomes = []
        tests = self

        @contextmanager
        def lease(exclusive=False, timeout=5):
            self.assertFalse(exclusive and self.leases)
            self.leases.append(exclusive)
            try:
                yield
            finally:
                self.leases.remove(exclusive)

        class NativeProcess:
            def __init__(self, run_config=None, window_loc=(50, 50), window_size=(1920, 1200)):
                tests.attempts.append((window_loc, window_size))
                try:
                    self._launch(None, ["sc2"])
                    if not tests.outcomes.pop(0):
                        raise ConnectionError("failed websocket")
                except Exception:
                    self.close()  # PySC2 closes the child before propagating.
                    raise

            def _launch(self, *args, **kwargs):
                return mock.Mock()

            def close(self):
                pass

        self.cls = launcher._guarded_class(NativeProcess)
        self.stack.enter_context(mock.patch.object(launcher, "game_lease", lease))
        self.stack.enter_context(mock.patch.object(launcher, "_monitor"))
        self.stack.enter_context(mock.patch.object(launcher.time, "sleep"))
        self.watch = self.stack.enter_context(mock.patch.object(launcher, "watch_launch"))
        self.watch.return_value.failed_log.return_value = "owned Graphics.txt"
        self.windows = self.stack.enter_context(mock.patch.object(launcher, "monitor_windows",
            return_value=[(16, 16, 1800, 1000), (-1500, 16, 1400, 880)]))
        self.other = self.stack.enter_context(mock.patch.object(launcher, "has_running_sc2", return_value=False))
        self.recover = self.stack.enter_context(mock.patch.object(launcher, "recover_display_paths",
            return_value={"status": "reapplied", "detail": "tested"}))

    def test_healthy_game_preserves_options_and_holds_shared_lease(self):
        self.outcomes = [True]
        game = self.cls(window_size=(777, 555))
        self.assertEqual(self.attempts, [((50, 50), (777, 555))])
        self.assertEqual(self.leases, [False])
        self.windows.assert_not_called()
        self.recover.assert_not_called()
        game.close()
        self.assertEqual(self.leases, [])

    def test_owned_failure_finds_secondary_screen_without_reconfiguring(self):
        self.outcomes = [False, False, True]
        game = self.cls()
        self.assertEqual(self.attempts[-1], ((-1500, 16), (1400, 880)))
        self.recover.assert_not_called()
        game.close()
        self.assertEqual(self.leases, [])

    def test_failure_on_all_work_areas_recovers_once_then_reenumerates(self):
        self.outcomes = [False, False, True]
        self.windows.side_effect = [[], [(16, 16, 1780, 900)]]
        game = self.cls()
        self.assertEqual(self.attempts[-1], ((16, 16), (1780, 900)))
        self.recover.assert_called_once()
        game.close()

    def test_unknown_error_propagates_without_screen_or_retry(self):
        self.outcomes = [False]
        self.watch.return_value.failed_log.return_value = None
        with self.assertRaises(ConnectionError):
            self.cls()
        self.assertEqual(len(self.attempts), 1)
        self.assertEqual(self.leases, [])
        self.windows.assert_not_called()

    def test_lock_permission_failure_still_allows_normal_game(self):
        self.outcomes = [True]
        with mock.patch.object(launcher, "game_lease", side_effect=PermissionError("read only profile")):
            game = self.cls()
        game.close()
        self.windows.assert_not_called()
        self.recover.assert_not_called()

    def test_other_process_prevents_layout_change_and_retries_are_bounded(self):
        self.windows.return_value = []
        self.other.return_value = True
        self.outcomes = [False, False, False]
        with self.assertRaises(ConnectionError):
            self.cls()
        self.assertEqual(len(self.attempts), 3)
        self.recover.assert_not_called()
        self.assertFalse(self.leases)

    def test_only_terminate_child_with_persistent_owned_failure(self):
        child = mock.Mock()
        child.poll.return_value = None
        signal = mock.Mock()
        signal.wait.side_effect = [False, False]
        watch = mock.Mock()
        watch.failed_log.side_effect = ["log", "log"]
        monitor_child(signal, child, watch)
        child.terminate.assert_called_once()
        child.reset_mock()
        signal.wait.side_effect = [False, False, True]
        watch.failed_log.side_effect = ["log", None]
        monitor_child(signal, child, watch)
        child.terminate.assert_not_called()
