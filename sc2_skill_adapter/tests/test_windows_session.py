import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from ctypes import wintypes
from display_guard.windows_session import _window_hint, game_lease, has_running_sc2, monitor_windows


class WindowHints(unittest.TestCase):
    def test_secondary_monitor_with_negative_coordinates_remains_on_its_screen(self):
        self.assertEqual(_window_hint(wintypes.RECT(-1620, -300, -20, 600)),
                         (-1604, -284, 1568, 828))

    def test_frame_fits_small_display_and_large_display_is_capped(self):
        self.assertEqual(_window_hint(wintypes.RECT(0, 0, 15, 5)), (7, 0, 1, 4))
        self.assertEqual(_window_hint(wintypes.RECT(0, 0, 3840, 2160)), (16, 16, 1920, 1200))
        self.assertIsNone(_window_hint(wintypes.RECT(0, 0, 0, 0)))


@unittest.skipUnless(os.name == 'nt', 'Windows desktop APIs')
class WindowsSession(unittest.TestCase):
    def test_monitor_probe_and_process_snapshot_are_read_only(self):
        for x, y, width, height in monitor_windows():
            self.assertIsInstance(x, int)
            self.assertIsInstance(y, int)
            self.assertTrue(0 < width <= 1920)
            self.assertTrue(0 < height <= 1200)
        self.assertIsInstance(has_running_sc2(), bool)

    def test_shared_lease_allows_other_shared_holders_and_survives_owner_crash(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'LOCALAPPDATA': temp}):
            ready = Path(temp) / 'child-ready'
            source = (
                "import sys, time\n"
                "from pathlib import Path\n"
                "from display_guard.windows_session import game_lease\n"
                "with game_lease(timeout=0):\n"
                "    Path(sys.argv[1]).touch()\n"
                "    time.sleep(60)\n"
            )
            worker = subprocess.Popen([sys.executable, '-B', '-c', source, str(ready)],
                                      cwd=Path(__file__).resolve().parents[1],
                                      creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                deadline = time.monotonic() + 10
                while not ready.exists() and time.monotonic() < deadline:
                    self.assertIsNone(worker.poll(), 'Lease owner exited before taking the lock')
                    time.sleep(0.05)
                self.assertTrue(ready.is_file())
                with game_lease(timeout=0):
                    pass
                with self.assertRaises(TimeoutError), game_lease(exclusive=True, timeout=0):
                    self.fail('Recovery cannot change displays while another game is running')
            finally:
                worker.kill()
                worker.wait(timeout=10)
            # An OS-released lease needs no deletion/recovery of stale lock files.
            with game_lease(exclusive=True, timeout=2):
                with self.assertRaises(TimeoutError), game_lease(timeout=0):
                    self.fail('A game cannot start during an exclusive recovery')
            with game_lease(timeout=0):
                pass


if __name__ == '__main__':
    unittest.main()
