"""Regression coverage for attribution of a native graphics startup failure."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from display_guard import graphics_log


TEMP = "C:\\A Folder\\Temp\\sc-owned"
STAMP = 1_800_000_000


def section(port="50123", directory=TEMP, lines=("CreateDevice failed:",)):
    start = [
        "StarCraft II (B98310)",
        f'<Parameters> -listen 127.0.0.1 -port {port} -tempDir "{directory}\\" -windowwidth 1920',
    ]
    return "".join(f"GFX  12:34:56.123 {line}\r\n" for line in [*start, *lines])


class GraphicsLogTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name)
        self.path = self.folder / graphics_log.log_filename(STAMP)
        self.args = ["SC2_x64.exe", "-port", "50123", "-tempDir", TEMP + "\\"]

    def watch(self):
        with patch.object(graphics_log.time, "time", return_value=STAMP):
            return graphics_log.watch_launch(self.args, self.folder)

    def test_positive_matches_exact_port_and_normalized_full_temp_directory(self):
        launch = self.watch()
        self.path.write_text(section(directory=TEMP.lower()), encoding="utf-8-sig")
        self.assertEqual(launch.failed_log(), self.path)

    def test_unquoted_native_arguments(self):
        args = ["SC2_x64.exe", "-port", "50123", "-tempDir", r"C:\Temp\sc-owned"]
        with patch.object(graphics_log.time, "time", return_value=STAMP):
            launch = graphics_log.watch_launch(args, self.folder)
        self.path.write_text(section(directory=r"C:\Temp\sc-owned").replace('"', ''), encoding="utf-8")
        self.assertEqual(launch.failed_log(), self.path)

    def test_foreign_port_or_prefix_similar_temp_never_matches(self):
        launch = self.watch()
        for content in [section(port="501230"), section(directory=TEMP + "2")]:
            with self.subTest(content=content):
                self.path.write_text(content, encoding="utf-8")
                self.assertIsNone(launch.failed_log())

    def test_concurrent_startup_records_in_same_file_rejected(self):
        launch = self.watch()
        self.path.write_text(section(lines=()) + section(port="50124"), encoding="utf-8")
        self.assertIsNone(launch.failed_log())

    def test_initialization_errors_and_subsequent_lost_device_are_distinct(self):
        launch = self.watch()
        for lines in [
                ("DirectX call failed with error code -2005530516",),
                ("CreateDevice failed:", "CreateDevice succeeded"),
                ("CreateDevice succeeded", "Lost D3D9 device", "CreateDevice failed:")]:
            with self.subTest(lines=lines):
                self.path.write_text(section(lines=lines), encoding="utf-8")
                self.assertIsNone(launch.failed_log())

    def test_preexisting_same_second_log_is_not_evidence_even_if_appended(self):
        self.path.touch()
        launch = self.watch()
        self.path.write_text(section(), encoding="utf-8")
        self.assertIsNone(launch.failed_log())

    def test_no_match_without_per_launch_temp_port_pair_or_fresh_file(self):
        for args in ([], ["-port", "0", "-tempDir", TEMP], ["-port", "1", "-tempDir"],
                     ["-port", "50123", "-tempDir", r"C:\Temp"]):
            self.assertIsNone(graphics_log.watch_launch(args, self.folder))
        launch = self.watch()
        old = self.folder / graphics_log.log_filename(STAMP - 60)
        old.write_text(section(), encoding="utf-8")
        self.assertIsNone(launch.failed_log())

    def test_bounded_read_refuses_partial_evidence(self):
        launch = self.watch()
        self.path.write_text(section() + "x" * (256 * 1024), encoding="utf-8")
        self.assertIsNone(launch.failed_log())


if __name__ == "__main__":
    unittest.main()
