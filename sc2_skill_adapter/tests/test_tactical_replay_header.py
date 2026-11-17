"""Exercise the exact launcher audit helper with both MPQ header layouts."""
from pathlib import Path
import struct
import tempfile
import unittest


LAUNCHER = Path(__file__).resolve().parents[1] / 'scripts/run_tactical_matrix.ps1'


def audit_namespace():
    source = LAUNCHER.read_text(encoding='utf8').split("$auditSource = @'\n", 1)[1]
    source = source.split("\n'@", 1)[0].split('\ntry:\n    mode, *args', 1)[0]
    namespace = {}
    exec(compile(source, str(LAUNCHER), 'exec'), namespace)
    return namespace


class NativeReplayHeaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = staticmethod(audit_namespace()['native_replay_header'])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'episode.SC2Replay'

    @staticmethod
    def archive(header_size=32, archive_size=64):
        return struct.pack('<4sIIHHIIII', b'MPQ\x1a', header_size, archive_size,
                           3, 5, 32, 32, 0, 0) + bytes(max(0, archive_size - 32))

    def check_bytes(self, data):
        self.path.write_bytes(data)
        return self.audit(self.path)

    def test_direct_archive_header(self):
        result = self.check_bytes(self.archive())
        self.assertEqual(result['archive_offset'], 0)
        self.assertFalse(result['user_data'])

    def test_native_sc2_user_data_layout(self):
        # Exact layout observed in all ten reported replays: 512 / 1024 / 114;
        # archive v3 uses a 208-byte header. Payload is inert, not a real game.
        data = struct.pack('<4sIII', b'MPQ\x1b', 512, 1024, 114)
        result = self.check_bytes(data + bytes(1024 - 16) + self.archive(208, 256))
        self.assertEqual(result, dict(user_data=True, archive_offset=1024,
                                     header_size=208, archive_size=256))

    def test_declared_offset_need_not_be_1024(self):
        data = struct.pack('<4sIII', b'MPQ\x1b', 128, 512, 48)
        self.assertEqual(self.check_bytes(data + bytes(512-16) + self.archive())['archive_offset'], 512)

    def test_truncated_or_unrecognized_headers_rejected(self):
        for data in (b'', b'MPQ\x1a', b'MPQ\x1b', b'not replay data', self.archive()[:-1]):
            with self.subTest(data=data[:16]), self.assertRaisesRegex(ValueError, 'Invalid native replay header'):
                self.check_bytes(data)

    def test_user_data_bounds_rejected(self):
        for user_size, offset, content in ((512, 4096, 114), (512, 8, 114),
                                           (10, 1024, 114), (2048, 1024, 1020)):
            data = struct.pack('<4sIII', b'MPQ\x1b', user_size, offset, content)
            with self.subTest(offset=offset, content=content), self.assertRaises(ValueError):
                self.check_bytes(data + bytes(1024-16) + self.archive())

    def test_archive_signature_at_wrong_offset_does_not_pass(self):
        data = struct.pack('<4sIII', b'MPQ\x1b', 512, 1024, 114)
        with self.assertRaisesRegex(ValueError, 'missing archive at declared offset'):
            self.check_bytes(data + bytes(496) + self.archive() + bytes(1024))

    def test_invalid_archive_sizes_rejected(self):
        for header_size, archive_size in ((16, 64), (208, 64), (32, 999)):
            data = struct.pack('<4sIIHHIIII', b'MPQ\x1a', header_size, archive_size,
                               3, 5, 32, 32, 0, 0) + bytes(32)
            with self.subTest(header_size=header_size, archive_size=archive_size), self.assertRaises(ValueError):
                self.check_bytes(data)


if __name__ == '__main__':
    unittest.main()
