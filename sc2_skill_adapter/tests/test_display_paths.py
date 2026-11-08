import ctypes
import unittest
from unittest.mock import patch

from display_guard import display_paths as dp


def path(target, source=0, *, active=True, available=True, tech=5):
    p = dp._Path()
    p.target.id = target
    p.source.id = source
    p.flags = dp.ACTIVE if active else 0
    p.target.available = available
    p.target.tech = tech
    return p


class FakeDriver:
    def __init__(self, active=(), available=(), saved=()):
        self.current = dp._Layout(list(active), [])
        self.all = dp._Layout(list(available), [])
        self.saved = dp._Layout(list(saved), [])
        self.calls = []
        self.reject = set()

    def query(self, flags):
        return {dp.QDC_ONLY_ACTIVE_PATHS: self.current, dp.QDC_ALL_PATHS: self.all,
                dp.QDC_DATABASE_CURRENT: self.saved}[flags]

    def describe(self, p):
        virtual_driver = p.target.id >= 50
        bus = r'\\?\ROOT#DISPLAY' if virtual_driver else r'\\?\PCI#VEN_8086'
        return {'kind': dp._kind(bus, p.target.tech), 'available': bool(p.target.available)}

    def configure(self, layout, *, validate):
        self.calls.append((validate, layout))
        if validate:
            if any(p.target.id in self.reject for p in layout.paths):
                raise OSError('unsupported display')
        else:
            self.current = layout


class DisplayPathsTests(unittest.TestCase):
    def test_abi_and_driver_detection(self):
        self.assertEqual(ctypes.sizeof(dp._Path), 72)
        self.assertEqual(ctypes.sizeof(dp._Mode), 64)
        self.assertEqual(dp._kind(r'\\?\ROOT#DISPLAY', 5), 'indirect')
        self.assertEqual(dp._kind(r'\\?\PCI#VEN_8086', 17), 'indirect')
        self.assertEqual(dp._kind(r'\\?\PCI#VEN_8086', 5), 'physical')
        self.assertEqual(dp._kind('', 5), 'unknown')

    def test_keep_virtual_privacy_even_with_active_physical_monitor(self):
        both = [path(1), path(51)]
        api = FakeDriver(both, both)
        self.assertEqual(dp._recover(api)['status'], 'protected_virtual')
        self.assertFalse(api.calls)

    def test_reapply_exact_dual_monitor_modes(self):
        now = [path(1), path(2, source=1)]
        api = FakeDriver(now, now)
        api.current.modes = [dp._Mode()]*4
        self.assertEqual(dp._recover(api)['status'], 'reapplied')
        self.assertEqual([v for v, _ in api.calls], [True, False])
        self.assertIs(api.calls[0][1], api.calls[1][1])
        self.assertEqual(api.calls[0][1].active_keys, api.current.active_keys)
        self.assertEqual(len(api.calls[1][1].modes), 4)

    def test_restore_saved_clone_before_guessing_a_single_monitor(self):
        saved = [path(1), path(2)]   # two targets can legitimately share one source
        api = FakeDriver(available=[path(2, active=False), path(1, active=False)], saved=saved)
        self.assertEqual(dp._recover(api)['status'], 'activated')
        self.assertIs(api.calls[1][1], api.saved)
        self.assertEqual(api.current.active_keys, api.saved.active_keys)

    def test_skip_disconnected_saved_targets_and_invalid_routes(self):
        routes = [path(1, active=False, available=False), path(2, active=False), path(3, active=False)]
        api = FakeDriver(available=routes, saved=[path(1)])
        api.reject = {2}
        self.assertEqual(dp._recover(api)['status'], 'activated')
        self.assertEqual([(validate, plan.paths[0].target.id) for validate, plan in api.calls],
                         [(True, 2), (True, 3), (False, 3)])
        self.assertEqual(api.current.paths[0].source.mode, dp.INVALID_MODE)
        self.assertFalse(api.current.modes)

    def test_indirect_only_when_no_live_desktop_and_physical_routes_fail(self):
        api = FakeDriver(available=[path(1, active=False), path(50, active=False)])
        api.reject = {1}
        self.assertEqual(dp._recover(api)['status'], 'activated')
        self.assertEqual(api.calls[-1][1].paths[0].target.id, 50)

    def test_stale_indirect_path_never_wakes_physical_screen(self):
        old = path(51, available=False)
        api = FakeDriver(active=[old], available=[path(1, active=False)])
        self.assertEqual(dp._recover(api)['status'], 'protected_virtual')
        self.assertFalse(api.calls)
        api.all.paths.append(path(52, active=False))
        self.assertEqual(dp._recover(api)['status'], 'protected_virtual')
        self.assertFalse(api.calls)
        api.all.paths.append(path(51, active=False))
        self.assertEqual(dp._recover(api)['status'], 'activated')
        self.assertEqual(api.calls[-1][1].paths[0].target.id, 51)

    def test_no_write_if_screen_changed_during_validation(self):
        api = FakeDriver(available=[path(1, active=False)])
        def changed(layout, *, validate):
            self.assertTrue(validate)
            api.current = dp._Layout([path(50)], [])
        api.configure = changed
        self.assertEqual(dp._recover(api)['status'], 'unchanged')

    def test_failed_verification_restores_original_dual_layout(self):
        now = [path(1), path(2, source=1)]
        api = FakeDriver(active=now, available=now)
        original = api.current
        saved_method = api.configure
        def drop_one(layout, *, validate):
            saved_method(layout, validate=validate)
            if not validate and len(api.calls) == 2:
                api.current = dp._Layout(now[:1], [])
        api.configure = drop_one
        with patch.object(dp.time, 'sleep'):
            report = dp._recover(api)
        self.assertEqual(report['status'], 'failed')
        self.assertIn('restored', report['detail'])
        self.assertIs(api.current, original)
        self.assertEqual([v for v, _ in api.calls], [True, False, True, False])

    def test_native_access_denied_is_reported(self):
        with patch.object(dp, '_Windows', side_effect=OSError('access denied')):
            result = dp.recover_display_paths()
        self.assertEqual(result['status'], 'unavailable')
        self.assertIn('access denied', result['detail'])
