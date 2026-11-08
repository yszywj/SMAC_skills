"""Transient Windows display recovery for a *confirmed* SC2 graphics startup failure.

inventory_display_paths is read-only. recover_display_paths never saves a Windows
display profile or switches away from an active indirect/unknown display.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
import os
import time


# Use explicit Windows ABI widths even when the planner is unit-tested elsewhere.
U32 = C.c_uint32
QDC_ALL_PATHS = 1
QDC_ONLY_ACTIVE_PATHS = 2
QDC_DATABASE_CURRENT = 4
SDC_USE_SUPPLIED_DISPLAY_CONFIG = 0x20
SDC_VALIDATE = 0x40
SDC_APPLY = 0x80
SDC_NO_OPTIMIZATION = 0x100
SDC_ALLOW_CHANGES = 0x400
ACTIVE = 1
INVALID_MODE = 0xFFFFFFFF


class _Luid(C.Structure):
    _fields_ = [('low', U32), ('high', C.c_int32)]


class _Source(C.Structure):
    _fields_ = [('adapter', _Luid), ('id', U32), ('mode', U32), ('status', U32)]


class _Target(C.Structure):
    _fields_ = [('adapter', _Luid), ('id', U32), ('mode', U32), ('tech', U32),
                ('rotation', U32), ('scaling', U32), ('refresh_num', U32),
                ('refresh_den', U32), ('scan', U32), ('available', C.c_int32),
                ('status', U32)]


class _Path(C.Structure):
    _fields_ = [('source', _Source), ('target', _Target), ('flags', U32)]


class _ModePayload(C.Union):
    # DISPLAYCONFIG_VIDEO_SIGNAL_INFO is the largest variant, with 8-byte alignment.
    _fields_ = [('data', C.c_ubyte * 48), ('align', C.c_uint64)]


class _Mode(C.Structure):
    _fields_ = [('type', U32), ('id', U32), ('adapter', _Luid), ('payload', _ModePayload)]


class _Header(C.Structure):
    _fields_ = [('type', U32), ('size', U32), ('adapter', _Luid), ('id', U32)]


class _AdapterName(C.Structure):
    _fields_ = [('header', _Header), ('path', C.c_wchar * 128)]


class _TargetName(C.Structure):
    _fields_ = [('header', _Header), ('flags', U32), ('tech', U32),
                ('manufacturer', C.c_uint16), ('product', C.c_uint16), ('connector', U32),
                ('name', C.c_wchar * 64), ('path', C.c_wchar * 128)]


def _id(endpoint):
    return (endpoint.adapter.high, endpoint.adapter.low, endpoint.id)


def _kind(adapter_path, technology):
    # Some virtual drivers advertise HDMI and an ordinary monitor name/EDID.
    bus = adapter_path.upper()
    if technology in (15, 16, 17, 18) or '#ROOT#' in bus or '\\ROOT#' in bus or '\\SWD#' in bus:
        return 'indirect'
    if ('\\PCI#' in bus or '#PCI#' in bus) and technology in {*range(15), 0x80000000}:
        return 'physical'
    return 'unknown'


@dataclass
class _Layout:
    paths: list
    modes: list
    allow_changes: bool = False

    @property
    def active_keys(self):
        return {(_id(p.source), _id(p.target)) for p in self.paths if p.flags & ACTIVE}

    def signature(self):
        return ([bytes(p) for p in self.paths], [bytes(m) for m in self.modes])


class _Windows:
    def __init__(self):
        if os.name != 'nt':
            raise OSError('Windows DisplayConfig is unavailable on this platform')
        self.user32 = C.WinDLL('user32')
        self.user32.GetDisplayConfigBufferSizes.argtypes = (U32, C.POINTER(U32), C.POINTER(U32))
        self.user32.QueryDisplayConfig.argtypes = (
            U32, C.POINTER(U32), C.POINTER(_Path), C.POINTER(U32), C.POINTER(_Mode), C.POINTER(U32))
        self.user32.DisplayConfigGetDeviceInfo.argtypes = (C.c_void_p,)
        self.user32.SetDisplayConfig.argtypes = (U32, C.POINTER(_Path), U32, C.POINTER(_Mode), U32)

    def query(self, flags):
        # Monitor hotplug can change the required buffer size between the two calls.
        for _ in range(4):
            n, m = U32(), U32()
            err = self.user32.GetDisplayConfigBufferSizes(flags, C.byref(n), C.byref(m))
            if err:
                raise C.WinError(err)
            paths, modes = (_Path * n.value)(), (_Mode * m.value)()
            topology = U32()
            err = self.user32.QueryDisplayConfig(flags, C.byref(n), paths, C.byref(m), modes,
                                                 C.byref(topology) if flags == QDC_DATABASE_CURRENT else None)
            if err == 122:  # ERROR_INSUFFICIENT_BUFFER
                continue
            if err:
                raise C.WinError(err)
            return _Layout(list(paths[:n.value]), list(modes[:m.value]))
        raise OSError('Display configuration changed repeatedly during enumeration')

    def describe(self, path):
        adapter = _AdapterName()
        adapter.header = _Header(4, C.sizeof(adapter), path.target.adapter, 0)
        if self.user32.DisplayConfigGetDeviceInfo(C.byref(adapter)):
            adapter_path = ''
        else:
            adapter_path = adapter.path
        name = _TargetName()
        name.header = _Header(2, C.sizeof(name), path.target.adapter, path.target.id)
        label = '' if self.user32.DisplayConfigGetDeviceInfo(C.byref(name)) else name.name
        return {'name': label, 'kind': _kind(adapter_path, path.target.tech),
                'target_id': _id(path.target), 'source_id': _id(path.source),
                'available': bool(path.target.available), 'active': bool(path.flags & ACTIVE)}

    def configure(self, layout, *, validate):
        paths = (_Path * len(layout.paths))(*layout.paths)
        modes = (_Mode * len(layout.modes))(*layout.modes) if layout.modes else None
        flags = SDC_USE_SUPPLIED_DISPLAY_CONFIG | (SDC_VALIDATE if validate else SDC_APPLY | SDC_NO_OPTIMIZATION)
        if layout.allow_changes:
            flags |= SDC_ALLOW_CHANGES
        err = self.user32.SetDisplayConfig(len(paths), paths, len(layout.modes), modes, flags)
        if err:
            raise C.WinError(err)


def inventory_display_paths():
    """Return read-only per-source/target paths. A target may have multiple possible routes."""
    api = _Windows()
    return [api.describe(p) for p in api.query(QDC_ALL_PATHS).paths if p.target.available or p.flags & ACTIVE]


def _candidates(api, original, all_paths, *, indirect_only):
    if not indirect_only and original.paths and all(p.target.available for p in original.paths):
        yield 'reapplied', original
    available = [p for p in all_paths.paths if p.target.available]
    if indirect_only:
        # "Indirect" can also mean a USB or wireless monitor. During a
        # possible private session only revive the exact previously active
        # indirect target; do not light up a different monitor.
        previous = {_id(p.target) for p in original.paths if api.describe(p)['kind'] == 'indirect'}
        available = [p for p in available if _id(p.target) in previous]
    kinds = {_id(p.target): api.describe(p)['kind'] for p in available}
    # Preserve the normal single-, extended- or cloned-screen layout if Windows
    # has a matching saved topology. Never guess which socket is "screen 1".
    try:
        saved = api.query(QDC_DATABASE_CURRENT)
    except OSError:
        saved = None
    # If a real desktop exists, preserve it. When Windows has no active path,
    # an available indirect output is a useful fallback if all real paths fail.
    if indirect_only:
        kinds_to_try = ('indirect',)
    elif any(p.target.available for p in original.paths):
        kinds_to_try = ('physical',)
    else:
        kinds_to_try = ('physical', 'indirect')
    for kind in kinds_to_try:
        usable = {key for key, value in kinds.items() if value == kind}
        if saved and saved.paths and all(p.flags & ACTIVE and _id(p.target) in usable for p in saved.paths):
            yield 'activated', saved
        for p in available:
            if _id(p.target) in usable:
                single = _Path.from_buffer_copy(p)
                single.flags = ACTIVE
                single.source.mode = single.target.mode = INVALID_MODE
                yield 'activated', _Layout([single], [], allow_changes=True)


def _recover(api):
    original = api.query(QDC_ONLY_ACTIVE_PATHS)
    active = [api.describe(p) for p in original.paths]

    def result(status, detail):
        return {'status': status, 'detail': detail, 'targets': active}

    indirect_active = any(p['kind'] != 'physical' for p in active)
    if indirect_active and any(p['available'] for p in active):
        return result('protected_virtual', 'Active indirect or unknown display; layout preserved')

    all_paths = api.query(QDC_ALL_PATHS)
    last_error = 'No available, identifiable output path'
    for status, plan in _candidates(api, original, all_paths, indirect_only=indirect_active):
        try:
            api.configure(plan, validate=True)
        except OSError as exc:
            last_error = str(exc)
            continue
        if api.query(QDC_ONLY_ACTIVE_PATHS).signature() != original.signature():
            return result('unchanged', 'Display configuration changed during recovery')
        try:
            api.configure(plan, validate=False)
        except OSError as exc:
            last_error = str(exc)
            # An unsuccessful driver call can still have affected the desktop.
            if api.query(QDC_ONLY_ACTIVE_PATHS).signature() == original.signature():
                continue
        else:
            for delay in (0, .15, .35):
                if delay:
                    time.sleep(delay)
                try:
                    current = api.query(QDC_ONLY_ACTIVE_PATHS)
                except OSError as exc:
                    last_error = 'Could not verify the active display: '+str(exc)
                    continue
                if (current.active_keys == plan.active_keys and current.paths
                        and all(p.target.available for p in current.paths)):
                    return {'status': status, 'detail': 'Validated, applied and verified',
                            'targets': [api.describe(p) for p in current.paths]}
                last_error = 'Requested active display paths were not verified'
        current = api.query(QDC_ONLY_ACTIVE_PATHS)
        # Do not undo a new virtual session or another user's newly active output.
        if (original.paths and current.active_keys <= original.active_keys | plan.active_keys
                and all(api.describe(p)['kind'] == 'physical' for p in current.paths)):
            api.configure(original, validate=True)
            api.configure(original, validate=False)
            restored = api.query(QDC_ONLY_ACTIVE_PATHS)
            last_error += ('; previous paths restored' if restored.active_keys == original.active_keys
                           else '; previous paths could not be verified')
        return result('failed', last_error)
    if indirect_active:
        return result('protected_virtual', 'Detached indirect path still marked active; '+last_error)
    return result('unavailable', last_error)


def recover_display_paths():
    """Recover after an actual graphics-init error. No persistence or default monitor IDs."""
    try:
        return _recover(_Windows())
    except OSError as exc:
        return {'status': 'unavailable', 'detail': str(exc), 'targets': []}
