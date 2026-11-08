"""Windows desktop discovery and per-user interprocess coordination for SC2."""

import ctypes
from ctypes import wintypes
from contextlib import contextmanager
import math
import os
from pathlib import Path
import time


def _library(name):
    if os.name != 'nt':
        raise OSError('The SC2 desktop guard requires Windows.')
    return ctypes.WinDLL(name, use_last_error=True)


def _signature(function, result, *arguments):
    function.argtypes = arguments
    function.restype = result
    return function


class _MonitorInfo(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('monitor', wintypes.RECT),
                ('work', wintypes.RECT), ('flags', wintypes.DWORD)]


def _window_hint(work):
    """Suggested client rectangle; leave space for the nonclient frame."""
    width = work.right - work.left
    height = work.bottom - work.top
    if width <= 0 or height <= 0:
        return None
    side = min(16, (width - 1) // 2)
    vertical = min(72, height // 3)
    return (work.left + side, work.top + min(16, vertical // 2),
            min(1920, width - 2 * side), min(1200, height - vertical))


def monitor_windows():
    """List (x, y, client width, client height) hints, primary monitor first."""
    user32 = _library('user32')
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HANDLE,
                                       wintypes.HDC, ctypes.POINTER(wintypes.RECT),
                                       wintypes.LPARAM)
    enum = _signature(user32.EnumDisplayMonitors, wintypes.BOOL, wintypes.HDC,
                      ctypes.POINTER(wintypes.RECT), callback_type, wintypes.LPARAM)
    info_for = _signature(user32.GetMonitorInfoW, wintypes.BOOL, wintypes.HANDLE,
                          ctypes.POINTER(_MonitorInfo))
    items = []
    errors = []

    @callback_type
    def visit(monitor, _dc, _rect, _data):
        info = _MonitorInfo()
        info.size = ctypes.sizeof(info)
        if not info_for(monitor, ctypes.byref(info)):
            errors.append(ctypes.get_last_error())
            return False
        hint = _window_hint(info.work)
        if hint is not None:
            # MONITORINFOF_PRIMARY == 1; all coordinates are desktop-relative.
            items.append((not bool(info.flags & 1), hint))
        return True

    if not enum(None, None, visit, 0):
        raise ctypes.WinError(errors[0] if errors else ctypes.get_last_error())
    return list(dict.fromkeys(hint for _, hint in sorted(items)))


class _Overlapped(ctypes.Structure):
    # OVERLAPPED offset zero is the first byte, even if the lock file is empty.
    _fields_ = [('internal', ctypes.c_size_t), ('internal_high', ctypes.c_size_t),
                ('offset', wintypes.DWORD), ('offset_high', wintypes.DWORD),
                ('event', wintypes.HANDLE)]


@contextmanager
def game_lease(exclusive=False, timeout=5.0):
    """Hold a shared game or exclusive desktop-recovery lease for this user.

    A Windows byte-range lock belongs to the live handle. The permanent file
    contains no lock state and must never be unlinked/recreated to 'unlock' it.
    """
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError('Lease timeout must be nonnegative and finite.')
    kernel32 = _library('kernel32')
    import msvcrt
    lock = _signature(kernel32.LockFileEx, wintypes.BOOL, wintypes.HANDLE,
                      wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                      wintypes.DWORD, ctypes.POINTER(_Overlapped))
    unlock = _signature(kernel32.UnlockFileEx, wintypes.BOOL, wintypes.HANDLE,
                        wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                        ctypes.POINTER(_Overlapped))
    directory = os.environ.get('LOCALAPPDATA')
    if not directory:
        raise OSError('Cannot locate the per-user SC2 desktop guard lock.')
    folder = Path(directory) / 'SMAC_agent' / 'display_guard'
    folder.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(folder / 'games.lock', os.O_CREAT | os.O_RDWR | os.O_NOINHERIT)
    try:
        handle = msvcrt.get_osfhandle(descriptor)
        flags = 1 | (2 if exclusive else 0)  # FAIL_IMMEDIATELY | EXCLUSIVE_LOCK
        offset = _Overlapped()
        deadline = time.monotonic() + timeout
        while not lock(handle, flags, 0, 1, 0, ctypes.byref(offset)):
            error = ctypes.get_last_error()
            if error != 33:  # ERROR_LOCK_VIOLATION
                raise ctypes.WinError(error)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Another SC2 launch or display recovery holds the desktop lease.')
            time.sleep(min(0.1, remaining))
        try:
            yield
        finally:
            if not unlock(handle, 0, 1, 0, ctypes.byref(offset)):
                raise ctypes.WinError(ctypes.get_last_error())
    finally:
        os.close(descriptor)


class _ProcessEntry(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('usage', wintypes.DWORD),
                ('pid', wintypes.DWORD), ('heap_id', ctypes.c_size_t),
                ('module_id', wintypes.DWORD), ('threads', wintypes.DWORD),
                ('parent_pid', wintypes.DWORD), ('priority', wintypes.LONG),
                ('flags', wintypes.DWORD), ('image', wintypes.WCHAR * 260)]


def has_running_sc2():
    """Return whether any SC2 game executable exists in the system snapshot."""
    kernel32 = _library('kernel32')
    snapshot = _signature(kernel32.CreateToolhelp32Snapshot, wintypes.HANDLE,
                          wintypes.DWORD, wintypes.DWORD)
    first = _signature(kernel32.Process32FirstW, wintypes.BOOL, wintypes.HANDLE,
                       ctypes.POINTER(_ProcessEntry))
    advance = _signature(kernel32.Process32NextW, wintypes.BOOL, wintypes.HANDLE,
                         ctypes.POINTER(_ProcessEntry))
    close = _signature(kernel32.CloseHandle, wintypes.BOOL, wintypes.HANDLE)
    handle = snapshot(2, 0)  # TH32CS_SNAPPROCESS
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        entry = _ProcessEntry()
        entry.size = ctypes.sizeof(entry)
        if not first(handle, ctypes.byref(entry)):
            raise ctypes.WinError(ctypes.get_last_error())
        while True:
            if entry.image.casefold() in ('sc2_x64.exe', 'sc2.exe'):
                return True
            if not advance(handle, ctypes.byref(entry)):
                error = ctypes.get_last_error()
                if error == 18:  # ERROR_NO_MORE_FILES
                    return False
                raise ctypes.WinError(error)
    finally:
        close(handle)
