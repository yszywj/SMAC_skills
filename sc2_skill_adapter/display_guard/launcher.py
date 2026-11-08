"""Bounded graphics-only recovery before PySC2 returns a game controller."""
from contextlib import contextmanager, ExitStack
import inspect
import logging
import os
import threading
import time

from .display_paths import recover_display_paths
from .graphics_log import watch_launch
from .windows_session import game_lease, has_running_sc2, monitor_windows


def _windows():
    try:
        return monitor_windows()
    except OSError as error:
        logging.warning("SC2 display enumeration unavailable: %s", error)
        return []


def _recover():
    try:
        with game_lease(exclusive=True, timeout=5):
            if has_running_sc2():
                return {"status": "busy", "detail": "Another SC2 process is running"}
            return recover_display_paths()
    except TimeoutError as error:
        return {"status": "busy", "detail": str(error)}
    except OSError as error:
        return {"status": "unavailable", "detail": str(error)}


def _placements(allow_reconfigure):
    # Normal launch has already failed with its own native error. First wait
    # for hotplug, then try each usable work area without changing the desktop.
    time.sleep(1)
    areas = _windows()
    for area in dict.fromkeys(areas):
        yield area
    if not areas:
        yield None  # Session may already have recovered without an enumerated work area.
    if not allow_reconfigure:
        return
    result = _recover()
    logging.warning("SC2 display recovery: %s", result)
    if result["status"] in ("activated", "reapplied", "protected_virtual", "busy", "unchanged"):
        time.sleep(1)
        yield from (_windows() or [None])


def _monitor(stop, child, watch):
    # PySC2's websocket connection may block beyond the campaign watchdog.
    # Only terminate the Popen created by this attempt, and only while its
    # own graphics log reports persistent failure before controller startup.
    while not stop.wait(0.6) and child.poll() is None:
        if watch.failed_log() and not stop.wait(2) and watch.failed_log():
            if child.poll() is None:
                try:
                    child.terminate()
                except OSError:
                    pass  # It may have exited while we read the log.
            return


def _guarded_class(original):
    signature = inspect.signature(original.__init__)

    class DisplayAwareProcess(original):
        def __init__(self, *args, **kwargs):
            initial = signature.bind(self, *args, **kwargs)
            self._display_lease = None
            self._watcher = None
            self._graphics = None
            alternatives = None
            placement = None
            while True:
                self._graphics = None
                self._stop = threading.Event()
                self._watcher = None
                bound = signature.bind(*initial.args, **initial.kwargs)
                if placement:
                    x, y, width, height = placement
                    bound.arguments["window_loc"] = (x, y)
                    bound.arguments["window_size"] = (width, height)
                lease = ExitStack()
                coordinated = True
                try:
                    try:
                        lease.enter_context(game_lease(timeout=15))
                    except TimeoutError:
                        raise  # A peer is changing the desktop.
                    except OSError as error:
                        coordinated = False
                        logging.warning("SC2 desktop lock unavailable; window-only recovery: %s", error)
                    self._display_lease = lease
                    original.__init__(*bound.args, **bound.kwargs)
                    return  # Keep the shared lease until PySC2 closes this game.
                except BaseException as error:
                    lease.close()
                    self._display_lease = None
                    failure = (self._graphics.failed_log() if isinstance(error, Exception)
                               and self._graphics else None)
                    if failure is None:
                        raise
                    logging.warning("SC2 native graphics initialization failed (%s): %s", failure, error)
                    if alternatives is None:
                        alternatives = _placements(coordinated)
                    try:
                        placement = next(alternatives)
                    except StopIteration:
                        raise error from None
                    logging.warning("Retrying SC2 before game creation; window=%s", placement or "default")
                finally:
                    self._stop_monitor()

        def _launch(self, run_config, args, **kwargs):
            self._graphics = watch_launch(args)
            child = super()._launch(run_config, args, **kwargs)
            if self._graphics:
                self._watcher = threading.Thread(target=_monitor, args=(self._stop, child, self._graphics),
                                                 name="sc2-graphics-startup", daemon=True)
                try:
                    self._watcher.start()
                except RuntimeError as error:
                    self._watcher = None
                    logging.warning("SC2 startup watcher unavailable: %s", error)
            return child

        def _stop_monitor(self):
            self._stop.set()
            if self._watcher:
                self._watcher.join()
                self._watcher = None

        def close(self):
            try:
                self._stop_monitor()
                super().close()
            finally:
                if self._display_lease is not None:
                    self._display_lease.close()
                    self._display_lease = None

    return DisplayAwareProcess


@contextmanager
def windows_startup_guard():
    if os.name != "nt" or os.environ.get("SMAC_SC2_DISPLAY_RECOVERY", "1") == "0":
        yield
        return
    from pysc2.lib import sc_process
    original = sc_process.StarcraftProcess
    sc_process.StarcraftProcess = _guarded_class(original)
    try:
        yield
    finally:
        sc_process.StarcraftProcess = original
