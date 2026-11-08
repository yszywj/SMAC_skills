"""Attribute a Windows SC2 graphics startup failure to one PySC2 launch."""
import ctypes
from dataclasses import dataclass
from datetime import datetime
import ntpath
import os
from pathlib import Path
import re
import time


def game_logs_folder():
    """Respect the current user's redirected Documents folder on Windows."""
    documents = Path.home() / "Documents"
    if os.name == "nt":
        buffer = ctypes.create_unicode_buffer(260)
        if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buffer) == 0:
            documents = Path(buffer.value)
    return documents / "StarCraft II" / "GameLogs"


def log_filename(second):
    return datetime.fromtimestamp(second).strftime("%Y-%m-%d %H.%M.%S Graphics.txt")


def _read_small_log(path):
    # Initialization logs are a few KiB. A partial prefix of a large log
    # could hide another launch header or a later successful device creation.
    limit = 256 * 1024
    try:
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        if len(data) <= limit:
            return data.decode("utf-8-sig")
    except (OSError, UnicodeError):
        pass
    return ""


_RECORD = r"^GFX\s+\d{2}:\d{2}:\d{2}\.\d{3}\s+"
_HEADERS = re.compile(_RECORD + r"<Parameters>[^\S\n]+([^\r\n]+)", re.MULTILINE)
_SESSIONS = re.compile(_RECORD + r"StarCraft II \(B\d+\)", re.MULTILINE)
_FAILED = re.compile(_RECORD + r"CreateDevice failed:\s*$", re.MULTILINE)
_SUCCEEDED = re.compile(_RECORD + r"CreateDevice succeeded\b", re.MULTILINE)
_TEMP_DIR = re.compile(r'(?:^|\s)-tempDir\s+(?:"([^"]+)"|([^\s"]+))(?=\s|$)', re.I)
_PORT = re.compile(r"(?:^|\s)-port\s+(\d+)(?=\s|$)", re.I)


def _owned_failure(content, temp_dir, port):
    headers = _HEADERS.findall(content)
    if len(headers) != 1 or len(_SESSIONS.findall(content)) != 1:
        return False  # Concurrent launches can share a second-resolution name.
    ports = _PORT.findall(headers[0])
    directories = _TEMP_DIR.findall(headers[0])
    if ports != [port] or len(directories) != 1:
        return False
    quoted, plain = directories[0]
    logged_dir = ntpath.normcase(ntpath.normpath(quoted or plain))
    return (logged_dir == temp_dir and _FAILED.search(content) is not None
            and _SUCCEEDED.search(content) is None)


@dataclass(frozen=True)
class GraphicsLaunch:
    directory: Path
    temp_dir: str
    port: str
    start_second: int
    started: float
    existing_names: frozenset

    def failed_log(self):
        """Return this launch's explicit creation failure, or None if uncertain."""
        # Only probe filenames for the startup window. Thousands of older game
        # logs may be present and unrelated workers may be writing new files.
        elapsed = min(120, max(0, int(time.monotonic() - self.started) + 1))
        for offset in range(0, elapsed + 1):
            path = self.directory / log_filename(self.start_second + offset)
            if path.name not in self.existing_names:
                content = _read_small_log(path)
                if content and _owned_failure(content, self.temp_dir, self.port):
                    return path
        return None


def watch_launch(args, log_dir=None):
    """Call immediately before spawning SC2 with PySC2's final argv."""
    if args.count("-port") != 1 or args.count("-tempDir") != 1:
        return None
    try:
        port = str(args[args.index("-port") + 1])
        temp_dir = ntpath.normcase(ntpath.normpath(args[args.index("-tempDir") + 1]))
    except (IndexError, TypeError):
        return None
    if not port.isdecimal() or not 0 < int(port) <= 65535:
        return None
    if not ntpath.isabs(temp_dir) or not ntpath.basename(temp_dir).startswith("sc-"):
        return None
    directory = Path(log_dir) if log_dir is not None else game_logs_folder()
    start_second = int(time.time())
    started = time.monotonic()
    # Do not attribute an append/overwrite of an existing same-second file.
    names = [log_filename(start_second + offset) for offset in (0, 1)]
    try:
        existing = frozenset(name for name in names if (directory / name).exists())
    except OSError:
        return None
    return GraphicsLaunch(directory, temp_dir, port, start_second, started, existing)
