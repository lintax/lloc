"""Per-module run lock: at most one mutating lloc run per module.

OS-level advisory lock (fcntl.flock on POSIX, msvcrt.locking on Windows —
both stdlib) held on <module>/.lloc.lock for the duration of the run. The
lock dies with the process: after a kill or crash NO manual cleanup is
needed — the file may remain on disk, but the lock itself is released by
the OS and the next run proceeds normally.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

LOCK_FILE = ".lloc.lock"


class LockBusy(Exception):
    """Another lloc process holds the module lock."""


def _try_lock(f) -> bool:
    if os.name == "nt":  # pragma: no cover - exercised on Windows only
        import msvcrt
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(f) -> None:
    if os.name == "nt":  # pragma: no cover - exercised on Windows only
        import msvcrt
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        return
    import fcntl
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@contextmanager
def module_lock(root: Path, label: str = ""):
    """Hold the advisory run lock for a module directory (non-blocking).
    Raises LockBusy immediately if another process holds it."""
    path = Path(root) / LOCK_FILE
    f = open(path, "a+", encoding="utf-8")
    if not _try_lock(f):
        f.close()
        where = label or str(root)
        raise LockBusy(f"another lloc run is active for this module ({where}); "
                       "wait for it to finish — no lock file cleanup is ever needed")
    try:
        yield
    finally:
        try:
            _unlock(f)
        finally:
            f.close()
