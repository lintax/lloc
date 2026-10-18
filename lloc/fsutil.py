"""Filesystem and time helpers: atomic writes, UTC timestamps, durations."""
from __future__ import annotations

import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file in the same directory + os.replace: readers never
    see a partial file, an interrupted write leaves the old content intact.
    os.replace is atomic on POSIX and on Windows (same volume)."""
    # per process AND thread: two threads may write the same file
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():  # replace failed mid-way: don't litter
            tmp.unlink(missing_ok=True)


def utc_now_iso() -> str:
    """Current UTC time, second precision, ISO-8601 with offset — readable in
    git diffs and parseable by datetime.fromisoformat on Python 3.10+."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_when(value: str) -> datetime | None:
    """Parse a stored timestamp; None for absent/foreign values."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


_DURATION_RE = re.compile(r"^(\d+)\s*([mhd]?)$")
_UNITS = {"m": 60, "h": 3600, "d": 86400, "": 3600}  # bare number = hours


def parse_duration(text: str) -> timedelta:
    """'90m' / '24h' / '7d' / '24' (hours) -> timedelta; ValueError otherwise."""
    m = _DURATION_RE.match(text.strip().lower())
    if not m:
        raise ValueError(f"invalid duration {text!r} (expected e.g. 90m, 24h, 7d)")
    return timedelta(seconds=int(m.group(1)) * _UNITS[m.group(2)])
