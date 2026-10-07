"""Translation store (locales/*.json), lock file, delta computation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import Project
from .fsutil import atomic_write_text
from .hashes import text_hash

LOCK_NAME = "l10n.lock.json"
LOCALES_DIR = "locales"


def base_hashes(project: Project) -> dict[str, str]:
    """Current canonical hashes of all base strings, in strings.toml order."""
    return {key: text_hash(e.text) for key, e in project.strings.items()}


# --- lock ---------------------------------------------------------------

def lock_path(root: Path) -> Path:
    return root / LOCK_NAME


def load_lock(root: Path) -> dict[str, str]:
    p = lock_path(root)
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return dict(data.get("keys", {}))


def save_lock(root: Path, keys: dict[str, str]) -> None:
    payload = {"version": 1, "keys": keys}
    atomic_write_text(lock_path(root), json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


# --- locale store -------------------------------------------------------

def locale_path(root: Path, lang: str) -> Path:
    return root / LOCALES_DIR / f"{lang}.json"


def load_locale(root: Path, lang: str) -> dict[str, dict]:
    """Return {key: {"text": ..., "src": ...}}; empty dict if file absent."""
    p = locale_path(root, lang)
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return dict(data.get("strings", {}))


def save_locale(root: Path, lang: str, entries: dict[str, dict], order: list[str]) -> None:
    """Write locale file with keys in `order` (strings.toml order). Keys not in
    `order` are dropped (orphan cleanup happens naturally on save)."""
    ordered = {k: entries[k] for k in order if k in entries}
    payload = {"version": 1, "language": lang, "strings": ordered}
    p = locale_path(root, lang)
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(p, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


# --- classification -----------------------------------------------------

@dataclass
class LangStatus:
    lang: str
    ok: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    orphan: list[str] = field(default_factory=list)

    @property
    def pending(self) -> list[str]:
        """Keys that need translation (missing + stale), in base order."""
        return self.missing + self.stale

    @property
    def clean(self) -> bool:
        return not self.missing and not self.stale and not self.orphan


def classify(project: Project, lang: str, locale: dict[str, dict] | None = None) -> LangStatus:
    if locale is None:
        locale = load_locale(project.root, lang)
    current = base_hashes(project)
    st = LangStatus(lang=lang)
    for key, h in current.items():
        entry = locale.get(key)
        if entry is None or not entry.get("text"):
            st.missing.append(key)
        elif entry.get("src") != h:
            st.stale.append(key)
        else:
            st.ok.append(key)
    for key in locale:
        if key not in current:
            st.orphan.append(key)
    return st
