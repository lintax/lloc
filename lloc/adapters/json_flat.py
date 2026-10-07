"""json adapter: flat key -> value JSON, one file per language.

Web-i18n friendly (i18next flat resources): `<dir>/<code>.json` with plain
string values, keys sorted, UTF-8, no nesting — dots in keys stay literal.
The base language is written to `<base_language>.json`. File names carry
plain BCP-47 codes; legacy Android codes in the config (in/iw) are converted
to id/he (langnames.canonical_code) — the web world has no in/iw legacy.

Unmanaged discipline mirrors android-xml: keys present in an existing file
but absent from strings.toml survive export unchanged (merged in, sorted with
the rest). import_ returns every string value; unknown keys are filtered by
the import command. Untranslated managed keys are omitted, not emitted empty.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..fsutil import atomic_write_text
from ..langnames import canonical_code
from .base import Adapter


class JsonAdapter(Adapter):
    name = "json"

    @property
    def out_dir(self) -> Path:
        rel = self.project.adapter_config.get("dir", ".")
        return (self.project.root / rel).resolve()

    def file_for(self, lang: str) -> Path:
        return self.out_dir / f"{canonical_code(lang)}.json"

    def _read_existing(self, path: Path) -> dict[str, str]:
        from ..config import ConfigError

        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as e:
            raise ConfigError(f"{path}: existing file is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected a flat JSON object")
        return {k: v for k, v in data.items() if isinstance(v, str)}

    # --- export ----------------------------------------------------------

    def export(self, locales: dict[str, dict[str, dict]],
               langs: list[str] | None = None) -> list[str]:
        base = {key: {"text": e.text} for key, e in self.project.strings.items()}
        if langs is None:
            targets = [(self.project.base_language, base)] + [
                (lang, locales.get(lang, {})) for lang in self.project.languages]
        else:  # incremental: selected languages only, base untouched
            targets = [(lang, locales.get(lang, {})) for lang in langs]
        written = []
        for lang, entries in targets:
            path = self.file_for(lang)
            out = {k: v for k, v in self._read_existing(path).items()
                   if k not in self.project.strings}  # unmanaged keys survive
            for key in self.project.strings:
                e = entries.get(key)
                if e and e.get("text"):
                    out[key] = e["text"]
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                path, json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            written.append(str(path))
        return written

    # --- import ----------------------------------------------------------

    def import_(self) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for lang in self.project.languages:
            path = self.file_for(lang)
            if path.is_file():
                result[lang] = self._read_existing(path)
        return result

    # --- bootstrap -------------------------------------------------------

    def bootstrap_source(self) -> tuple[dict[str, str], int]:
        from ..config import ConfigError

        path = self.file_for(self.project.base_language)
        if not path.is_file():
            raise ConfigError(f"bootstrap: base file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected a flat JSON object")
        entries = {k: v for k, v in data.items() if isinstance(v, str)}
        return entries, len(data) - len(entries)

    # --- check -----------------------------------------------------------

    def check(self, text: str) -> list[str]:
        return []  # any Python string is representable in JSON
