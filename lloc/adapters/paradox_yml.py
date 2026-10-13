"""paradox-yml adapter: Paradox game localization (`key:0 "text"` files).

Covers Stellaris, CK3, EU, Victoria, HOI4, Imperator and their mods. Unlike
the other adapters the base language is not generated: the English files of
a game or mod ARE the source, and `import --bootstrap-strings` turns them
into strings.toml (with `type`/`context` derived from the key and file name).

  source_dir  scanned recursively for `*_l_<language>.yml`
  out_dir     translations go to `<out_dir>/<language>/<out_name>_l_<language>.yml`
              (default `out`, next to l10n.toml — the source tree is never
              written to unless pointed at)
  out_name    default "lloc"
  exclude     fnmatch patterns of file names to ignore on import
  languages   BCP-47 code -> Paradox language name, on top of the built-ins

Text is stored unescaped: the two-character `\\n` of the files becomes a real
line break (so the usual escape validation applies) and `\\"` a plain quote.

Name lists (the key is the name itself, or the file is a `*_names` /
`name_list_*` file) are left out of strings.toml like script-only values.

import_ returns the translations that already ship with the source, minus
the ones that are plainly still English — mods routinely copy the base text
into every language file. Export writes machine/manual translations only:
imported entries already load from their own files.
"""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from ..fsutil import atomic_write_text
from ..langnames import canonical_code, expected_scripts
from ..validate import script_error, strip_placeholders
from .base import Adapter

STYLE = "paradox"
BOM = "\ufeff"

LANGUAGE_NAMES = {
    "en": "english", "ru": "russian", "de": "german", "fr": "french",
    "es": "spanish", "pl": "polish", "ja": "japanese", "ko": "korean",
    "zh": "simp_chinese", "zh-Hans": "simp_chinese", "pt": "braz_por",
    "pt-BR": "braz_por", "it": "italian", "nl": "dutch", "tr": "turkish",
    "ar": "arabic",
}

_KEY = r"([A-Za-z0-9_.\-']+):(\d*)"
# well-formed value (inner quotes escaped), optional trailing comment
_STRICT_RE = re.compile(_KEY + r'\s*"((?:[^"\\]|\\.)*)"\s*(?:#.*)?$')
# the game itself reads up to the LAST quote of the line
_LOOSE_RE = re.compile(_KEY + r'\s*"(.*)"[^"]*$')
_HEADER_RE = re.compile(r"l_[a-z_]+:\s*(?:#.*)?$")
_UNESCAPE = {"n": "\n", '"': '"', "t": "\t"}

# key suffix -> string type shown to the translator
_TYPES = (
    (re.compile(r"(?i)[._](desc|description|flavor|flavour)$"), "description"),
    (re.compile(r"(?i)[._](tooltip|tt|effect|custom_tooltip)$"), "tooltip"),
    (re.compile(r"(?i)\.\d+\.(name|title|t)$"), "event title"),
    (re.compile(r"(?i)\.\d+\.(d|[a-z]\.response)$"), "event text"),
    (re.compile(r"(?i)\.\d+\.[a-z]\d?$"), "event option"),
    (re.compile(r"(?i)[._](name|title|plural|adj)$"), "name"),
)


# name-list files: star, ship, species and leader names
_NAME_FILE_RE = re.compile(r"(?i)(^|_)name_?lists?(_|$)|(^|_|\d)(random_)?names?(_|$)")


def is_name(key: str, text: str, stem: str) -> bool:
    """Name-list entries are proper names the game draws at random: there is
    nothing to translate, and thousands of them per mod. The game's own
    convention marks them (the key IS the name); the file name catches the
    rest."""
    return text.strip() == key or bool(_NAME_FILE_RE.search(stem))


def string_type(key: str) -> str:
    for pattern, kind in _TYPES:
        if pattern.search(key):
            return kind
    return ""


def unescape(raw: str) -> str:
    return re.sub(r'\\([n"t])', lambda m: _UNESCAPE[m.group(1)], raw)


def escape(text: str) -> str:
    return text.replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")


def parse(text: str) -> list[tuple[str, str, str]]:
    """[(key, version, unescaped text)] in document order. Lines before the
    `l_<language>:` header, comments and malformed lines are ignored."""
    out = []
    seen_header = False
    for line in text.lstrip(BOM).splitlines():
        line = line.strip()
        if not seen_header:
            seen_header = bool(_HEADER_RE.match(line))
            continue
        m = _STRICT_RE.match(line) or _LOOSE_RE.match(line)
        if m:
            out.append((m.group(1), m.group(2), unescape(m.group(3))))
    return out


class ParadoxYmlAdapter(Adapter):
    name = "paradox-yml"

    @property
    def cfg(self) -> dict:
        return self.project.adapter_config

    @property
    def source_dir(self) -> Path:
        from ..config import ConfigError

        rel = self.cfg.get("source_dir")
        if not rel:
            raise ConfigError("[output.paradox-yml].source_dir is required")
        return (self.project.root / rel).resolve()

    @property
    def out_dir(self) -> Path:
        return (self.project.root / self.cfg.get("out_dir", "out")).resolve()

    def language_name(self, lang: str) -> str:
        from ..config import ConfigError

        names = {**LANGUAGE_NAMES, **self.cfg.get("languages", {})}
        code = canonical_code(lang)
        name = names.get(lang) or names.get(code) or names.get(code.split("-")[0])
        if not name:
            raise ConfigError(f"paradox-yml: no Paradox language name for {lang!r}; "
                              "add it to [output.paradox-yml].languages")
        return name

    def file_for(self, lang: str) -> Path:
        name = self.language_name(lang)
        return self.out_dir / name / f"{self.cfg.get('out_name', 'lloc')}_l_{name}.yml"

    def _scan(self, lang: str) -> dict[str, tuple[str, str, str]]:
        """{key: (text, version, file stem)} over every file of the language;
        later files (path order) win, as a rough match of the game's own
        override behaviour."""
        suffix = f"_l_{self.language_name(lang)}.yml"
        exclude = self.cfg.get("exclude", [])
        own = self.file_for(lang)
        found: dict[str, tuple[str, str, str]] = {}
        for path in sorted(self.source_dir.rglob("*.yml")):
            if not path.name.lower().endswith(suffix) or path == own \
                    or any(fnmatch.fnmatch(path.name, pat) for pat in exclude):
                continue
            stem = path.name[:-len(suffix)]
            for key, version, text in parse(path.read_text(encoding="utf-8", errors="replace")):
                found[key] = (text, version, stem)
        return found

    # --- export ----------------------------------------------------------

    def export(self, locales: dict[str, dict[str, dict]],
               langs: list[str] | None = None) -> list[str]:
        versions = {k: v[1] for k, v in self._scan(self.project.base_language).items()}
        written = []
        for lang in (self.project.languages if langs is None else langs):
            entries = locales.get(lang, {})
            name = self.language_name(lang)
            lines = [f"l_{name}:"]
            for key in sorted(self.project.strings):
                e = entries.get(key)
                if e and e.get("text") and e.get("model") != "imported":
                    lines.append(f' {key}:{versions.get(key, "0")} "{escape(e["text"])}"')
            path = self.file_for(lang)
            if len(lines) == 1:
                path.unlink(missing_ok=True)  # nothing of ours left to ship
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(path, BOM + "\n".join(lines) + "\n")
            written.append(str(path))
        return written

    # --- import ----------------------------------------------------------

    def _untranslated(self, key: str, text: str, lang: str) -> bool:
        base = self.project.strings.get(key)
        if base is None:
            return False
        expected = expected_scripts(lang)
        if "LATIN" not in expected:
            return script_error(text, expected, style=STYLE) is not None
        return text == base.text and len(strip_placeholders(text, STYLE).split()) >= 3

    def import_(self) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for lang in self.project.languages:
            found = {key: text for key, (text, _, _) in self._scan(lang).items()
                     if not self._untranslated(key, text, lang)}
            if found:
                result[lang] = found
        return result

    # --- bootstrap -------------------------------------------------------

    def _translatable(self) -> tuple[dict[str, tuple[str, str, str]], int]:
        found = self._scan(self.project.base_language)
        keep = {k: v for k, v in found.items()
                if any(c.isalpha() for c in strip_placeholders(v[0], STYLE))
                and not is_name(k, v[0], v[2])}
        return keep, len(found) - len(keep)

    def bootstrap_source(self) -> tuple[dict[str, str], int]:
        keep, skipped = self._translatable()
        return {k: v[0] for k, v in keep.items()}, skipped

    def bootstrap_meta(self) -> dict[str, dict[str, str]]:
        return {k: {"context": f"file: {stem}", "type": string_type(k)}
                for k, (_, _, stem) in self._translatable()[0].items()}

    # --- check -----------------------------------------------------------

    def check(self, text: str) -> list[str]:
        return []  # quotes and line breaks are escaped on export
