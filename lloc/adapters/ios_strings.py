"""ios adapter: <lang>.lproj/Localizable.strings.

Xcode conventions: per-language `<code>.lproj` directories with BCP-47 codes
as-is (`es-419.lproj`, `pt-BR.lproj`, `zh-Hant.lproj`); the base language goes
to `<base>.lproj` (e.g. `en.lproj`) — the development-language lproj.
`Base.lproj` is a storyboard/xib mechanism, not a strings home, so it is not
used. Legacy Android codes in the config (in/iw) are written as their BCP-47
equivalents (id/he), see langnames.canonical_code.

Format: UTF-8 `"key" = "value";` pairs (modern Xcode reads UTF-8; UTF-16 is
a legacy default), `\\" \\n \\t \\\\` escapes, `/* ... */` comments. The
string's `context` is emitted as a comment above each managed entry.

Unmanaged discipline mirrors android-xml ("skip and keep"): pairs whose key
is not in strings.toml survive export unchanged, together with the comment
directly above them; comments above managed pairs describe regenerated
content and are dropped. import_ returns every pair; unknown keys are
filtered by the import command.
"""
from __future__ import annotations

from pathlib import Path

from ..fsutil import atomic_write_text
from ..langnames import canonical_code
from .base import Adapter

_ESCAPES = [("\\", "\\\\"), ('"', '\\"'), ("\n", "\\n"), ("\t", "\\t"), ("\r", "\\r")]
_UNESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}


def escape(text: str) -> str:
    for a, b in _ESCAPES:
        text = text.replace(a, b)
    return text


def _read_quoted(text: str, i: int) -> tuple[str, int]:
    """Read a double-quoted string starting at text[i] == '\"'; returns
    (unescaped value, index past the closing quote)."""
    out: list[str] = []
    i += 1
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text):
            nxt = text[i + 1]
            out.append(_UNESCAPES.get(nxt, "\\" + nxt))
            i += 2
            continue
        if c == '"':
            return "".join(out), i + 1
        out.append(c)
        i += 1
    raise ValueError("unterminated string literal")


def _skip_ws(text: str, i: int) -> int:
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    return i


def parse_strings(text: str) -> list[tuple]:
    """Parse a .strings document into a list of ("comment", raw) and
    ("pair", key, value) items, in document order."""
    items: list[tuple] = []
    i, n = 0, len(text)
    while i < n:
        i = _skip_ws(text, i)
        if i >= n:
            break
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            if j == -1:
                raise ValueError("unterminated /* comment")
            items.append(("comment", text[i:j + 2]))
            i = j + 2
            continue
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j == -1 else j
            items.append(("comment", text[i:j]))
            i = j
            continue
        if text[i] == '"':
            key, i = _read_quoted(text, i)
            i = _skip_ws(text, i)
            if i >= n or text[i] != "=":
                raise ValueError(f"expected '=' after key {key!r}")
            i = _skip_ws(text, i + 1)
            if i >= n or text[i] != '"':
                raise ValueError(f"expected a quoted value for key {key!r}")
            value, i = _read_quoted(text, i)
            i = _skip_ws(text, i)
            if i >= n or text[i] != ";":
                raise ValueError(f"missing ';' after entry {key!r}")
            i += 1
            items.append(("pair", key, value))
            continue
        raise ValueError(f"unexpected character {text[i]!r} at offset {i}")
    return items


class IosStringsAdapter(Adapter):
    name = "ios"

    @property
    def lproj_dir(self) -> Path:
        rel = self.project.adapter_config.get("lproj_dir", ".")
        return (self.project.root / rel).resolve()

    @property
    def filename(self) -> str:
        return str(self.project.adapter_config.get("filename", "Localizable.strings"))

    def dir_for(self, lang: str) -> str:
        return f"{canonical_code(lang)}.lproj"

    def _path(self, lang: str) -> Path:
        return self.lproj_dir / self.dir_for(lang) / self.filename

    # --- export ----------------------------------------------------------

    def _preserved_blocks(self, path: Path) -> list[str]:
        """Unmanaged pairs (keys not in strings.toml) from an existing file,
        each with the comment directly above it; trailing comments kept."""
        if not path.is_file():
            return []
        items = parse_strings(path.read_text(encoding="utf-8"))
        blocks: list[str] = []
        pending: list[str] = []
        for item in items:
            if item[0] == "comment":
                pending.append(item[1])
                continue
            _, key, value = item
            if key in self.project.strings:
                pending = []  # comment described regenerated content
                continue
            blocks.extend(pending)
            blocks.append(f'"{escape(key)}" = "{escape(value)}";')
            pending = []
        blocks.extend(pending)  # trailing comments after the last entry
        return blocks

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
            path = self._path(lang)
            preserved = self._preserved_blocks(path)
            lines: list[str] = []
            for key, se in self.project.strings.items():
                e = entries.get(key)
                if not (e and e.get("text")):
                    continue
                if se.context:
                    lines.append(f"/* {se.context.replace('*/', '* /')} */")
                lines.append(f'"{escape(key)}" = "{escape(e["text"])}";')
            lines.extend(preserved)
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(path, "\n".join(lines) + "\n")
            written.append(str(path))
        return written

    # --- import ----------------------------------------------------------

    def import_(self) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for lang in self.project.languages:
            path = self._path(lang)
            if not path.is_file():
                continue
            strings: dict[str, str] = {}
            for item in parse_strings(path.read_text(encoding="utf-8")):
                if item[0] == "pair":
                    strings[item[1]] = item[2]
            result[lang] = strings
        return result

    # --- bootstrap -------------------------------------------------------

    def bootstrap_source(self) -> tuple[dict[str, str], int]:
        from ..config import ConfigError

        path = self._path(self.project.base_language)
        if not path.is_file():
            raise ConfigError(f"bootstrap: base file not found: {path}")
        entries: dict[str, str] = {}
        for item in parse_strings(path.read_text(encoding="utf-8")):
            if item[0] == "pair":
                entries[item[1]] = item[2]
        return entries, 0

    # --- check -----------------------------------------------------------

    def check(self, text: str) -> list[str]:
        """Serialize + parse roundtrip: guards the escaper/parser pair."""
        try:
            items = parse_strings(f'"k" = "{escape(text)}";')
        except ValueError as e:
            return [f"not representable in .strings: {e}"]
        if items != [("pair", "k", text)]:
            return [".strings escaping roundtrip failed"]
        return []
