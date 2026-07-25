"""Post-translation validation. SPEC.md §7.

A translation identical to the source is NOT an error: in closely related
languages many strings legitimately coincide ("Start", "Status", "OK").
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

# printf-style placeholders (%s, %d, %1$s, %.2f, %%) incl. iOS %@ / %1$@,
# and escape sequences.
PLACEHOLDER_RE = re.compile(r"%(?:\d+\$)?(?:\.\d+)?[sdfxXeg@]|%%")
ESCAPE_RE = re.compile(r"\n|\t")

MIN_LENGTH_BASE = 10  # ratio check uses max(len(base), this) to avoid noise on tiny strings
# Absolute floor for the length limit: short bases in long-script languages
# (Devanagari, Tamil, Sinhala...) legitimately blow past ratio*len — e.g.
# "Exhale — 8s" (11 chars) in Nepali. Anything under this many chars is
# never "too long".
MIN_LENGTH_LIMIT = 48


def placeholders(text: str) -> Counter:
    return Counter(PLACEHOLDER_RE.findall(text)) + Counter(ESCAPE_RE.findall(text))


def check_translation(base: str, translated: str, length_ratio: float = 3.0) -> list[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    errors: list[str] = []
    if not isinstance(translated, str) or not translated.strip():
        return ["translation is empty"]

    got, want = placeholders(translated), placeholders(base)
    if got != want:
        def fmt(c: Counter) -> str:
            return ", ".join(sorted(c.elements())) or "(none)"
        errors.append(
            f"placeholders mismatch: source has [{fmt(want)}], translation has [{fmt(got)}]; "
            "keep every placeholder exactly as in the source"
        )

    limit = max(max(len(base), MIN_LENGTH_BASE) * length_ratio, MIN_LENGTH_LIMIT)
    if len(translated) > limit:
        errors.append(
            f"translation is too long: {len(translated)} chars vs {len(base)} in source; "
            "keep it concise, this is UI text"
        )
    return errors


# --- script-mix heuristic (verify warning, no LLM) -----------------------

DEFAULT_LATIN_TOKEN_MAX = 5  # latin tokens this short pass anywhere ("OK", "EPG", "min")

_LETTER_RUN_RE = re.compile(r"[^\W\d_]+")


def _char_script(ch: str) -> str | None:
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:  # unnamed characters carry no script signal
        return None


def script_error(text: str, expected: frozenset[str],
                 latin_token_max: int = DEFAULT_LATIN_TOKEN_MAX,
                 allow: frozenset[str] = frozenset()) -> str | None:
    """Inline candidate validation (translate): the DOMINANT script of the
    translation must be the language's expected script — both cloud models
    happily write Serbian in Latin unless rejected. Placeholders are
    stripped; a latin-dominant text is still fine when all its latin content
    passes the usual allowances (short tokens, brands) — "OK" or "%1$d h"
    must not be rejected for a Cyrillic language."""
    counts: dict[str, int] = {}
    for ch in PLACEHOLDER_RE.sub(" ", text):
        s = _char_script(ch) if ch.isalpha() else None
        if s:
            counts[s] = counts.get(s, 0) + 1
    if not counts:
        return None  # digits/punctuation only
    dominant = max(counts, key=counts.get)
    if dominant in expected:
        return None
    if dominant == "LATIN" and not script_mix(text, expected, latin_token_max, allow):
        return None  # latin-only but within allowances (brands, "OK", units)
    want = "/".join(sorted(expected)).title()
    return (f"wrong script: the translation is mostly {dominant.title()}, "
            f"but this language must be written in {want} script — "
            f"rewrite it in {want}")


def script_mix(text: str, expected: frozenset[str],
               latin_token_max: int = DEFAULT_LATIN_TOKEN_MAX,
               allow: frozenset[str] = frozenset()) -> list[str]:
    """Warnings for letters of a foreign Unicode script in a translation
    (live example: Burmese letters inside Armenian text).

    Latin is special-cased for non-Latin languages: digits, punctuation and
    placeholders carry no script; short latin tokens (<= latin_token_max) and
    tokens from `allow` (casefolded; brands, glossary terms) are fine —
    anything longer is reported (usually untranslated English).

    Placeholders are stripped before scanning: otherwise the `d` of `%1$d`
    glues to an adjacent CJK/Cyrillic run ("%1$d個まで", "%1$dч") and
    false-flags perfectly fine strings.

    Spaceless scripts (CJK/Hangul/kana) legitimately glue latin tokens to
    native particles without a boundary ("Play를", "1000Hzのトーン") — for
    those languages a mixed run is split into script segments and the latin
    rules apply per segment. Alphabetic languages keep strict mixed-run
    flagging: a latin letter inside a Cyrillic word is a defect."""
    spaceless = bool(expected & {"CJK", "HANGUL", "HIRAGANA", "KATAKANA"})
    warnings: list[str] = []
    for m in _LETTER_RUN_RE.finditer(PLACEHOLDER_RE.sub(" ", text)):
        run = m.group(0)
        scripts = {s for s in (_char_script(c) for c in run) if s}
        foreign = scripts - expected
        if not foreign:
            continue
        segments = _script_segments(run) if spaceless and len(scripts) > 1 else [run]
        for seg in segments:
            seg_scripts = {s for s in (_char_script(c) for c in seg) if s}
            if not seg_scripts - expected:
                continue
            if seg_scripts == {"LATIN"} and "LATIN" not in expected:
                if len(seg) <= latin_token_max or seg.casefold() in allow:
                    continue
                warnings.append(f'untranslated latin token "{seg}"')
                continue
            warnings.append(
                f'foreign script {"+".join(sorted(seg_scripts - expected))} in "{seg}"')
    return warnings


def _script_segments(run: str) -> list[str]:
    """Split a letter run into maximal same-script segments."""
    segments: list[str] = []
    current = ""
    current_script = None
    for ch in run:
        s = _char_script(ch)
        if current and s != current_script:
            segments.append(current)
            current = ""
        current += ch
        current_script = s
    if current:
        segments.append(current)
    return segments
