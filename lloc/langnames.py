"""Human-readable language names for prompts. SPEC.md §6.

LLMs reliably understand "Latin American Spanish", not "es-419": every
{target_language}/{base_language} placeholder gets a spelled-out name.
Resolution: exact full-code match first, then primary subtag + spelled-out
region/script qualifiers, unknown codes pass through as-is (never an error).
Legacy Android codes (in, iw, ji) are treated as their BCP-47 equivalents.
"""
from __future__ import annotations

# legacy / grandfathered codes -> BCP-47
_ALIASES = {"in": "id", "iw": "he", "ji": "yi", "tl": "fil"}

# full-code overrides where plain composition reads unnaturally, PLUS
# script disambiguation for languages with ambiguous real-world practice:
# LLMs default to the "wrong" script easily (live case: both cloud models
# wrote Serbian in Latin). The prompt name pins the CLDR-default script,
# consistent with expected_scripts() below.
_FULL = {
    # sr: user decision (июль 2026) — Latin is the more common practice for
    # this audience, overriding the CLDR default (Cyrillic). sr-Cyrl subtag
    # still available for the opposite choice.
    "sr": "Serbian (Latin script)",
    "bs": "Bosnian (Latin script)",
    "kk": "Kazakh (Cyrillic script)",
    "ky": "Kyrgyz (Cyrillic script)",
    "mn": "Mongolian (Cyrillic script)",
    "uz": "Uzbek (Latin script)",
    "az": "Azerbaijani (Latin script)",
    "pa": "Punjabi (Gurmukhi script)",
    "ms": "Malay (Latin script)",
    "ha": "Hausa (Latin script)",
    "es-419": "Latin American Spanish",
    "pt-BR": "Brazilian Portuguese",
    "pt-PT": "European Portuguese",
    "zh": "Simplified Chinese",
    "zh-Hans": "Simplified Chinese",
    "zh-Hant": "Traditional Chinese (Taiwan/Hong Kong)",
    "zh-HK": "Traditional Chinese (Hong Kong)",
    "zh-TW": "Traditional Chinese (Taiwan)",
    "sr-Latn": "Serbian (Latin script)",
    "jv-Latn": "Javanese (Latin script)",
    "my-Qaag": "Burmese (Zawgyi script)",
    "fr-CA": "Canadian French",
    "en-GB": "British English",
    "en-US": "American English",
}

_PRIMARY = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese",
    "az": "Azerbaijani", "be": "Belarusian", "bg": "Bulgarian", "bn": "Bengali",
    "bs": "Bosnian", "ca": "Catalan", "cs": "Czech", "cy": "Welsh",
    "da": "Danish", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "et": "Estonian", "eu": "Basque", "fa": "Persian (Farsi)",
    "fi": "Finnish", "fil": "Filipino", "fr": "French", "ga": "Irish",
    "gl": "Galician", "gu": "Gujarati", "ha": "Hausa", "he": "Hebrew",
    "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian", "hy": "Armenian",
    "id": "Indonesian", "ig": "Igbo", "is": "Icelandic", "it": "Italian",
    "ja": "Japanese", "jv": "Javanese", "ka": "Georgian", "kk": "Kazakh",
    "km": "Khmer", "kn": "Kannada", "ko": "Korean", "ky": "Kyrgyz",
    "lo": "Lao", "lt": "Lithuanian", "lv": "Latvian", "mk": "Macedonian",
    "ml": "Malayalam", "mn": "Mongolian", "mr": "Marathi", "ms": "Malay",
    "mt": "Maltese", "my": "Burmese", "ne": "Nepali", "nl": "Dutch",
    "no": "Norwegian", "or": "Odia", "pa": "Punjabi", "pl": "Polish",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "rw": "Kinyarwanda",
    "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "so": "Somali",
    "sq": "Albanian", "sr": "Serbian", "sv": "Swedish", "sw": "Swahili",
    "ta": "Tamil", "te": "Telugu", "th": "Thai", "tr": "Turkish",
    "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese",
    "yi": "Yiddish", "yo": "Yoruba", "zu": "Zulu",
}

# spelled-out subtags for generic composition: scripts (4 letters, Title case)
_SCRIPTS = {"Latn": "Latin script", "Cyrl": "Cyrillic script", "Arab": "Arabic script",
            "Hans": "Simplified script", "Hant": "Traditional script"}
# regions (2 letters upper / 3 digits)
_REGIONS = {"419": "Latin America", "BR": "Brazil", "PT": "Portugal", "US": "USA",
            "GB": "UK", "CA": "Canada", "TW": "Taiwan", "HK": "Hong Kong",
            "IN": "India", "MX": "Mexico", "AR": "Argentina", "AU": "Australia"}


# Expected Unicode script(s) per primary subtag — first word of the
# unicodedata.name() of a letter ("LATIN SMALL LETTER A" -> "LATIN").
# Languages not listed default to LATIN. Script subtags (-Latn/-Cyrl/-Arab)
# override. Used by the script-mix heuristic in `verify` (validate.script_mix).
_CYRILLIC = frozenset({"CYRILLIC"})
_ARABIC = frozenset({"ARABIC"})
_DEVANAGARI = frozenset({"DEVANAGARI"})
_SCRIPTS_BY_LANG: dict[str, frozenset[str]] = {
    "ru": _CYRILLIC, "uk": _CYRILLIC, "be": _CYRILLIC, "bg": _CYRILLIC,
    "kk": _CYRILLIC, "ky": _CYRILLIC, "mk": _CYRILLIC, "mn": _CYRILLIC,
    # sr intentionally NOT mapped to Cyrillic: project decision is Latin
    # (default LATIN applies); use "sr-Cyrl" in [app].languages to flip.
    "ar": _ARABIC, "fa": _ARABIC, "ur": _ARABIC,
    "he": frozenset({"HEBREW"}), "yi": frozenset({"HEBREW"}),
    "el": frozenset({"GREEK"}),
    "hy": frozenset({"ARMENIAN"}),
    "ka": frozenset({"GEORGIAN"}),
    "am": frozenset({"ETHIOPIC"}),
    "my": frozenset({"MYANMAR"}),
    "km": frozenset({"KHMER"}),
    "lo": frozenset({"LAO"}),
    "th": frozenset({"THAI"}),
    "si": frozenset({"SINHALA"}),
    "ta": frozenset({"TAMIL"}),
    "te": frozenset({"TELUGU"}),
    "kn": frozenset({"KANNADA"}),
    "ml": frozenset({"MALAYALAM"}),
    "gu": frozenset({"GUJARATI"}),
    "pa": frozenset({"GURMUKHI"}),
    "bn": frozenset({"BENGALI"}), "as": frozenset({"BENGALI"}),
    "hi": _DEVANAGARI, "mr": _DEVANAGARI, "ne": _DEVANAGARI,
    "or": frozenset({"ORIYA"}),
    "ja": frozenset({"CJK", "HIRAGANA", "KATAKANA", "KATAKANA-HIRAGANA",
                     "IDEOGRAPHIC"}),
    "zh": frozenset({"CJK", "IDEOGRAPHIC", "BOPOMOFO"}),
    "ko": frozenset({"HANGUL", "CJK"}),
}
_SCRIPT_SUBTAGS = {"Latn": frozenset({"LATIN"}), "Cyrl": _CYRILLIC,
                   "Arab": _ARABIC,
                   "Hans": _SCRIPTS_BY_LANG["zh"], "Hant": _SCRIPTS_BY_LANG["zh"],
                   "Qaag": _SCRIPTS_BY_LANG["my"]}
_LATIN = frozenset({"LATIN"})


def expected_scripts(code: str) -> frozenset[str]:
    """Unicode scripts a translation into `code` is expected to use.
    Latin-script languages -> {"LATIN"}; script subtags override."""
    parts = canonical_code(code).split("-")
    for p in parts[1:]:
        if p in _SCRIPT_SUBTAGS:
            return _SCRIPT_SUBTAGS[p]
    return _SCRIPTS_BY_LANG.get(parts[0], _LATIN)


def canonical_code(code: str) -> str:
    """BCP-47 form of a possibly legacy code: in -> id, iw -> he, _ -> -.
    Used by adapters whose file names carry plain BCP-47 (ios, json)."""
    parts = code.replace("_", "-").split("-")
    parts[0] = _ALIASES.get(parts[0], parts[0])
    return "-".join(parts)


def language_name(code: str) -> str:
    """Spelled-out English name for a BCP-47 (or legacy Android) code.
    Unknown codes are returned unchanged — never an error."""
    if not code:
        return code
    canonical = canonical_code(code)
    parts = canonical.split("-")
    if canonical in _FULL:
        return _FULL[canonical]
    primary = _PRIMARY.get(parts[0])
    if primary is None:
        return code  # unknown primary subtag: pass through as-is
    quals = [_SCRIPTS.get(p) or _REGIONS.get(p) or p for p in parts[1:]]
    return f"{primary} ({', '.join(quals)})" if quals else primary
