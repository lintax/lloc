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

# full-code overrides where plain composition reads unnaturally
_FULL = {
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
