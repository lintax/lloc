"""Post-translation validation. SPEC.md §7.

A translation identical to the source is NOT an error: in closely related
languages many strings legitimately coincide ("Start", "Status", "OK").
"""
from __future__ import annotations

import re
from collections import Counter

# printf-style placeholders (%s, %d, %1$s, %.2f, %%) and escape sequences.
PLACEHOLDER_RE = re.compile(r"%(?:\d+\$)?(?:\.\d+)?[sdfxXeg]|%%")
ESCAPE_RE = re.compile(r"\n|\t")

MIN_LENGTH_BASE = 10  # ratio check uses max(len(base), this) to avoid noise on tiny strings


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

    limit = max(len(base), MIN_LENGTH_BASE) * length_ratio
    if len(translated) > limit:
        errors.append(
            f"translation is too long: {len(translated)} chars vs {len(base)} in source; "
            "keep it concise, this is UI text"
        )
    return errors
