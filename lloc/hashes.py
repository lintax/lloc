"""Canonical hash of a base string. See SPEC.md §4.3."""
import hashlib


def text_hash(text: str) -> str:
    """sha256 of canonical representation, truncated to 16 hex chars."""
    canon = "t\x00" + text
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]
