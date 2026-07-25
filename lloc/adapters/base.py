"""Adapter interface. SPEC.md §8."""
from __future__ import annotations

from ..config import Project


class Adapter:
    name: str = ""

    def __init__(self, project: Project):
        self.project = project

    def export(self, locales: dict[str, dict[str, dict]]) -> list[str]:
        """Write platform files from the locale store. Returns list of written paths."""
        raise NotImplementedError

    def import_(self) -> dict[str, dict[str, str]]:
        """Read platform files. Returns {lang: {key: raw_text}}."""
        raise NotImplementedError

    def check(self, text: str) -> list[str]:
        """Format-specific validation of a raw stored text. Returns error list."""
        return []

    def bootstrap_source(self) -> tuple[dict[str, str], int]:
        """Read the base-language platform file for `import --bootstrap-strings`.
        Returns ({key: raw_text} in document order, count of skipped
        non-translatable entries)."""
        raise NotImplementedError
