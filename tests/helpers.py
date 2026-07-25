"""Shared test fixtures: minimal project on disk + mock Ollama transport."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from lloc.config import load_project

L10N_TOML = """\
[app]
description = "Demo TV app"
tone = "neutral, concise"
base_language = "en"
languages = ["ru", "de"]

[glossary]
"EPG" = { "*" = "EPG" }

[ollama]
batch_size = 3
max_retries = 1
workers = 1

[output]
adapter = "android-xml"

[output.android-xml]
res_dir = "res"
legacy_codes = { id = "in", he = "iw" }
"""

STRINGS_TOML = """\
[menu_search]
text = "Search"
context = "Main menu item"
type = "button"

[playback_error]
text = "Failed to open stream %1$s. Check your connection."
context = "Error message; %1$s is the channel name"
type = "message"

[items_count]
text = "Items: %d"
type = "message"
"""


def make_project(tmp: Path, l10n: str = L10N_TOML, strings: str = STRINGS_TOML):
    (tmp / "l10n.toml").write_text(l10n, encoding="utf-8")
    (tmp / "strings.toml").write_text(strings, encoding="utf-8")
    return load_project(tmp)


def make_tmpdir(testcase) -> Path:
    d = tempfile.TemporaryDirectory()
    testcase.addCleanup(d.cleanup)
    return Path(d.name)


class MockTransport:
    """Callable transport: answers with a translator function base_text -> text.

    Records every payload for assertions."""

    def __init__(self, translator=None, fail_first=None):
        self.calls: list[dict] = []
        self.translator = translator or (lambda lang, key, text: f"[{lang}]{text}")
        self.fail_first = dict(fail_first or {})  # key -> bad value returned on first call

    def __call__(self, url: str, payload: dict, timeout: int) -> dict:
        self.calls.append(payload)
        keys = payload["format"]["required"]
        lang = self._target_lang(payload)
        out = {}
        for key in keys:
            if key in self.fail_first:
                out[key] = self.fail_first.pop(key)
            else:
                out[key] = self.translator(lang, key, self._base_text(payload, key))
        return {"message": {"content": json.dumps(out, ensure_ascii=False)}}

    @staticmethod
    def _target_lang(payload: dict) -> str:
        # prompt contains "to <lang>." per translate.txt template
        content = payload["messages"][0]["content"]
        for line in content.splitlines():
            if " to " in line and line.startswith("TASK:"):
                return line.rsplit(" to ", 1)[1].rstrip(".")
        return "xx"

    @staticmethod
    def _base_text(payload: dict, key: str) -> str:
        content = payload["messages"][-1]["content"]
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if line == f"key: {key}" and i + 1 < len(lines):
                return lines[i + 1].removeprefix("text: ")
        return key
