"""Manual fixes: `set` command, protection of manual entries, few-shot priority."""
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

from lloc.__main__ import main
from lloc.engine import plan, run_translate
from lloc.hashes import text_hash
from lloc.fsutil import utc_now_iso
from lloc.prompts import select_examples
from lloc.store import load_locale, save_locale

from .helpers import MockTransport, make_project, make_tmpdir


def run_cli(*argv):
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(out):
        rc = main(list(argv))
    return rc, out.getvalue()


class TestSetCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmpdir(self)
        self.project = make_project(self.tmp)

    def test_set_writes_manual_entry_and_builds(self):
        rc, out = run_cli("--dir", str(self.tmp), "set", "menu_search",
                          "Найти", "--lang", "ru")
        self.assertEqual(rc, 0)
        entry = load_locale(self.tmp, "ru")["menu_search"]
        self.assertEqual(entry["text"], "Найти")
        self.assertEqual(entry["model"], "manual")
        self.assertEqual(entry["src"], text_hash("Search"))
        self.assertTrue(entry["translated_at"])
        # incremental build: this language's XML written, base untouched
        ru = self.tmp / "res" / "values-b+ru" / "strings.xml"
        self.assertIn('<string name="menu_search">Найти</string>',
                      ru.read_text(encoding="utf-8"))
        self.assertFalse((self.tmp / "res" / "values" / "strings.xml").exists())
        self.assertFalse((self.tmp / "res" / "values-b+de").exists())

    def test_set_validates_placeholders(self):
        rc, out = run_cli("--dir", str(self.tmp), "set", "playback_error",
                          "Потерял плейсхолдер", "--lang", "ru")
        self.assertEqual(rc, 1)
        self.assertIn("placeholders mismatch", out)
        self.assertIn("--force-set", out)
        self.assertEqual(load_locale(self.tmp, "ru"), {})  # nothing written

    def test_force_set_overrides_validation(self):
        rc, out = run_cli("--dir", str(self.tmp), "set", "playback_error",
                          "Без плейсхолдера", "--lang", "ru", "--force-set")
        self.assertEqual(rc, 0)
        self.assertEqual(load_locale(self.tmp, "ru")["playback_error"]["model"], "manual")

    def test_errors(self):
        rc, out = run_cli("--dir", str(self.tmp), "set", "nope", "x", "--lang", "ru")
        self.assertEqual(rc, 2)
        self.assertIn("unknown key", out)
        rc, out = run_cli("--dir", str(self.tmp), "set", "menu_search", "x")
        self.assertEqual(rc, 2)
        self.assertIn("--lang is required", out)
        rc, out = run_cli("--dir", str(self.tmp), "set", "menu_search", "x",
                          "--lang", "ru,de")
        self.assertEqual(rc, 2)
        rc, out = run_cli("--dir", str(self.tmp), "set", "menu_search", "x",
                          "--lang", "xx")
        self.assertEqual(rc, 2)


class TestManualProtection(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmpdir(self)
        self.project = make_project(self.tmp)
        entries = {}
        for i, (key, e) in enumerate(self.project.strings.items()):
            model = "manual" if i == 0 else "gemma4:12b-it-qat"
            entries[key] = {"text": f"[ru]{e.text}", "src": text_hash(e.text),
                            "model": model, "translated_at": utc_now_iso()}
        save_locale(self.tmp, "ru", entries, list(self.project.strings))
        self.manual_key = list(self.project.strings)[0]

    def test_force_means_force_manual_included(self):
        work = plan(self.project, ["ru"], force=True, limit=None)
        self.assertIn(self.manual_key, work["ru"])
        self.assertEqual(len(work["ru"]), 3)  # everything, no exceptions

    def test_retranslate_model_is_honest_fnmatch(self):
        # model patterns naturally do not match "manual"...
        work = plan(self.project, ["ru"], force=False, limit=None,
                    retranslate_model="gemma*")
        self.assertNotIn(self.manual_key, work["ru"])
        self.assertEqual(len(work["ru"]), 2)
        # ...but an explicit request for manual entries matches them
        work = plan(self.project, ["ru"], force=False, limit=None,
                    retranslate_model="manual")
        self.assertEqual(work["ru"], [self.manual_key])
        # and "*" matches everything, manual included — no special cases
        work = plan(self.project, ["ru"], force=False, limit=None,
                    retranslate_model="*")
        self.assertEqual(len(work["ru"]), 3)

    def test_stale_manual_is_retranslated_and_flagged(self):
        # change the base text under the manual fix -> stale -> back to the LLM
        from lloc.config import load_project
        from lloc.engine import manual_stale

        strings = (self.tmp / "strings.toml").read_text(encoding="utf-8")
        (self.tmp / "strings.toml").write_text(
            strings.replace('text = "Search"', 'text = "Find"'), encoding="utf-8")
        project2 = load_project(self.tmp)
        self.assertEqual(manual_stale(project2, ["ru"]), {"ru": ["menu_search"]})
        results = run_translate(project2, ["ru"], transport=MockTransport())
        self.assertEqual(results[0].translated, 1)
        entry = load_locale(self.tmp, "ru")["menu_search"]
        self.assertEqual(entry["text"], "[ru]Find")
        self.assertNotEqual(entry["model"], "manual")  # retranslated by the LLM

    def test_stale_manual_report_line(self):
        from lloc.__main__ import _report_stale_manual
        from lloc.engine import LangResult

        out = io.StringIO()
        with redirect_stdout(out):
            _report_stale_manual({"ru": ["menu_search", "items_count"]},
                                 [LangResult(lang="ru", translated=2)])
        self.assertIn("[ru] 2 manual translation(s) were stale and retranslated: "
                      "menu_search, items_count", out.getvalue())
        # failed keys are not reported as retranslated
        out = io.StringIO()
        with redirect_stdout(out):
            _report_stale_manual({"ru": ["menu_search"]},
                                 [LangResult(lang="ru", failed={"menu_search": ["e"]})])
        self.assertEqual(out.getvalue(), "")


class TestManualFewShotPriority(unittest.TestCase):
    def test_manual_examples_rank_first(self):
        tmp = make_tmpdir(self)
        strings = """\
[search_title]
text = "Search everywhere"

[search_hint]
text = "Search in the list"

[search_button]
text = "Search now"
"""
        project = make_project(tmp, strings=strings)
        locale = {
            "search_title": {"text": "Искать всюду", "src": "x",
                             "model": "gemma4:12b-it-qat"},
            "search_hint": {"text": "Поиск по списку", "src": "x",
                            "model": "manual"},
        }
        # mark the LLM entry as judge-verified — unverified would be excluded
        from lloc.semantic import _pair_hash, save_semantic
        save_semantic(project, "ru", {
            "version": 1, "language": "ru", "entries": {
                "search_title": {"src": _pair_hash("Search everywhere", "Искать всюду"),
                                 "verdict": "ok", "back": ""}}})
        batch = [project.strings["search_button"]]
        examples = select_examples(project, batch, locale, lang="ru")
        # manual outranks verified
        self.assertEqual(examples[0], ("Search in the list", "Поиск по списку"))
        self.assertEqual(examples[1], ("Search everywhere", "Искать всюду"))
        self.assertEqual(len(examples), 2)


class TestSetWorkspaceGuard(unittest.TestCase):
    def test_set_from_workspace_root_requires_dir(self):
        from .test_workspace import make_workspace
        tmp = make_workspace(self)
        rc, out = run_cli("--dir", str(tmp), "set", "channel_open", "x", "--lang", "ru")
        self.assertEqual(rc, 2)
        self.assertIn("use --dir <member>", out)


if __name__ == "__main__":
    unittest.main()
