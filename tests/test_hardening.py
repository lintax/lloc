"""Hardening batch: run lock, atomic writes, model metadata,
--retranslate-model, report, Windows-portability static checks."""
import io
import json
import re
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

from lloc.__main__ import main
from lloc.engine import plan, run_translate
from lloc.fsutil import atomic_write_text, parse_duration, utc_now_iso
from lloc.hashes import text_hash
from lloc.runlock import LOCK_FILE, LockBusy, module_lock
from lloc.store import load_locale, save_locale

from .helpers import MockTransport, make_project, make_tmpdir


def run_cli(*argv):
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(out):
        rc = main(list(argv))
    return rc, out.getvalue()


class TestRunLock(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmpdir(self)
        make_project(self.tmp)

    def test_busy_lock_rejects_mutating_commands(self):
        with module_lock(self.tmp):
            rc, out = run_cli("--dir", str(self.tmp), "build")
            self.assertEqual(rc, 3)
            self.assertIn("another lloc run is active for this module", out)
            # read-only commands are not blocked
            rc, _ = run_cli("--dir", str(self.tmp), "status")
            self.assertEqual(rc, 1)  # pending work, but ran fine

    def test_lock_released_on_exit_no_cleanup_needed(self):
        with module_lock(self.tmp):
            pass
        # the lock file is still on disk — and that must NOT matter
        self.assertTrue((self.tmp / LOCK_FILE).exists())
        with module_lock(self.tmp):
            pass  # re-acquire succeeds without any manual cleanup
        rc, _ = run_cli("--dir", str(self.tmp), "build")
        self.assertEqual(rc, 0)

    def test_second_acquire_raises(self):
        with module_lock(self.tmp):
            with self.assertRaises(LockBusy):
                with module_lock(self.tmp):
                    pass


class TestAtomicWrites(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmpdir(self)

    def test_replaces_existing_and_leaves_no_tmp(self):
        p = self.tmp / "data.json"
        p.write_text("old", encoding="utf-8")
        atomic_write_text(p, "new")
        self.assertEqual(p.read_text(encoding="utf-8"), "new")
        self.assertEqual(list(self.tmp.glob("*.tmp")), [])

    def test_save_locale_is_atomic(self):
        project = make_project(self.tmp)
        entries = {"menu_search": {"text": "Поиск", "src": text_hash("Search")}}
        save_locale(self.tmp, "ru", entries, list(project.strings))
        self.assertEqual(load_locale(self.tmp, "ru")["menu_search"]["text"], "Поиск")
        self.assertEqual(list((self.tmp / "locales").glob("*.tmp")), [])


class TestModelMetadata(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmpdir(self)
        self.project = make_project(self.tmp)

    def test_translate_stamps_model_and_time(self):
        run_translate(self.project, ["ru"], transport=MockTransport())
        entry = load_locale(self.tmp, "ru")["menu_search"]
        self.assertEqual(entry["model"], self.project.ollama.model)
        when = datetime.fromisoformat(entry["translated_at"])
        self.assertLess(abs((datetime.now(timezone.utc) - when).total_seconds()), 60)

    def test_import_stamps_imported(self):
        d = self.tmp / "res" / "values-b+ru"
        d.mkdir(parents=True)
        (d / "strings.xml").write_text(
            '<resources><string name="menu_search">Поиск</string></resources>',
            encoding="utf-8")
        rc, _ = run_cli("--dir", str(self.tmp), "import")
        self.assertEqual(rc, 0)
        entry = load_locale(self.tmp, "ru")["menu_search"]
        self.assertEqual(entry["model"], "imported")
        self.assertTrue(entry["translated_at"])

    def test_retranslate_model_pattern(self):
        # ru fully translated: one entry by an old model, the rest by current
        entries = {}
        for i, (key, e) in enumerate(self.project.strings.items()):
            model = "gemma3:12b" if i == 0 else "gemma4:12b-it-qat"
            entries[key] = {"text": f"[ru]{e.text}", "src": text_hash(e.text),
                            "model": model, "translated_at": utc_now_iso()}
        save_locale(self.tmp, "ru", entries, list(self.project.strings))
        work = plan(self.project, ["ru"], force=False, limit=None,
                    retranslate_model="gemma3*")
        self.assertEqual(work["ru"], [list(self.project.strings)[0]])
        # entries without a model field (pre-metadata) do not match
        work = plan(self.project, ["ru"], force=False, limit=None,
                    retranslate_model="imported")
        self.assertEqual(work["ru"], [])
        # CLI dry-run picks the flag up
        rc, out = run_cli("--dir", str(self.tmp), "translate", "--dry-run",
                          "--lang", "ru", "--retranslate-model", "gemma3*")
        self.assertEqual(rc, 0)
        self.assertIn("1 key", out)


class TestReport(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmpdir(self)
        self.project = make_project(self.tmp)
        old = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(timespec="seconds")
        keys = list(self.project.strings)
        entries = {
            keys[0]: {"text": "свежий", "src": text_hash(self.project.strings[keys[0]].text),
                      "model": "gemma4:12b-it-qat", "translated_at": utc_now_iso()},
            keys[1]: {"text": "старый", "src": text_hash(self.project.strings[keys[1]].text),
                      "model": "gemma4:12b-it-qat", "translated_at": old},
            keys[2]: {"text": "без метки", "src": text_hash(self.project.strings[keys[2]].text)},
        }
        save_locale(self.tmp, "ru", entries, keys)

    def test_default_window_24h(self):
        rc, out = run_cli("--dir", str(self.tmp), "report", "--last")
        self.assertEqual(rc, 0)
        self.assertIn("свежий", out)
        self.assertNotIn("старый", out)
        self.assertNotIn("без метки", out)
        self.assertIn("report: 1 strings in the last 24h", out)

    def test_since_widens_window(self):
        rc, out = run_cli("--dir", str(self.tmp), "report", "--since", "7d")
        self.assertEqual(rc, 0)
        self.assertIn("свежий", out)
        self.assertIn("старый", out)
        self.assertIn("report: 2 strings in the last 7d", out)

    def test_bad_duration_is_config_error(self):
        rc, out = run_cli("--dir", str(self.tmp), "report", "--since", "fortnight")
        self.assertEqual(rc, 2)
        self.assertIn("invalid duration", out)

    def test_parse_duration_units(self):
        self.assertEqual(parse_duration("90m"), timedelta(minutes=90))
        self.assertEqual(parse_duration("24h"), timedelta(hours=24))
        self.assertEqual(parse_duration("7d"), timedelta(days=7))
        self.assertEqual(parse_duration("24"), timedelta(hours=24))


class TestWindowsSmoke(unittest.TestCase):
    """Static portability checks — we have no real Windows box, so at least
    make sure path handling stays on pathlib and never hardcodes '/'."""

    def _sources(self):
        pkg = Path(__file__).resolve().parent.parent / "lloc"
        for p in pkg.rglob("*.py"):
            yield p, p.read_text(encoding="utf-8")

    def test_no_os_path_join_or_os_sep(self):
        for p, src in self._sources():
            self.assertNotIn("os.path.join", src, p.name)
            self.assertNotIn("os.sep", src, p.name)

    def test_no_slash_string_concat_for_paths(self):
        # building paths via `x + "/" + y` breaks nothing on Windows but is a
        # smell; all joins must go through pathlib's `/` operator
        pattern = re.compile(r"""\+\s*["']/["']|["']/["']\s*\+""")
        for p, src in self._sources():
            self.assertIsNone(pattern.search(src), p.name)

    def test_no_slash_inside_path_literals(self):
        # Path("a/b") is legal but mixes separators into literals; require
        # Path("a") / "b" so every join is explicit
        pattern = re.compile(r"""Path\(\s*["'][^"']*/""")
        for p, src in self._sources():
            self.assertIsNone(pattern.search(src), p.name)


if __name__ == "__main__":
    unittest.main()
