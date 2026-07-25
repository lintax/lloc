"""`import` (full mode): existing platform translations -> locales/ + lock. SPEC.md §5, §9."""
import io
import json
import unittest
from contextlib import redirect_stdout

from lloc.__main__ import main
from lloc.hashes import text_hash
from lloc.store import base_hashes, classify, load_lock, load_locale
from lloc.config import load_project

from .helpers import L10N_TOML, make_project, make_tmpdir

RU_XML = """<resources>
    <string name="menu_search">Поиск</string>
    <string name="playback_error">Не удалось открыть поток %1$s. Проверьте подключение.</string>
    <string name="app_key" translatable="false">com.example</string>
    <string name="legacy_key">Больше не существует в strings.toml</string>
    <string name="items_count"> </string>
</resources>
"""

IW_XML = """<resources>
    <string name="menu_search">חיפוש</string>
</resources>
"""


class TestImportTranslations(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmpdir(self)
        toml = L10N_TOML.replace('languages = ["ru", "de"]',
                                 'languages = ["ru", "de", "he"]')
        self.project = make_project(self.tmp, l10n=toml)
        (self.tmp / "res" / "values-b+ru").mkdir(parents=True)
        (self.tmp / "res" / "values-b+ru" / "strings.xml").write_text(RU_XML, encoding="utf-8")
        (self.tmp / "res" / "values-b+iw").mkdir(parents=True)
        (self.tmp / "res" / "values-b+iw" / "strings.xml").write_text(IW_XML, encoding="utf-8")

    def _run(self, *extra: str) -> tuple[int, str]:
        out = io.StringIO()
        with redirect_stdout(out):
            rc = main(["--dir", str(self.tmp), "import", *extra])
        return rc, out.getvalue()

    def test_import_basic(self):
        rc, out = self._run()
        self.assertEqual(rc, 0)
        ru = load_locale(self.tmp, "ru")
        self.assertEqual(ru["menu_search"], {"text": "Поиск", "src": text_hash("Search")})
        self.assertIn("playback_error", ru)
        # unknown, non-translatable and empty entries are skipped
        self.assertNotIn("legacy_key", ru)
        self.assertNotIn("app_key", ru)
        self.assertNotIn("items_count", ru)
        self.assertIn("skipped 2", out)  # legacy_key + empty; translatable="false" is invisible to the adapter
        # imported entries are ok, not stale
        st = classify(self.project, "ru")
        self.assertEqual(sorted(st.ok), ["menu_search", "playback_error"])
        self.assertEqual(st.orphan, [])

    def test_legacy_dir_maps_to_bcp47(self):
        rc, _ = self._run()
        self.assertEqual(rc, 0)
        he = load_locale(self.tmp, "he")
        self.assertEqual(he["menu_search"]["text"], "חיפוש")

    def test_lock_written(self):
        self._run()
        self.assertEqual(load_lock(self.tmp), base_hashes(self.project))

    def test_idempotent(self):
        self._run()
        first = (self.tmp / "locales" / "ru.json").read_text(encoding="utf-8")
        rc, out = self._run()
        self.assertEqual(rc, 0)
        self.assertIn("import: 0 imported", out)
        self.assertEqual(first, (self.tmp / "locales" / "ru.json").read_text(encoding="utf-8"))

    def test_existing_entries_kept_without_overwrite(self):
        self._run()
        p = self.tmp / "locales" / "ru.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        data["strings"]["menu_search"]["text"] = "Ручная правка"
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        self._run()
        self.assertEqual(load_locale(self.tmp, "ru")["menu_search"]["text"], "Ручная правка")
        rc, _ = self._run("--overwrite")
        self.assertEqual(rc, 0)
        self.assertEqual(load_locale(self.tmp, "ru")["menu_search"]["text"], "Поиск")

    def test_assume_stale(self):
        rc, _ = self._run("--assume-stale")
        self.assertEqual(rc, 0)
        st = classify(self.project, "ru")
        self.assertEqual(st.ok, [])
        self.assertEqual(sorted(st.stale), ["menu_search", "playback_error"])

    def test_lang_filter(self):
        rc, _ = self._run("--lang", "he")
        self.assertEqual(rc, 0)
        self.assertTrue((self.tmp / "locales" / "he.json").is_file())
        self.assertFalse((self.tmp / "locales" / "ru.json").is_file())

    def test_no_platform_files(self):
        tmp2 = make_tmpdir(self)
        make_project(tmp2)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = main(["--dir", str(tmp2), "import"])
        self.assertEqual(rc, 0)
        self.assertIn("no platform file for: ru, de", out.getvalue())

    def test_import_then_build_roundtrip(self):
        self._run()
        out = io.StringIO()
        with redirect_stdout(out):
            rc = main(["--dir", str(self.tmp), "build"])
        self.assertEqual(rc, 0)
        ru = (self.tmp / "res" / "values-b+ru" / "strings.xml").read_text(encoding="utf-8")
        self.assertIn('<string name="menu_search">Поиск</string>', ru)
        # special key survives build, legacy stray string is gone
        self.assertIn('translatable="false"', ru)
        self.assertNotIn("legacy_key", ru)


if __name__ == "__main__":
    unittest.main()
