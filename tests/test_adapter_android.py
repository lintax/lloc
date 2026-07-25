import unittest

from lloc.adapters import get_adapter
from lloc.adapters.android_xml import escape, unescape
from lloc.hashes import text_hash

from .helpers import L10N_TOML, make_project, make_tmpdir  # noqa: F401


class TestEscaping(unittest.TestCase):
    CASES = [
        ("Don't stop", "Don\\'t stop"),
        ('Say "hi"', 'Say \\"hi\\"'),
        ("a & b < c", "a &amp; b &lt; c"),
        ("line1\nline2\ttab", "line1\\nline2\\ttab"),
        ("@starts_with_at", "\\@starts_with_at"),
        ("100%", "100%"),
    ]

    def test_escape(self):
        for raw, esc in self.CASES:
            self.assertEqual(escape(raw), esc, raw)

    def test_roundtrip(self):
        for raw, _ in self.CASES:
            xml_unescaped = escape(raw).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            self.assertEqual(unescape(xml_unescaped), raw, raw)


class TestAdapter(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmpdir(self)
        toml = L10N_TOML.replace('languages = ["ru", "de"]',
                                 'languages = ["ru", "id", "he", "es-419", "sr-Latn"]')
        self.project = make_project(self.tmp, l10n=toml)
        self.adapter = get_adapter(self.project)

    def test_dir_mapping(self):
        self.assertEqual(self.adapter.dir_for("ru"), "values-b+ru")
        self.assertEqual(self.adapter.dir_for("id"), "values-b+in")     # legacy
        self.assertEqual(self.adapter.dir_for("he"), "values-b+iw")     # legacy
        self.assertEqual(self.adapter.dir_for("es-419"), "values-b+es+419")
        self.assertEqual(self.adapter.dir_for("sr-Latn"), "values-b+sr+Latn")
        for lang in self.project.languages:
            self.assertEqual(self.adapter.lang_for(self.adapter.dir_for(lang)), lang)

    def _locales(self):
        def entry(text):
            return {"text": text, "src": text_hash(text)}
        return {
            "ru": {
                "menu_search": entry("Поиск"),
                "playback_error": entry("Не удалось открыть поток %1$s. Проверьте подключение."),
                "items_count": entry("Элементов: %d"),
            },
            "he": {"menu_search": entry("חיפוש")},
        }

    def test_export_format(self):
        self.adapter.export(self._locales())
        f = self.tmp / "res" / "values-b+ru" / "strings.xml"
        content = f.read_text(encoding="utf-8")
        self.assertFalse(content.startswith("<?xml"))       # no declaration
        self.assertIn('    <string name="menu_search">Поиск</string>', content)
        self.assertTrue(content.startswith("<resources>"))
        # missing translations are skipped, not emitted empty
        he = (self.tmp / "res" / "values-b+iw" / "strings.xml").read_text(encoding="utf-8")
        self.assertNotIn("playback_error", he)

    def test_export_import_roundtrip(self):
        locales = self._locales()
        self.adapter.export(locales)
        imported = self.adapter.import_()
        self.assertEqual(imported["ru"]["playback_error"],
                         locales["ru"]["playback_error"]["text"])
        self.assertEqual(imported["he"]["menu_search"], "חיפוש")

    def test_check_valid(self):
        self.assertEqual(self.adapter.check("Don't & <b>"), [])

    def test_export_writes_base_values(self):
        self.adapter.export(self._locales())
        f = self.tmp / "res" / "values" / "strings.xml"
        content = f.read_text(encoding="utf-8")
        self.assertFalse(content.startswith("<?xml"))
        self.assertIn('    <string name="menu_search">Search</string>', content)
        self.assertIn('<string name="items_count">Items: %d</string>', content)


SPECIAL_XML = """<resources>
    <string name="menu_search">STALE, must be regenerated</string>
    <!-- do not translate -->
    <string name="app_key" translatable="false">com.example.app &amp; co</string>
    <string-array name="modes">
        <!-- keep inner comment -->
        <item>Auto</item>
        <item>Don\\'t</item>
    </string-array>
    <plurals name="days">
        <item quantity="other">%d days</item>
    </plurals>
    <string name="stray_unmanaged">Plain string not in strings.toml</string>
</resources>
"""


class TestSpecialKeys(unittest.TestCase):
    """SPEC §8.1: special keys are skipped on import and kept on build."""

    def setUp(self):
        self.tmp = make_tmpdir(self)
        self.project = make_project(self.tmp)
        self.adapter = get_adapter(self.project)
        for d in ("values", "values-b+ru"):
            (self.tmp / "res" / d).mkdir(parents=True)
            (self.tmp / "res" / d / "strings.xml").write_text(SPECIAL_XML, encoding="utf-8")

    def _export(self):
        entry = {"text": "Поиск", "src": text_hash("Search")}
        self.adapter.export({"ru": {"menu_search": entry}})

    def test_build_preserves_special_keys_with_comments(self):
        self._export()
        for d in ("values", "values-b+ru"):
            content = (self.tmp / "res" / d / "strings.xml").read_text(encoding="utf-8"), d
            self.assertIn('<string name="app_key" translatable="false">'
                          "com.example.app &amp; co</string>", content[0], d)
            self.assertIn("<!-- do not translate -->", content[0], d)   # attached comment
            self.assertIn("<!-- keep inner comment -->", content[0], d)
            self.assertIn("<item>Don\\'t</item>", content[0], d)        # array intact
            self.assertIn('<plurals name="days">', content[0], d)
            # managed strings regenerated, stray plain strings dropped
            self.assertNotIn("STALE", content[0], d)
            self.assertNotIn("stray_unmanaged", content[0], d)
        base = (self.tmp / "res" / "values" / "strings.xml").read_text(encoding="utf-8")
        self.assertIn('<string name="menu_search">Search</string>', base)
        ru = (self.tmp / "res" / "values-b+ru" / "strings.xml").read_text(encoding="utf-8")
        self.assertIn('<string name="menu_search">Поиск</string>', ru)

    def test_build_idempotent(self):
        self._export()
        first = (self.tmp / "res" / "values" / "strings.xml").read_text(encoding="utf-8")
        self._export()
        second = (self.tmp / "res" / "values" / "strings.xml").read_text(encoding="utf-8")
        self.assertEqual(first, second)

    def test_import_skips_special_keys(self):
        imported = self.adapter.import_()
        self.assertEqual(set(imported["ru"]), {"menu_search", "stray_unmanaged"})
        self.assertNotIn("app_key", imported["ru"])                     # translatable=false
        # arrays/plurals contribute nothing
        self.assertEqual(imported["ru"]["menu_search"], "STALE, must be regenerated")


if __name__ == "__main__":
    unittest.main()
