"""ios adapter (.strings / <lang>.lproj). SPEC.md §8.2."""
import unittest

from lloc.adapters import get_adapter
from lloc.adapters.ios_strings import escape, parse_strings
from lloc.hashes import text_hash
from lloc.validate import check_translation

from .helpers import L10N_TOML, STRINGS_TOML, make_project, make_tmpdir

IOS_TOML = L10N_TOML.replace(
    'adapter = "android-xml"', 'adapter = "ios"').replace(
    '[output.android-xml]\nres_dir = "res"\nlegacy_codes = { id = "in", he = "iw" }',
    '[output.ios]\nlproj_dir = "Resources"')


class TestParser(unittest.TestCase):
    def test_roundtrip_escapes(self):
        for raw in ['Say "hi"', "line1\nline2\ttab", "back\\slash", "plain"]:
            items = parse_strings(f'"k" = "{escape(raw)}";')
            self.assertEqual(items, [("pair", "k", raw)], raw)

    def test_comments_and_pairs(self):
        doc = '/* block */\n"a" = "1";\n// line\n"b" = "2";\n'
        items = parse_strings(doc)
        self.assertEqual(items, [("comment", "/* block */"), ("pair", "a", "1"),
                                 ("comment", "// line"), ("pair", "b", "2")])

    def test_errors(self):
        for bad in ['"a" = "1"', '"a" "1";', '"a" = 1;', '/* oops', '"unterminated']:
            with self.assertRaises(ValueError, msg=bad):
                parse_strings(bad)


class TestIosAdapter(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmpdir(self)
        toml = IOS_TOML.replace('languages = ["ru", "de"]',
                                'languages = ["ru", "es-419", "pt-BR", "zh-Hant", "id", "he"]')
        self.project = make_project(self.tmp, l10n=toml)
        self.adapter = get_adapter(self.project)

    def _entry(self, text):
        return {"text": text, "src": text_hash(text)}

    def test_lproj_naming_bcp47_and_legacy(self):
        self.assertEqual(self.adapter.dir_for("es-419"), "es-419.lproj")
        self.assertEqual(self.adapter.dir_for("pt-BR"), "pt-BR.lproj")
        self.assertEqual(self.adapter.dir_for("zh-Hant"), "zh-Hant.lproj")
        # legacy Android codes are converted, never emitted as in/iw
        self.assertEqual(self.adapter.dir_for("in"), "id.lproj")
        self.assertEqual(self.adapter.dir_for("iw"), "he.lproj")

    def test_export_import_roundtrip(self):
        locales = {"ru": {
            "menu_search": self._entry("Поиск"),
            "playback_error": self._entry('Не удалось открыть "поток" %1$s.\nПроверьте.'),
        }}
        self.adapter.export(locales)
        f = self.tmp / "Resources" / "ru.lproj" / "Localizable.strings"
        content = f.read_text(encoding="utf-8")
        self.assertIn('"menu_search" = "Поиск";', content)
        self.assertIn("/* Main menu item */", content)  # context as comment
        imported = self.adapter.import_()
        self.assertEqual(imported["ru"]["playback_error"],
                         locales["ru"]["playback_error"]["text"])

    def test_base_language_lproj(self):
        self.adapter.export({})
        base = self.tmp / "Resources" / "en.lproj" / "Localizable.strings"
        self.assertIn('"menu_search" = "Search";', base.read_text(encoding="utf-8"))
        self.assertFalse((self.tmp / "Resources" / "Base.lproj").exists())

    def test_unmanaged_pairs_preserved_with_comment(self):
        d = self.tmp / "Resources" / "ru.lproj"
        d.mkdir(parents=True)
        (d / "Localizable.strings").write_text(
            '/* regenerated */\n"menu_search" = "STALE";\n'
            '/* keep me */\n"special_key" = "untouched";\n',
            encoding="utf-8")
        self.adapter.export({"ru": {"menu_search": self._entry("Поиск")}})
        content = (d / "Localizable.strings").read_text(encoding="utf-8")
        self.assertIn('"special_key" = "untouched";', content)
        self.assertIn("/* keep me */", content)
        self.assertNotIn("STALE", content)
        self.assertNotIn("/* regenerated */", content)  # comment of a managed pair

    def test_bootstrap_source(self):
        d = self.tmp / "Resources" / "en.lproj"
        d.mkdir(parents=True)
        (d / "Localizable.strings").write_text('"hello" = "Hello";\n', encoding="utf-8")
        entries, skipped = self.adapter.bootstrap_source()
        self.assertEqual(entries, {"hello": "Hello"})
        self.assertEqual(skipped, 0)

    def test_check_roundtrip(self):
        self.assertEqual(self.adapter.check('tricky "quote" \n %1$@'), [])

    def test_placeholder_at_validated(self):
        errs = check_translation("Open %1$@ now", "Открыть")
        self.assertTrue(any("placeholders mismatch" in e for e in errs))
        self.assertEqual(check_translation("Open %1$@ now", "Открыть %1$@"), [])


IOS_TOML_DEFAULTS = IOS_TOML  # reuse for CLI-level smoke


class TestIosCliRoundtrip(unittest.TestCase):
    def test_import_command_via_cli(self):
        import io
        from contextlib import redirect_stdout
        from lloc.__main__ import main
        from lloc.store import load_locale

        tmp = make_tmpdir(self)
        make_project(tmp, l10n=IOS_TOML_DEFAULTS)
        d = tmp / "Resources" / "ru.lproj"
        d.mkdir(parents=True)
        (d / "Localizable.strings").write_text(
            '"menu_search" = "Поиск";\n"unknown_key" = "skip me";\n', encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            rc = main(["--dir", str(tmp), "import"])
        self.assertEqual(rc, 0)
        ru = load_locale(tmp, "ru")
        self.assertEqual(ru["menu_search"]["text"], "Поиск")
        self.assertNotIn("unknown_key", ru)


if __name__ == "__main__":
    unittest.main()
