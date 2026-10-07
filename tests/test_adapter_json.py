"""json adapter (flat i18next-style, file per language)."""
import json
import unittest

from lloc.adapters import get_adapter
from lloc.hashes import text_hash

from .helpers import L10N_TOML, make_project, make_tmpdir

JSON_TOML = L10N_TOML.replace(
    'adapter = "android-xml"', 'adapter = "json"').replace(
    '[output.android-xml]\nres_dir = "res"\nlegacy_codes = { id = "in", he = "iw" }',
    '[output.json]\ndir = "i18n"')


class TestJsonAdapter(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmpdir(self)
        toml = JSON_TOML.replace('languages = ["ru", "de"]',
                                 'languages = ["ru", "es-419", "in", "iw"]')
        self.project = make_project(self.tmp, l10n=toml)
        self.adapter = get_adapter(self.project)

    def _entry(self, text):
        return {"text": text, "src": text_hash(text)}

    def test_file_naming_bcp47_and_legacy(self):
        self.assertEqual(self.adapter.file_for("es-419").name, "es-419.json")
        # legacy Android codes are converted to BCP-47 file names
        self.assertEqual(self.adapter.file_for("in").name, "id.json")
        self.assertEqual(self.adapter.file_for("iw").name, "he.json")

    def test_export_flat_sorted_and_base(self):
        self.adapter.export({"ru": {
            "menu_search": self._entry("Поиск"),
            "items_count": self._entry("Элементов: %d"),
        }})
        ru = json.loads((self.tmp / "i18n" / "ru.json").read_text(encoding="utf-8"))
        self.assertEqual(ru, {"items_count": "Элементов: %d", "menu_search": "Поиск"})
        raw = (self.tmp / "i18n" / "ru.json").read_text(encoding="utf-8")
        self.assertLess(raw.index("items_count"), raw.index("menu_search"))  # sorted
        self.assertIn("Поиск", raw)  # ensure_ascii=False
        base = json.loads((self.tmp / "i18n" / "en.json").read_text(encoding="utf-8"))
        self.assertEqual(base["menu_search"], "Search")
        # untranslated keys are omitted, not emitted empty
        self.assertNotIn("open_error", ru)

    def test_export_import_roundtrip(self):
        locales = {"ru": {k: self._entry(f"[ru]{e.text}")
                          for k, e in self.project.strings.items()}}
        self.adapter.export(locales)
        imported = self.adapter.import_()
        for key in self.project.strings:
            self.assertEqual(imported["ru"][key], locales["ru"][key]["text"])

    def test_unmanaged_keys_preserved(self):
        d = self.tmp / "i18n"
        d.mkdir()
        (d / "ru.json").write_text(json.dumps({
            "menu_search": "STALE",
            "web_only_key": "keep me",
        }), encoding="utf-8")
        self.adapter.export({"ru": {"menu_search": self._entry("Поиск")}})
        ru = json.loads((d / "ru.json").read_text(encoding="utf-8"))
        self.assertEqual(ru["web_only_key"], "keep me")
        self.assertEqual(ru["menu_search"], "Поиск")

    def test_export_idempotent(self):
        locales = {"ru": {"menu_search": self._entry("Поиск")}}
        self.adapter.export(locales)
        first = (self.tmp / "i18n" / "ru.json").read_text(encoding="utf-8")
        self.adapter.export(locales)
        self.assertEqual(first, (self.tmp / "i18n" / "ru.json").read_text(encoding="utf-8"))

    def test_bootstrap_source(self):
        d = self.tmp / "i18n"
        d.mkdir()
        (d / "en.json").write_text(json.dumps({"hello": "Hello", "count": 5}),
                                   encoding="utf-8")
        entries, skipped = self.adapter.bootstrap_source()
        self.assertEqual(entries, {"hello": "Hello"})
        self.assertEqual(skipped, 1)  # non-string value

    def test_corrupt_existing_is_config_error(self):
        from lloc.config import ConfigError
        d = self.tmp / "i18n"
        d.mkdir()
        (d / "ru.json").write_text("{broken", encoding="utf-8")
        with self.assertRaises(ConfigError):
            self.adapter.export({})


if __name__ == "__main__":
    unittest.main()
