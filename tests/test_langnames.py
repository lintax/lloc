"""Spelled-out language names for prompts (langnames.py)."""
import unittest

from lloc.langnames import language_name
from lloc.prompts import build_translate_prompt

from .helpers import L10N_TOML, make_project, make_tmpdir


class TestLanguageName(unittest.TestCase):
    def test_plain_codes(self):
        self.assertEqual(language_name("mk"), "Macedonian")
        self.assertEqual(language_name("ha"), "Hausa (Latin script)")
        self.assertEqual(language_name("zu"), "Zulu")
        self.assertEqual(language_name("lo"), "Lao")
        self.assertEqual(language_name("en"), "English")

    def test_regional_variants_differ_from_base(self):
        self.assertEqual(language_name("es"), "Spanish")
        self.assertEqual(language_name("es-419"), "Latin American Spanish")
        self.assertNotEqual(language_name("es"), language_name("es-419"))
        self.assertEqual(language_name("pt"), "Portuguese")
        self.assertEqual(language_name("pt-BR"), "Brazilian Portuguese")

    def test_chinese_scripts(self):
        self.assertEqual(language_name("zh"), "Simplified Chinese")
        self.assertEqual(language_name("zh-Hant"), "Traditional Chinese (Taiwan/Hong Kong)")

    def test_script_composition(self):
        self.assertEqual(language_name("sr-Latn"), "Serbian (Latin script)")
        self.assertEqual(language_name("uz-Cyrl"), "Uzbek (Cyrillic script)")

    def test_ambiguous_script_languages_pinned(self):
        # live case: both cloud models wrote Serbian in Latin — the prompt
        # name must pin the CLDR-default script for ambiguous-practice codes
        self.assertEqual(language_name("sr"), "Serbian (Latin script)")  # решение пользователя
        self.assertEqual(language_name("kk"), "Kazakh (Cyrillic script)")
        self.assertEqual(language_name("mn"), "Mongolian (Cyrillic script)")
        self.assertEqual(language_name("uz"), "Uzbek (Latin script)")
        self.assertEqual(language_name("az"), "Azerbaijani (Latin script)")
        self.assertEqual(language_name("pa"), "Punjabi (Gurmukhi script)")
        self.assertEqual(language_name("bs"), "Bosnian (Latin script)")
        self.assertEqual(language_name("ms"), "Malay (Latin script)")
        # unambiguous languages stay plain
        self.assertEqual(language_name("ru"), "Russian")
        self.assertEqual(language_name("de"), "German")

    def test_legacy_android_codes(self):
        self.assertEqual(language_name("in"), "Indonesian")
        self.assertEqual(language_name("iw"), "Hebrew")

    def test_unknown_codes_pass_through(self):
        self.assertEqual(language_name("xx"), "xx")
        self.assertEqual(language_name("xx-YY"), "xx-YY")
        self.assertEqual(language_name(""), "")
        # unknown qualifier of a known language: spelled out where possible
        self.assertEqual(language_name("ru-XQ"), "Russian (XQ)")

    def test_prompt_uses_spelled_out_names(self):
        tmp = make_tmpdir(self)
        toml = L10N_TOML.replace('languages = ["ru", "de"]',
                                 'languages = ["es-419", "kk"]')
        project = make_project(tmp, l10n=toml)
        batch = list(project.strings.values())[:1]
        prompt = build_translate_prompt(project, "es-419", batch, {})
        self.assertIn("Latin American Spanish", prompt)
        self.assertNotIn("es-419", prompt)
        self.assertIn("English", prompt)  # base_language spelled out too


if __name__ == "__main__":
    unittest.main()
