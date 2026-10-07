"""Script-mix heuristic: foreign Unicode scripts in translations (no LLM)."""
import io
import json
import unittest
from contextlib import redirect_stdout

from lloc.__main__ import main
from lloc.langnames import expected_scripts
from lloc.validate import script_mix

from .helpers import L10N_TOML, make_project, make_tmpdir


class TestExpectedScripts(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(expected_scripts("ru"), frozenset({"CYRILLIC"}))
        self.assertEqual(expected_scripts("de"), frozenset({"LATIN"}))
        self.assertEqual(expected_scripts("hy"), frozenset({"ARMENIAN"}))
        self.assertEqual(expected_scripts("kk"), frozenset({"CYRILLIC"}))
        self.assertIn("HIRAGANA", expected_scripts("ja"))
        self.assertIn("CJK", expected_scripts("zh-Hant"))
        # script subtag wins over the language default
        self.assertEqual(expected_scripts("sr-Latn"), frozenset({"LATIN"}))
        self.assertEqual(expected_scripts("sr-Cyrl"), frozenset({"CYRILLIC"}))
        self.assertEqual(expected_scripts("sr"), frozenset({"LATIN"}))  # решение пользователя
        # legacy codes canonicalized
        self.assertEqual(expected_scripts("iw"), frozenset({"HEBREW"}))
        # unknown language defaults to Latin
        self.assertEqual(expected_scripts("xx"), frozenset({"LATIN"}))


class TestScriptMix(unittest.TestCase):
    HY = expected_scripts("hy")
    RU = expected_scripts("ru")
    DE = expected_scripts("de")
    NE = expected_scripts("ne")

    def test_live_case_burmese_inside_armenian(self):
        # the real hy defect from the production run: MYANMAR LETTER NA
        # (န) glued into an Armenian word
        w = script_mix("Կարճ နային հանգիստ", self.HY)
        self.assertTrue(any("MYANMAR" in x for x in w), w)

    def test_clean_texts_pass(self):
        self.assertEqual(script_mix("Короткий дневной отдых", self.RU), [])
        self.assertEqual(script_mix("Kurze Pause am Tag", self.DE), [])
        self.assertEqual(script_mix("छोटो दिउँसोको आराम", self.NE), [])

    def test_digits_punct_placeholders_ignored(self):
        self.assertEqual(script_mix("Осталось: %1$d мин (%2$s)…", self.RU), [])
        self.assertEqual(script_mix("%1$dh %2$dm", self.RU), [])

    def test_placeholder_glued_to_native_script_not_flagged(self):
        # live false positives: %1$d個まで (ja), %1$dч (ru), %1$s에 (ko)
        self.assertEqual(script_mix("最大%1$d個まで保存できます。", expected_scripts("ja")), [])
        self.assertEqual(script_mix("%1$dч %2$dм", self.RU), [])
        self.assertEqual(script_mix("%1$s에 깨움", expected_scripts("ko")), [])

    def test_brace_placeholders_not_flagged(self):
        self.assertEqual(script_mix("Проект {projectName}: {{taskCount}}", self.RU), [])

    def test_short_latin_tokens_allowed(self):
        self.assertEqual(script_mix("Экспорт в PDF и CSV: OK", self.RU), [])

    def test_long_latin_token_flagged(self):
        w = script_mix("Хатагч preset хадгалах", self.RU, latin_token_max=5)
        self.assertEqual(w, ['untranslated latin token "preset"'])

    def test_allowlist_and_threshold_configurable(self):
        self.assertEqual(
            script_mix("Осталось Premium", self.RU, allow=frozenset({"premium"})), [])
        self.assertEqual(script_mix("Осталось Premium", self.RU, latin_token_max=7), [])
        w = script_mix("Осталось Premium", self.RU)
        self.assertEqual(len(w), 1)

    def test_cyrillic_inside_latin_language_flagged(self):
        w = script_mix("Einstellungen сохранить", self.DE)
        self.assertTrue(any("CYRILLIC" in x for x in w))

    def test_japanese_mixed_scripts_ok(self):
        self.assertEqual(script_mix("設定を保存する（オン）", expected_scripts("ja")), [])

    def test_spaceless_scripts_allow_glued_latin_tokens(self):
        # latin token glued to a native particle without a space boundary
        self.assertEqual(script_mix("1000Hzのトーン", expected_scripts("ja")), [])
        self.assertEqual(script_mix("Play를 통해", expected_scripts("ko")), [])
        # a LONG glued latin token is still reported
        w = script_mix("Downloadのトーン", expected_scripts("ja"), latin_token_max=5)
        self.assertEqual(w, ['untranslated latin token "Download"'])
        # alphabetic languages stay strict: latin letter inside a cyrillic word
        w = script_mix("таймерaў", expected_scripts("be"))
        self.assertTrue(any("LATIN" in x for x in w), w)


class TestVerifyIntegration(unittest.TestCase):
    def test_verify_warns_on_script_mix(self):
        tmp = make_tmpdir(self)
        # glossary term "PDF" -> allowlisted; add long brand via [validation]
        toml = L10N_TOML.replace(
            'semantic = "off"', 'semantic = "off"\nlatin_allowlist = ["Acmeboard"]')
        project = make_project(tmp, l10n=toml)
        from lloc.hashes import text_hash
        from lloc.store import save_locale
        entries = {
            "menu_search": {"text": "Поиск preset-ов", "src": text_hash("Search")},
            "items_count": {"text": "Acmeboard: %d", "src": text_hash("Items: %d")},
        }
        save_locale(tmp, "ru", entries, list(project.strings))
        out = io.StringIO()
        with redirect_stdout(out):
            rc = main(["--dir", str(tmp), "verify", "--lang", "ru"])
        text = out.getvalue()
        self.assertEqual(rc, 0)  # warnings only, exit code untouched
        self.assertIn('untranslated latin token "preset"', text)
        self.assertNotIn("Acmeboard", [w for w in text.splitlines()
                                       if "warning" in w and "Acmeboard" in w])
        self.assertIn("warning(s)", text)


if __name__ == "__main__":
    unittest.main()
