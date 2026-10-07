import io
import json
import unittest
from contextlib import redirect_stdout

from lloc.config import load_project
from lloc.hashes import text_hash
from lloc.ollama import OllamaClient, string_schema
from lloc.store import save_locale
from lloc.terms import (
    Divergence, build_term_map, divergences, extract_terms, fix_language,
    heuristic_divergences, load_term_map, text_terms,
)

from .helpers import make_tmpdir

L10N = """\
[app]
description = "Project tracker"
base_language = "en"
languages = ["ru", "ja"]
[ollama]
batch_size = 10
max_retries = 1
[output]
adapter = "android-xml"
[output.android-xml]
res_dir = "res"
"""

# "the" appears in 5/6 strings -> DF stopword; "project(s)" in 4 -> term;
# "cloud sync" bigram in 2 -> term; "settings" only 1 -> not a term.
STRINGS = """\
[open_project]
text = "Open the project"
[project_list]
text = "All the projects"
[project_lock]
text = "The project is locked"
[project_notes]
text = "No notes for the project"
[sync_start]
text = "Starting the cloud sync"
[sync_stop]
text = "Cloud sync stopped"
"""


def _project(testcase):
    tmp = make_tmpdir(testcase)
    (tmp / "l10n.toml").write_text(L10N, encoding="utf-8")
    (tmp / "strings.toml").write_text(STRINGS, encoding="utf-8")
    return load_project(tmp)


def _locale_ru(project, project_words):
    """Build an ok-status ru locale; project strings translated with given words."""
    texts = {
        "open_project": f"Открыть {project_words[0]}",
        "project_list": f"Все {project_words[1]}",
        "project_lock": f"{project_words[2]} заблокирован",
        "project_notes": f"Нет заметок для {project_words[3]}",
        "sync_start": "Запуск облачной синхронизации",
        "sync_stop": "Облачная синхронизация остановлена",
    }
    return {k: {"text": v, "src": text_hash(project.strings[k].text)}
            for k, v in texts.items()}


class TestExtraction(unittest.TestCase):
    def setUp(self):
        self.project = _project(self)

    def test_terms_extracted(self):
        terms = extract_terms(self.project)
        self.assertIn("project", terms)                      # prefix-merged projects
        self.assertEqual(len(terms["project"]), 4)
        self.assertIn("cloud sync", terms)                  # bigram, 2 strings
        self.assertNotIn("the", terms)                       # DF stopword
        self.assertNotIn("settings", terms)                  # below min_occurrences

    def test_text_terms_matching(self):
        self.assertEqual(text_terms(self.project, "Projects are here"), {"project"})
        self.assertIn("cloud sync", text_terms(self.project, "a cloud sync runs"))
        self.assertEqual(text_terms(self.project, "Nothing relevant"), set())

    def test_fewshot_prefers_term_examples(self):
        from lloc.config import StringEntry
        from lloc.prompts import select_examples
        # the few-shot pool is trusted-only: mark entries manual
        locale = {
            # term-sharing example (project)
            "project_list": {"text": "Все проекты", "src": "x", "model": "manual"},
            # word-overlap only ("open" not a term)
            "sync_start": {"text": "Запуск", "src": "x", "model": "manual"},
        }
        batch = [StringEntry(key="new_key", text="Lock the project now")]
        self.project.strings["new_key"] = StringEntry(key="new_key", text="Lock the project now")
        examples = select_examples(self.project, batch, locale)
        self.assertEqual(examples[0][1], "Все проекты")       # term match ranked first


STRINGS_PLAY = """\
[app_tagline]
text = "A fast and simple text editor"
[play_button]
text = "Edit"
[play_trailer]
text = "Edit title"
[play_game]
text = "Edit a note"
[editor_settings]
text = "Editor settings"
"""


class TestGlossaryBoundary(unittest.TestCase):
    """derivation (edit/editor) has no lexical criterion, so a
    [glossary] word is the boundary: its own term, never prefix-merged."""

    def _project(self, glossary="", strings=STRINGS_PLAY, l10n=L10N):
        tmp = make_tmpdir(self)
        (tmp / "l10n.toml").write_text(l10n + glossary, encoding="utf-8")
        (tmp / "strings.toml").write_text(strings, encoding="utf-8")
        return load_project(tmp)

    def test_without_glossary_known_limitation(self):
        # edit swallows editor when nothing marks the boundary
        terms = extract_terms(self._project())
        self.assertIn("app_tagline", terms["edit"])
        self.assertNotIn("editor", terms)

    def test_glossary_editor_not_merged_into_edit(self):
        project = self._project('[glossary]\n"editor" = { ru = "редактор" }\n')
        terms = extract_terms(project)
        self.assertEqual(set(terms["edit"]),
                         {"play_button", "play_trailer", "play_game"})
        # own term even below min_occurrences (glossary is the last word)
        self.assertEqual(set(terms["editor"]), {"app_tagline", "editor_settings"})

    def test_inflection_still_merges_into_glossary_term(self):
        # projects -> project must keep merging when "project" itself is glossaried
        project = self._project('[glossary]\n"project" = { ru = "проект" }\n',
                                strings=STRINGS)
        terms = extract_terms(project)
        self.assertIn("project", terms)
        self.assertEqual(len(terms["project"]), 4)
        self.assertNotIn("projects", terms)

    def test_ru_base_inflection_still_merges(self):
        # no glossary involved: plain declension merging must be untouched
        strings = """\
[a]
text = "Открыть проект"
[b]
text = "Нет заметок для проекта"
[c]
text = "Все проекты здесь"
"""
        l10n = L10N.replace('base_language = "en"', 'base_language = "ru"') \
                   .replace('languages = ["ru", "ja"]', 'languages = ["en", "ja"]')
        project = self._project(strings=strings, l10n=l10n)
        terms = extract_terms(project)
        self.assertEqual(set(terms["проект"]), {"a", "b", "c"})

    def test_text_terms_glossary_word_matches_only_itself(self):
        project = self._project('[glossary]\n"editor" = { ru = "редактор" }\n')
        self.assertEqual(text_terms(project, "A fast text editor"), {"editor"})
        self.assertEqual(text_terms(project, "Edit the note"), {"edit"})


class TestHeuristic(unittest.TestCase):
    def setUp(self):
        self.project = _project(self)

    def test_consistent_quiet(self):
        locale = _locale_ru(self.project, ["проект", "проекты", "Проект", "проекта"])
        self.assertEqual(heuristic_divergences(self.project, "ru", locale), [])

    def test_divergent_flagged(self):
        locale = _locale_ru(self.project, ["проект", "проекты", "Подпроект", "проекта"])
        warnings = heuristic_divergences(self.project, "ru", locale)
        self.assertEqual(len(warnings), 1)
        self.assertIn('"project"', warnings[0])
        self.assertIn("project_lock", warnings[0])

    def test_non_segmented_skipped(self):
        locale = _locale_ru(self.project, ["проект", "проекты", "Подпроект", "проекта"])
        self.assertEqual(heuristic_divergences(self.project, "ja", locale), [])

    def test_stale_strings_ignored(self):
        locale = _locale_ru(self.project, ["проект", "проекты", "Подпроект", "проекта"])
        locale["project_lock"]["src"] = "0000000000000000"   # stale -> not analyzed
        self.assertEqual(heuristic_divergences(self.project, "ru", locale), [])


class TestVariantMatching(unittest.TestCase):
    """Morphology-tolerant comparison of translated term variants."""

    def test_ru_inflection_merges(self):
        from lloc.terms import _variants_match
        self.assertTrue(_variants_match("этот проект", "этого проекта", "ru"))   # user's FP
        self.assertTrue(_variants_match("проект", "проекты", "ru"))
        self.assertTrue(_variants_match("проекта", "этот проект", "ru"))         # word-count diff
        self.assertFalse(_variants_match("проект", "подпроект", "ru"))          # derivation
        self.assertFalse(_variants_match("проект", "задача", "ru"))

    def test_de_real_divergence_stays(self):
        from lloc.terms import _variants_match
        self.assertFalse(_variants_match("projekte", "vorhaben", "de"))            # real problem
        self.assertTrue(_variants_match("vorhaben", "vorhabens", "de"))            # inflection

    def test_ar_article_and_broken_plural_merge(self):
        from lloc.terms import _variants_match
        self.assertTrue(_variants_match("المشروع", "مشاريع", "ar"))              # user's FP
        self.assertTrue(_variants_match("مشروع", "المشروع", "ar"))
        self.assertFalse(_variants_match("مشروع", "مهمة", "ar"))                # different word

    def test_ja_substring_containment(self):
        from lloc.terms import _variants_match
        self.assertTrue(_variants_match("ロック", "ロックされています", "ja"))
        self.assertFalse(_variants_match("プロジェクト", "タスク", "ja"))

    def test_short_words_not_overmerged(self):
        from lloc.terms import _words_match
        self.assertFalse(_words_match("кот", "код", "ru"))
        self.assertFalse(_words_match("die", "der", "de"))

    def test_grouping_uses_language(self):
        from lloc.terms import divergences
        data = {"version": 1, "language": "ru",
                "terms": {"this project": {
                    "a": {"src": "x", "tr": "этого проекта"},
                    "b": {"src": "x", "tr": "этот проект"},
                    "c": {"src": "x", "tr": "Этот проект"}}},
                "accepted": {}}
        self.assertEqual(divergences(data), [])                                # merged, quiet


class TermMapTransport:
    """Answers terms.txt extraction prompts using a fixed word map, and
    choose_term / translate prompts via handlers."""

    def __init__(self, extract_map, translate=None, choice=None):
        self.extract_map = extract_map        # key -> translated term word
        self.translate = translate or {}      # key -> full translation for fix
        self.choice = choice
        self.extract_calls = 0
        self.translate_prompts = []

    def __call__(self, url, payload, timeout):
        content = payload["messages"][-1]["content"]
        keys = payload["format"].get("required", [])
        if "term under analysis" in content:
            self.extract_calls += 1
            out = {k: self.extract_map.get(k, "-") for k in keys}
        elif "choose the ONE" in content:
            out = {"variant": self.choice}
        else:  # translate prompt (fix-terms re-translation)
            self.translate_prompts.append(content)
            out = {k: self.translate[k] for k in keys}
        return {"message": {"content": json.dumps(out, ensure_ascii=False)}}


class TestTermMap(unittest.TestCase):
    def setUp(self):
        self.project = _project(self)

    def test_build_incremental_and_divergences(self):
        locale = _locale_ru(self.project, ["проект", "проекты", "подпроект", "проекта"])
        save_locale(self.project.root, "ru", locale, list(self.project.strings))
        transport = TermMapTransport({
            "open_project": "проект", "project_list": "проекты",
            "project_lock": "подпроект", "project_notes": "проекта",
            "sync_start": "облачной синхронизации", "sync_stop": "облачная синхронизация",
        })
        client = OllamaClient(self.project.ollama, transport=transport)
        data = build_term_map(self.project, "ru", locale, client)
        first_calls = transport.extract_calls
        self.assertGreater(first_calls, 0)
        self.assertTrue((self.project.root / "terms" / "ru.json").is_file())

        divs = divergences(data)
        self.assertEqual(len(divs), 1)                        # sync variants prefix-merge
        d = divs[0]
        self.assertEqual(d.term, "project")
        self.assertEqual(d.majority, "проект")                 # проект/проекты/проекта merge
        self.assertEqual(d.variants["подпроект"], ["project_lock"])
        self.assertAlmostEqual(d.fraction, 0.75)

        # incremental: nothing changed -> no new requests
        build_term_map(self.project, "ru", locale, client)
        self.assertEqual(transport.extract_calls, first_calls)

    def test_accepted_suppresses(self):
        data = {"version": 1, "language": "ru",
                "terms": {"project": {
                    "a": {"src": "x", "tr": "проект"},
                    "b": {"src": "x", "tr": "подпроект"}}},
                "accepted": {"project": ["проект", "подпроект"]}}
        self.assertEqual(divergences(data), [])


class TestFixTerms(unittest.TestCase):
    def setUp(self):
        self.project = _project(self)
        self.locale = _locale_ru(self.project, ["проект", "проекты", "подпроект", "проекта"])
        save_locale(self.project.root, "ru", self.locale, list(self.project.strings))
        self.extract_map = {
            "open_project": "проект", "project_list": "проекты",
            "project_lock": "подпроект", "project_notes": "проекта",
            "sync_start": "синхронизации", "sync_stop": "синхронизация",
        }

    def test_fix_retranslates_minority_with_pinned_note(self):
        transport = TermMapTransport(
            self.extract_map, translate={"project_lock": "Проект заблокирован"})
        client = OllamaClient(self.project.ollama, transport=transport)

        def obedient(url, payload, timeout):
            resp = transport(url, payload, timeout)
            content = payload["messages"][-1]["content"]
            if "term under analysis" in content and "project_lock" in content \
                    and transport.translate_prompts:
                # after the fix, extraction sees the majority variant
                keys = payload["format"]["required"]
                out = {k: ("проект" if k == "project_lock" else self.extract_map.get(k, "-"))
                       for k in keys}
                return {"message": {"content": json.dumps(out, ensure_ascii=False)}}
            return resp
        client.transport = obedient

        with redirect_stdout(io.StringIO()):
            stats = fix_language(self.project, "ru", client)
        self.assertEqual(stats["retranslated"], 1)
        self.assertEqual(stats["accepted"], [])
        self.assertEqual(stats["failed"], {})
        # pinned note present in the re-translation prompt
        self.assertTrue(transport.translate_prompts)
        note = transport.translate_prompts[0]
        self.assertIn('TERMINOLOGY: in this app "project" is translated as "проект"', note)
        self.assertIn("unless", note)
        # locale updated
        from lloc.store import load_locale
        self.assertEqual(load_locale(self.project.root, "ru")["project_lock"]["text"],
                         "Проект заблокирован")

    def test_persistent_divergence_becomes_accepted_and_idempotent(self):
        # model insists on its variant: re-translation keeps "подпроект"
        transport = TermMapTransport(
            self.extract_map, translate={"project_lock": "Подпроект заблокирован"})
        client = OllamaClient(self.project.ollama, transport=transport)
        with redirect_stdout(io.StringIO()):
            stats = fix_language(self.project, "ru", client)
        self.assertEqual(stats["accepted"], ["project"])
        data = load_term_map(self.project, "ru")
        self.assertIn("проект", data["accepted"]["project"])
        self.assertIn("подпроект", data["accepted"]["project"])

        # second run: divergence accepted -> nothing to do, no re-translation
        before = len(transport.translate_prompts)
        with redirect_stdout(io.StringIO()):
            stats2 = fix_language(self.project, "ru", client)
        self.assertEqual(stats2["retranslated"], 0)
        self.assertEqual(len(transport.translate_prompts), before)


if __name__ == "__main__":
    unittest.main()
