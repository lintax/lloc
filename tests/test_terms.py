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
description = "TV app"
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

# "the" appears in 5/6 strings -> DF stopword; "channel(s)" in 4 -> term;
# "live stream" bigram in 2 -> term; "settings" only 1 -> not a term.
STRINGS = """\
[open_channel]
text = "Open the channel"
[channel_list]
text = "All the channels"
[channel_lock]
text = "The channel is locked"
[channel_epg]
text = "No guide for the channel"
[stream_start]
text = "Starting the live stream"
[stream_stop]
text = "Live stream stopped"
"""


def _project(testcase):
    tmp = make_tmpdir(testcase)
    (tmp / "l10n.toml").write_text(L10N, encoding="utf-8")
    (tmp / "strings.toml").write_text(STRINGS, encoding="utf-8")
    return load_project(tmp)


def _locale_ru(project, channel_words):
    """Build an ok-status ru locale; channel strings translated with given words."""
    texts = {
        "open_channel": f"Открыть {channel_words[0]}",
        "channel_list": f"Все {channel_words[1]}",
        "channel_lock": f"{channel_words[2]} заблокирован",
        "channel_epg": f"Нет программы для {channel_words[3]}",
        "stream_start": "Запуск прямой трансляции",
        "stream_stop": "Прямая трансляция остановлена",
    }
    return {k: {"text": v, "src": text_hash(project.strings[k].text)}
            for k, v in texts.items()}


class TestExtraction(unittest.TestCase):
    def setUp(self):
        self.project = _project(self)

    def test_terms_extracted(self):
        terms = extract_terms(self.project)
        self.assertIn("channel", terms)                      # prefix-merged channels
        self.assertEqual(len(terms["channel"]), 4)
        self.assertIn("live stream", terms)                  # bigram, 2 strings
        self.assertNotIn("the", terms)                       # DF stopword
        self.assertNotIn("settings", terms)                  # below min_occurrences

    def test_text_terms_matching(self):
        self.assertEqual(text_terms(self.project, "Channels are here"), {"channel"})
        self.assertIn("live stream", text_terms(self.project, "a live stream plays"))
        self.assertEqual(text_terms(self.project, "Nothing relevant"), set())

    def test_disabled(self):
        self.project.terms.enabled = False
        self.assertEqual(text_terms(self.project, "channel"), set())

    def test_fewshot_prefers_term_examples(self):
        from lloc.config import StringEntry
        from lloc.prompts import select_examples
        locale = {
            # term-sharing example (channel)
            "channel_list": {"text": "Все каналы", "src": "x"},
            # word-overlap only ("open" not a term)
            "stream_start": {"text": "Запуск", "src": "x"},
        }
        batch = [StringEntry(key="new_key", text="Lock the channel now")]
        self.project.strings["new_key"] = StringEntry(key="new_key", text="Lock the channel now")
        examples = select_examples(self.project, batch, locale)
        self.assertEqual(examples[0][1], "Все каналы")       # term match ranked first


STRINGS_PLAY = """\
[app_tagline]
text = "A fast and simple TV player"
[play_button]
text = "Play"
[play_trailer]
text = "Play trailer"
[play_game]
text = "Play a game"
[player_settings]
text = "Player settings"
"""


class TestGlossaryBoundary(unittest.TestCase):
    """SPEC §13.5: derivation (play/player) has no lexical criterion, so a
    [glossary] word is the boundary: its own term, never prefix-merged."""

    def _project(self, glossary="", strings=STRINGS_PLAY, l10n=L10N):
        tmp = make_tmpdir(self)
        (tmp / "l10n.toml").write_text(l10n + glossary, encoding="utf-8")
        (tmp / "strings.toml").write_text(strings, encoding="utf-8")
        return load_project(tmp)

    def test_without_glossary_known_limitation(self):
        # documents §13.5: play swallows player when nothing marks the boundary
        terms = extract_terms(self._project())
        self.assertIn("app_tagline", terms["play"])
        self.assertNotIn("player", terms)

    def test_glossary_player_not_merged_into_play(self):
        project = self._project('[glossary]\n"player" = { ru = "плеер" }\n')
        terms = extract_terms(project)
        self.assertEqual(set(terms["play"]),
                         {"play_button", "play_trailer", "play_game"})
        # own term even below min_occurrences (glossary is the last word)
        self.assertEqual(set(terms["player"]), {"app_tagline", "player_settings"})

    def test_inflection_still_merges_into_glossary_term(self):
        # channels -> channel must keep merging when "channel" itself is glossaried
        project = self._project('[glossary]\n"channel" = { ru = "канал" }\n',
                                strings=STRINGS)
        terms = extract_terms(project)
        self.assertIn("channel", terms)
        self.assertEqual(len(terms["channel"]), 4)
        self.assertNotIn("channels", terms)

    def test_ru_base_inflection_still_merges(self):
        # no glossary involved: plain declension merging must be untouched
        strings = """\
[a]
text = "Открыть канал"
[b]
text = "Нет программы для канала"
[c]
text = "Все каналы здесь"
"""
        l10n = L10N.replace('base_language = "en"', 'base_language = "ru"') \
                   .replace('languages = ["ru", "ja"]', 'languages = ["en", "ja"]')
        project = self._project(strings=strings, l10n=l10n)
        terms = extract_terms(project)
        self.assertEqual(set(terms["канал"]), {"a", "b", "c"})

    def test_text_terms_glossary_word_matches_only_itself(self):
        project = self._project('[glossary]\n"player" = { ru = "плеер" }\n')
        self.assertEqual(text_terms(project, "A fast TV player"), {"player"})
        self.assertEqual(text_terms(project, "Play the game"), {"play"})


class TestHeuristic(unittest.TestCase):
    def setUp(self):
        self.project = _project(self)

    def test_consistent_quiet(self):
        locale = _locale_ru(self.project, ["канал", "каналы", "Канал", "канала"])
        self.assertEqual(heuristic_divergences(self.project, "ru", locale), [])

    def test_divergent_flagged(self):
        locale = _locale_ru(self.project, ["канал", "каналы", "Телеканал", "канала"])
        warnings = heuristic_divergences(self.project, "ru", locale)
        self.assertEqual(len(warnings), 1)
        self.assertIn('"channel"', warnings[0])
        self.assertIn("channel_lock", warnings[0])

    def test_non_segmented_skipped(self):
        locale = _locale_ru(self.project, ["канал", "каналы", "Телеканал", "канала"])
        self.assertEqual(heuristic_divergences(self.project, "ja", locale), [])

    def test_stale_strings_ignored(self):
        locale = _locale_ru(self.project, ["канал", "каналы", "Телеканал", "канала"])
        locale["channel_lock"]["src"] = "0000000000000000"   # stale -> not analyzed
        self.assertEqual(heuristic_divergences(self.project, "ru", locale), [])


class TestVariantMatching(unittest.TestCase):
    """Morphology-tolerant comparison of translated term variants (SPEC §13.3)."""

    def test_ru_inflection_merges(self):
        from lloc.terms import _variants_match
        self.assertTrue(_variants_match("этот канал", "этого канала", "ru"))   # user's FP
        self.assertTrue(_variants_match("канал", "каналы", "ru"))
        self.assertTrue(_variants_match("канала", "этот канал", "ru"))         # word-count diff
        self.assertFalse(_variants_match("канал", "телеканал", "ru"))          # derivation
        self.assertFalse(_variants_match("канал", "передача", "ru"))

    def test_de_real_divergence_stays(self):
        from lloc.terms import _variants_match
        self.assertFalse(_variants_match("kanäle", "sender", "de"))            # real problem
        self.assertTrue(_variants_match("sender", "sendern", "de"))            # inflection

    def test_ar_article_and_broken_plural_merge(self):
        from lloc.terms import _variants_match
        self.assertTrue(_variants_match("القناة", "قنوات", "ar"))              # user's FP
        self.assertTrue(_variants_match("قناة", "القناة", "ar"))
        self.assertFalse(_variants_match("قناة", "محطة", "ar"))                # different word

    def test_ja_substring_containment(self):
        from lloc.terms import _variants_match
        self.assertTrue(_variants_match("ロック", "ロックされています", "ja"))
        self.assertFalse(_variants_match("チャンネル", "番組", "ja"))

    def test_short_words_not_overmerged(self):
        from lloc.terms import _words_match
        self.assertFalse(_words_match("кот", "код", "ru"))
        self.assertFalse(_words_match("die", "der", "de"))

    def test_grouping_uses_language(self):
        from lloc.terms import divergences
        data = {"version": 1, "language": "ru",
                "terms": {"this channel": {
                    "a": {"src": "x", "tr": "этого канала"},
                    "b": {"src": "x", "tr": "этот канал"},
                    "c": {"src": "x", "tr": "Этот канал"}}},
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
        locale = _locale_ru(self.project, ["канал", "каналы", "телеканал", "канала"])
        save_locale(self.project.root, "ru", locale, list(self.project.strings))
        transport = TermMapTransport({
            "open_channel": "канал", "channel_list": "каналы",
            "channel_lock": "телеканал", "channel_epg": "канала",
            "stream_start": "прямой трансляции", "stream_stop": "прямая трансляция",
        })
        client = OllamaClient(self.project.ollama, transport=transport)
        data = build_term_map(self.project, "ru", locale, client)
        first_calls = transport.extract_calls
        self.assertGreater(first_calls, 0)
        self.assertTrue((self.project.root / "terms" / "ru.json").is_file())

        divs = divergences(data)
        self.assertEqual(len(divs), 1)                        # stream variants prefix-merge
        d = divs[0]
        self.assertEqual(d.term, "channel")
        self.assertEqual(d.majority, "канал")                 # канал/каналы/канала merge
        self.assertEqual(d.variants["телеканал"], ["channel_lock"])
        self.assertAlmostEqual(d.fraction, 0.75)

        # incremental: nothing changed -> no new requests
        build_term_map(self.project, "ru", locale, client)
        self.assertEqual(transport.extract_calls, first_calls)

    def test_accepted_suppresses(self):
        data = {"version": 1, "language": "ru",
                "terms": {"channel": {
                    "a": {"src": "x", "tr": "канал"},
                    "b": {"src": "x", "tr": "телеканал"}}},
                "accepted": {"channel": ["канал", "телеканал"]}}
        self.assertEqual(divergences(data), [])


class TestFixTerms(unittest.TestCase):
    def setUp(self):
        self.project = _project(self)
        self.locale = _locale_ru(self.project, ["канал", "каналы", "телеканал", "канала"])
        save_locale(self.project.root, "ru", self.locale, list(self.project.strings))
        self.extract_map = {
            "open_channel": "канал", "channel_list": "каналы",
            "channel_lock": "телеканал", "channel_epg": "канала",
            "stream_start": "трансляции", "stream_stop": "трансляция",
        }

    def test_fix_retranslates_minority_with_pinned_note(self):
        transport = TermMapTransport(
            self.extract_map, translate={"channel_lock": "Канал заблокирован"})
        client = OllamaClient(self.project.ollama, transport=transport)

        def obedient(url, payload, timeout):
            resp = transport(url, payload, timeout)
            content = payload["messages"][-1]["content"]
            if "term under analysis" in content and "channel_lock" in content \
                    and transport.translate_prompts:
                # after the fix, extraction sees the majority variant
                keys = payload["format"]["required"]
                out = {k: ("канал" if k == "channel_lock" else self.extract_map.get(k, "-"))
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
        self.assertIn('TERMINOLOGY: in this app "channel" is translated as "канал"', note)
        self.assertIn("unless", note)
        # locale updated
        from lloc.store import load_locale
        self.assertEqual(load_locale(self.project.root, "ru")["channel_lock"]["text"],
                         "Канал заблокирован")

    def test_persistent_divergence_becomes_accepted_and_idempotent(self):
        # model insists on its variant: re-translation keeps "телеканал"
        transport = TermMapTransport(
            self.extract_map, translate={"channel_lock": "Телеканал заблокирован"})
        client = OllamaClient(self.project.ollama, transport=transport)
        with redirect_stdout(io.StringIO()):
            stats = fix_language(self.project, "ru", client)
        self.assertEqual(stats["accepted"], ["channel"])
        data = load_term_map(self.project, "ru")
        self.assertIn("канал", data["accepted"]["channel"])
        self.assertIn("телеканал", data["accepted"]["channel"])

        # second run: divergence accepted -> nothing to do, no re-translation
        before = len(transport.translate_prompts)
        with redirect_stdout(io.StringIO()):
            stats2 = fix_language(self.project, "ru", client)
        self.assertEqual(stats2["retranslated"], 0)
        self.assertEqual(len(transport.translate_prompts), before)


if __name__ == "__main__":
    unittest.main()
