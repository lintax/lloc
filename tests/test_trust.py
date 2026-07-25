"""Reliability hardening (SPEC §13.6): trusted-only few-shot pool,
anchor-first cold start, conflict escalation."""
import unittest

from lloc.config import StringEntry
from lloc.engine import plan
from lloc.hashes import text_hash
from lloc.prompts import select_examples
from lloc.semantic import _pair_hash, save_semantic
from lloc.store import save_locale

from .helpers import make_project, make_tmpdir

SEARCH_STRINGS = """\
[search_title]
text = "Search everywhere"

[search_hint]
text = "Search in the list"

[search_saved]
text = "Search saved items"

[search_recent]
text = "Search recent files"

[search_button]
text = "Search now"
"""


class TestTrustedPool(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmpdir(self)
        self.project = make_project(self.tmp, strings=SEARCH_STRINGS)
        self.locale = {
            "search_title": {"text": "manual-tr", "src": "x", "model": "manual"},
            "search_hint": {"text": "verified-tr", "src": "x",
                            "model": "gemma4:12b-it-qat"},
            "search_saved": {"text": "imported-tr", "src": "x", "model": "imported"},
            # translated with semantic="off": no verdict anywhere
            "search_recent": {"text": "unverified-tr", "src": "x",
                              "model": "gemma4:12b-it-qat"},
        }
        save_semantic(self.project, "ru", {
            "version": 1, "language": "ru", "entries": {
                "search_hint": {"src": _pair_hash("Search in the list", "verified-tr"),
                                "verdict": "ok", "back": ""}}})
        self.batch = [self.project.strings["search_button"]]

    def test_pool_is_trusted_only_and_ranked(self):
        examples = select_examples(self.project, self.batch, self.locale, lang="ru")
        self.assertEqual([t for _, t in examples],
                         ["manual-tr", "verified-tr", "imported-tr"])
        # the unverified entry is not in the pool at all
        self.assertNotIn("unverified-tr", [t for _, t in examples])

    def test_stale_verdict_does_not_verify_new_text(self):
        # translation changed after the verdict -> pair hash mismatch -> untrusted
        self.locale["search_hint"]["text"] = "retranslated differently"
        examples = select_examples(self.project, self.batch, self.locale, lang="ru")
        self.assertEqual([t for _, t in examples], ["manual-tr", "imported-tr"])

    def test_force_rerun_with_unverified_legacy_pool_gets_zero_examples(self):
        """The lv-incident scenario: a --force rerun over a locale full of OLD
        unverified entries (a pre-QA bulk run) must start with an EMPTY
        few-shot pool — no bypass path may leak the legacy translations in."""
        from lloc.prompts import build_translate_prompt
        tmp = make_tmpdir(self)
        project = make_project(tmp, strings=SEARCH_STRINGS)
        legacy = {k: {"text": f"legacy({k})", "src": "stale-hash",
                      "model": "gemma4:12b-it-qat",
                      "translated_at": "2026-07-25T16:00:00+00:00"}
                  for k in project.strings}  # no semantic verdicts anywhere
        save_locale(tmp, "lv" if "lv" in project.languages else "ru",
                    legacy, list(project.strings))
        lang = "ru"
        batch = [project.strings["search_button"]]
        self.assertEqual(select_examples(project, batch, legacy, lang=lang), [])
        prompt = build_translate_prompt(project, lang, batch, legacy)
        self.assertIn("(none yet)", prompt)  # examples block is empty
        for k in project.strings:
            self.assertNotIn(f"legacy({k})", prompt)

    def test_suspect_verdict_is_not_trusted(self):
        save_semantic(self.project, "ru", {
            "version": 1, "language": "ru", "entries": {
                "search_hint": {"src": _pair_hash("Search in the list", "verified-tr"),
                                "verdict": "suspect", "reason": "bad", "back": ""}}})
        examples = select_examples(self.project, self.batch, self.locale, lang="ru")
        self.assertEqual([t for _, t in examples], ["manual-tr", "imported-tr"])


ANCHOR_STRINGS = """\
[channel_rich]
text = "Open channel"
context = "Button on the player screen; opens the currently highlighted channel"

[channel_poor]
text = "The channel could not be loaded, check the connection and try again"

[channel_mid]
text = "Channel list"
context = "Screen title"

[misc_hello]
text = "Welcome"

[misc_bye]
text = "Goodbye"

[misc_yes]
text = "Yes"
"""


class TestTermsAlwaysOn(unittest.TestCase):
    """Decision record (SPEC §13.6): the term system is unconditional — the
    former [terms].enabled kill switch was removed. Setting it is a config
    error with a pointer to the decision, and anchors always apply."""

    def test_enabled_key_is_config_error(self):
        from lloc.config import ConfigError
        from .helpers import L10N_TOML
        tmp = make_tmpdir(self)
        toml = L10N_TOML + "\n[terms]\nenabled = false\n"
        with self.assertRaisesRegex(ConfigError, r"\[terms\].enabled was removed"):
            make_project(tmp, l10n=toml, strings=ANCHOR_STRINGS)
        # `enabled = true` is equally rejected: the key itself is gone
        tmp2 = make_tmpdir(self)
        toml = L10N_TOML + "\n[terms]\nenabled = true\n"
        with self.assertRaisesRegex(ConfigError, "SPEC.md §13.6"):
            make_project(tmp2, l10n=toml, strings=ANCHOR_STRINGS)

    def test_terms_and_anchors_unconditional(self):
        from lloc.terms import text_terms
        tmp = make_tmpdir(self)
        project = make_project(tmp, strings=ANCHOR_STRINGS)  # no [terms] section
        self.assertEqual(text_terms(project, "Open channel now"), {"channel"})
        self.assertEqual(plan(project, ["ru"], False, None)["ru"][0], "channel_rich")


class TestAnchorColdStart(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmpdir(self)
        self.project = make_project(self.tmp, strings=ANCHOR_STRINGS)

    def test_cold_start_puts_term_anchor_first(self):
        work = plan(self.project, ["ru"], force=False, limit=None)
        keys = work["ru"]
        self.assertEqual(set(keys), set(self.project.strings))
        # "channel" is the frequent term; its representative (richest context,
        # then shortest text) leads the plan
        self.assertEqual(keys[0], "channel_rich")
        # non-anchor keys keep base order after the anchors
        tail = [k for k in keys if k.startswith("misc_")]
        self.assertEqual(tail, ["misc_hello", "misc_bye", "misc_yes"])

    def test_incremental_run_keeps_base_order(self):
        # > 10% accepted -> plain strings.toml order for the remaining delta
        entries = {k: {"text": f"[ru]{e.text}", "src": text_hash(e.text)}
                   for k, e in list(self.project.strings.items())[:3]}
        save_locale(self.tmp, "ru", entries, list(self.project.strings))
        work = plan(self.project, ["ru"], force=False, limit=None)
        self.assertEqual(work["ru"], ["misc_hello", "misc_bye", "misc_yes"])

    def test_empty_delta_untouched(self):
        entries = {k: {"text": "x", "src": text_hash(e.text)}
                   for k, e in self.project.strings.items()}
        save_locale(self.tmp, "ru", entries, list(self.project.strings))
        self.assertEqual(plan(self.project, ["ru"], False, None)["ru"], [])


if __name__ == "__main__":
    unittest.main()
