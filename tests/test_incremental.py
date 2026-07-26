"""Incremental-translate guarantees: a 2-string delta pays ONLY for itself;
the backlog is swept solely by explicit fix-semantic; twice-arbitrated
entries become resolved and stop being re-judged."""
import io
import unittest
from contextlib import redirect_stdout

from lloc.engine import run_translate
from lloc.hashes import text_hash
from lloc.semantic import (_pair_hash, is_resolved, load_semantic,
                           mark_arbitrated, save_semantic)
from lloc.store import load_locale, save_locale

from .helpers import make_project, make_tmpdir
from .test_ensemble import ENS, ENSEMBLE_TOML, EnsembleTransport, _PH
from .test_weak_models import BIG_STRINGS

NEW_KEYS = """
[fresh_one]
text = "A brand new string"

[fresh_two]
text = "Another brand new string"
"""


class TestIncrementalTranslate(unittest.TestCase):
    def _project_with_backlog(self):
        """20 old keys, ALL arbitrated (fat backlog) + 2 fresh untranslated."""
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=ENSEMBLE_TOML,
                               strings=BIG_STRINGS + NEW_KEYS)
        old = {k: {"text": f"A({k})", "src": text_hash(e.text), "model": "m"}
               for k, e in project.strings.items() if k.startswith("key_")}
        save_locale(tmp, "ru", old, list(project.strings))
        save_semantic(project, "ru", {"version": 1, "language": "ru", "entries": {
            k: {"src": _pair_hash(project.strings[k].text, old[k]["text"]),
                "verdict": "arbitrated", "attempts": 1, "back": ""}
            for k in old}})
        return tmp, project

    def test_two_string_delta_pays_only_for_itself(self):
        tmp, project = self._project_with_backlog()
        tr = EnsembleTransport()
        with redirect_stdout(io.StringIO()):
            results = run_translate(project, ["ru"], transport=tr)
        translated_keys = {k for _m, kind, ks in tr.events
                           for k in ks if kind == "translate"}
        self.assertEqual(translated_keys, {"fresh_one", "fresh_two"})  # delta only
        judged_keys = {k for _m, kind, ks in tr.events
                       for k in ks if kind == "ensemble_judge"}
        self.assertEqual(judged_keys, {"fresh_one", "fresh_two"})
        self.assertEqual(results[0].translated, 2)
        # the backlog is untouched: still arbitrated with attempts=1
        entries = load_semantic(project, "ru")["entries"]
        self.assertEqual(entries["key_00"]["attempts"], 1)

    def test_fix_semantic_sweeps_backlog(self):
        tmp, project = self._project_with_backlog()
        # translate the fresh delta first so only the backlog remains
        with redirect_stdout(io.StringIO()):
            run_translate(project, ["ru"], transport=EnsembleTransport())
        tr = EnsembleTransport()  # judge agrees everywhere -> backlog becomes ok
        with redirect_stdout(io.StringIO()):
            run_translate(project, ["ru"], transport=tr, include_backlog=True)
        swept = {k for _m, kind, ks in tr.events
                 for k in ks if kind == "translate"}
        self.assertEqual(swept, {k for k in project.strings if k.startswith("key_")})
        entries = load_semantic(project, "ru")["entries"]
        self.assertTrue(all(entries[k]["verdict"] == "ok" for k in swept))

    def test_twice_arbitrated_becomes_resolved_and_left_alone(self):
        tmp, project = self._project_with_backlog()
        with redirect_stdout(io.StringIO()):
            run_translate(project, ["ru"], transport=EnsembleTransport())
        # sweep 1: judge STILL disagrees on everything -> attempts 1 -> 2
        tr1 = EnsembleTransport(verdicts={k: "c2" for k in project.strings})
        with redirect_stdout(io.StringIO()):
            run_translate(project, ["ru"], transport=tr1, include_backlog=True)
        entries = load_semantic(project, "ru")["entries"]
        self.assertEqual(entries["key_00"]["attempts"], 2)
        self.assertTrue(is_resolved(entries["key_00"]))
        # sweep 2: resolved entries are NOT re-judged
        tr2 = EnsembleTransport(verdicts={k: "c2" for k in project.strings})
        with redirect_stdout(io.StringIO()):
            run_translate(project, ["ru"], transport=tr2, include_backlog=True)
        self.assertEqual([e for e in tr2.events if e[1] == "translate"], [])
        # ...until the translation itself changes (src mismatch reopens)
        locale = load_locale(tmp, "ru")
        self.assertTrue(locale["key_00"]["text"])  # winner is in place

    def test_attempts_reset_on_new_dispute(self):
        tmp = make_tmpdir(self)
        project = make_project(tmp)
        mark_arbitrated(project, "ru", {"menu_search": ("Вариант", "spor")})
        self.assertEqual(load_semantic(project, "ru")["entries"]
                         ["menu_search"]["attempts"], 1)
        mark_arbitrated(project, "ru", {"menu_search": ("Вариант2", "spor")})
        self.assertEqual(load_semantic(project, "ru")["entries"]
                         ["menu_search"]["attempts"], 2)
        # an ok verdict in between resets the chain
        from lloc.semantic import mark_ok
        mark_ok(project, "ru", {"menu_search": "Вариант3"})
        mark_arbitrated(project, "ru", {"menu_search": ("Вариант4", "spor")})
        self.assertEqual(load_semantic(project, "ru")["entries"]
                         ["menu_search"]["attempts"], 1)


if __name__ == "__main__":
    unittest.main()
