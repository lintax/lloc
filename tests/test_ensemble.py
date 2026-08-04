"""Ensemble mode (SPEC §13.7): two models translate every string, a judge
arbitrates by meaning; strictly phased for minimal model swaps."""
import io
import json
import unittest
from contextlib import redirect_stdout

from lloc import engine
from lloc.config import ConfigError
from lloc.engine import run_translate
from lloc.semantic import load_semantic, save_semantic, verified_pairs, _pair_hash
from lloc.store import load_locale, save_locale
from lloc.hashes import text_hash
from lloc.prompts import select_examples

from .helpers import L10N_TOML, LOCAL_MODEL, make_project, make_tmpdir

ENS = "ens-model"

ENSEMBLE_TOML = L10N_TOML.replace(
    'semantic = "off"', f'semantic = "batch"\nensemble_model = "{ENS}"')

_PH = {"playback_error": " %1$s", "items_count": " %d"}


class EnsembleTransport:
    """Routes by payload['model'] and prompt kind. Phase A texts are
    "A(key)", phase B texts "B(key)" (or a forced value via b_same)."""

    def __init__(self, verdicts=None, b_same=()):
        self.verdicts = dict(verdicts or {})
        self.b_same = set(b_same)   # keys where B repeats A's text
        self.events = []            # (model, kind, keys)
        self.on_call = None

    def __call__(self, url, payload, timeout):
        model = payload["model"]
        keys = list(payload["format"].get("required", []))
        first = payload["messages"][0]["content"]
        if "comparing TWO independent translations" in first:
            kind = "ensemble_judge"
            out = {k: self.verdicts.get(k, "agree") for k in keys}
        elif "literal translator" in first:
            kind = "back"
            out = {k: f"back({k})" for k in keys}
        else:
            kind = "translate"
            prefix = "B" if model == ENS else "A"
            out = {}
            for k in keys:
                text = f"A({k})" if (prefix == "B" and k in self.b_same) \
                    else f"{prefix}({k})"
                out[k] = text + _PH.get(k, "")
        self.events.append((model, kind, keys))
        if self.on_call:
            self.on_call(model, kind, keys)
        return {"message": {"content": json.dumps(out, ensure_ascii=False)}}


class TestEnsembleConfig(unittest.TestCase):
    def test_parse_and_modes(self):
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=ENSEMBLE_TOML)
        self.assertTrue(project.ensemble)
        self.assertEqual(project.ensemble_judge, ENS)      # default judge = B model
        self.assertFalse(project.semantic_phased)          # ensemble supersedes
        self.assertFalse(project.semantic_inline_judge)

    def test_judge_override_via_semantic_model(self):
        tmp = make_tmpdir(self)
        toml = ENSEMBLE_TOML.replace('ensemble_model =',
                                     'semantic_model = "third-judge"\nensemble_model =')
        project = make_project(tmp, l10n=toml)
        self.assertEqual(project.ensemble_judge, "third-judge")

    def test_requires_semantic_batch(self):
        tmp = make_tmpdir(self)
        toml = L10N_TOML.replace('semantic = "off"',
                                 f'semantic = "off"\nensemble_model = "{ENS}"')
        with self.assertRaisesRegex(ConfigError, "ensemble_model requires"):
            make_project(tmp, l10n=toml)


class TestEnsembleRun(unittest.TestCase):
    def _run(self, project, tr, langs=("ru",)):
        with redirect_stdout(io.StringIO()) as out:
            results = run_translate(project, list(langs), transport=tr)
        return results, out.getvalue()

    def test_both_phases_translate_same_delta_and_swap_order(self):
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=ENSEMBLE_TOML)
        tr = EnsembleTransport()
        results, out = self._run(project, tr, langs=("ru", "de"))
        # phase A translated everything with the main model, phase B with ENS
        a_keys = {k for m, kind, ks in tr.events for k in ks
                  if kind == "translate" and m == LOCAL_MODEL}
        b_keys = {k for m, kind, ks in tr.events for k in ks
                  if kind == "translate" and m == ENS}
        self.assertEqual(a_keys, set(project.strings))
        self.assertEqual(b_keys, set(project.strings))
        # strict phasing: every main-model call precedes the first ENS call
        models_seq = [m for m, _kind, _ks in tr.events]
        first_ens = models_seq.index(ENS)
        self.assertNotIn(LOCAL_MODEL, models_seq[first_ens:])
        self.assertIn("phase A", out)
        self.assertIn("phase B", out)
        self.assertIn("phase C", out)

    def test_agreement_yields_ok_and_trusted_pool(self):
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=ENSEMBLE_TOML)
        tr = EnsembleTransport()  # judge says "agree" everywhere
        results, _ = self._run(project, tr)
        locale = load_locale(tmp, "ru")
        self.assertEqual(locale["menu_search"]["text"], "A(menu_search)")
        self.assertEqual(load_semantic(project, "ru")["entries"]
                         ["menu_search"]["verdict"], "ok")
        self.assertIn("menu_search", verified_pairs(tmp, "ru"))
        self.assertEqual(results[0].arbitrated, 0)

    def test_verbatim_identical_candidates_agree_without_judge(self):
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=ENSEMBLE_TOML)
        tr = EnsembleTransport(b_same=set(L10N_KEYS := ["menu_search",
                                                        "playback_error",
                                                        "items_count"]))
        self._run(project, tr)
        # identical pairs: no judge calls at all
        self.assertEqual([e for e in tr.events if e[1] == "ensemble_judge"], [])
        self.assertEqual(load_semantic(project, "ru")["entries"]
                         ["menu_search"]["verdict"], "ok")

    def test_disagreement_winner_written_as_arbitrated(self):
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=ENSEMBLE_TOML)
        tr = EnsembleTransport(verdicts={"menu_search": "c2"})
        results, out = self._run(project, tr)
        locale = load_locale(tmp, "ru")
        self.assertEqual(locale["menu_search"]["text"], "B(menu_search)")
        self.assertEqual(locale["menu_search"]["model"], ENS)
        entries = load_semantic(project, "ru")["entries"]
        self.assertEqual(entries["menu_search"]["verdict"], "arbitrated")
        self.assertNotIn("menu_search", verified_pairs(tmp, "ru"))
        self.assertEqual(results[0].arbitrated, 1)
        self.assertIn("ARBITRATED menu_search", out)
        # the other keys agreed -> trusted
        self.assertEqual(entries["items_count"]["verdict"], "ok")

    def test_suspect_entries_swept_only_with_backlog_flag(self):
        # empty delta + a cached suspect: a PLAIN translate must not touch it
        # (incremental runs pay only for their delta); the explicit backlog
        # sweep (--fix-semantic path) re-resolves it
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=ENSEMBLE_TOML)
        entries = {k: {"text": f"A({k})" + _PH.get(k, ""),
                       "src": text_hash(e.text), "model": LOCAL_MODEL}
                   for k, e in project.strings.items()}
        save_locale(tmp, "ru", entries, list(project.strings))
        save_semantic(project, "ru", {"version": 1, "language": "ru", "entries": {
            k: {"src": _pair_hash(project.strings[k].text, entries[k]["text"]),
                "verdict": "suspect" if k == "menu_search" else "ok",
                "back": ""} for k in entries}})
        # plain incremental translate: ZERO LLM calls (delta is empty)
        tr = EnsembleTransport(verdicts={"menu_search": "c2"})
        with redirect_stdout(io.StringIO()):
            run_translate(project, ["ru"], transport=tr)
        self.assertEqual([e for e in tr.events if e[1] == "translate"], [])
        # explicit backlog sweep re-resolves the suspect
        tr2 = EnsembleTransport(verdicts={"menu_search": "c2"})
        with redirect_stdout(io.StringIO()):
            run_translate(project, ["ru"], transport=tr2, include_backlog=True)
        b_keys = {k for m, kind, ks in tr2.events for k in ks
                  if kind == "translate" and m == ENS}
        self.assertEqual(b_keys, {"menu_search"})  # only the suspect was due
        self.assertEqual(load_locale(tmp, "ru")["menu_search"]["text"],
                         "B(menu_search)")

    def test_interrupt_after_phase_a_leaves_unverified_pending(self):
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=ENSEMBLE_TOML)
        tr = EnsembleTransport()

        def hook(model, kind, keys):
            if kind == "translate" and model == LOCAL_MODEL:
                engine._cancel.set()  # interrupted right after phase A work

        tr.on_call = hook
        engine._cancel.clear()
        try:
            from lloc.engine import _ensemble_run
            with redirect_stdout(io.StringIO()):
                _ensemble_run([(project, "ru", list(project.strings))], transport=tr)
        finally:
            engine._cancel.clear()
        locale = load_locale(tmp, "ru")
        self.assertEqual(len(locale), 3)  # phase A saved
        # ...but unverified: no verdicts, hence NOT in the few-shot pool
        self.assertEqual(load_semantic(project, "ru")["entries"], {})
        batch = [project.strings["menu_search"]]
        self.assertEqual(select_examples(project, batch, locale, lang="ru"), [])

    def test_parallel_workers_keep_phase_barriers(self):
        # cloud profile: languages run in parallel INSIDE a phase, but phases
        # remain strict barriers (all A calls before any B call)
        tmp = make_tmpdir(self)
        toml = ENSEMBLE_TOML.replace("workers = 1", "workers = 4")
        project = make_project(tmp, l10n=toml)
        tr = EnsembleTransport()
        results, _ = self._run(project, tr, langs=("ru", "de"))
        models_seq = [m for m, _k, _ in tr.events]
        first_ens = models_seq.index(ENS)
        self.assertNotIn(LOCAL_MODEL, models_seq[first_ens:])
        self.assertEqual(sum(r.translated for r in results), 6)
        for lang in ("ru", "de"):
            self.assertEqual(len(load_locale(tmp, lang)), 3)
            self.assertEqual(len(load_semantic(project, lang)["entries"]), 3)

    def test_ensemble_weak_counts_lost_arbitrations_not_disagreements(self):
        from .test_weak_models import BIG_STRINGS
        # all 10 keys disagree and ALL are lost to the ensemble model -> weak
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=ENSEMBLE_TOML, strings=BIG_STRINGS)
        tr = EnsembleTransport(verdicts={f"key_{i:02d}": "c2" for i in range(20)})
        results, out = self._run(project, tr)
        self.assertTrue(results[0].weak)
        self.assertIn("of arbitrations lost to", out)
        self.assertIn(f'[models] "ru" = "{ENS}"', out)
        # ...but disagreements WON by phase A are not weakness
        tmp2 = make_tmpdir(self)
        project2 = make_project(tmp2, l10n=ENSEMBLE_TOML, strings=BIG_STRINGS)
        tr2 = EnsembleTransport(verdicts={f"key_{i:02d}": "c1" for i in range(20)})
        results2, out2 = self._run(project2, tr2)
        self.assertFalse(results2[0].weak)
        self.assertEqual(results2[0].arbitrated, 20)  # 100% disagreement, no warning
        self.assertNotIn("looks weak", out2)

    def test_backlog_rejudging_never_triggers_weak_warning(self):
        """Live misfire: 6 fresh strings + a big arbitrated backlog -> bogus
        'model looks weak' on 31 languages. Backlog re-judging must not feed
        the weak stats — only fresh phase-A pairs count."""
        from .test_weak_models import BIG_STRINGS
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=ENSEMBLE_TOML, strings=BIG_STRINGS)
        # locale fully translated, all 20 entries carry an ARBITRATED verdict
        # (the backlog); plan delta is EMPTY
        entries = {k: {"text": f"A({k})", "src": text_hash(e.text),
                       "model": LOCAL_MODEL}
                   for k, e in project.strings.items()}
        save_locale(tmp, "ru", entries, list(project.strings))
        save_semantic(project, "ru", {"version": 1, "language": "ru", "entries": {
            k: {"src": _pair_hash(project.strings[k].text, entries[k]["text"]),
                "verdict": "arbitrated", "back": ""} for k in entries}})
        # judge hands EVERY backlog pair to the B model -> 100% c2
        # (explicit sweep: a plain translate would not touch the backlog)
        tr = EnsembleTransport(verdicts={k: "c2" for k in project.strings})
        with redirect_stdout(io.StringIO()) as out_io:
            results = run_translate(project, ["ru"], transport=tr,
                                    include_backlog=True)
        out = out_io.getvalue()
        self.assertEqual(results[0].arbitrated, 20)  # backlog was re-resolved
        self.assertFalse(results[0].weak)            # ...but it is NOT weakness
        self.assertNotIn("looks weak", out)

    def test_models_pin_overrides_phase_a(self):
        tmp = make_tmpdir(self)
        toml = ENSEMBLE_TOML + '\n[models]\n"ru" = "pinned-a"\n'
        project = make_project(tmp, l10n=toml)
        tr = EnsembleTransport()
        self._run(project, tr)
        a_models = {m for m, kind, ks in tr.events if kind == "translate" and
                    any(not k.startswith("B") for k in ks)}
        self.assertIn("pinned-a", {m for m, kind, _ in tr.events
                                   if kind == "translate"})
        self.assertEqual(load_locale(tmp, "ru")["menu_search"]["model"],
                         "pinned-a")  # agree keeps A's entry with its model


CLOUD_A, CLOUD_B, CLOUD_J = "a397-cloud", "b-flash:cloud", "judge:cloud"

CLOUD_TOML = L10N_TOML.replace(
    "batch_size = 3", f'model = "{CLOUD_A}"\nbatch_size = 3').replace(
    'semantic = "off"',
    f'semantic = "batch"\nensemble_model = "{CLOUD_B}"\n'
    f'semantic_model = "{CLOUD_J}"')


class CloudEnsembleTransport(EnsembleTransport):
    """Routes translate calls by explicit A/B model names."""

    def __call__(self, url, payload, timeout):
        model = payload["model"]
        keys = list(payload["format"].get("required", []))
        first = payload["messages"][0]["content"]
        if "comparing TWO independent translations" in first:
            kind, out = "ensemble_judge", {k: self.verdicts.get(k, "agree")
                                           for k in keys}
        elif "literal translator" in first:
            kind, out = "back", {k: f"back({k})" for k in keys}
        else:
            kind = "translate"
            prefix = "B" if model == CLOUD_B else "A"
            out = {}
            for k in keys:
                text = f"A({k})" if (prefix == "B" and k in self.b_same) \
                    else f"{prefix}({k})"
                out[k] = text + _PH.get(k, "")
        with getattr(self, "_lock", threading.Lock()):
            self.events.append((model, kind, keys))
        if self.on_call:
            self.on_call(model, kind, keys)
        return {"message": {"content": json.dumps(out, ensure_ascii=False)}}


import threading  # noqa: E402


class TestConcurrentBatchEnsemble(unittest.TestCase):
    def _run(self, project, tr, langs=("ru",)):
        with redirect_stdout(io.StringIO()) as out:
            results = run_translate(project, list(langs), transport=tr)
        return results, out.getvalue()

    def test_auto_detection(self):
        # all-cloud trio -> concurrent; local trio -> phased; overrides win
        tmp = make_tmpdir(self)
        self.assertTrue(make_project(tmp, l10n=CLOUD_TOML).ensemble_batch)
        tmp2 = make_tmpdir(self)
        self.assertFalse(make_project(tmp2, l10n=ENSEMBLE_TOML).ensemble_batch)
        tmp3 = make_tmpdir(self)
        toml = ENSEMBLE_TOML.replace(
            'semantic = "batch"',
            'semantic = "batch"\nensemble_concurrency = "batch"')
        self.assertTrue(make_project(tmp3, l10n=toml).ensemble_batch)  # local, explicit
        tmp4 = make_tmpdir(self)
        toml = CLOUD_TOML.replace(
            'semantic = "batch"',
            'semantic = "batch"\nensemble_concurrency = "phased"')
        self.assertFalse(make_project(tmp4, l10n=toml).ensemble_batch)
        tmp5 = make_tmpdir(self)
        bad = CLOUD_TOML.replace('semantic = "batch"',
                                 'semantic = "batch"\nensemble_concurrency = "warp"')
        from lloc.config import ConfigError
        with self.assertRaisesRegex(ConfigError, "ensemble_concurrency"):
            make_project(tmp5, l10n=bad)

    def test_a_and_b_translate_concurrently_then_judge(self):
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=CLOUD_TOML)
        tr = CloudEnsembleTransport()
        barrier = threading.Barrier(2)

        def hook(model, kind, keys):
            if kind == "translate":
                # A and B must be in flight at the same time — a sequential
                # implementation deadlocks here (barrier timeout -> error)
                barrier.wait(timeout=10)

        tr.on_call = hook
        results, out = self._run(project, tr)
        self.assertIsNone(results[0].error)
        self.assertEqual(results[0].translated, 3)
        self.assertIn("A+B", out)
        self.assertNotIn("phase A", out)  # no phases in concurrent mode
        # judge ran after the pair, verdicts recorded
        self.assertEqual(load_semantic(project, "ru")["entries"]
                         ["menu_search"]["verdict"], "ok")
        self.assertIn("menu_search", verified_pairs(tmp, "ru"))

    def test_disagreement_and_agree_semantics(self):
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=CLOUD_TOML)
        tr = CloudEnsembleTransport(verdicts={"menu_search": "c2"})
        results, out = self._run(project, tr)
        locale = load_locale(tmp, "ru")
        self.assertEqual(locale["menu_search"]["text"], "B(menu_search)")
        self.assertEqual(locale["menu_search"]["model"], CLOUD_B)
        self.assertEqual(load_semantic(project, "ru")["entries"]
                         ["menu_search"]["verdict"], "arbitrated")
        self.assertEqual(load_semantic(project, "ru")["entries"]
                         ["items_count"]["verdict"], "ok")
        self.assertEqual(results[0].arbitrated, 1)

    def test_third_party_judge_uses_direct_path_no_backtranslations(self):
        # judge (CLOUD_J) produced neither candidate -> ONE direct call per
        # batch, zero back-translation calls
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=CLOUD_TOML)
        tr = CloudEnsembleTransport(verdicts={"menu_search": "c2"})
        results, out = self._run(project, tr)
        kinds = [kind for _m, kind, _k in tr.events]
        self.assertNotIn("back", kinds)                    # no back-translations
        self.assertEqual(kinds.count("ensemble_judge"), 1)  # one call per batch
        # verdicts fully honored on the direct path
        self.assertEqual(load_locale(tmp, "ru")["menu_search"]["text"],
                         "B(menu_search)")
        # progress log with verdict counts
        self.assertRegex(out, r"C 1/1: agree 2, c1 0, c2 1 in \d+\.\ds")

    def test_self_judge_keeps_backtranslation_path(self):
        # judge == ensemble model (phased local profile): back-translations
        # remain mandatory — the judge must not "remember" its own translation
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=ENSEMBLE_TOML)  # judge = ENS = B model
        tr = EnsembleTransport(verdicts={"menu_search": "c1"})
        with redirect_stdout(io.StringIO()) as out:
            run_translate(project, ["ru"], transport=tr)
        kinds = [kind for _m, kind, _k in tr.events]
        self.assertIn("back", kinds)
        self.assertRegex(out.getvalue(), r"C \d+/\d+: agree \d+, c1 \d+, c2 \d+")

    def test_interrupt_writes_no_unjudged_pairs(self):
        tmp = make_tmpdir(self)
        toml = CLOUD_TOML.replace("batch_size = 3", "batch_size = 1")
        project = make_project(tmp, l10n=toml)
        tr = CloudEnsembleTransport()

        def hook(model, kind, keys):
            if kind == "ensemble_judge":
                engine._cancel.set()  # cancel right after the FIRST verdict

        tr.on_call = hook
        engine._cancel.clear()
        try:
            from lloc.engine import _ensemble_batch_language
            with redirect_stdout(io.StringIO()):
                result = _ensemble_batch_language(project, "ru",
                                                  list(project.strings), tr)
        finally:
            engine._cancel.clear()
        # batch 1 fully judged and written; later batches left untouched
        self.assertEqual(len(load_locale(tmp, "ru")), 1)
        self.assertEqual(len(load_semantic(project, "ru")["entries"]), 1)
        self.assertEqual(result.translated, 1)


if __name__ == "__main__":
    unittest.main()
