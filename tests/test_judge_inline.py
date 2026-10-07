"""Concurrent inline judge: small judge model runs in parallel with the
main translator ([validation] semantic_mode = "inline")."""
import io
import json
import threading
import unittest
from contextlib import redirect_stdout

from lloc import engine
from lloc.engine import run_translate
from lloc.semantic import load_semantic
from lloc.store import load_locale

from .helpers import L10N_TOML, NO_ENSEMBLE, make_project, make_tmpdir

JUDGE = "tiny-judge"


def judged_toml(batch_size=1):
    return L10N_TOML.replace(
        'semantic = "off"',
        f'semantic = "batch"\n{NO_ENSEMBLE}\nsemantic_model = "{JUDGE}"\n'
    'semantic_mode = "inline"'
    ).replace("batch_size = 3", f"batch_size = {batch_size}")


class TwoModelTransport:
    """Routes by payload['model']: the main model translates, the judge model
    answers back-translation + verdict prompts. Thread-safe event log."""

    def __init__(self, bad=(), always_bad=False):
        self.bad = set(bad)
        self.always_bad = always_bad
        self.events = []          # (model, kind, keys)
        self.lock = threading.Lock()
        self.on_call = None       # optional hook(model, kind, keys)

    def __call__(self, url, payload, timeout):
        model = payload["model"]
        keys = payload["format"]["required"]
        first = payload["messages"][0]["content"]
        if model == JUDGE and "FORCED CHOICE" in first:
            kind = "arbitrate"
            out = {"choice": "c2", "reason": "slightly closer in meaning"}
        elif model == JUDGE:
            if "literal translator" in first:
                kind = "back"
                out = {k: (f"wrong-meaning({k})" if k in self.bad else f"back({k})")
                       for k in keys}
            else:
                kind = "judge"
                out = {k: ("meaning drifted" if k in self.bad else "ok") for k in keys}
        else:
            kind = "translate"
            retry = len(payload["messages"]) > 1 or "QA judge flagged" in first
            out = {}
            ph = {"open_error": " %1$s", "items_count": " %d"}
            self.attempt = getattr(self, "attempt", 0) + 1
            for k in keys:
                if k in self.bad and retry and not self.always_bad:
                    self.bad.discard(k)
                # persistent-bad texts differ per attempt -> real candidates
                text = f"tr({k})" if k not in self.bad else f"BAD{self.attempt}({k})"
                out[k] = text + ph.get(k, "")
        with self.lock:
            self.events.append((model, kind, list(keys)))
        if self.on_call:
            self.on_call(model, kind, keys)
        return {"message": {"content": json.dumps(out, ensure_ascii=False)}}


class TestConcurrentJudge(unittest.TestCase):
    def _run(self, project, tr, langs=("ru",)):
        with redirect_stdout(io.StringIO()):
            return run_translate(project, list(langs), transport=tr)

    def test_all_ok_written_and_cached(self):
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=judged_toml())
        tr = TwoModelTransport()
        results = self._run(project, tr)
        self.assertEqual(results[0].failed, {})
        self.assertEqual(results[0].translated, 3)
        locale = load_locale(tmp, "ru")
        self.assertEqual(len(locale), 3)
        self.assertEqual(locale["menu_search"]["model"], project.ollama.model)
        # cache marked verified
        self.assertEqual(load_semantic(project, "ru")["entries"]
                         ["menu_search"]["verdict"], "ok")
        # models used: main for translate, judge for back/judge only
        for model, kind, _ in tr.events:
            self.assertEqual(model == JUDGE, kind in ("back", "judge"))

    def test_pipeline_is_concurrent(self):
        """The judge of batch N-1 runs WHILE batch N is being translated:
        the judge blocks until translate of batch 2 starts — a sequential
        implementation would deadlock (guarded by a timeout)."""
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=judged_toml(batch_size=1))
        tr = TwoModelTransport()
        second_translate_started = threading.Event()
        translate_count = {"n": 0}

        def hook(model, kind, keys):
            if kind == "translate":
                translate_count["n"] += 1
                if translate_count["n"] >= 2:
                    second_translate_started.set()
            if model == JUDGE and kind == "back":
                # judge waits for proof that the main model moved on
                assert second_translate_started.wait(timeout=10), \
                    "pipeline is sequential: judge blocked the next batch"

        tr.on_call = hook
        results = self._run(project, tr)
        self.assertEqual(results[0].translated, 3)
        self.assertEqual(len(load_locale(tmp, "ru")), 3)

    def test_suspect_retried_by_main_model_with_feedback(self):
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=judged_toml())
        tr = TwoModelTransport(bad={"menu_search"})
        results = self._run(project, tr)
        self.assertEqual(results[0].failed, {})
        self.assertEqual(load_locale(tmp, "ru")["menu_search"]["text"], "tr(menu_search)")
        # the retry went to the MAIN model and carried the judge's feedback
        retries = [(m, k) for m, kind, k in tr.events
                   if kind == "translate" and k == ["menu_search"]]
        self.assertTrue(any(m != JUDGE for m, _ in retries))

    def test_persistent_suspect_arbitrated_forced_choice(self):
        # the model insists, the judge keeps flagging -> the judge makes a
        # FORCED CHOICE among the real candidates; the winner ships
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=judged_toml())
        tr = TwoModelTransport(bad={"menu_search"}, always_bad=True)
        out = io.StringIO()
        with redirect_stdout(out):
            results = run_translate(project, ["ru"], transport=tr)
        self.assertEqual(results[0].failed, {})   # semantic dispute != failure
        # the judge chose c2 = the retry candidate; both were real candidates
        arb_calls = [(m, k) for m, kind, k in tr.events if kind == "arbitrate"]
        self.assertEqual(len(arb_calls), 1)
        self.assertEqual(arb_calls[0][0], JUDGE)  # arbitration by the judge model
        locale = load_locale(tmp, "ru")
        self.assertTrue(locale["menu_search"]["text"].startswith("BAD"))
        self.assertIn("2 candidate(s)", out.getvalue())
        self.assertIn("ARBITRATED menu_search", out.getvalue())
        # arbitrated verdict: excluded from the trusted pool, re-checked later
        from lloc.semantic import verified_pairs
        self.assertEqual(load_semantic(project, "ru")["entries"]
                         ["menu_search"]["verdict"], "arbitrated")
        self.assertNotIn("menu_search", verified_pairs(tmp, "ru"))
        self.assertEqual(len(locale), 3)          # healthy strings all landed

    def test_interrupt_leaves_unverified_pending(self):
        """Cancel between batches: the translated-but-unjudged batch is NOT
        written — it stays pending for the next run."""
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=judged_toml(batch_size=1))
        tr = TwoModelTransport()

        def hook(model, kind, keys):
            if kind == "translate":
                engine._cancel.set()  # "Ctrl+C" right after the first batch

        tr.on_call = hook
        engine._cancel.clear()
        try:
            from lloc.engine import translate_language
            with redirect_stdout(io.StringIO()):
                result = translate_language(project, "ru", list(project.strings),
                                            transport=tr)
        finally:
            engine._cancel.clear()
        self.assertIsNone(result.error)
        self.assertEqual(load_locale(tmp, "ru"), {})  # unverified: not written
        self.assertEqual(result.translated, 0)

    def test_phased_flow_not_broken_by_new_key(self):
        # semantic_mode = "auto" + judge model -> still phased, no inline judge
        tmp = make_tmpdir(self)
        toml = L10N_TOML.replace(
            'semantic = "off"',
            f'semantic = "batch"\n{NO_ENSEMBLE}\nsemantic_model = "big-judge"')
        project = make_project(tmp, l10n=toml)
        self.assertTrue(project.semantic_phased)
        self.assertFalse(project.semantic_inline_judge)
        tr = TwoModelTransport()
        results = self._run(project, tr)  # plain translate, no judge calls
        self.assertEqual(results[0].translated, 3)
        self.assertEqual({kind for _, kind, _ in tr.events}, {"translate"})

    def test_bad_semantic_mode_is_config_error(self):
        from lloc.config import ConfigError
        tmp = make_tmpdir(self)
        toml = L10N_TOML.replace('semantic = "off"',
                                 'semantic = "batch"\nsemantic_mode = "turbo"')
        with self.assertRaisesRegex(ConfigError, "semantic_mode"):
            make_project(tmp, l10n=toml)


if __name__ == "__main__":
    unittest.main()
