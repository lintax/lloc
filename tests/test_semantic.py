"""Semantic QA: back-translation + judge, cache delta, fix-semantic."""
import io
import json
import unittest
from contextlib import redirect_stdout

from lloc.hashes import text_hash
from lloc.ollama import OllamaClient
from lloc.semantic import (build_semantic, clear_keys, load_semantic,
                           prune_semantic_caches, suspects)
from lloc.store import load_locale, save_locale

from .helpers import NO_ENSEMBLE, make_project, make_tmpdir


class SemanticTransport:
    """Scripted transport: answers back-translation and judge prompts.

    back[key] -> literal back-translation returned in step 1
    verdict[key] -> judge's answer in step 2 ("ok" or a reason)
    """

    def __init__(self, back: dict, verdict: dict):
        self.back = back
        self.verdict = verdict
        self.calls = []

    def __call__(self, url, payload, timeout):
        prompt = payload["messages"][0]["content"]
        keys = payload["format"]["required"]
        if "literal translator" in prompt:
            kind = "back"
            out = {k: self.back.get(k, f"back({k})") for k in keys}
        elif "QA judge" in prompt:
            kind = "judge"
            out = {k: self.verdict.get(k, "ok") for k in keys}
        else:  # regular translate prompt (fix-semantic path)
            kind = "translate"
            out = {k: f"fixed({k})" for k in keys}
        self.calls.append((kind, keys, prompt))
        return {"message": {"content": json.dumps(out, ensure_ascii=False)}}


class TestBuildSemantic(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmpdir(self)
        self.project = make_project(self.tmp)
        self.locale = {
            "menu_search": {"text": "Meklēt", "src": text_hash("Search")},
            # "avīze" = newspaper: the judge must flag it
            "open_error": {"text": "Avīzes kļūda %1$s", "src": "x"},
        }
        save_locale(self.tmp, "ru", self.locale, list(self.project.strings))

    def _client(self, transport):
        return OllamaClient(self.project.ollama, transport=transport)

    def test_backtranslate_judge_and_cache(self):
        tr = SemanticTransport(
            back={"menu_search": "Search", "open_error": "Newspaper error %1$s"},
            verdict={"menu_search": "ok",
                     "open_error": 'the project became a "newspaper"'})
        data = build_semantic(self.project, "ru", self.locale, self._client(tr))
        self.assertEqual(data["entries"]["menu_search"]["verdict"], "ok")
        sus = suspects(data)
        self.assertEqual(len(sus), 1)
        key, reason, back = sus[0]
        self.assertEqual(key, "open_error")
        self.assertIn("newspaper", reason)
        self.assertEqual(back, "Newspaper error %1$s")
        # cache file written
        cached = json.loads(
            (self.tmp / "semantic" / "ru.json").read_text(encoding="utf-8"))
        self.assertEqual(cached["entries"]["open_error"]["verdict"], "suspect")
        # back-translation prompt must NOT leak the key's context
        back_prompts = [p for kind, _, p in tr.calls if kind == "back"]
        self.assertTrue(back_prompts)
        self.assertNotIn("Main menu item", back_prompts[0])
        self.assertNotIn("project name", back_prompts[0])
        # judge prompt gets the context (menu_search skipped the judge: its
        # back-translation matched the original verbatim) and instructs the
        # judge to accept synonyms/paraphrases
        judge_prompts = [p for kind, _, p in tr.calls if kind == "judge"]
        self.assertIn("project name", judge_prompts[0])
        self.assertNotIn("key: menu_search", judge_prompts[0])
        self.assertIn("Synonyms, paraphrases and natural rewording are OK",
                      judge_prompts[0])
        self.assertIn("newspaper", judge_prompts[0])  # wrong-domain-term example

    def test_verbatim_back_translation_skips_judge(self):
        # both back-translations match the originals -> zero judge calls
        tr = SemanticTransport(
            back={"menu_search": "Search",
                  "open_error": "Failed to open project %1$s.  check your connection."},
            verdict={})
        data = build_semantic(self.project, "ru", self.locale, self._client(tr))
        self.assertEqual([k for k, _, _ in tr.calls], ["back"])  # no "judge"
        self.assertEqual(suspects(data), [])
        for key in ("menu_search", "open_error"):
            self.assertEqual(data["entries"][key]["verdict"], "ok")

    def test_rerun_is_delta(self):
        tr = SemanticTransport(back={}, verdict={})
        client = self._client(tr)
        build_semantic(self.project, "ru", self.locale, client)
        first = len(tr.calls)
        data = build_semantic(self.project, "ru", self.locale, client)
        self.assertEqual(len(tr.calls), first)  # nothing changed -> no requests
        # change one translation -> only it is re-checked
        self.locale["menu_search"]["text"] = "Atrast"
        build_semantic(self.project, "ru", self.locale, client)
        kind, keys, _ = tr.calls[first]
        self.assertEqual(keys, ["menu_search"])

    def test_prune_and_clear(self):
        tr = SemanticTransport(back={}, verdict={"menu_search": "bad"})
        build_semantic(self.project, "ru", self.locale, self._client(tr))
        clear_keys(self.project, "ru", ["menu_search"])
        data = load_semantic(self.project, "ru")
        self.assertNotIn("menu_search", data["entries"])
        # orphaned keys are pruned offline
        data["entries"]["ghost_key"] = {"src": "x", "verdict": "suspect"}
        from lloc.semantic import save_semantic
        save_semantic(self.project, "ru", data)
        prune_semantic_caches(self.project)
        self.assertNotIn("ghost_key", load_semantic(self.project, "ru")["entries"])


class TestFixSemantic(unittest.TestCase):
    def test_fix_semantic_retranslates_suspects(self):
        import unittest.mock
        from lloc import __main__ as cli
        from lloc.__main__ import _run_fix_semantic

        tmp = make_tmpdir(self)
        project = make_project(tmp)
        locale = {"menu_search": {"text": "Avīze", "src": text_hash("Search")}}
        save_locale(tmp, "ru", locale, list(project.strings))
        # seed the cache as verify --semantic would
        tr = SemanticTransport(back={"menu_search": "Newspaper"},
                               verdict={"menu_search": "search became a newspaper"})
        build_semantic(project, "ru", locale, OllamaClient(project.ollama, transport=tr))

        fix_tr = SemanticTransport(back={}, verdict={})
        real_init = OllamaClient.__init__

        def patched_init(self_, cfg, transport=None):
            real_init(self_, cfg, transport=fix_tr)

        out = io.StringIO()
        with unittest.mock.patch.object(OllamaClient, "__init__", patched_init), \
             redirect_stdout(out):
            rc = _run_fix_semantic(project, ["ru"])
        self.assertEqual(rc, 0)
        entry = load_locale(tmp, "ru")["menu_search"]
        self.assertEqual(entry["text"], "fixed(menu_search)")
        self.assertEqual(entry["model"], project.ollama.model)
        # the retranslation prompt carried the judge's findings
        kind, keys, prompt = fix_tr.calls[0]
        self.assertEqual(kind, "translate")
        self.assertIn("search became a newspaper", prompt)
        self.assertIn("Newspaper", prompt)
        # fixed key dropped from the cache for the next --semantic pass
        self.assertNotIn("menu_search", load_semantic(project, "ru")["entries"])
        self.assertIn("retranslated 1", out.getvalue())

    def test_no_suspects_noop(self):
        from lloc.__main__ import _run_fix_semantic
        tmp = make_tmpdir(self)
        project = make_project(tmp)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = _run_fix_semantic(project, ["ru"])
        self.assertEqual(rc, 0)
        self.assertIn("no suspects", out.getvalue())


class InlineTransport:
    """Scripted transport for inline-QA tests: translate -> back -> judge.

    `bad_first` keys get a semantically wrong first translation; the judge
    flags it; the retry produces a good one (unless `always_bad`)."""

    def __init__(self, bad_first=(), always_bad=False):
        self.bad = set(bad_first)
        self.always_bad = always_bad
        self.calls = []  # (kind, keys, prompt)

    def __call__(self, url, payload, timeout):
        prompt = payload["messages"][-1]["content"]
        first = payload["messages"][0]["content"]
        keys = payload["format"]["required"]
        if "FORCED CHOICE" in first:
            kind, out = "arbitrate", {"choice": "c1", "reason": "least bad"}
            self.calls.append((kind, keys, first))
            return {"message": {"content": json.dumps(out)}}
        if "literal translator" in first:
            kind, out = "back", {
                k: ("newspaper!" if k in self.bad else f"back-ok({k})") for k in keys}
        elif "QA judge" in first:
            kind, out = "judge", {
                k: ('it says "newspaper"' if k in self.bad else "ok") for k in keys}
        else:
            kind, out = "translate", {}
            retry = len(payload["messages"]) > 1
            # keep the fixture keys' placeholders so validation passes
            ph = {"open_error": " %1$s", "items_count": " %d"}
            for k in keys:
                if k in self.bad and retry and not self.always_bad:
                    self.bad.discard(k)  # the retry fixes it
                text = f"tr({k})" if k not in self.bad else f"BAD({k})"
                out[k] = text + ph.get(k, "")
        self.calls.append((kind, list(keys), prompt))
        return {"message": {"content": json.dumps(out, ensure_ascii=False)}}


def _inline_project(testcase):
    from .helpers import L10N_TOML
    tmp = make_tmpdir(testcase)
    toml = L10N_TOML.replace('semantic = "off"', f'semantic = "batch"\n{NO_ENSEMBLE}')
    return tmp, make_project(tmp, l10n=toml)


class TestInlineSemanticQA(unittest.TestCase):
    def _run(self, project, transport):
        from lloc.engine import run_translate
        with redirect_stdout(io.StringIO()):
            return run_translate(project, ["ru"], transport=transport)

    def test_suspect_bounces_to_retry_with_feedback_then_accepted(self):
        tmp, project = _inline_project(self)
        tr = InlineTransport(bad_first={"menu_search"})
        results = self._run(project, tr)
        self.assertEqual(results[0].failed, {})
        self.assertEqual(load_locale(tmp, "ru")["menu_search"]["text"], "tr(menu_search)")
        kinds = [k for k, _, _ in tr.calls]
        self.assertEqual(kinds, ["translate", "back", "judge",
                                 "translate", "back", "judge"])
        # the retry schema narrowed to the suspect and carried judge feedback
        _, retry_keys, retry_prompt = tr.calls[3]
        self.assertEqual(retry_keys, ["menu_search"])
        self.assertIn("meaning drift", retry_prompt)
        self.assertIn("newspaper", retry_prompt)

    def test_persistent_suspect_arbitrated_never_failed(self):
        # semantic disputes auto-resolve: after retries the judge
        # makes a forced choice; the winner ships, nothing blocks
        tmp, project = _inline_project(self)
        tr = InlineTransport(bad_first={"menu_search"}, always_bad=True)
        out = io.StringIO()
        with redirect_stdout(out):
            from lloc.engine import run_translate
            results = run_translate(project, ["ru"], transport=tr)
        self.assertEqual(results[0].failed, {})   # not a failure
        locale = load_locale(tmp, "ru")
        self.assertEqual(locale["menu_search"]["text"], "BAD(menu_search)")
        self.assertEqual(len(locale), 3)
        self.assertIn("ARBITRATED menu_search", out.getvalue())
        # verdict "arbitrated": not trusted for few-shot...
        from lloc.semantic import verified_pairs
        data = load_semantic(project, "ru")
        self.assertEqual(data["entries"]["menu_search"]["verdict"], "arbitrated")
        self.assertNotIn("menu_search", verified_pairs(tmp, "ru"))
        # ...and re-checked by the next semantic pass despite a matching hash
        audit = SemanticTransport(back={}, verdict={})
        build_semantic(project, "ru", load_locale(tmp, "ru"),
                       OllamaClient(project.ollama, transport=audit))
        self.assertTrue(any(kind == "back" and "menu_search" in keys
                            for kind, keys, _ in audit.calls))

    def test_broken_placeholders_still_fail_mechanically(self):
        # a candidate losing %1$s must never ship, arbitration or not
        tmp, project = _inline_project(self)
        good = InlineTransport()

        def transport(url, payload, timeout):
            resp = good(url, payload, timeout)
            kind, keys, _ = good.calls[-1]
            if kind == "translate" and "open_error" in keys:
                out = json.loads(resp["message"]["content"])
                out["open_error"] = "потерял плейсхолдер навсегда"
                resp = {"message": {"content": json.dumps(out, ensure_ascii=False)}}
            return resp

        with redirect_stdout(io.StringIO()):
            from lloc.engine import run_translate
            results = run_translate(project, ["ru"], transport=transport)
        self.assertIn("open_error", results[0].failed)
        self.assertTrue(any("placeholders mismatch" in e
                            for e in results[0].failed["open_error"]))
        self.assertNotIn("open_error", load_locale(tmp, "ru"))

    def test_accepted_batches_marked_verified_in_cache(self):
        tmp, project = _inline_project(self)
        tr = InlineTransport()
        self._run(project, tr)
        data = load_semantic(project, "ru")
        self.assertEqual(data["entries"]["menu_search"]["verdict"], "ok")
        # a later audit pass re-checks nothing for unchanged pairs
        audit = SemanticTransport(back={}, verdict={})
        locale = load_locale(tmp, "ru")
        build_semantic(project, "ru", locale, OllamaClient(project.ollama, transport=audit))
        self.assertEqual(audit.calls, [])

    def test_semantic_off_disables_qa(self):
        from .helpers import L10N_TOML
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=L10N_TOML)  # semantic = "off"
        tr = InlineTransport()
        self._run(project, tr)
        self.assertEqual({k for k, _, _ in tr.calls}, {"translate"})
        self.assertEqual(len(load_locale(tmp, "ru")), 3)


class TestPhasedJudgeModel(unittest.TestCase):
    def test_phased_check_uses_judge_model_then_fix_uses_main(self):
        import unittest.mock
        from .helpers import L10N_TOML
        from lloc.__main__ import _phased_semantic, _phased_semantic_check, _run_fix_semantic

        tmp = make_tmpdir(self)
        toml = L10N_TOML.replace(
            'semantic = "off"',
            f'semantic = "batch"\n{NO_ENSEMBLE}\nsemantic_model = "qwen3.5:9b"')
        project = make_project(tmp, l10n=toml)
        self.assertTrue(_phased_semantic(project))
        # inline QA must be OFF when a judge model is configured
        from lloc.engine import run_translate
        tr = InlineTransport()
        with redirect_stdout(io.StringIO()):
            run_translate(project, ["ru"], transport=tr)
        self.assertEqual({k for k, _, _ in tr.calls}, {"translate"})

        # phase 2 (judge model) + phase 3 (main model): track models used
        models_used = []
        judge_tr = SemanticTransport(back={}, verdict={"menu_search": "wrong thing"})
        fix_tr = SemanticTransport(back={}, verdict={})
        real_init = OllamaClient.__init__

        def patched_init(self_, cfg, transport=None):
            models_used.append(cfg.model)
            real_init(self_, cfg,
                      transport=judge_tr if cfg.model == "qwen3.5:9b" else fix_tr)

        out = io.StringIO()
        with unittest.mock.patch.object(OllamaClient, "__init__", patched_init), \
             redirect_stdout(out):
            sus = _phased_semantic_check(project, ["ru"])
            self.assertEqual(sus, ["ru"])
            rc = _run_fix_semantic(project, sus)
        self.assertEqual(rc, 0)
        self.assertEqual(models_used, ["qwen3.5:9b", project.ollama.model])
        # one judge sweep, then the main model fixed the suspect
        self.assertEqual(load_locale(tmp, "ru")["menu_search"]["text"],
                         "fixed(menu_search)")
        self.assertIn("1 suspect", out.getvalue())


class TestParallelSemanticAudit(unittest.TestCase):
    def test_languages_audited_in_parallel_and_all_complete(self):
        import threading
        import unittest.mock
        from lloc import __main__ as cli
        from .helpers import L10N_TOML

        tmp = make_tmpdir(self)
        toml = L10N_TOML.replace("workers = 1", "workers = 2")
        project = make_project(tmp, l10n=toml)
        for lang in ("ru", "de"):
            entries = {k: {"text": f"[{lang}]{e.text}", "src": text_hash(e.text)}
                       for k, e in project.strings.items()}
            save_locale(tmp, lang, entries, list(project.strings))

        barrier = threading.Barrier(2)

        def transport(url, payload, timeout):
            first = payload["messages"][0]["content"]
            keys = payload["format"]["required"]
            if "literal translator" in first:
                # both languages' back-translation calls must be in flight
                # simultaneously — a sequential audit deadlocks here
                barrier.wait(timeout=10)
                out = {k: f"back({k})" for k in keys}
            else:
                out = {k: "ok" for k in keys}
            return {"message": {"content": json.dumps(out)}}

        real_init = OllamaClient.__init__

        def patched(self_, cfg, transport_=None, stream_transport=None):
            real_init(self_, cfg, transport=transport)

        out = io.StringIO()
        with unittest.mock.patch.object(OllamaClient, "__init__", patched), \
             redirect_stdout(out):
            rc = cli.main(["--dir", str(tmp), "verify", "--semantic"])
        self.assertEqual(rc, 0)
        for lang in ("ru", "de"):  # both languages fully audited and cached
            entries = load_semantic(project, lang)["entries"]
            self.assertEqual(len(entries), 3, lang)
            self.assertTrue(all(e["verdict"] == "ok" for e in entries.values()))


class TestLengthFloor(unittest.TestCase):
    def test_short_base_long_script_not_flagged(self):
        from lloc.validate import check_translation
        # ne case from the live run: "Exhale — 8s" (11 chars), Devanagari
        # translation legitimately ~3.5x longer
        self.assertEqual(
            check_translation("Exhale — 8s", "सास बाहिर फाल्नुहोस् — ८ सेकेन्ड"), [])
        # the ratio check still exists for genuinely long output
        errs = check_translation("Exhale — 8s", "x" * 60)
        self.assertTrue(any("too long" in e for e in errs))


if __name__ == "__main__":
    unittest.main()
