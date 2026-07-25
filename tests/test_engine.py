import io
import json
import unittest
import unittest.mock
from contextlib import redirect_stdout

from lloc.engine import run_translate, translate_batch
from lloc.ollama import OllamaClient, OllamaError, string_schema
from lloc.store import classify, load_locale, load_lock

from .helpers import MockTransport, make_project, make_tmpdir


class TestOllamaClient(unittest.TestCase):
    def test_schema(self):
        s = string_schema(["a", "b"])
        self.assertEqual(s["required"], ["a", "b"])
        self.assertEqual(s["properties"]["a"], {"type": "string"})

    def test_chat_parses_json(self):
        transport = MockTransport()
        p = make_project(make_tmpdir(self))
        client = OllamaClient(p.ollama, transport=transport)
        parsed, raw = client.chat([{"role": "user", "content": "TASK: x to ru.\nkey: k\ntext: t"}],
                                  string_schema(["k"]))
        self.assertIn("k", parsed)
        self.assertEqual(json.loads(raw), parsed)
        from lloc.config import DEFAULT_MODEL
        self.assertEqual(transport.calls[0]["model"], DEFAULT_MODEL)
        self.assertFalse(transport.calls[0]["stream"])

    def test_invalid_json_raises(self):
        def bad(url, payload, timeout):
            return {"message": {"content": "not json"}}
        p = make_project(make_tmpdir(self))
        with self.assertRaises(OllamaError):
            OllamaClient(p.ollama, transport=bad).chat([], {})

    def test_error_field_raises(self):
        def err(url, payload, timeout):
            return {"error": "model not found"}
        p = make_project(make_tmpdir(self))
        with self.assertRaisesRegex(OllamaError, "model not found"):
            OllamaClient(p.ollama, transport=err).chat([], {})


class TestTranslateBatch(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmpdir(self)
        self.project = make_project(self.tmp)

    def test_retry_recovers_lost_placeholder(self):
        # first response loses %1$s -> validation error -> retry fixes it
        transport = MockTransport(fail_first={"playback_error": "Плейсхолдер потерян"})
        client = OllamaClient(self.project.ollama, transport=transport)
        accepted, failed = translate_batch(
            self.project, client, "ru", list(self.project.strings), {})
        self.assertEqual(failed, {})
        self.assertEqual(len(accepted), 3)
        self.assertEqual(len(transport.calls), 2)
        # retry conversation includes assistant echo + error text
        retry_msgs = transport.calls[1]["messages"]
        self.assertEqual([m["role"] for m in retry_msgs], ["user", "assistant", "user"])
        self.assertIn("placeholders mismatch", retry_msgs[2]["content"])
        # retry schema narrowed to the failed key only
        self.assertEqual(transport.calls[1]["format"]["required"], ["playback_error"])

    def test_persistent_failure_reported(self):
        transport = MockTransport(fail_first={"items_count": "без числа"})
        transport.fail_first["items_count"] = "без числа"  # keep failing

        def always_bad(lang, key, text):
            return "no placeholder" if key == "items_count" else f"[{lang}]{text}"
        transport.translator = always_bad
        transport.fail_first = {}
        client = OllamaClient(self.project.ollama, transport=transport)
        accepted, failed = translate_batch(
            self.project, client, "ru", list(self.project.strings), {})
        self.assertIn("items_count", failed)
        self.assertEqual(len(accepted), 2)
        self.assertEqual(len(transport.calls), 2)  # max_retries=1 -> 2 attempts


class TestRunTranslate(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmpdir(self)
        self.project = make_project(self.tmp)

    def test_full_run_and_idempotence(self):
        transport = MockTransport()
        results = run_translate(self.project, self.project.languages, transport=transport)
        self.assertEqual(sum(r.translated for r in results), 6)  # 3 strings x 2 langs
        for lang in self.project.languages:
            st = classify(self.project, lang)
            self.assertTrue(st.clean, f"{lang}: {st}")
            locale = load_locale(self.project.root, lang)
            self.assertIn(f"[{lang}]", locale["menu_search"]["text"])
        first_calls = len(transport.calls)

        # second run: no delta -> no requests
        results = run_translate(self.project, self.project.languages, transport=transport)
        self.assertEqual(sum(r.translated for r in results), 0)
        self.assertEqual(len(transport.calls), first_calls)

    def test_stale_only_delta(self):
        transport = MockTransport()
        run_translate(self.project, ["ru"], transport=transport)
        # change one base string -> only it should be retranslated
        strings = (self.tmp / "strings.toml").read_text(encoding="utf-8")
        strings = strings.replace('text = "Search"', 'text = "Find"')
        (self.tmp / "strings.toml").write_text(strings, encoding="utf-8")
        from lloc.config import load_project
        project2 = load_project(self.tmp)
        before = len(transport.calls)
        results = run_translate(project2, ["ru"], transport=transport)
        self.assertEqual(results[0].translated, 1)
        self.assertEqual(len(transport.calls), before + 1)
        self.assertEqual(load_locale(self.tmp, "ru")["menu_search"]["text"], "[ru]Find")

    def test_workers_parallel(self):
        self.project.ollama.workers = 2
        transport = MockTransport()
        results = run_translate(self.project, self.project.languages, transport=transport)
        self.assertEqual(sum(r.translated for r in results), 6)
        self.assertEqual([r.lang for r in results], self.project.languages)  # stable order


class TestCli(unittest.TestCase):
    def test_status_and_lock_via_cli(self):
        from lloc.__main__ import main
        tmp = make_tmpdir(self)
        make_project(tmp)
        self.assertEqual(main(["--dir", str(tmp), "status"]), 1)  # everything missing
        self.assertEqual(main(["--dir", str(tmp), "translate", "--dry-run"]), 0)
        self.assertEqual(load_lock(tmp), {})  # dry-run touches nothing

    def test_flag_order_tolerance(self):
        from lloc.__main__ import build_parser, main
        tmp = make_tmpdir(self)
        make_project(tmp)

        # subcommand flags before the subcommand (the reported failure case)
        args = build_parser().parse_args(
            ["--force", "--dry-run", "--dir", str(tmp), "translate"])
        self.assertEqual(args.command, "translate")
        self.assertTrue(args.force)
        self.assertTrue(args.dry_run)
        self.assertEqual(args.dir, str(tmp))

        # global flags after the subcommand
        args = build_parser().parse_args(
            ["translate", "--dir", str(tmp), "--lang", "ru", "--dry-run", "--limit", "2"])
        self.assertEqual(args.dir, str(tmp))
        self.assertEqual(args.lang, "ru")
        self.assertTrue(args.dry_run)
        self.assertEqual(args.limit, 2)

        # mixed positions, full run path
        self.assertEqual(main(["--dry-run", "--dir", str(tmp), "translate", "--force"]), 0)
        self.assertEqual(main(["translate", "--dry-run", "--dir", str(tmp)]), 0)

        # defaults are filled for flags never mentioned (verbose used by cmd_build)
        self.assertEqual(main(["--dir", str(tmp), "build"]), 0)

    def test_total_time_and_model_override(self):
        from lloc import __main__ as cli
        from lloc.engine import LangResult
        tmp = make_tmpdir(self)
        make_project(tmp)
        seen = {}

        def fake_run_translate(project, langs, force=False, limit=None, transport=None, **kw):
            seen["model"] = project.ollama.model
            return [LangResult(lang=lang, translated=3) for lang in langs]

        with unittest.mock.patch.object(cli, "run_translate", fake_run_translate):
            out = io.StringIO()
            with redirect_stdout(out):
                rc = cli.main(["--dir", str(tmp), "translate",
                               "--model", "gemma4:12b-it-qat", "--no-build"])
        self.assertEqual(rc, 0)
        self.assertEqual(seen["model"], "gemma4:12b-it-qat")  # --model overrides config
        self.assertRegex(
            out.getvalue(),
            r"translated 6 strings across 2 languages in \d+\.\ds \(model gemma4:12b-it-qat\)")


if __name__ == "__main__":
    unittest.main()
