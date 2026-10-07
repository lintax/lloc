"""Global-audit blocker fixes: fenced-JSON tolerance, batch-skip instead of
crash, English stopwords in term extraction, capped term-warning output."""
import io
import json
import unittest
from contextlib import redirect_stdout

from lloc.ollama import OllamaBadResponse, OllamaClient, extract_json
from lloc.semantic import build_semantic, load_semantic
from lloc.store import save_locale
from lloc.hashes import text_hash

from .helpers import make_project, make_tmpdir


class TestFencedJson(unittest.TestCase):
    def test_extract_json_variants(self):
        obj = '{"k": "v"}'
        self.assertEqual(extract_json(obj), obj)
        self.assertEqual(json.loads(extract_json(f"```json\n{obj}\n```")), {"k": "v"})
        self.assertEqual(json.loads(extract_json(f"```\n{obj}\n```")), {"k": "v"})
        self.assertEqual(json.loads(extract_json(f"Here you go:\n{obj}\nHope it helps!")),
                         {"k": "v"})

    def test_client_parses_fenced_reply(self):
        # the live gpt-oss:120b failure mode: JSON inside markdown fences
        def transport(url, payload, timeout):
            keys = payload["format"]["required"]
            body = json.dumps({k: "ok" for k in keys})
            return {"message": {"content": f"```json\n{body}\n```"}}

        p = make_project(make_tmpdir(self))
        client = OllamaClient(p.ollama, transport=transport)
        parsed, _ = client.chat([{"role": "user", "content": "x"}],
                                {"required": ["a"], "properties": {}})
        self.assertEqual(parsed, {"a": "ok"})


class TestAuditBatchSkip(unittest.TestCase):
    def test_unusable_judge_reply_skips_batch_not_crashes(self):
        tmp = make_tmpdir(self)
        project = make_project(tmp)
        entries = {k: {"text": f"[ru]{e.text}", "src": text_hash(e.text)}
                   for k, e in project.strings.items()}
        save_locale(tmp, "ru", entries, list(project.strings))
        calls = {"n": 0}

        def garbage(url, payload, timeout):
            calls["n"] += 1
            return {"message": {"content": "utter nonsense, no json here"}}

        client = OllamaClient(project.ollama, transport=garbage)
        out = io.StringIO()
        with redirect_stdout(out):
            data = build_semantic(project, "ru", entries, client)  # no crash
        self.assertIn("skipping a semantic batch", out.getvalue())
        self.assertEqual(data["entries"], {})          # nothing recorded
        self.assertGreaterEqual(calls["n"], 2)          # retried before skipping
        self.assertEqual(load_semantic(project, "ru")["entries"], {})


class TestTermNoise(unittest.TestCase):
    def test_english_function_words_are_not_terms(self):
        from lloc.terms import extract_terms
        strings = "\n".join(
            f'[k{i}]\ntext = "Turn off the alarm and check your {w} project"\n'
            for i, w in enumerate(["settings", "backup", "sync", "export"]))
        tmp = make_tmpdir(self)
        project = make_project(tmp, strings=strings)
        terms = extract_terms(project)
        for noise in ("the", "and", "your", "off"):
            self.assertNotIn(noise, terms)
        self.assertIn("alarm", terms)  # real terms survive
        self.assertIn("project", terms)

    def test_verify_caps_term_warnings(self):
        from lloc.__main__ import main
        # 12 strings sharing the term "kanaal" with wildly divergent
        # translations -> more than the cap of heuristic warnings
        strings = "\n".join(
            f'[k{i:02d}]\ntext = "Open project number {i} now quickly"\n'
            for i in range(12))
        tmp = make_tmpdir(self)
        project = make_project(tmp, strings=strings)
        entries = {f"k{i:02d}": {"text": f"Wort{i} übersetzung{i} nummer",
                                 "src": text_hash(f"Open project number {i} now quickly")}
                   for i in range(12)}
        save_locale(tmp, "de", entries, list(project.strings))
        out = io.StringIO()
        with redirect_stdout(out):
            rc = main(["--dir", str(tmp), "verify", "--lang", "de"])
        text = out.getvalue()
        warn_lines = [ln for ln in text.splitlines()
                      if "possibly divergent" in ln]
        if warn_lines:  # heuristic fired: output must be capped
            self.assertLessEqual(len(warn_lines), 8)
        if "more term warning(s)" in text:
            self.assertIn("--verbose", text)
        # --verbose shows everything
        out2 = io.StringIO()
        with redirect_stdout(out2):
            main(["--dir", str(tmp), "verify", "--lang", "de", "--verbose"])
        self.assertNotIn("more term warning(s)", out2.getvalue())


if __name__ == "__main__":
    unittest.main()
