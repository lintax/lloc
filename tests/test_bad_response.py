"""Empty/invalid model replies: think:false, num_ctx, retry instead of FATAL."""
import io
import json
import unittest
from contextlib import redirect_stdout

from lloc.engine import run_translate, translate_batch
from lloc.ollama import OllamaBadResponse, OllamaClient, OllamaError, string_schema

from .helpers import MockTransport, make_project, make_tmpdir


def _ok(keys, lang="ru"):
    return {"message": {"content": json.dumps(
        {k: f"[{lang}]tr" if "%" not in k else k for k in keys})}}


class TestPayloadOptions(unittest.TestCase):
    def setUp(self):
        self.project = make_project(make_tmpdir(self))

    def test_think_false_and_num_ctx_sent(self):
        transport = MockTransport()
        client = OllamaClient(self.project.ollama, transport=transport)
        client.chat([{"role": "user", "content": "TASK: x to ru.\nkey: k\ntext: t"}],
                    string_schema(["k"]))
        payload = transport.calls[0]
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["options"]["num_ctx"], 8192)  # built-in default

    def test_glm_gets_think_level_not_false(self):
        # glm ignores think:false and leaks its reasoning into content
        from lloc.ollama import _think_value
        self.assertEqual(_think_value("glm-5.3-flash:cloud"), "low")
        self.assertIs(_think_value("deepseek-v4.1-flash:cloud"), False)

    def test_num_ctx_configurable(self):
        from .helpers import L10N_TOML
        toml = L10N_TOML.replace("[ollama]", "[ollama]\nnum_ctx = 16384")
        project = make_project(make_tmpdir(self), l10n=toml)
        self.assertEqual(project.ollama.num_ctx, 16384)

    def test_think_unsupported_fallback(self):
        calls = []

        def transport(url, payload, timeout):
            calls.append(payload)
            if "think" in payload:
                return {"error": 'registry.ollama.ai/library/x does not support thinking'}
            return {"message": {"content": json.dumps({"k": "v"})}}
        client = OllamaClient(self.project.ollama, transport=transport)
        parsed, _ = client.chat([{"role": "user", "content": "x"}], string_schema(["k"]))
        self.assertEqual(parsed, {"k": "v"})
        self.assertEqual(len(calls), 2)
        self.assertNotIn("think", calls[1])
        # subsequent requests skip `think` immediately
        client.chat([{"role": "user", "content": "x"}], string_schema(["k"]))
        self.assertNotIn("think", calls[2])

    def test_empty_content_raises_bad_response(self):
        def empty(url, payload, timeout):
            return {"message": {"content": "", "thinking": "hmm, let me think"}}
        client = OllamaClient(self.project.ollama, transport=empty)
        with self.assertRaisesRegex(OllamaBadResponse, "thinking"):
            client.chat([{"role": "user", "content": "x"}], string_schema(["k"]))


class FlakyTransport:
    """Returns empty content N times, then behaves like MockTransport."""

    def __init__(self, empty_times: int):
        self.empty_times = empty_times
        self.inner = MockTransport()
        self.calls = []

    def __call__(self, url, payload, timeout):
        self.calls.append(payload)
        if self.empty_times > 0:
            self.empty_times -= 1
            return {"message": {"content": "", "thinking": "..."}}
        return self.inner(url, payload, timeout)


class TestRetryOnBadResponse(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmpdir(self)
        self.project = make_project(self.tmp)  # max_retries = 1

    def test_empty_reply_retried_then_ok(self):
        transport = FlakyTransport(empty_times=1)
        client = OllamaClient(self.project.ollama, transport=transport)
        accepted, failed, _arb = translate_batch(
            self.project, client, "ru", list(self.project.strings), {})
        self.assertEqual(failed, {})
        self.assertEqual(len(accepted), 3)
        self.assertEqual(len(transport.calls), 2)
        # retry conversation carries the error explanation
        retry_msgs = transport.calls[1]["messages"]
        self.assertEqual([m["role"] for m in retry_msgs], ["user", "assistant", "user"])
        self.assertIn("unusable", retry_msgs[2]["content"])

    def test_persistent_empty_fails_keys_not_language(self):
        transport = FlakyTransport(empty_times=99)
        out = io.StringIO()
        with redirect_stdout(out):
            results = run_translate(self.project, ["ru"], transport=transport)
        r = results[0]
        self.assertIsNone(r.error)                       # NOT fatal
        self.assertEqual(r.translated, 0)
        self.assertEqual(set(r.failed), set(self.project.strings))
        self.assertNotIn("FATAL", out.getvalue())
        self.assertEqual(len(transport.calls), 2)        # max_retries=1 -> 2 attempts

    def test_transport_error_still_fatal(self):
        def down(url, payload, timeout):
            raise OllamaError("cannot reach Ollama at http://x: refused")
        out = io.StringIO()
        with redirect_stdout(out):
            results = run_translate(self.project, ["ru"], transport=down)
        self.assertIsNotNone(results[0].error)
        self.assertIn("FATAL", out.getvalue())


if __name__ == "__main__":
    unittest.main()
