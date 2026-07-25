"""Auto-pull of a missing model: 404 -> /api/pull (stream) -> retry."""
import io
import json
import re
import unittest
from contextlib import redirect_stdout

from lloc.config import DEFAULT_MODEL
from lloc.ollama import OllamaClient, OllamaError, string_schema

from .helpers import make_project, make_tmpdir


def _ok_response(keys):
    return {"message": {"content": json.dumps({k: f"tr:{k}" for k in keys})}}


class MissingThenOkTransport:
    """First chat call: 404 model not found. After a pull: success."""

    def __init__(self, model=DEFAULT_MODEL, fail_forever=False):
        self.model = model
        self.fail_forever = fail_forever
        self.chat_calls = 0
        self.pulled = False

    def __call__(self, url, payload, timeout):
        assert url.endswith("/api/chat")
        self.chat_calls += 1
        if not self.pulled or self.fail_forever:
            raise OllamaError(f"Ollama HTTP 404: model '{self.model}' not found", status=404)
        return _ok_response(payload["format"]["required"])


class PullStream:
    def __init__(self, owner, chunks=None, error=None):
        self.owner = owner          # MissingThenOkTransport to flip .pulled
        self.chunks = chunks
        self.error = error
        self.calls = []

    def __call__(self, url, payload, timeout):
        assert url.endswith("/api/pull")
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        self.owner.pulled = True
        yield from self.chunks if self.chunks is not None else [
            {"status": "pulling manifest"},
            {"status": "downloading", "total": 100 * 2**20, "completed": 10 * 2**20},
            {"status": "downloading", "total": 100 * 2**20, "completed": 100 * 2**20},
            {"status": "verifying sha256 digest"},
            {"status": "success"},
        ]


class TestAutoPull(unittest.TestCase):
    def setUp(self):
        self.project = make_project(make_tmpdir(self))
        self.transport = MissingThenOkTransport()

    def _client(self, stream):
        return OllamaClient(self.project.ollama, transport=self.transport, stream_transport=stream)

    def test_pull_then_retry_succeeds(self):
        stream = PullStream(self.transport)
        client = self._client(stream)
        out = io.StringIO()
        with redirect_stdout(out):
            parsed, _ = client.chat([{"role": "user", "content": "x"}], string_schema(["k"]))
        self.assertEqual(parsed, {"k": "tr:k"})
        self.assertEqual(self.transport.chat_calls, 2)          # 404, then success
        self.assertEqual(len(stream.calls), 1)
        self.assertEqual(stream.calls[0], {"model": DEFAULT_MODEL, "stream": True})
        text = out.getvalue()
        self.assertIn("pulling", text)
        self.assertIn("100%", text)                             # progress was rendered
        self.assertIn("is ready", text)

    def test_pull_stream_error_gives_hint(self):
        stream = PullStream(self.transport, chunks=[{"error": "pull model manifest: file does not exist"}])
        client = self._client(stream)
        with redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(OllamaError, "ollama pull " + re.escape(DEFAULT_MODEL)) as ctx:
                client.chat([{"role": "user", "content": "x"}], string_schema(["k"]))
        self.assertIn("file does not exist", str(ctx.exception))

    def test_pull_network_error_gives_hint(self):
        stream = PullStream(self.transport, error=OllamaError("cannot reach Ollama at http://x: refused"))
        client = self._client(stream)
        with redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(OllamaError, "failed to pull model"):
                client.chat([{"role": "user", "content": "x"}], string_schema(["k"]))

    def test_unrelated_404_not_pulled(self):
        def other_404(url, payload, timeout):
            raise OllamaError("Ollama HTTP 404: page not found", status=404)
        stream = PullStream(self.transport)
        client = OllamaClient(self.project.ollama, transport=other_404, stream_transport=stream)
        with self.assertRaises(OllamaError):
            client.chat([{"role": "user", "content": "x"}], string_schema(["k"]))
        self.assertEqual(stream.calls, [])                      # no pull attempted

    def test_error_field_variant_triggers_pull(self):
        # some proxies return 200 with {"error": "model ... not found"}
        transport = self.transport

        def error_field(url, payload, timeout):
            transport.chat_calls += 1
            if not transport.pulled:
                return {"error": f"model '{DEFAULT_MODEL}' not found, try pulling it first"}
            return _ok_response(payload["format"]["required"])
        stream = PullStream(transport)
        client = OllamaClient(self.project.ollama, transport=error_field, stream_transport=stream)
        with redirect_stdout(io.StringIO()):
            parsed, _ = client.chat([{"role": "user", "content": "x"}], string_schema(["k"]))
        self.assertEqual(parsed, {"k": "tr:k"})
        self.assertEqual(len(stream.calls), 1)


if __name__ == "__main__":
    unittest.main()
