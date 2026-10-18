"""Timeouts, dropped connections and 5xx: retried like a throttle instead of
killing the run; a refused connection (daemon not running) stays fatal."""
import http.client
import io
import json
import socket
import threading
import unittest
import unittest.mock
import urllib.error

from lloc import ollama as ollama_mod
from lloc.ollama import AdaptiveLimiter, OllamaClient, OllamaError

from .helpers import make_project, make_tmpdir

OK = {"message": {"content": json.dumps({"k": "v"})}}
SCHEMA = {"required": ["k"], "properties": {}}
MSGS = [{"role": "user", "content": "x"}]


class TestTransportMapping(unittest.TestCase):
    """The real urllib transport must turn low-level failures into OllamaError."""

    def _serve(self, handler):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen()
        self.addCleanup(srv.close)
        conns = []
        self.addCleanup(lambda: [c.close() for c in conns])

        def loop():
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            conns.append(conn)
            handler(conn)
        threading.Thread(target=loop, daemon=True).start()
        return f"http://127.0.0.1:{srv.getsockname()[1]}/api/chat"

    def test_read_timeout_is_transient_ollama_error(self):
        url = self._serve(lambda conn: None)  # accept, never answer
        with self.assertRaises(OllamaError) as cm:
            ollama_mod._http_transport(url, {}, 1)
        self.assertTrue(cm.exception.timed_out)
        self.assertTrue(cm.exception.transient)
        self.assertIn("timed out after 1s", str(cm.exception))

    def test_dropped_connection_is_transient(self):
        def drop(conn):
            conn.recv(65536)
            conn.close()
        url = self._serve(drop)
        with self.assertRaises(OllamaError) as cm:
            ollama_mod._http_transport(url, {}, 5)
        self.assertTrue(cm.exception.transient)
        self.assertFalse(cm.exception.timed_out)
        self.assertIn("connection to Ollama lost", str(cm.exception))

    def test_refused_connection_is_fatal(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()  # nothing listens there now
        with self.assertRaises(OllamaError) as cm:
            ollama_mod._http_transport(f"http://127.0.0.1:{port}/api/chat", {}, 2)
        self.assertFalse(cm.exception.transient)
        self.assertIn("cannot reach Ollama", str(cm.exception))

    def test_http_503_keeps_status(self):
        err = urllib.error.HTTPError("http://x", 503, "Service Unavailable",
                                     {}, io.BytesIO(b'{"error": "overloaded"}'))
        with unittest.mock.patch.object(ollama_mod.urllib.request, "urlopen",
                                        side_effect=err):
            with self.assertRaises(OllamaError) as cm:
                ollama_mod._http_transport("http://x/api/chat", {}, 5)
        self.assertEqual(cm.exception.status, 503)


class TestTransientRetry(unittest.TestCase):
    def setUp(self):
        ollama_mod._limiter = AdaptiveLimiter()
        self.sleeps: list[float] = []
        patcher = unittest.mock.patch.object(
            ollama_mod.time, "sleep", side_effect=self.sleeps.append)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.project = make_project(make_tmpdir(self))

    def _chat(self, transport):
        return OllamaClient(self.project.ollama, transport=transport).chat(MSGS, SCHEMA)

    def _scripted(self, errors):
        calls = {"n": 0}

        def transport(url, payload, timeout):
            calls["n"] += 1
            if calls["n"] <= len(errors):
                raise errors[calls["n"] - 1]
            return OK
        return transport, calls

    def test_timeout_retried_then_succeeds(self):
        transport, calls = self._scripted(
            [OllamaError("Ollama request timed out after 600s", timed_out=True)])
        with unittest.mock.patch("builtins.print") as printed:
            parsed, _ = self._chat(transport)
        self.assertEqual(parsed, {"k": "v"})
        self.assertEqual(calls["n"], 2)
        self.assertEqual(len(self.sleeps), 1)
        line = printed.call_args_list[0].args[0]
        self.assertTrue(line.startswith("retrying after Ollama request timed out"), line)
        self.assertTrue(line.endswith("(attempt 2/3)"), line)
        self.assertEqual(ollama_mod._limiter.limit, AdaptiveLimiter.CEILING)

    def test_persistent_timeout_gives_up_after_three_tries(self):
        transport, calls = self._scripted(
            [OllamaError("timed out", timed_out=True)] * 10)
        with unittest.mock.patch("builtins.print"):
            with self.assertRaises(OllamaError):
                self._chat(transport)
        self.assertEqual(calls["n"], ollama_mod.TIMEOUT_TRIES)

    def test_503_retried_then_succeeds_without_shrinking_limit(self):
        transport, calls = self._scripted([
            OllamaError("Ollama HTTP 503: overloaded", status=503),
            OllamaError("Ollama HTTP 502: bad gateway", status=502),
        ])
        with unittest.mock.patch("builtins.print"):
            parsed, _ = self._chat(transport)
        self.assertEqual(parsed, {"k": "v"})
        self.assertEqual(calls["n"], 3)
        self.assertEqual(len(self.sleeps), 2)
        self.assertEqual(ollama_mod._limiter.limit, AdaptiveLimiter.CEILING)

    def test_persistent_5xx_finally_raises(self):
        transport, calls = self._scripted(
            [OllamaError("Ollama HTTP 500: boom", status=500)] * 20)
        with unittest.mock.patch("builtins.print"):
            with self.assertRaises(OllamaError):
                self._chat(transport)
        self.assertEqual(calls["n"], ollama_mod.BACKOFF_TRIES)

    def test_refused_not_retried(self):
        transport, calls = self._scripted(
            [OllamaError("cannot reach Ollama at http://x: refused")])
        with self.assertRaises(OllamaError):
            self._chat(transport)
        self.assertEqual(calls["n"], 1)
        self.assertEqual(self.sleeps, [])

    def test_400_not_retried(self):
        transport, calls = self._scripted(
            [OllamaError("Ollama HTTP 400: bad request", status=400)])
        with self.assertRaises(OllamaError):
            self._chat(transport)
        self.assertEqual(calls["n"], 1)


if __name__ == "__main__":
    unittest.main()
