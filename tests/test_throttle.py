"""HTTP 429 handling: exponential backoff instead of FATAL + the global
adaptive concurrency limiter."""
import json
import unittest
import unittest.mock

from lloc import ollama as ollama_mod
from lloc.ollama import AdaptiveLimiter, OllamaClient, OllamaError

from .helpers import make_project, make_tmpdir


class TestBackoffOn429(unittest.TestCase):
    def setUp(self):
        # fresh limiter per test; sleeps recorded, not slept
        ollama_mod._limiter = AdaptiveLimiter()
        self.sleeps: list[float] = []
        patcher = unittest.mock.patch.object(
            ollama_mod.time, "sleep", side_effect=self.sleeps.append)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.project = make_project(make_tmpdir(self))

    def _client(self, transport):
        return OllamaClient(self.project.ollama, transport=transport)

    def test_429_retried_then_succeeds_no_fatal(self):
        calls = {"n": 0}

        def transport(url, payload, timeout):
            calls["n"] += 1
            if calls["n"] <= 3:
                raise OllamaError("Ollama HTTP 429: too many concurrent requests",
                                  status=429)
            return {"message": {"content": json.dumps({"k": "v"})}}

        parsed, _ = self._client(transport).chat(
            [{"role": "user", "content": "x"}], {"required": ["k"], "properties": {}})
        self.assertEqual(parsed, {"k": "v"})
        self.assertEqual(calls["n"], 4)
        self.assertEqual(len(self.sleeps), 3)
        # exponential-ish: each wait no smaller than the previous base
        self.assertLess(self.sleeps[0], self.sleeps[2])

    def test_retry_after_header_honoured(self):
        calls = {"n": 0}

        def transport(url, payload, timeout):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OllamaError("Ollama HTTP 429: slow down", status=429,
                                  retry_after=7.5)
            return {"message": {"content": json.dumps({"k": "v"})}}

        self._client(transport).chat([{"role": "user", "content": "x"}],
                                     {"required": ["k"], "properties": {}})
        self.assertEqual(self.sleeps, [7.5])

    def test_persistent_429_finally_raises(self):
        def transport(url, payload, timeout):
            raise OllamaError("Ollama HTTP 429: nope", status=429)

        with self.assertRaises(OllamaError):
            self._client(transport).chat([{"role": "user", "content": "x"}],
                                         {"required": ["k"], "properties": {}})
        self.assertEqual(len(self.sleeps), ollama_mod.BACKOFF_TRIES - 1)

    def test_non_retryable_error_not_retried(self):
        # 5xx and timeouts are retried too (tests/test_transient.py); a
        # client error is not
        calls = {"n": 0}

        def transport(url, payload, timeout):
            calls["n"] += 1
            raise OllamaError("Ollama HTTP 400: bad request", status=400)

        with self.assertRaises(OllamaError):
            self._client(transport).chat([{"role": "user", "content": "x"}],
                                         {"required": ["k"], "properties": {}})
        self.assertEqual(calls["n"], 1)
        self.assertEqual(self.sleeps, [])


class TestAdaptiveLimiter(unittest.TestCase):
    def test_shrink_and_recover(self):
        lim = AdaptiveLimiter()
        self.assertEqual(lim.limit, AdaptiveLimiter.CEILING)
        lim.on_throttle()
        self.assertEqual(lim.limit, int(AdaptiveLimiter.CEILING * 0.7))
        for _ in range(10):
            lim.on_throttle()
        self.assertEqual(lim.limit, AdaptiveLimiter.MIN)  # floor holds
        # recovery: +1 per RECOVERY_STREAK consecutive successes
        for _ in range(AdaptiveLimiter.RECOVERY_STREAK):
            lim.on_success()
        self.assertEqual(lim.limit, AdaptiveLimiter.MIN + 1)
        # a throttle resets the streak
        for _ in range(AdaptiveLimiter.RECOVERY_STREAK - 1):
            lim.on_success()
        lim.on_throttle()
        for _ in range(AdaptiveLimiter.RECOVERY_STREAK - 1):
            lim.on_success()
        self.assertEqual(lim.limit, max(AdaptiveLimiter.MIN,
                                        int((AdaptiveLimiter.MIN + 1) * 0.7)))

    def test_limiter_gates_concurrency(self):
        import threading
        lim = AdaptiveLimiter()
        lim._limit = 2
        active = []
        peak = []
        lock = threading.Lock()

        def job():
            with lim:
                with lock:
                    active.append(1)
                    peak.append(len(active))
                import time as _t
                _t.sleep(0.01)
                with lock:
                    active.pop()

        threads = [threading.Thread(target=job) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertLessEqual(max(peak), 2)


if __name__ == "__main__":
    unittest.main()
