"""Batches of one module run in parallel when they are independent (no inline
QA), never more requests in flight than `workers` across the whole run."""
import io
import threading
import time
import unittest
from contextlib import redirect_stdout

from unittest import mock

from lloc import engine
from lloc.engine import run_translate, run_translate_jobs
from lloc.ollama import OllamaError
from lloc.semantic import load_semantic
from lloc.store import load_locale

from .helpers import L10N_TOML, MockTransport, make_project, make_tmpdir

STRINGS = "".join(f'[s{i}]\ntext = "String number {i}"\n\n' for i in range(12))


class Tracking(MockTransport):
    """MockTransport that holds every request a little and records how many
    were in flight at once."""

    def __init__(self, fail_at: int | None = None):
        super().__init__()
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.fail_at = fail_at

    def __call__(self, url, payload, timeout):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            n = len(self.calls) + 1
            self.calls.append(payload)
        try:
            time.sleep(0.05)
            if self.fail_at is not None and n == self.fail_at:
                raise OllamaError("connection to Ollama lost")
            self.calls.pop()  # MockTransport records it again
            return super().__call__(url, payload, timeout)
        finally:
            with self.lock:
                self.active -= 1


def project(tmp, workers: int, semantic: str = "off"):
    l10n = (L10N_TOML.replace("workers = 1", f"workers = {workers}")
            .replace("batch_size = 3", "batch_size = 1")
            .replace('semantic = "off"', f'semantic = "{semantic}"'))
    return make_project(tmp, l10n=l10n, strings=STRINGS)


class TestParallelBatches(unittest.TestCase):
    def test_batches_of_one_module_run_in_parallel(self):
        tmp = make_tmpdir(self)
        p = project(tmp, workers=4)
        transport = Tracking()
        with redirect_stdout(io.StringIO()):
            results = run_translate(p, ["ru"], transport=transport)
        self.assertEqual(results[0].translated, 12)
        self.assertEqual(len(load_locale(tmp, "ru")), 12)
        self.assertGreater(transport.peak, 1)
        self.assertLessEqual(transport.peak, 4)

    def test_workers_cap_requests_across_modules(self):
        tmp_a, tmp_b = make_tmpdir(self), make_tmpdir(self)
        a, b = project(tmp_a, workers=3), project(tmp_b, workers=3)
        transport = Tracking()
        jobs = [(a, "ru", list(a.strings)), (b, "ru", list(b.strings))]
        with redirect_stdout(io.StringIO()):
            run_translate_jobs(jobs, workers=3, transport=transport)
        self.assertEqual(len(load_locale(tmp_a, "ru")), 12)
        self.assertEqual(len(load_locale(tmp_b, "ru")), 12)
        self.assertLessEqual(transport.peak, 3)

    def test_one_worker_stays_sequential(self):
        tmp = make_tmpdir(self)
        transport = Tracking()
        with redirect_stdout(io.StringIO()):
            run_translate(project(tmp, workers=1), ["ru"], transport=transport)
        self.assertEqual(transport.peak, 1)

    def test_fatal_error_starts_no_more_batches(self):
        tmp = make_tmpdir(self)
        transport = Tracking(fail_at=3)
        out = io.StringIO()
        with redirect_stdout(out):
            results = run_translate(project(tmp, workers=2), ["ru"], transport=transport)
        self.assertIn("connection to Ollama lost", results[0].error)
        # the batches in flight finish and are kept, the rest stay pending
        self.assertLess(len(transport.calls), 12)
        self.assertEqual(len(load_locale(tmp, "ru")), results[0].translated)
        self.assertIn("FATAL", out.getvalue())


class TimedEnsemble:
    """Wraps the ensemble mock transport: holds every call a little and
    records the interval of each phase-A translate call and its prompt."""

    def __init__(self, fail_key: str | None = None):
        from .test_ensemble import CloudEnsembleTransport
        self.inner = CloudEnsembleTransport()
        self.lock = threading.Lock()
        self.a_calls: list[tuple[float, float, tuple]] = []
        self.prompts: dict[tuple, dict[str, str]] = {}
        self.active_a = 0
        self.peak_a = 0
        self.fail_key = fail_key

    def __call__(self, url, payload, timeout):
        from .test_ensemble import CLOUD_A
        first = payload["messages"][0]["content"]
        keys = tuple(payload["format"].get("required", []))
        is_a = payload["model"] == CLOUD_A and "comparing TWO" not in first
        if is_a or "comparing TWO" not in first:
            with self.lock:
                self.prompts.setdefault(keys, {})[payload["model"]] = first
        start = time.monotonic()
        if is_a:
            with self.lock:
                self.active_a += 1
                self.peak_a = max(self.peak_a, self.active_a)
        try:
            time.sleep(0.03)
            if is_a and self.fail_key in keys:
                raise OllamaError("connection to Ollama lost")
            return self.inner(url, payload, timeout)
        finally:
            if is_a:
                with self.lock:
                    self.active_a -= 1
                    self.a_calls.append((start, time.monotonic(), keys))


def ensemble_project(tmp, workers: int):
    from .test_ensemble import CLOUD_TOML
    l10n = (CLOUD_TOML.replace("workers = 1", f"workers = {workers}")
            .replace("batch_size = 3", "batch_size = 1"))
    return make_project(tmp, l10n=l10n, strings=STRINGS)


class TestParallelEnsembleBatches(unittest.TestCase):
    def run_ensemble(self, p, tr, anchors=()):
        with mock.patch.object(engine, "_anchors", lambda _p, _k: list(anchors)), \
                redirect_stdout(io.StringIO()) as out:
            results = run_translate(p, ["ru"], transport=tr)
        return results[0], out.getvalue()

    def test_batches_run_in_a_window_without_lost_verdicts(self):
        tmp = make_tmpdir(self)
        p = ensemble_project(tmp, workers=4)
        tr = TimedEnsemble()
        result, _ = self.run_ensemble(p, tr)
        self.assertIsNone(result.error)
        self.assertEqual(result.translated, 12)
        self.assertGreater(tr.peak_a, 1)
        self.assertLessEqual(tr.peak_a, 4)
        # every batch's verdicts survived the parallel writes
        self.assertEqual(len(load_locale(tmp, "ru")), 12)
        entries = load_semantic(p, "ru")["entries"]
        self.assertEqual(len(entries), 12)
        self.assertTrue(all(e["verdict"] == "ok" for e in entries.values()))

    def test_prompt_is_built_once_for_a_and_b(self):
        from .test_ensemble import CLOUD_A, CLOUD_B
        tr = TimedEnsemble()
        with mock.patch.object(engine, "build_translate_prompt",
                               wraps=engine.build_translate_prompt) as build:
            self.run_ensemble(ensemble_project(make_tmpdir(self), workers=4), tr)
        self.assertEqual(build.call_count, 12)  # one per batch, not two
        pairs = [v for v in tr.prompts.values() if CLOUD_A in v and CLOUD_B in v]
        self.assertEqual(len(pairs), 12)
        for v in pairs:
            self.assertEqual(v[CLOUD_A], v[CLOUD_B])

    def test_cold_start_runs_anchor_batches_one_by_one(self):
        tr = TimedEnsemble()
        p = ensemble_project(make_tmpdir(self), workers=4)
        self.run_ensemble(p, tr, anchors=["s0", "s1"])
        calls = sorted(tr.a_calls)
        self.assertEqual([c[2] for c in calls[:2]], [("s0",), ("s1",)])
        # the second anchor starts after the first ends, the window after both
        self.assertLessEqual(calls[0][1], calls[1][0])
        self.assertLessEqual(calls[1][1], min(c[0] for c in calls[2:]))
        self.assertGreater(tr.peak_a, 1)

    def test_fatal_error_keeps_judged_batches_and_starts_no_more(self):
        tmp = make_tmpdir(self)
        p = ensemble_project(tmp, workers=2)
        tr = TimedEnsemble(fail_key="s3")
        result, out = self.run_ensemble(p, tr)
        self.assertIn("connection to Ollama lost", result.error)
        self.assertEqual(out.count("FATAL"), 1)
        self.assertLess(len(tr.a_calls), 12)
        # nothing unjudged: every written key has its verdict
        written = set(load_locale(tmp, "ru"))
        self.assertNotIn("s3", written)
        self.assertEqual(written, set(load_semantic(p, "ru")["entries"]))


class Looping(MockTransport):
    """A model that never finishes a reply containing one of `bad` keys,
    or any reply at all when `bad` is None."""

    def __init__(self, bad=None):
        super().__init__()
        self.bad = bad
        self.timeouts = 0
        self.waits: list[float] = []

    def __call__(self, url, payload, timeout):
        keys = payload["format"]["required"]
        if self.bad is None or set(keys) & set(self.bad):
            self.calls.append(payload)
            self.timeouts += 1
            self.waits.append(timeout)
            raise OllamaError(f"Ollama request timed out after {timeout}s",
                              timed_out=True)
        return super().__call__(url, payload, timeout)


def batch_project(tmp, batch_size: int, workers: int = 1):
    l10n = (L10N_TOML.replace("batch_size = 3", f"batch_size = {batch_size}")
            .replace("workers = 1", f"workers = {workers}"))
    return make_project(tmp, l10n=l10n, strings=STRINGS)


class TestTimedOutBatchIsSplit(unittest.TestCase):
    def test_one_looping_string_takes_no_others_down(self):
        tmp = make_tmpdir(self)
        transport = Looping(bad=["s5"])
        with redirect_stdout(io.StringIO()) as out:
            results = run_translate(batch_project(tmp, 12), ["ru"], transport=transport)
        r = results[0]
        self.assertIsNone(r.error)
        self.assertEqual(r.translated, 11)
        self.assertEqual(list(r.failed), ["s5"])
        self.assertNotIn("s5", load_locale(tmp, "ru"))
        self.assertIn("splitting 12 strings", out.getvalue())
        # halving: 12 -> 6 -> 3 -> 2 -> 1, one timeout per level, each part
        # waited for in proportion to its size, never under 30 s
        self.assertEqual(transport.timeouts, 5)
        self.assertEqual(transport.waits, [120, 60, 30, 30, 30])

    def test_server_down_is_still_fatal(self):
        tmp = make_tmpdir(self)
        transport = Looping(bad=None)
        with redirect_stdout(io.StringIO()):
            results = run_translate(batch_project(tmp, 12), ["ru"], transport=transport)
        self.assertIn("timed out", results[0].error)
        self.assertEqual(len(load_locale(tmp, "ru")), 0)
        self.assertLessEqual(len(transport.calls), 4)  # the breaker: max(4, 2 * workers)


if __name__ == "__main__":
    unittest.main()
