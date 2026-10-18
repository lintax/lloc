"""Batches of one module run in parallel when they are independent (no inline
QA), never more requests in flight than `workers` across the whole run."""
import io
import threading
import time
import unittest
from contextlib import redirect_stdout

from lloc.engine import run_translate, run_translate_jobs
from lloc.ollama import OllamaError
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


if __name__ == "__main__":
    unittest.main()
