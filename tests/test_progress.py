"""Progress output: batch start/finish lines, model-load note, heartbeat."""
import io
import re
import time
import unittest
from contextlib import redirect_stdout

from lloc.engine import _heartbeat, run_translate

from .helpers import MockTransport, make_project, make_tmpdir


class TestProgressOutput(unittest.TestCase):
    def test_batch_start_finish_and_model_note(self):
        project = make_project(make_tmpdir(self))
        out = io.StringIO()
        with redirect_stdout(out):
            run_translate(project, ["ru"], transport=MockTransport())
        from lloc.config import DEFAULT_MODEL
        lines = out.getvalue().splitlines()
        self.assertIn(f"loading model '{DEFAULT_MODEL}' into memory, "
                      "first response may take a while...", lines)
        self.assertEqual(1, sum("loading model" in l for l in lines))  # printed once
        self.assertIn("[ru] batch 1/1: translating 3 strings...", lines)
        # finish line reports count and elapsed time
        self.assertTrue(any(re.fullmatch(r"\[ru\] batch 1/1: \+3 in \d+\.\ds", l) for l in lines),
                        lines)
        # start line comes before finish line
        start = lines.index("[ru] batch 1/1: translating 3 strings...")
        finish = next(i for i, l in enumerate(lines) if l.startswith("[ru] batch 1/1: +3"))
        self.assertLess(start, finish)

    def test_heartbeat_fires_on_slow_request(self):
        out = io.StringIO()
        with redirect_stdout(out):
            with _heartbeat("[xx] batch 1/1", interval=0.02):
                time.sleep(0.09)
        text = out.getvalue()
        self.assertIn("[xx] batch 1/1: still working,", text)

    def test_heartbeat_silent_on_fast_request(self):
        out = io.StringIO()
        with redirect_stdout(out):
            with _heartbeat("[xx] batch 1/1", interval=1.0):
                pass
        self.assertEqual(out.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
