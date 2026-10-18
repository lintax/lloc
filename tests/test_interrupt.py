"""Ctrl+C during translate: no hang, progress saved, exit 130."""
import io
import sys
import time
import unittest
import unittest.mock
from contextlib import redirect_stderr, redirect_stdout

from lloc import engine
from lloc.engine import TranslateInterrupted, run_translate, run_translate_jobs
from lloc.store import load_locale

from .helpers import MockTransport, make_project, make_tmpdir


class TestSequentialInterrupt(unittest.TestCase):
    def test_interrupt_between_batches_keeps_saved_progress(self):
        tmp = make_tmpdir(self)
        project = make_project(tmp)
        project.ollama.batch_size = 1  # 3 strings -> 3 batches
        good = MockTransport()
        calls = {"n": 0}

        def transport(url, payload, timeout):
            calls["n"] += 1
            if calls["n"] >= 2:  # Ctrl+C in the middle of the second request
                raise KeyboardInterrupt
            return good(url, payload, timeout)

        out = io.StringIO()
        with redirect_stdout(out):
            with self.assertRaises(TranslateInterrupted):
                run_translate(project, ["ru"], transport=transport)
        locale = load_locale(tmp, "ru")
        self.assertEqual(list(locale), ["menu_search"])  # batch 1 written, rest not
        self.assertTrue(engine._cancel.is_set())

    def test_rerun_after_interrupt_continues(self):
        tmp = make_tmpdir(self)
        project = make_project(tmp)
        project.ollama.batch_size = 1
        flaky = {"n": 0}
        good = MockTransport()

        def transport(url, payload, timeout):
            flaky["n"] += 1
            if flaky["n"] == 2:
                raise KeyboardInterrupt
            return good(url, payload, timeout)

        with redirect_stdout(io.StringIO()):
            with self.assertRaises(TranslateInterrupted):
                run_translate(project, ["ru"], transport=transport)
            results = run_translate(project, ["ru"], transport=good)  # clean rerun
        self.assertEqual(results[0].translated, 2)  # only the remaining delta
        self.assertEqual(len(load_locale(tmp, "ru")), 3)


class TestPoolInterrupt(unittest.TestCase):
    def test_no_hang_when_a_worker_is_stuck(self):
        import threading

        tmp = make_tmpdir(self)
        project = make_project(tmp)
        project.ollama.timeout_sec = 600  # a "long HTTP request"
        state = {"first": True}
        stuck_done = threading.Event()

        def transport(url, payload, timeout):
            if state["first"]:
                state["first"] = False
                raise KeyboardInterrupt  # SIGINT surfaces via fut.result()
            # the other worker sits in a "request" until cancellation,
            # then aborts quietly (no writes after the interrupt)
            try:
                engine._cancel.wait(timeout=10)
                raise KeyboardInterrupt
            finally:
                stuck_done.set()

        jobs = [(project, lang, list(project.strings)) for lang in project.languages]
        started = time.monotonic()
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(TranslateInterrupted):
                run_translate_jobs(jobs, workers=2, transport=transport)
        self.assertLess(time.monotonic() - started, 5)  # returned immediately
        self.assertTrue(engine._cancel.is_set())        # stuck worker was unblocked
        self.assertTrue(stuck_done.wait(timeout=5))     # ... and exited its request
        time.sleep(0.05)  # let the worker unwind before tmpdir cleanup


class TestCliInterrupt(unittest.TestCase):
    def test_exit_130_message_and_released_lock(self):
        from lloc import __main__ as cli
        from lloc.runlock import module_lock

        tmp = make_tmpdir(self)
        make_project(tmp)
        out, err = io.StringIO(), io.StringIO()
        with unittest.mock.patch.object(cli, "run_translate",
                                        side_effect=TranslateInterrupted), \
             unittest.mock.patch("os._exit") as fake_exit, \
             redirect_stdout(out), redirect_stderr(err):
            rc = cli.main(["--dir", str(tmp), "translate", "--no-build"])
        fake_exit.assert_called_once_with(130)
        self.assertEqual(rc, 130)
        self.assertIn("interrupted — progress saved, rerun to continue", err.getvalue())
        with module_lock(tmp):  # module lock was released on unwind
            pass

    def test_bare_keyboard_interrupt_handled_too(self):
        from lloc import __main__ as cli

        tmp = make_tmpdir(self)
        make_project(tmp)
        err = io.StringIO()
        with unittest.mock.patch.object(cli, "run_translate",
                                        side_effect=KeyboardInterrupt), \
             unittest.mock.patch("os._exit") as fake_exit, \
             redirect_stdout(io.StringIO()), redirect_stderr(err):
            rc = cli.main(["--dir", str(tmp), "translate", "--no-build"])
        fake_exit.assert_called_once_with(130)
        self.assertEqual(rc, 130)

    def test_exits_even_when_output_pipe_is_closed(self):
        # a closed GUI leaves lloc writing into a pipe nobody reads: the
        # message fails with BrokenPipeError, the exit must still happen
        from lloc import __main__ as cli

        class Broken(io.StringIO):
            def write(self, _s):
                raise BrokenPipeError(32, "Broken pipe")

            def flush(self):
                raise BrokenPipeError(32, "Broken pipe")

        with unittest.mock.patch("os._exit") as fake_exit, \
             unittest.mock.patch.object(sys, "stderr", Broken()), \
             unittest.mock.patch.object(sys, "stdout", Broken()):
            cli._interrupted()
        fake_exit.assert_called_once_with(130)


if __name__ == "__main__":
    unittest.main()
