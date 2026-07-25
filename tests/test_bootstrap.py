import io
import unittest
from contextlib import redirect_stdout, redirect_stderr

from lloc.__main__ import main
from lloc.config import load_project

from .helpers import L10N_TOML, make_tmpdir

BASE_XML = """<resources>
    <string name="app_key" translatable="false">com.example.app</string>
    <string name="menu_search">Search</string>
    <string name="quoted">Don\\'t stop</string>
    <string name="with_ph">Open %1$s now</string>
    <string-array name="modes">
        <item>Auto</item>
        <item>Manual</item>
    </string-array>
    <plurals name="days">
        <item quantity="other">%d days</item>
    </plurals>
</resources>
"""


class TestBootstrap(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmpdir(self)
        (self.tmp / "l10n.toml").write_text(L10N_TOML, encoding="utf-8")
        base_dir = self.tmp / "res" / "values"
        base_dir.mkdir(parents=True)
        (base_dir / "strings.xml").write_text(BASE_XML, encoding="utf-8")

    def _run(self, *extra):
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(out):
            rc = main(["--dir", str(self.tmp), "import", "--bootstrap-strings", *extra])
        return rc, out.getvalue()

    def test_bootstrap_generates_loadable_strings_toml(self):
        rc, out = self._run()
        self.assertEqual(rc, 0)
        self.assertIn("3 strings", out)
        # 1 translatable="false" + string-array + plurals
        self.assertIn("skipped 3 special/non-translatable", out)

        project = load_project(self.tmp)  # generated file parses as a valid project
        self.assertEqual(list(project.strings), ["menu_search", "quoted", "with_ph"])
        self.assertEqual(project.strings["quoted"].text, "Don't stop")     # unescaped
        self.assertEqual(project.strings["with_ph"].text, "Open %1$s now")
        self.assertNotIn("app_key", project.strings)

    def test_refuses_to_overwrite_without_flag(self):
        self.assertEqual(self._run()[0], 0)
        rc, out = self._run()
        self.assertEqual(rc, 2)
        self.assertIn("--overwrite", out)
        self.assertEqual(self._run("--overwrite")[0], 0)

    def test_missing_base_file_clear_error(self):
        (self.tmp / "res" / "values" / "strings.xml").unlink()
        rc, out = self._run()
        self.assertEqual(rc, 2)
        self.assertIn("base file not found", out)


if __name__ == "__main__":
    unittest.main()
