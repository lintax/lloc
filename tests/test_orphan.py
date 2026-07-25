"""End-to-end orphan cleanup: deleting a string from strings.toml + one
translate run removes it from locales/*.json, l10n.lock.json, terms/<lang>.json
and (via the automatic build) from every generated XML — while preserved
special keys in the XML stay untouched (SPEC.md §4.4, §8.1, §13.3)."""
import io
import json
import unittest
from contextlib import redirect_stdout

from lloc.__main__ import main
from lloc.config import load_project
from lloc.hashes import text_hash
from lloc.store import load_locale, load_lock, save_locale

from .helpers import L10N_TOML, make_tmpdir

STRINGS_V1 = """\
[channel_open]
text = "Open the channel"
[channel_list]
text = "All the channels"
[channel_lock]
text = "The channel is locked"
[obsolete]
text = "Old channel banner"
"""
STRINGS_V2 = "\n".join(STRINGS_V1.splitlines()[:6]) + "\n"  # [obsolete] removed


class TestOrphanCleanup(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmpdir(self)
        (self.tmp / "l10n.toml").write_text(
            L10N_TOML.replace('languages = ["ru", "de"]', 'languages = ["ru"]'),
            encoding="utf-8")
        (self.tmp / "strings.toml").write_text(STRINGS_V1, encoding="utf-8")
        project = load_project(self.tmp)
        self.assertIn("obsolete", project.strings)

        # fully translated locale (so translate has zero delta -> offline run)
        entries = {k: {"text": f"ru:{e.text}", "src": text_hash(e.text)}
                   for k, e in project.strings.items()}
        save_locale(self.tmp, "ru", entries, list(project.strings))

        # terms cache: live term with the doomed key + a fully dead term
        (self.tmp / "terms").mkdir()
        (self.tmp / "terms" / "ru.json").write_text(json.dumps({
            "version": 1, "language": "ru",
            "terms": {
                "channel": {k: {"src": "x", "tr": "канал"}
                            for k in ("channel_open", "channel_lock", "obsolete")},
                "deadterm": {"obsolete": {"src": "x", "tr": "старый"}},
            },
            "accepted": {"deadterm": ["старый", "древний"]},
        }, ensure_ascii=False), encoding="utf-8")

        # XML with the key present + a special block that must survive
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--dir", str(self.tmp), "translate"]), 0)
        xml_path = self.tmp / "res" / "values-b+ru" / "strings.xml"
        xml = xml_path.read_text(encoding="utf-8")
        self.assertIn("obsolete", xml)
        xml_path.write_text(xml.replace(
            "</resources>",
            '    <string name="app_key" translatable="false">keep.me</string>\n'
            "</resources>"), encoding="utf-8")

    def test_deleted_key_vanishes_everywhere(self):
        (self.tmp / "strings.toml").write_text(STRINGS_V2, encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            rc = main(["--dir", str(self.tmp), "translate"])  # no delta: no LLM calls
        self.assertEqual(rc, 0)

        self.assertNotIn("obsolete", load_locale(self.tmp, "ru"))      # locale
        self.assertNotIn("obsolete", load_lock(self.tmp))              # lock

        data = json.loads((self.tmp / "terms" / "ru.json").read_text(encoding="utf-8"))
        self.assertNotIn("obsolete", data["terms"]["channel"])         # key pruned
        self.assertNotIn("deadterm", data["terms"])                    # dead term pruned
        self.assertNotIn("deadterm", data["accepted"])                 # accepted pruned

        for d in ("values", "values-b+ru"):
            xml = (self.tmp / "res" / d / "strings.xml").read_text(encoding="utf-8")
            self.assertNotIn("obsolete", xml, d)                       # XML rebuilt
        ru_xml = (self.tmp / "res" / "values-b+ru" / "strings.xml").read_text(encoding="utf-8")
        self.assertIn('name="app_key"', ru_xml)                        # special key kept


if __name__ == "__main__":
    unittest.main()
