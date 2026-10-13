"""paradox-yml adapter, Paradox placeholder style and reference modules."""
import io
import json
import unittest
from contextlib import redirect_stdout

from lloc.__main__ import main
from lloc.adapters import get_adapter
from lloc.adapters.paradox_yml import parse, string_type
from lloc.config import load, load_project
from lloc.hashes import text_hash
from lloc.prompts import select_examples
from lloc.store import load_locale
from lloc.validate import check_translation, placeholders

from .helpers import make_tmpdir

L10N = """\
[app]
description = "Demo mod"
base_language = "en"
languages = ["ru"]

[output]
adapter = "paradox-yml"
[output.paradox-yml]
source_dir = "mod"
exclude = ["*consolidated_l_*"]
"""

ENGLISH = """\
﻿l_english:
 # comment
 tech_alpha:0 "Alpha Drive"
 tech_alpha_desc:1 "Moves §Yships§! faster.\\nCosts $COST$ £energy£."
 demo.1.a: "They said \\"no\\"." # trailing comment
 loose_quotes:0 "An "odd" line"
 only_syntax:0 "$tech_alpha$"
 empty:0 ""
 untouched:0 "Left as English by the author"
"""

RUSSIAN = """\
﻿l_russian:
 tech_alpha:0 "Альфа-двигатель"
 untouched:0 "Left as English by the author"
"""

OLD_OUTPUT = '﻿l_russian:\n demo.1.a:0 "Machine output of another tool"\n'


def run(*argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(list(argv))
    return code, buf.getvalue()


class TestParse(unittest.TestCase):
    def test_parse_and_unescape(self):
        rows = {k: (v, t) for k, v, t in parse(ENGLISH)}
        self.assertEqual(rows["tech_alpha"], ("0", "Alpha Drive"))
        self.assertEqual(rows["tech_alpha_desc"],
                         ("1", "Moves §Yships§! faster.\nCosts $COST$ £energy£."))
        self.assertEqual(rows["demo.1.a"], ("", 'They said "no".'))
        self.assertEqual(rows["loose_quotes"][1], 'An "odd" line')

    def test_nothing_before_header(self):
        self.assertEqual(parse(' key:0 "x"\nl_english:\n other:0 "y"\n'),
                         [("other", "0", "y")])

    def test_string_type(self):
        self.assertEqual(string_type("tech_alpha_desc"), "description")
        self.assertEqual(string_type("demo.1.a"), "event option")
        self.assertEqual(string_type("demo.1.name"), "event title")
        self.assertEqual(string_type("building_x_tooltip"), "tooltip")
        self.assertEqual(string_type("tech_alpha"), "")


class TestParadoxPlaceholders(unittest.TestCase):
    def test_tokens(self):
        c = placeholders("§Y$NAME$§! pays £energy£ to [Root.GetName] #bold now#! @icon!",
                         "paradox")
        for token in ("§Y", "§!", "$NAME$", "£energy£", "[Root.GetName]", "#bold", "#!", "@icon!"):
            self.assertEqual(c[token], 1, token)

    def test_text_between_markers_is_translatable(self):
        self.assertEqual(check_translation("Unlocks §YAlloys§!", "Открывает §YСплавы§!",
                                           style="paradox"), [])

    def test_lost_or_translated_syntax_is_rejected(self):
        self.assertTrue(check_translation("Unlocks §YAlloys§!", "Открывает Сплавы",
                                          style="paradox"))
        self.assertTrue(check_translation("Costs $COST$", "Стоит $ЦЕНА$", style="paradox"))

    def test_default_style_ignores_paradox_syntax(self):
        self.assertEqual(check_translation("See [Beta] at $5$", "См. [Бета] за $5$"), [])


class TestParadoxAdapter(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmpdir(self)
        loc = self.tmp / "mod" / "localisation"
        (loc / "english").mkdir(parents=True)
        (loc / "russian").mkdir()
        (loc / "english" / "demo_l_english.yml").write_text(ENGLISH, encoding="utf-8")
        (loc / "russian" / "demo_l_russian.yml").write_text(RUSSIAN, encoding="utf-8")
        (loc / "russian" / "0_1_consolidated_l_russian.yml").write_text(
            OLD_OUTPUT, encoding="utf-8")
        (self.tmp / "l10n.toml").write_text(L10N, encoding="utf-8")
        self.assertEqual(run("--dir", str(self.tmp), "import", "--bootstrap-strings")[0], 0)
        self.project = load_project(self.tmp)
        self.adapter = get_adapter(self.project)

    def test_bootstrap_skips_untranslatable_and_derives_meta(self):
        s = self.project.strings
        self.assertEqual(list(s), ["tech_alpha", "tech_alpha_desc", "demo.1.a",
                                   "loose_quotes", "untouched"])
        self.assertEqual(s["tech_alpha_desc"].type, "description")
        self.assertEqual(s["tech_alpha_desc"].context, "file: demo")
        self.assertEqual(self.project.placeholder_style, "paradox")

    def test_import_drops_english_leftovers_and_excluded_files(self):
        self.assertEqual(self.adapter.import_(), {"ru": {"tech_alpha": "Альфа-двигатель"}})

    def test_export_writes_only_own_translations(self):
        def entry(key, text, model):
            return {"text": text, "src": text_hash(self.project.strings[key].text),
                    "model": model}
        written = self.adapter.export({"ru": {
            "tech_alpha": entry("tech_alpha", "Альфа-двигатель", "imported"),
            "tech_alpha_desc": entry("tech_alpha_desc", "Быстрее.\nЦена: $COST$", "m"),
            "demo.1.a": entry("demo.1.a", 'Они сказали "нет".', "manual"),
        }})
        path = self.tmp / "out" / "russian" / "lloc_l_russian.yml"
        self.assertEqual(written, [str(path)])
        self.assertEqual(path.read_text(encoding="utf-8"),
                         '﻿l_russian:\n'
                         ' demo.1.a: "Они сказали \\"нет\\"."\n'
                         ' tech_alpha_desc:1 "Быстрее.\\nЦена: $COST$"\n')
        # own output is never re-imported, wherever out_dir points
        self.assertEqual(parse(path.read_text(encoding="utf-8"))[1][2],
                         "Быстрее.\nЦена: $COST$")

    def test_export_without_own_translations_removes_file(self):
        path = self.adapter.file_for("ru")
        path.parent.mkdir(parents=True)
        path.write_text("stale", encoding="utf-8")
        self.assertEqual(self.adapter.export({"ru": {}}), [])
        self.assertFalse(path.exists())


class TestReferenceModule(unittest.TestCase):
    def setUp(self):
        tmp = make_tmpdir(self)
        files = {
            "l10n/config.toml": 'members = ["game", "mod"]\n\n[app]\nlanguages = ["ru"]\n'
                                '[validation]\nsemantic = "off"\n',
            "l10n/g.toml": 'context = "A game."\n',
            "game/l10n.toml": 'group = "g"\nreference = true\n[output]\nadapter = "json"\n',
            "game/strings.toml": '[alloys]\ntext = "Alloys"\n'
                                 '[lore]\ntext = "' + "Alloys were forged long ago. " * 8 + '"\n'
                                 '[other]\ntext = "Unity"\n',
            "mod/l10n.toml": 'group = "g"\n[output]\nadapter = "json"\n',
            "mod/strings.toml": '[cost]\ntext = "Alloy cost"\n',
        }
        for rel, content in files.items():
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp / rel).write_text(content, encoding="utf-8")
        self.tmp = tmp
        (tmp / "game" / "locales").mkdir()
        (tmp / "game" / "locales" / "ru.json").write_text(
            '{"version": 1, "language": "ru", "strings": {'
            '"alloys": {"text": "Сплавы", "model": "imported"},'
            '"lore": {"text": "Давно.", "model": "imported"},'
            '"other": {"text": "Единство", "model": "imported"}}}', encoding="utf-8")

    def test_reference_is_not_a_workspace_member(self):
        ws = load(self.tmp)
        self.assertEqual([m.module for m in ws.members], ["mod"])
        self.assertEqual([m.module for m in ws.references], ["game"])
        self.assertTrue(load_project(self.tmp / "game").reference)  # still usable directly
        self.assertEqual(load_locale(self.tmp / "game", "ru")["alloys"]["text"], "Сплавы")

    def test_examples_come_from_short_matching_reference_strings(self):
        mod = load_project(self.tmp / "mod")
        self.assertEqual(mod.siblings, [])  # no part in term statistics
        examples = select_examples(mod, [mod.strings["cost"]], {}, lang="ru")
        self.assertEqual(examples, [("Alloys", "Сплавы")])

    def test_ungrouped_reference_serves_ungrouped_modules(self):
        for name in ("game", "mod"):
            cfg = self.tmp / name / "l10n.toml"
            cfg.write_text(cfg.read_text(encoding="utf-8").replace('group = "g"\n', ""),
                           encoding="utf-8")
        mod = load_project(self.tmp / "mod")
        self.assertEqual((mod.group, [r.module for r in mod.references]), ("mod", ["game"]))
        self.assertEqual(select_examples(mod, [mod.strings["cost"]], {}, lang="ru"),
                         [("Alloys", "Сплавы")])

    def test_established_name_is_looked_up_as_a_whole(self):
        game = self.tmp / "game"
        (game / "strings.toml").write_text(
            '[gov]\ntext = "Stagnant Ascendancy"\n'
            + "".join(f'[s{i}]\ntext = "Stagnant water {i}"\n[a{i}]\ntext = "Ascendancy {i}"\n'
                      for i in range(40)), encoding="utf-8")
        entries = {"gov": "Стагнационный режим",
                   **{f"s{i}": f"Вода {i}" for i in range(40)},
                   **{f"a{i}": f"Возвышение {i}" for i in range(40)}}
        (game / "locales" / "ru.json").write_text(json.dumps({"strings": {
            k: {"text": v, "model": "imported"} for k, v in entries.items()}}),
            encoding="utf-8")
        (self.tmp / "mod" / "strings.toml").write_text(
            '[x]\ntext = "The Stagnant Ascendancies use these devices."\n', encoding="utf-8")
        mod = load_project(self.tmp / "mod")
        examples = select_examples(mod, [mod.strings["x"]], {}, lang="ru")
        self.assertEqual(examples[0], ("Stagnant Ascendancy", "Стагнационный режим"))


if __name__ == "__main__":
    unittest.main()
