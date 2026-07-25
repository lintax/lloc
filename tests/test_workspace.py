"""Workspace mode for monorepos (SPEC.md §14): discovery, inheritance,
groups, group-scoped terms/few-shot, duplicate keys, aggregated CLI."""
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

from lloc.__main__ import main
from lloc.config import ConfigError, Project, Workspace, load, load_project
from lloc.hashes import text_hash
from lloc.store import load_locale, load_lock, save_locale

from .helpers import MockTransport, make_tmpdir

WS_CONFIG = """\
members = ["apps/*", "libs/*"]

[app]
tone = "neutral, concise"
base_language = "en"
languages = ["ru", "de"]

[validation]
semantic = "off"   # tests use a dumb mock transport; see helpers.L10N_TOML
script_check = false

[ollama]
batch_size = 3
workers = 2
model = "ws-model"
"""

GROUP_PLAYER = 'context = "TinyPlayer product family: IPTV player apps and a shared core."\n'

MEMBER_OUTPUT = """
[output]
adapter = "android-xml"
[output.android-xml]
res_dir = "res"
"""

TV_CFG = 'group = "player"\n\n[app]\ndescription = "TV variant; remote-control UI."\n' \
         + MEMBER_OUTPUT
MOBILE_CFG = 'group = "player"\n\n[app]\ndescription = "Mobile variant."\n' \
             'languages = ["ru"]\n' + MEMBER_OUTPUT
CORE_CFG = 'group = "player"\n\n[app]\ndescription = "Shared playback core."\n' + MEMBER_OUTPUT
SOLO_CFG = '[app]\ndescription = "Unrelated tool."\n' + MEMBER_OUTPUT

TV_STRINGS = """\
[channel_open]
text = "Open the channel"
[channel_next]
text = "Next channel"
[tv_hello]
text = "Welcome"
"""
MOBILE_STRINGS = '[mob_hello]\ntext = "Welcome back"\n'
CORE_STRINGS = '[core_lock]\ntext = "Channel is locked"\n'
SOLO_STRINGS = """\
[solo_a]
text = "Open the channel"
[solo_b]
text = "Next channel"
"""


def make_workspace(testcase, files=None):
    """Monorepo: l10n/config.toml + l10n/player.toml + 3 apps + 1 lib."""
    tmp = make_tmpdir(testcase)
    defaults = {
        "l10n/config.toml": WS_CONFIG,
        "l10n/player.toml": GROUP_PLAYER,
        "apps/tv/l10n.toml": TV_CFG,
        "apps/tv/strings.toml": TV_STRINGS,
        "apps/mobile/l10n.toml": MOBILE_CFG,
        "apps/mobile/strings.toml": MOBILE_STRINGS,
        "apps/solo/l10n.toml": SOLO_CFG,
        "apps/solo/strings.toml": SOLO_STRINGS,
        "libs/core/l10n.toml": CORE_CFG,
        "libs/core/strings.toml": CORE_STRINGS,
    }
    defaults.update(files or {})
    for rel, content in defaults.items():
        if content is None:
            continue
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return tmp


def member(ws: Workspace, name: str) -> Project:
    return next(m for m in ws.members if m.module == name)


class TestDiscovery(unittest.TestCase):
    def test_members_globs(self):
        ws = load(make_workspace(self))
        self.assertIsInstance(ws, Workspace)
        self.assertEqual([m.module for m in ws.members],
                         ["apps/mobile", "apps/solo", "apps/tv", "libs/core"])
        self.assertEqual(ws.workers, 2)

    def test_pattern_without_members_is_error(self):
        tmp = make_workspace(self, {
            "l10n/config.toml": WS_CONFIG.replace('"libs/*"', '"nope/*"')})
        with self.assertRaisesRegex(ConfigError, "matched no directory"):
            load(tmp)

    def test_nested_workspace_forbidden(self):
        tmp = make_workspace(self, {
            "apps/tv/l10n.toml": 'members = ["x/*"]\n' + TV_CFG})
        with self.assertRaisesRegex(ConfigError, "nested workspaces"):
            load(tmp)

    def test_members_in_single_config_is_error(self):
        tmp = make_tmpdir(self)
        (tmp / "l10n.toml").write_text('members = ["a/*"]\n', encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, "workspace config"):
            load(tmp)

    def test_single_mode_untouched(self):
        from .helpers import make_project
        project = make_project(make_tmpdir(self))
        self.assertIsInstance(project, Project)
        self.assertEqual(project.module, "")
        self.assertEqual(project.siblings, [])
        self.assertEqual(project.full_context, project.description)


class TestInheritance(unittest.TestCase):
    def setUp(self):
        self.tmp = make_workspace(self)
        self.ws = load(self.tmp)

    def test_defaults_inherited(self):
        tv = member(self.ws, "apps/tv")
        self.assertEqual(tv.languages, ["ru", "de"])
        self.assertEqual(tv.tone, "neutral, concise")
        self.assertEqual(tv.ollama.model, "ws-model")
        self.assertEqual(tv.ollama.batch_size, 3)
        self.assertEqual(tv.description, "TV variant; remote-control UI.")

    def test_member_overrides(self):
        mobile = member(self.ws, "apps/mobile")
        self.assertEqual(mobile.languages, ["ru"])            # overrides workspace list
        self.assertEqual(mobile.ollama.model, "ws-model")     # still inherits the rest

    def test_dir_on_member_attaches_workspace(self):
        tv = load(self.tmp / "apps" / "tv")
        self.assertIsInstance(tv, Project)
        self.assertEqual(tv.module, "apps/tv")
        self.assertEqual(tv.group, "player")
        self.assertEqual(tv.languages, ["ru", "de"])          # inherited via parent walk
        self.assertEqual(sorted(s.module for s in tv.siblings),
                         ["apps/mobile", "libs/core"])
        # load_project entry point behaves the same (used by tests/tools)
        self.assertEqual(load_project(self.tmp / "apps" / "tv").group, "player")


class TestGroups(unittest.TestCase):
    def setUp(self):
        self.tmp = make_workspace(self)
        self.ws = load(self.tmp)

    def test_group_context_from_l10n_dir(self):
        tv = member(self.ws, "apps/tv")
        self.assertIn("TinyPlayer product family", tv.group_context)

    def test_module_without_group_is_own_group(self):
        solo = member(self.ws, "apps/solo")
        self.assertEqual(solo.group, "apps/solo")
        self.assertEqual(solo.group_context, "")
        self.assertEqual(solo.siblings, [])

    def test_missing_group_file_is_error(self):
        tmp = make_workspace(self, {
            "apps/solo/l10n.toml": 'group = "ghost"\n' + SOLO_CFG})
        with self.assertRaisesRegex(ConfigError, "ghost"):
            load(tmp)

    def test_layered_prompt_context(self):
        from lloc.prompts import build_translate_prompt
        tv = member(self.ws, "apps/tv")
        prompt = build_translate_prompt(
            tv, "ru", [tv.strings["channel_open"]], {})
        self.assertIn("TinyPlayer product family", prompt)     # group context
        self.assertIn("TV variant; remote-control UI.", prompt)  # module context
        self.assertIn("neutral, concise", prompt)              # workspace tone
        pos_group = prompt.find("TinyPlayer product family")
        pos_module = prompt.find("TV variant")
        self.assertLess(pos_group, pos_module)                 # group before module


class TestGroupBoundaries(unittest.TestCase):
    def setUp(self):
        self.tmp = make_workspace(self)
        self.ws = load(self.tmp)

    def test_terms_counted_group_wide(self):
        from lloc.terms import extract_terms
        # tv alone has "channel" in 2 strings (< min_occurrences=3); the core
        # sibling adds the 3rd -> term exists, keys stay module-local
        tv = member(self.ws, "apps/tv")
        terms = extract_terms(tv)
        self.assertIn("channel", terms)
        self.assertEqual(terms["channel"], ["channel_open", "channel_next"])

    def test_other_group_is_isolated(self):
        from lloc.terms import extract_terms
        # solo also has "channel" in 2 strings, but no same-group sibling
        solo = member(self.ws, "apps/solo")
        self.assertNotIn("channel", extract_terms(solo))

    def test_fewshot_from_sibling_locale(self):
        from lloc.prompts import select_examples
        core = member(self.ws, "libs/core")
        # trusted-only pool (SPEC §13.6): the sibling's entry must be trusted
        save_locale(core.root, "ru",
                    {"core_lock": {"text": "Канал заблокирован",
                                   "src": text_hash("Channel is locked"),
                                   "model": "imported"}},
                    ["core_lock"])
        tv = member(self.ws, "apps/tv")
        examples = select_examples(tv, [tv.strings["channel_open"]], {}, lang="ru")
        self.assertIn(("Channel is locked", "Канал заблокирован"), examples)
        # solo (different group) never sees it
        solo = member(self.ws, "apps/solo")
        self.assertEqual(
            select_examples(solo, [solo.strings["solo_a"]], {}, lang="ru"), [])

    def test_duplicate_keys_in_group_warn_in_verify(self):
        tmp = make_workspace(self, {
            "libs/core/strings.toml": CORE_STRINGS + '[channel_open]\ntext = "Open"\n'})
        out = io.StringIO()
        with redirect_stdout(out):
            main(["--dir", str(tmp), "verify"])
        self.assertIn('duplicate keys with module "libs/core"', out.getvalue())
        self.assertIn("channel_open", out.getvalue())

    def test_no_duplicate_warning_when_clean(self):
        out = io.StringIO()
        with redirect_stdout(out):
            main(["--dir", str(self.tmp), "verify"])
        self.assertNotIn("duplicate keys", out.getvalue())


class TestWorkspaceCli(unittest.TestCase):
    def setUp(self):
        self.tmp = make_workspace(self)

    def _run(self, *argv):
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(out):
            rc = main(["--dir", str(self.tmp), *argv])
        return rc, out.getvalue()

    def test_status_aggregated(self):
        rc, out = self._run("status")
        for header in ("=== apps/tv (group: player) ===",
                       "=== apps/solo (group: apps/solo) ===",
                       "=== libs/core (group: player) ==="):
            self.assertIn(header, out)
        self.assertEqual(rc, 1)                                # everything missing

    def test_lang_filter_skips_members_without_language(self):
        rc, out = self._run("status", "--lang", "de")
        self.assertIn("apps/tv", out)
        self.assertNotIn("apps/mobile", out)                   # mobile is ru-only
        rc, out = self._run("status", "--lang", "xx")
        self.assertEqual(rc, 2)
        self.assertIn("not in any member", out)

    def test_translate_dry_run_groups_languages_by_keyset(self):
        rc, out = self._run("translate", "--dry-run")
        self.assertEqual(rc, 0)
        # languages with an identical pending keyset collapse into one line
        self.assertIn("[apps/tv] all 3 keys → 2 languages (ru, de)", out)
        self.assertIn("[libs/core] all 1 keys → 2 languages (ru, de)", out)
        self.assertIn("[apps/mobile] all 1 keys → ru", out)
        self.assertIn("no requests made", out)

    def test_translate_dry_run_partial_keysets_split_groups(self):
        # de is fully translated except one key -> its own group; ru needs all
        ws = load(self.tmp)
        tv = member(ws, "apps/tv")
        entries = {k: {"text": f"[de]{e.text}", "src": text_hash(e.text)}
                   for k, e in tv.strings.items() if k != "tv_hello"}
        save_locale(tv.root, "de", entries, list(tv.strings))
        rc, out = self._run("translate", "--dry-run")
        self.assertEqual(rc, 0)
        self.assertIn("[apps/tv] all 3 keys → ru", out)
        self.assertIn("[apps/tv] 1 key (tv_hello) → de", out)

    def test_import_requires_module_dir(self):
        rc, out = self._run("import", "--bootstrap-strings")
        self.assertEqual(rc, 2)
        self.assertIn("--dir <member>", out)

    def test_translate_offline_when_no_delta_saves_locks_and_builds(self):
        # pre-fill all locales as ok -> translate makes zero LLM calls
        ws = load(self.tmp)
        for m in ws.members:
            for lang in m.languages:
                entries = {k: {"text": f"[{lang}]{e.text}", "src": text_hash(e.text)}
                           for k, e in m.strings.items()}
                save_locale(m.root, lang, entries, list(m.strings))
        rc, out = self._run("translate")
        self.assertEqual(rc, 0)
        self.assertIn("nothing to translate", out)
        for m in ws.members:
            self.assertTrue(load_lock(m.root), m.module)       # per-module lock
            xml = m.root / "res" / "values" / "strings.xml"
            self.assertTrue(xml.is_file(), m.module)           # per-module build

    def test_shared_pool_translates_all_members(self):
        from lloc.engine import plan, run_translate_jobs
        ws = load(self.tmp)
        transport = MockTransport()
        jobs = []
        for m in ws.members:
            work = plan(m, m.languages, force=False, limit=None)
            jobs += [(m, lang, keys) for lang, keys in work.items() if keys]
        with redirect_stdout(io.StringIO()):
            results = run_translate_jobs(jobs, workers=ws.workers, transport=transport)
        self.assertEqual(len(results), len(jobs))
        self.assertTrue(all(r.error is None for _, r in results))
        tv = member(ws, "apps/tv")
        self.assertEqual(load_locale(tv.root, "ru")["tv_hello"]["text"], "[ru]Welcome")
        core = member(ws, "libs/core")
        self.assertEqual(load_locale(core.root, "de")["core_lock"]["text"],
                         "[de]Channel is locked")


if __name__ == "__main__":
    unittest.main()
