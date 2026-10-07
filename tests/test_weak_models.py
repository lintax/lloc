"""lv incident follow-ups: meaning-only arbitration prompt, weak-language
escalation, per-language [models] overrides with swap-minimizing job order."""
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from lloc.config import ConfigError
from lloc.engine import order_jobs, run_translate
from lloc.store import load_locale

from .helpers import L10N_TOML, NO_ENSEMBLE, make_project, make_tmpdir


class TestArbitratePromptWording(unittest.TestCase):
    def test_meaning_only_criterion(self):
        text = (Path(__file__).resolve().parent.parent
                / "prompts" / "arbitrate.txt").read_text(encoding="utf-8")
        self.assertIn("whose MEANING is closest to the source", text)
        self.assertIn("Style, length, conciseness, fluency and word choice "
                      "are IRRELEVANT", text)
        self.assertIn("least bad candidate BY MEANING", text)

    def test_prompt_reaches_the_judge(self):
        # through the real arbitrate() call, not just the template file
        from lloc.ollama import OllamaClient
        from lloc.semantic import arbitrate
        tmp = make_tmpdir(self)
        project = make_project(tmp)
        seen = {}

        def transport(url, payload, timeout):
            seen["prompt"] = payload["messages"][0]["content"]
            return {"message": {"content": json.dumps(
                {"choice": "c1", "reason": "meaning"})}}

        client = OllamaClient(project.ollama, transport=transport)
        arbitrate(project, "ru", "menu_search", ["Вариант А", "Вариант Б"], client)
        self.assertIn("are IRRELEVANT", seen["prompt"])


class WeakTransport:
    """Every translation is semantically flagged -> everything arbitrates."""

    def __call__(self, url, payload, timeout):
        first = payload["messages"][0]["content"]
        keys = payload["format"]["required"]
        if "FORCED CHOICE" in first:
            return {"message": {"content": json.dumps(
                {"choice": "c1", "reason": "least bad"})}}
        if "literal translator" in first:
            out = {k: f"garbage({k})" for k in keys}
        elif "QA judge" in first:
            out = {k: "gibberish" for k in keys}
        else:
            ph = {"open_error": " %1$s", "items_count": " %d"}
            out = {k: f"tr({k})" + ph.get(k, "") for k in keys}
        return {"message": {"content": json.dumps(out, ensure_ascii=False)}}


BIG_STRINGS = "\n".join(f'[key_{i:02d}]\ntext = "Common word number {i}"\n'
                        for i in range(20))


class TestWeakLanguageEscalation(unittest.TestCase):
    def test_weak_warning_and_summary(self):
        from lloc.__main__ import _report_weak
        tmp = make_tmpdir(self)
        toml = L10N_TOML.replace('semantic = "off"', f'semantic = "batch"\n{NO_ENSEMBLE}')
        project = make_project(tmp, l10n=toml, strings=BIG_STRINGS)
        out = io.StringIO()
        with redirect_stdout(out):
            results = run_translate(project, ["ru"], transport=WeakTransport())
        self.assertEqual(results[0].arbitrated, 20)   # 100% arbitrated
        self.assertTrue(results[0].weak)
        text = out.getvalue()
        self.assertIn("WARNING: model looks weak in ru (100% arbitrated)", text)
        self.assertIn("translate --force --lang ru --model qwen3.5:9b", text)
        out2 = io.StringIO()
        with redirect_stdout(out2):
            _report_weak([("", r) for r in results])
        self.assertIn("model looks weak in: ru", out2.getvalue())

    def test_no_warning_below_threshold_or_small_sample(self):
        from lloc.engine import LangResult, _maybe_warn_weak
        tmp = make_tmpdir(self)
        project = make_project(tmp)
        out = io.StringIO()
        with redirect_stdout(out):
            # small sample: 3 of 3 arbitrated but < WEAK_MIN_SAMPLE
            r = LangResult(lang="ru", translated=3, arbitrated=3)
            _maybe_warn_weak(project, "ru", r, "ru")
            # big sample, low ratio
            r2 = LangResult(lang="de", translated=100, arbitrated=5)
            _maybe_warn_weak(project, "de", r2, "de")
        self.assertEqual(out.getvalue(), "")
        self.assertFalse(r.weak or r2.weak)


MODELS_TOML = L10N_TOML.replace(
    'languages = ["ru", "de"]',
    'languages = ["ru", "lv", "de"]') + """
[models]
"lv" = "qwen3.5:9b"
"""


class TestPerLanguageModels(unittest.TestCase):
    def test_config_parse_and_validation(self):
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=MODELS_TOML)
        self.assertEqual(project.models, {"lv": "qwen3.5:9b"})
        bad = L10N_TOML + '\n[models]\n"xx" = "some-model"\n'
        with self.assertRaisesRegex(ConfigError, "not in \\[app\\].languages"):
            make_project(make_tmpdir(self), l10n=bad)

    def test_override_used_for_translation_and_stamped(self):
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=MODELS_TOML)
        models_seen = {}

        def transport(url, payload, timeout):
            keys = payload["format"]["required"]
            lang_models = models_seen.setdefault(payload["model"], set())
            lang_models.update(keys)
            ph = {"open_error": " %1$s", "items_count": " %d"}
            return {"message": {"content": json.dumps(
                {k: f"tr({k})" + ph.get(k, "") for k in keys})}}

        with redirect_stdout(io.StringIO()):
            run_translate(project, ["ru", "lv"], transport=transport)
        self.assertIn("qwen3.5:9b", models_seen)          # lv went to the override
        self.assertIn(project.ollama.model, models_seen)  # ru to the main model
        self.assertEqual(load_locale(tmp, "lv")["menu_search"]["model"], "qwen3.5:9b")
        self.assertEqual(load_locale(tmp, "ru")["menu_search"]["model"],
                         project.ollama.model)

    def test_jobs_grouped_by_model(self):
        tmp = make_tmpdir(self)
        toml = L10N_TOML.replace(
            'languages = ["ru", "de"]',
            'languages = ["ru", "lv", "de", "lt"]') + \
            '\n[models]\n"lv" = "alt-a"\n"lt" = "alt-a"\n'
        project = make_project(tmp, l10n=toml)
        jobs = [(project, lang, ["menu_search"]) for lang in project.languages]
        ordered = [lang for _, lang, _ in order_jobs(jobs)]
        # main-model languages first (first-seen model order), overrides grouped
        self.assertEqual(ordered, ["ru", "de", "lv", "lt"])

    def test_workspace_inherits_models_section(self):
        from .test_workspace import WS_CONFIG, make_workspace, member
        from lloc.config import load
        files = {"l10n/config.toml": WS_CONFIG + '\n[models]\n"ru" = "alt-ru"\n'}
        ws = load(make_workspace(self, files))
        self.assertEqual(member(ws, "apps/web").models, {"ru": "alt-ru"})

    def test_cli_model_flag_beats_models_section(self):
        import unittest.mock
        from lloc import __main__ as cli
        from lloc.engine import LangResult
        tmp = make_tmpdir(self)
        make_project(tmp, l10n=MODELS_TOML)
        seen = {}

        def fake_run_translate(project, langs, **kw):
            seen["models"] = dict(project.models)
            return [LangResult(lang=lang) for lang in langs]

        with unittest.mock.patch.object(cli, "run_translate", fake_run_translate), \
             redirect_stdout(io.StringIO()):
            cli.main(["--dir", str(tmp), "translate", "--model", "ab-test",
                      "--no-build"])
        self.assertEqual(seen["models"], {})  # per-language pins cleared


if __name__ == "__main__":
    unittest.main()
