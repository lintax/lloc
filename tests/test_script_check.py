"""Inline wrong-script rejection (the wrong-script incident class): candidates whose
dominant script differs from the language's expected script are bounced."""
import io
import json
import unittest
from contextlib import redirect_stdout

from lloc.engine import run_translate, translate_batch
from lloc.langnames import expected_scripts
from lloc.ollama import OllamaClient
from lloc.store import load_locale
from lloc.validate import script_error

from .helpers import L10N_TOML, make_project, make_tmpdir

KK = expected_scripts("kk")

# helpers config with the check ON (fixtures default it off for mocks)
KK_TOML = L10N_TOML.replace("script_check = false", "script_check = true").replace(
    'languages = ["ru", "de"]', 'languages = ["kk", "de"]')


class TestScriptError(unittest.TestCase):
    def test_serbian_latin_rejected(self):
        err = script_error("Nema alarma", KK)
        self.assertIsNotNone(err)
        self.assertIn("Cyrillic", err)
        self.assertIn("mostly Latin", err)

    def test_serbian_cyrillic_ok(self):
        self.assertIsNone(script_error("Нема аларма", KK))

    def test_short_latin_and_placeholders_tolerated(self):
        self.assertIsNone(script_error("OK", KK))
        self.assertIsNone(script_error("%1$dh %2$dm", KK))
        self.assertIsNone(script_error("Premium", KK, allow=frozenset({"premium"})))

    def test_latin_language_untouched(self):
        self.assertIsNone(script_error("Kein Alarm", expected_scripts("de")))


class TestInlineRejection(unittest.TestCase):
    def _project(self):
        tmp = make_tmpdir(self)
        return tmp, make_project(tmp, l10n=KK_TOML)

    def test_wrong_script_bounced_then_fixed_on_retry(self):
        tmp, project = self._project()
        attempts = {"n": 0}

        def transport(url, payload, timeout):
            keys = payload["format"]["required"]
            attempts["n"] += 1
            latin = attempts["n"] == 1
            ph = {"playback_error": " %1$s", "items_count": " %d"}
            out = {k: (f"Latinica {k}" if latin else f"Ћирилички превод {'х' * (len(k) % 5 + 1)}") + ph.get(k, "")
                   for k in keys}
            return {"message": {"content": json.dumps(out, ensure_ascii=False)}}

        client = OllamaClient(project.ollama, transport=transport)
        accepted, failed, _ = translate_batch(project, client, "kk",
                                              list(project.strings), {})
        self.assertEqual(failed, {})
        self.assertTrue(all(t.startswith("Ћирилички") for t in accepted.values()))

    def test_persistent_wrong_script_fails_with_reason(self):
        tmp, project = self._project()

        def transport(url, payload, timeout):
            keys = payload["format"]["required"]
            ph = {"playback_error": " %1$s", "items_count": " %d"}
            return {"message": {"content": json.dumps(
                {k: f"Uvek latinica {k}" + ph.get(k, "") for k in keys})}}

        client = OllamaClient(project.ollama, transport=transport)
        accepted, failed, _ = translate_batch(project, client, "kk",
                                              list(project.strings), {})
        self.assertEqual(accepted, {})
        self.assertTrue(all(any("wrong script" in e for e in errs)
                            for errs in failed.values()))

    def test_ensemble_survivor_when_a_is_wrong_script(self):
        # A insists on Latin (rejected), B answers in Cyrillic -> B ships
        # as the only valid candidate, not a failure
        tmp = make_tmpdir(self)
        toml = KK_TOML.replace(
            "batch_size = 3", 'model = "a-cloud"\nbatch_size = 3').replace(
            'semantic = "off"',
            'semantic = "batch"\nensemble_model = "b-cloud"\n'
            'semantic_model = "j-cloud"')
        project = make_project(tmp, l10n=toml)
        self.assertTrue(project.ensemble_batch)

        def transport(url, payload, timeout):
            model = payload["model"]
            keys = payload["format"]["required"]
            first = payload["messages"][0]["content"]
            if "comparing TWO independent translations" in first:
                return {"message": {"content": json.dumps(
                    {k: "agree" for k in keys})}}
            ph = {"playback_error": " %1$s", "items_count": " %d"}
            if model == "b-cloud":
                out = {k: f"Ћирилички превод {'х' * (len(k) % 5 + 1)}" + ph.get(k, "") for k in keys}
            else:
                out = {k: f"Latinica {k}" + ph.get(k, "") for k in keys}
            return {"message": {"content": json.dumps(out, ensure_ascii=False)}}

        with redirect_stdout(io.StringIO()) as out:
            results = run_translate(project, ["kk"], transport=transport)
        self.assertEqual(results[0].failed, {})
        locale = load_locale(tmp, "kk")
        self.assertEqual(len(locale), 3)
        for k, e in locale.items():
            self.assertTrue(e["text"].startswith("Ћирилички"), e)
            self.assertEqual(e["model"], "b-cloud")
        self.assertIn("only valid", out.getvalue())


if __name__ == "__main__":
    unittest.main()
