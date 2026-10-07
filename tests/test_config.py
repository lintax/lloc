import unittest

from lloc.config import PROFILES, ConfigError, load_project

from .helpers import LOCAL_MODEL, L10N_TOML, STRINGS_TOML, make_project, make_tmpdir


class TestConfig(unittest.TestCase):
    def test_valid_project(self):
        p = make_project(make_tmpdir(self))
        self.assertEqual(p.base_language, "en")
        self.assertEqual(p.languages, ["ru", "de"])
        self.assertEqual(list(p.strings), ["menu_search", "open_error", "items_count"])
        self.assertEqual(p.strings["menu_search"].type, "button")
        self.assertEqual(p.ollama.batch_size, 3)
        self.assertEqual(p.ollama.workers, 1)
        self.assertEqual(p.ollama.model, LOCAL_MODEL)  # from [ollama].profile
        self.assertEqual(p.adapter_name, "android-xml")
        self.assertEqual(p.adapter_config["legacy_codes"]["he"], "iw")

    def test_model_override(self):
        toml = L10N_TOML.replace("[ollama]", '[ollama]\nmodel = "qwen3.5:9b"')
        p = make_project(make_tmpdir(self), l10n=toml)
        self.assertEqual(p.ollama.model, "qwen3.5:9b")

    def test_plural_table_rejected(self):
        strings = STRINGS_TOML + '\n[days]\ncontext = "x"\n[days.text]\none = "%d day"\nother = "%d days"\n'
        with self.assertRaisesRegex(ConfigError, "plural.*not supported"):
            make_project(make_tmpdir(self), strings=strings)

    def test_missing_text_rejected(self):
        with self.assertRaisesRegex(ConfigError, "missing or empty 'text'"):
            make_project(make_tmpdir(self), strings='[key1]\ncontext = "no text"\n')

    def test_base_language_in_targets_rejected(self):
        toml = L10N_TOML.replace('languages = ["ru", "de"]', 'languages = ["ru", "en"]')
        with self.assertRaisesRegex(ConfigError, "base_language"):
            make_project(make_tmpdir(self), l10n=toml)

    def test_missing_adapter_rejected(self):
        toml = L10N_TOML.replace('adapter = "android-xml"', "")
        with self.assertRaisesRegex(ConfigError, "adapter"):
            make_project(make_tmpdir(self), l10n=toml)


class TestProfiles(unittest.TestCase):
    """[ollama].profile: the whole ensemble setup in one word."""

    # The shared fixture pins the local profile and tunes [ollama]; these tests
    # need the opposite — a config that states nothing beyond the required
    # minimum, so what remains is exactly what the profile provides.
    BARE = """\
[app]
base_language = "en"
languages = ["ru"]

[validation]

[ollama]

[output]
adapter = "android-xml"

[output.android-xml]
res_dir = "res"
"""

    def _load(self, toml):
        return make_project(make_tmpdir(self), l10n=toml)

    def test_default_is_the_cloud_ensemble(self):
        p = self._load(self.BARE)
        self.assertEqual(p.ollama.profile, "cloud")
        self.assertEqual(p.ollama.model, PROFILES["cloud"]["model"])
        self.assertEqual(p.ensemble_model, PROFILES["cloud"]["ensemble_model"])
        self.assertEqual(p.semantic_model, PROFILES["cloud"]["semantic_model"])
        self.assertEqual(p.ollama.timeout_sec, 600)
        self.assertEqual(p.ollama.workers, 10)
        self.assertTrue(p.ensemble)
        # three cloud tags -> concurrent per-batch ensemble, no swaps
        self.assertTrue(p.ensemble_batch)

    def test_local_profile_is_the_two_model_phased_pair(self):
        p = self._load(self.BARE.replace("[ollama]", '[ollama]\nprofile = "local"'))
        self.assertEqual(p.ollama.model, PROFILES["local"]["model"])
        self.assertEqual(p.ensemble_model, PROFILES["local"]["ensemble_model"])
        self.assertIsNone(p.semantic_model)
        self.assertEqual(p.ensemble_judge, PROFILES["local"]["ensemble_model"])
        self.assertTrue(p.ensemble)
        self.assertFalse(p.ensemble_batch)  # local models swap: stay phased

    def test_individual_keys_override_the_profile(self):
        toml = self.BARE.replace("[ollama]", '[ollama]\nworkers = 3\nmodel = "my-model"')
        toml = toml.replace("[validation]", '[validation]\nsemantic_model = "qwen3.5:4b"')
        p = self._load(toml)
        self.assertEqual(p.ollama.model, "my-model")
        self.assertEqual(p.ollama.workers, 3)
        self.assertEqual(p.semantic_model, "qwen3.5:4b")
        # untouched keys still come from the profile
        self.assertEqual(p.ollama.batch_size, PROFILES["cloud"]["batch_size"])
        self.assertEqual(p.ensemble_model, PROFILES["cloud"]["ensemble_model"])

    def test_false_switches_a_profile_model_off(self):
        toml = self.BARE.replace("[validation]", "[validation]\nensemble_model = false")
        p = self._load(toml)
        self.assertIsNone(p.ensemble_model)
        self.assertFalse(p.ensemble)
        self.assertEqual(p.semantic_mode, "batch")  # semantic QA itself stays on
        self.assertTrue(p.semantic_phased)          # via the profile's judge

    def test_semantic_off_disables_the_profile_ensemble_silently(self):
        p = self._load(self.BARE.replace("[validation]", '[validation]\nsemantic = "off"'))
        self.assertFalse(p.ensemble)
        self.assertIsNone(p.ensemble_model)

    def test_explicit_ensemble_model_with_semantic_off_is_an_error(self):
        toml = self.BARE.replace(
            "[validation]", '[validation]\nsemantic = "off"\nensemble_model = "b"')
        with self.assertRaisesRegex(ConfigError, "ensemble_model requires"):
            self._load(toml)

    def test_unknown_profile_rejected(self):
        toml = self.BARE.replace("[ollama]", '[ollama]\nprofile = "turbo"')
        with self.assertRaisesRegex(ConfigError, "profile"):
            self._load(toml)

    def test_bad_model_key_rejected(self):
        toml = self.BARE.replace("[validation]", "[validation]\nensemble_model = 7")
        with self.assertRaisesRegex(ConfigError, "model name or false"):
            self._load(toml)


if __name__ == "__main__":
    unittest.main()
