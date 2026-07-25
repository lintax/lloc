import unittest

from lloc.config import DEFAULT_MODEL, ConfigError, load_project

from .helpers import L10N_TOML, STRINGS_TOML, make_project, make_tmpdir


class TestConfig(unittest.TestCase):
    def test_valid_project(self):
        p = make_project(make_tmpdir(self))
        self.assertEqual(p.base_language, "en")
        self.assertEqual(p.languages, ["ru", "de"])
        self.assertEqual(list(p.strings), ["menu_search", "playback_error", "items_count"])
        self.assertEqual(p.strings["menu_search"].type, "button")
        self.assertEqual(p.ollama.batch_size, 3)
        self.assertEqual(p.ollama.workers, 1)
        self.assertEqual(p.ollama.model, DEFAULT_MODEL)  # built-in default
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


if __name__ == "__main__":
    unittest.main()
