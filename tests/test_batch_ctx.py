"""batch_size configurability + num_ctx policy for cloud vs local models."""
import json
import unittest

from lloc.config import DEFAULT_NUM_CTX, load
from lloc.ollama import OllamaClient

from .helpers import L10N_TOML, make_project, make_tmpdir


class TestBatchSize(unittest.TestCase):
    def test_configurable_and_used_for_batching(self):
        tmp = make_tmpdir(self)
        project = make_project(tmp, l10n=L10N_TOML.replace("batch_size = 3",
                                                           "batch_size = 20"))
        self.assertEqual(project.ollama.batch_size, 20)

    def test_workspace_inheritance_and_override(self):
        from .test_workspace import WS_CONFIG, make_workspace, member
        tmp = make_workspace(self, {
            "l10n/config.toml": WS_CONFIG.replace("batch_size = 3",
                                                  "batch_size = 20"),
            "apps/tv/l10n.toml":
                'group = "player"\n\n[app]\ndescription = "TV."\n'
                "[ollama]\nbatch_size = 5\n"
                '\n[output]\nadapter = "android-xml"\n'
                '[output.android-xml]\nres_dir = "res"\n',
        })
        ws = load(tmp)
        self.assertEqual(member(ws, "apps/mobile").ollama.batch_size, 20)  # inherited
        self.assertEqual(member(ws, "apps/tv").ollama.batch_size, 5)       # override


class TestNumCtxPolicy(unittest.TestCase):
    def _payload(self, toml_extra: str, model_line: str) -> dict:
        tmp = make_tmpdir(self)
        toml = L10N_TOML.replace("batch_size = 3", model_line + "\nbatch_size = 3")
        if toml_extra:
            toml = toml.replace("workers = 1", "workers = 1\n" + toml_extra)
        project = make_project(tmp, l10n=toml)
        seen = {}

        def transport(url, payload, timeout):
            seen.update(payload)
            return {"message": {"content": json.dumps({"k": "v"})}}

        OllamaClient(project.ollama, transport=transport).chat(
            [{"role": "user", "content": "x"}], {"required": ["k"], "properties": {}})
        return seen

    def test_cloud_model_without_explicit_num_ctx_sends_none(self):
        seen = self._payload("", 'model = "glm-5.3-flash:cloud"')
        self.assertNotIn("num_ctx", seen["options"])  # server decides

    def test_local_model_without_explicit_num_ctx_gets_default(self):
        seen = self._payload("", 'model = "gemma4:12b-it-qat"')
        self.assertEqual(seen["options"]["num_ctx"], DEFAULT_NUM_CTX)

    def test_explicit_num_ctx_always_wins(self):
        seen = self._payload("num_ctx = 16384", 'model = "glm-5.3-flash:cloud"')
        self.assertEqual(seen["options"]["num_ctx"], 16384)
        seen = self._payload("num_ctx = 4096", 'model = "gemma4:12b-it-qat"')
        self.assertEqual(seen["options"]["num_ctx"], 4096)


if __name__ == "__main__":
    unittest.main()
