import unittest

from lloc.hashes import text_hash
from lloc.store import base_hashes, classify, load_lock, load_locale, save_lock, save_locale

from .helpers import make_project, make_tmpdir


class TestHashes(unittest.TestCase):
    def test_stable_and_short(self):
        h = text_hash("Search")
        self.assertEqual(h, text_hash("Search"))
        self.assertEqual(len(h), 16)
        self.assertNotEqual(h, text_hash("search"))


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmpdir(self)
        self.project = make_project(self.tmp)

    def _locale(self, **overrides):
        entries = {
            key: {"text": f"ru:{e.text}", "src": text_hash(e.text)}
            for key, e in self.project.strings.items()
        }
        entries.update(overrides)
        return entries

    def test_classify_all_ok(self):
        st = classify(self.project, "ru", self._locale())
        self.assertTrue(st.clean)
        self.assertEqual(len(st.ok), 3)

    def test_classify_missing_stale_orphan(self):
        locale = self._locale(
            menu_search={"text": "Поиск", "src": "0000000000000000"},   # stale
            old_key={"text": "gone", "src": "1111111111111111"},        # orphan
        )
        del locale["items_count"]                                       # missing
        st = classify(self.project, "ru", locale)
        self.assertEqual(st.stale, ["menu_search"])
        self.assertEqual(st.missing, ["items_count"])
        self.assertEqual(st.orphan, ["old_key"])
        self.assertEqual(st.pending, ["items_count", "menu_search"])

    def test_locale_roundtrip_drops_orphans_and_orders(self):
        locale = self._locale(zzz={"text": "orphan", "src": "x"})
        save_locale(self.project.root, "ru", locale, list(self.project.strings))
        loaded = load_locale(self.project.root, "ru")
        self.assertNotIn("zzz", loaded)
        self.assertEqual(list(loaded), list(self.project.strings))

    def test_lock_roundtrip(self):
        hashes = base_hashes(self.project)
        save_lock(self.project.root, hashes)
        self.assertEqual(load_lock(self.project.root), hashes)
        self.assertEqual(load_lock(make_tmpdir(self)), {})


if __name__ == "__main__":
    unittest.main()
