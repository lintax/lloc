import unittest

from lloc.validate import check_translation, placeholders


class TestPlaceholders(unittest.TestCase):
    def test_extraction(self):
        c = placeholders("Load %1$s of %2$s (%d%%)\nnext")
        self.assertEqual(c["%1$s"], 1)
        self.assertEqual(c["%2$s"], 1)
        self.assertEqual(c["%d"], 1)
        self.assertEqual(c["%%"], 1)
        self.assertEqual(c["\n"], 1)

    def test_brace_placeholders(self):
        c = placeholders("Hi {name}, {{count}} of {0} ({{value, number}})")
        self.assertEqual(c["{name}"], 1)
        self.assertEqual(c["{{count}}"], 1)
        self.assertEqual(c["{0}"], 1)
        self.assertEqual(c["{{value, number}}"], 1)
        self.assertEqual(placeholders("a { b } c {}"), placeholders(""))  # plain braces


class TestCheckTranslation(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(check_translation("Failed to open %1$s", "Не удалось открыть %1$s"), [])

    def test_identical_is_not_an_error(self):
        self.assertEqual(check_translation("Status", "Status"), [])

    def test_empty(self):
        self.assertEqual(check_translation("x", "  "), ["translation is empty"])
        self.assertTrue(check_translation("x", None))

    def test_placeholder_lost(self):
        errs = check_translation("Open %1$s now", "Открыть сейчас")
        self.assertEqual(len(errs), 1)
        self.assertIn("placeholders mismatch", errs[0])
        self.assertIn("%1$s", errs[0])

    def test_brace_placeholder_translated(self):
        self.assertEqual(check_translation("Hi {name}", "Привет, {name}"), [])
        errs = check_translation("Tasks: {{count}}", "Задачи: {{количество}}")
        self.assertIn("placeholders mismatch", errs[0])

    def test_placeholder_mutated(self):
        errs = check_translation("Open %1$s", "Открыть %s")
        self.assertTrue(errs)

    def test_too_long(self):
        errs = check_translation("Very long base string here", "x" * 100)
        self.assertTrue(any("too long" in e for e in errs))

    def test_short_strings_tolerant(self):
        # 3x on tiny strings must not fire: "OK" -> "Хорошо" is legitimate
        self.assertEqual(check_translation("OK", "Хорошо"), [])


if __name__ == "__main__":
    unittest.main()
