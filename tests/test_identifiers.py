import unittest

from signals_to_semantics.identifiers import canonical_window_key, safe_window_stem


class IdentifierTests(unittest.TestCase):
    def test_canonical_window_key_rounds_timestamps(self):
        key = canonical_window_key("log-a", 1.23456789, 2.34567891)
        self.assertEqual(key, "log-a|1.234568|2.345679")

    def test_safe_window_stem_is_stable_and_portable(self):
        key = "log-a|1.234568|2.345679"
        first = safe_window_stem(key)
        second = safe_window_stem(key)

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("log-a-"))
        self.assertNotIn("|", first)
        self.assertNotIn("/", first)

    def test_safe_window_stem_changes_with_window(self):
        first = safe_window_stem("log-a|1.000000|2.000000")
        second = safe_window_stem("log-a|2.000000|3.000000")
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
