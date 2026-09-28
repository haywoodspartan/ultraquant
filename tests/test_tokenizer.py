"""Command-R's text becomes exactly llama.cpp's token ids.

11.124: a tokenizer built only from the GGUF's metadata. These pins hold
the pre-tokenizer's rules everywhere, llama.cpp's own ids for a handful
of texts wherever the model is on disk, and the whole gate wherever the
llama.cpp oracle is running.
"""

from __future__ import annotations

import unittest
import urllib.request

from ultraquant.infer import tokenizer as T


def _fragments(text: str) -> list:
    return list(T._command_r_fragments(text))


class PreTokenizerTests(unittest.TestCase):
    """The rules, as pure functions - no model needed."""

    def test_every_digit_is_its_own_fragment(self) -> None:
        self.assertEqual(_fragments("a12345b"),
                         ["a", "1", "2", "3", "4", "5", "b"])

    def test_contractions_split_like_gpt2(self) -> None:
        self.assertEqual(_fragments("it's"), ["it", "'s"])
        self.assertEqual(_fragments("they're"), ["they", "'re"])

    def test_the_last_space_of_a_run_joins_the_next_word(self) -> None:
        """The \\s+(?!\\S) lookahead - the defect the gate plants."""
        self.assertEqual(_fragments("a  b"), ["a", " ", " b"])

    def test_the_byte_alphabet_is_a_bijection(self) -> None:
        alphabet = T._byte_alphabet()
        self.assertEqual(len(alphabet), 256)
        self.assertEqual(len(set(alphabet)), 256)


class GoldenIdTests(unittest.TestCase):
    """llama.cpp's own ids, captured from its /tokenize on this model."""

    GOLDEN = {
        "Hello, world! The tower is 300 meters tall.":
            [5, 28339, 19, 3845, 8, 1896, 22804, 1801, 228, 26, 23, 23,
             24754, 20383, 21],
        "   leading spaces\tand\ttabs\n":
            [5, 1667, 8588, 18716, 205, 1807, 205, 33074, 206],
        "naïve café 東京 🙂": [5, 2540, 7342, 2279, 42632, 74429, 82199],
        "a12345b": [5, 72, 24, 25, 26, 27, 28, 73],
    }

    @classmethod
    def setUpClass(cls) -> None:
        from ultraquant.experiments import tokenizer_gate
        if not tokenizer_gate.COMMAND_R.exists():
            raise unittest.SkipTest("Command-R is not on this machine")
        cls.tok = T.Tokenizer.from_gguf(tokenizer_gate.COMMAND_R)

    def test_llama_cpps_ids(self) -> None:
        for text, ids in self.GOLDEN.items():
            with self.subTest(text=text):
                self.assertEqual(self.tok.encode(text, add_special=True), ids)

    def test_decoding_is_lossless(self) -> None:
        for text, ids in self.GOLDEN.items():
            with self.subTest(text=text):
                self.assertEqual(self.tok.decode(ids[1:]), text)


class GateVerdictTests(unittest.TestCase):

    def setUp(self) -> None:
        from ultraquant.experiments import tokenizer_gate
        self.gate = tokenizer_gate
        self.doc = " ".join(tokenizer_gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("Identical ids", "The exam can fail",
                       "Lossless both ways"):
            self.assertIn(phrase, self.doc)

    def test_the_void_first_run_is_kept_with_its_reason(self) -> None:
        self.assertIn("The first run was VOID, and the reason is a finding",
                      self.doc)
        self.assertIn("0 of Command-R's 253,333 merges touch a digit",
                      self.doc)

    def test_the_result_is_recorded(self) -> None:
        self.assertIn("with the harness shown able to fail", self.doc)
        self.assertIn("The whitespace defect mismatched 15 texts", self.doc)

    def test_the_gate_passes_against_the_live_oracle(self) -> None:
        try:
            urllib.request.urlopen(self.gate.ORACLE + "/health", timeout=2)
        except OSError:
            self.skipTest("the llama.cpp oracle is not running")
        self.assertTrue(self.gate.run_gate().passes)


if __name__ == "__main__":
    unittest.main()
