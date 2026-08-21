"""The native reader says the same sentence - §11.89, pinned.

The probe is a built executable, so the live-parity pins skip when it
is absent; the recorded verdict is pinned unconditionally, because
that is documentation and does not need a compiler.
"""

from __future__ import annotations

import subprocess
import unittest

from ultraquant.experiments import nativecalc_gate
from ultraquant.reason import calculate

_PROBE = nativecalc_gate.probe_path()
_BUILT = _PROBE.exists()


def _native(questions: list[str]) -> list[str]:
    got = subprocess.run([str(_PROBE)], input="\n".join(questions) + "\n",
                         capture_output=True, text=True, check=True)
    return got.stdout.splitlines()


def _python(question: str) -> str:
    if question.startswith("list "):
        return nativecalc_gate.record_of(calculate.read_list(question[5:]))
    return nativecalc_gate.record_of(calculate.evaluate(question))


@unittest.skipUnless(_BUILT, "native reader not built")
class ParityTests(unittest.TestCase):
    """The cases where two arithmetic readers usually diverge."""

    def _same(self, questions: list[str]) -> None:
        native = _native(questions)
        for index, question in enumerate(questions):
            self.assertEqual(native[index], _python(question),
                             f"tiers disagreed on {question!r}")

    def test_exactness_and_precedence(self) -> None:
        self._same(["what is 0.1 + 0.2?", "what is 3 + 4 * 5?",
                    "what is (3 + 4) * 5?", "what is 1 / 3?",
                    "what is 2 / 3 + 1 / 3?", "what is -2 ^ 2?",
                    "what is 2 ^ 3 ^ 2?", "what is 2 ^ -3?"])

    def test_roots_and_their_bounds(self) -> None:
        self._same(["what is the square root of 16?",
                    "what is the square root of 2?",
                    "what is the cube root of -8?",
                    "what is the square root of -4?",
                    "what is the square root of 2 to 3 decimal places?"])

    def test_rounding_and_its_labels(self) -> None:
        self._same(["what is 100 / 3 to 2 decimal places?",
                    "what is 1 / 2 to 2 decimal places?",
                    "what is 110 / 11 to 2 decimal places?",
                    "what is 100 / 3 to the nearest whole number?",
                    "what is 2 / 3 to four decimal places?",
                    "what is 100 / 3 to 3 significant figures?"])

    def test_every_refusal_word_for_word(self) -> None:
        """The wording is the product, not a detail of it."""
        self._same(["what is 10 / 0?", "what is 300 + 20%?",
                    "what is 0 ^ 0?", "what is 2 ^ 5000?",
                    "list the sum of 3 meters and 5 kilograms",
                    "list the sum of 3 meters and 5",
                    "list the maximum of 22 meters and 51 kilograms"])

    def test_not_my_question_is_symmetric(self) -> None:
        """A tier that answers MORE than Python is as wrong as one
        that answers less."""
        for question in ("what is the tower height?", "hello there",
                         "what is 5?", "is 2 + 2 = 4?",
                         "the tower height is 300 meters",
                         "what is the plus size?"):
            self.assertEqual(_native([question])[0], "none")
            self.assertEqual(_python(question), "none")

    def test_a_generated_corpus_agrees(self) -> None:
        report = nativecalc_gate.run_gate(cases=400, seed=11)
        self.assertEqual(report.mismatches, 0, report.reason)


class GateVerdictTests(unittest.TestCase):
    """The recorded PASS and the ordering bug it caught."""

    def setUp(self) -> None:
        self.doc = " ".join(nativecalc_gate.__doc__.split())

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        self.assertIn("| records that differed | **0** |", self.doc)

    def test_the_caught_bug_is_recorded(self) -> None:
        self.assertIn("They named the two units in opposite ORDER",
                      self.doc)
        self.assertIn("A gate that compared values would have called "
                      "that a pass", self.doc)

    def test_symmetry_is_a_criterion(self) -> None:
        self.assertIn("A native tier that answers MORE than Python is "
                      "exactly as wrong as one that answers less",
                      self.doc)


if __name__ == "__main__":
    unittest.main()
