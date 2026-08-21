"""A root that has no value still has a place - §11.81, pinned.

Whole powers exact, rational roots exact, irrational roots enclosed
in proved bounds rather than named, and three shapes refused.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from ultraquant.interpreter.thoughts import build_session, run_pipeline

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


class PowerTests(unittest.TestCase):
    """Powers, in symbols and in words."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="uq_power_t_"))
        self.session = build_session(self.root, seed=0)
        run_pipeline("the tower height is 300 meters", self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _ask(self, text: str) -> str:
        response, _t = run_pipeline(text, self.session)
        return response

    def test_whole_powers_are_exact(self) -> None:
        self.assertIn("= 1024", self._ask("what is 2 ^ 10?"))
        self.assertIn("= 0.125", self._ask("what is 2 ^ -3?"))
        self.assertIn("= 0.01", self._ask("what is 0.1 ^ 2?"))

    def test_the_sign_sits_outside_the_power(self) -> None:
        """-2^2 is -4, which is what everyone writing it means."""
        self.assertIn("= -4", self._ask("what is -2 ^ 2?"))

    def test_the_exponent_binds_tighter_than_the_product(self) -> None:
        self.assertIn("= 48", self._ask("what is 3 * 2 ^ 4?"))

    def test_powers_are_right_associative(self) -> None:
        self.assertIn("= 512", self._ask("what is 2 ^ 3 ^ 2?"))

    def test_spoken_powers_read_as_the_symbols(self) -> None:
        self.assertIn("= 25", self._ask("what is 5 squared?"))
        self.assertIn("= 8", self._ask("what is 2 cubed?"))
        self.assertIn("= 81",
                      self._ask("what is 3 to the power of 4?"))


class RootTests(unittest.TestCase):
    """Roots: exact where they can be, enclosed where they cannot."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="uq_root_t_"))
        self.session = build_session(self.root, seed=0)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _ask(self, text: str) -> str:
        response, _t = run_pipeline(text, self.session)
        return response

    def test_a_rational_root_is_given_exactly(self) -> None:
        self.assertIn("= 4", self._ask("what is the square root of "
                                       "16?"))
        self.assertIn("= 0.5", self._ask("what is the square root of "
                                         "0.25?"))
        self.assertIn("= 3", self._ask("what is the cube root of "
                                       "27?"))
        self.assertIn("= -2", self._ask("what is the cube root of "
                                        "-8?"))

    def test_an_irrational_root_is_enclosed_not_named(self) -> None:
        answer = self._ask("what is the square root of 2?")
        self.assertIn("is between 1.414213562 and 1.414213563",
                      answer)
        self.assertIn("proved bounds, not the number", answer)
        self.assertNotIn("=", answer)

    def test_the_bounds_are_true(self) -> None:
        """A wrong bracket would be a firmer lie than the float."""
        for radicand in (2, 3, 5, 10, 99, 200):
            answer = self._ask(f"what is the square root of "
                               f"{radicand}?")
            low, high = _NUMBER_RE.findall(
                answer.split(" is between ", 1)[1])[:2]
            self.assertLessEqual(Fraction(low) ** 2, radicand)
            self.assertGreater(Fraction(high) ** 2, radicand)

    def test_three_shapes_refuse(self) -> None:
        for question, phrase in (
                ("what is the square root of -4?",
                 "no real number raised to the power 2"),
                ("what is 0 ^ 0?", "a convention, not a value"),
                ("what is 2 ^ 5000?", "more digits than an answer")):
            answer = self._ask(question)
            self.assertTrue(answer.startswith("I can't compute"),
                            f"{question!r} was not refused: {answer}")
            self.assertIn(phrase, answer)

    def test_a_unit_never_gains_a_power(self) -> None:
        run_pipeline("the tower height is 300 meters", self.session)
        answer = self._ask("what is the tower height squared?")
        self.assertIn("would name a unit no definition I hold covers",
                      answer)


class GateVerdictTests(unittest.TestCase):
    """The PASS and the proved-bounds line."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import power_gate

        doc = " ".join(power_gate.__doc__.split())
        self.assertIn("+0.298 at 3.56x seed sd", doc)
        self.assertIn("zero false bounds", doc)

    def test_the_float_comparison_is_on_the_record(self) -> None:
        from ultraquant.experiments import power_gate

        doc = " ".join(power_gate.__doc__.split())
        self.assertIn("2.0000000000000004", doc)
        self.assertIn("The enclosure is shorter, and it is true", doc)


if __name__ == "__main__":
    unittest.main()
