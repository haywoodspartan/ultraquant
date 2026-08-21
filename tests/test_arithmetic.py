"""Arithmetic that is exactly right - §11.76, pinned.

Precedence, parentheses, signs and word operators; exact rationals
where floats drift; division by zero refused; and the branch taking
nothing that is not an expression.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from ultraquant.interpreter.thoughts import build_session, run_pipeline
from ultraquant.reason import calculate


class ReaderTests(unittest.TestCase):
    """The module alone: grammar, exactness, and strictness."""

    def _value(self, text: str) -> str:
        result = calculate.evaluate(text)
        self.assertIsNotNone(result, f"{text!r} was not read as math")
        self.assertEqual(result.refusal, "")
        return result.shown

    def test_precedence_is_structural(self) -> None:
        self.assertEqual(self._value("what is 3 + 4 * 5?"), "23")
        self.assertEqual(self._value("what is (3 + 4) * 5?"), "35")

    def test_word_operators_are_the_same_operators(self) -> None:
        self.assertEqual(self._value("what is 12 times 7?"), "84")
        self.assertEqual(self._value("what is 100 divided by 4?"), "25")
        self.assertEqual(self._value("what is 9 minus 4?"), "5")
        self.assertEqual(self._value("what is 8 over 2?"), "4")

    def test_signs_and_nesting(self) -> None:
        self.assertEqual(self._value("what is -3 + 5?"), "2")
        self.assertEqual(self._value("what is 5 - -3?"), "8")
        self.assertEqual(
            self._value("what is 2 * (3 + (4 - 1)) / 3?"), "4")

    def test_tenths_are_held_exactly(self) -> None:
        """The whole design choice, in one line: binary floating
        point answers this 0.30000000000000004."""
        self.assertEqual(self._value("what is 0.1 + 0.2?"), "0.3")
        self.assertEqual(self._value("what is 1.1 * 3?"), "3.3")

    def test_a_third_is_a_third(self) -> None:
        """0.3333333333 is a different number, so it is not given."""
        result = calculate.evaluate("what is 1 / 3?")
        self.assertEqual(result.shown, "1/3")
        self.assertTrue(result.fractional)
        self.assertEqual(self._value("what is 2 / 3 + 1 / 3?"), "1")

    def test_division_by_zero_refuses(self) -> None:
        for text in ("what is 10 / 0?", "what is 5 / (3 - 3)?",
                     "what is 7 / (0.1 + 0.2 - 0.3)?"):
            result = calculate.evaluate(text)
            self.assertIsNotNone(result)
            self.assertIn("undefined", result.refusal)
            self.assertEqual(result.shown, "")

    def test_anything_with_words_is_not_our_question(self) -> None:
        """A math branch that ate ordinary questions would be a worse
        bug than having no math."""
        for text in ("what is the tower height?",
                     "what is the tower material?",
                     "the tower height is 300 meters",
                     "what is 300 meters + 2 kilometers?",
                     "what is 5?",
                     "hello there"):
            self.assertIsNone(calculate.evaluate(text),
                              f"{text!r} was read as arithmetic")

    def test_the_echo_shows_the_reading(self) -> None:
        result = calculate.evaluate("what is 5 - -3?")
        self.assertEqual(result.expression, "5 - -3")
        result = calculate.evaluate("what is (3 + 4) * 5?")
        self.assertEqual(result.expression, "(3 + 4) * 5")

    def test_render_is_exact_for_every_terminating_value(self) -> None:
        for top, bottom, shown in ((1, 8, "0.125"), (5, 2, "2.5"),
                                   (3, 10, "0.3"), (-7, 4, "-1.75"),
                                   (4, 1, "4"), (1, 3, "1/3"),
                                   (2, 7, "2/7")):
            self.assertEqual(
                calculate.render_exact(Fraction(top, bottom)), shown)


class SurfaceTests(unittest.TestCase):
    """The branch in the live pipeline."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="uq_arith_t_"))
        self.session = build_session(self.root, seed=0)
        run_pipeline("the tower height is 300 meters", self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _ask(self, text: str) -> str:
        response, _t = run_pipeline(text, self.session)
        return response

    def test_the_answer_is_marked_computed_not_stored(self) -> None:
        response = self._ask("what is 2 + 2?")
        self.assertIn("2 + 2 = 4", response)
        self.assertIn("computed, not stored", response)

    def test_a_bare_expression_is_a_question(self) -> None:
        """"6 * 7" asks in a shape no interrogative lead covers."""
        for text in ("6 * 7", "2+2", "calculate 6 * 7",
                     "compute (2 + 3) * 4"):
            self.assertIn("=", self._ask(text))

    def test_statements_are_still_statements(self) -> None:
        self._ask("the keep material is oak")
        self.assertEqual(
            self.session.memory.recall_fact("keep material")["value"],
            "oak")

    def test_stored_questions_still_answer(self) -> None:
        self.assertIn("300 meters", self._ask("what is the tower "
                                              "height?"))

    def test_nothing_is_remembered_from_a_calculation(self) -> None:
        before = set(self.session.memory.fact_keys())
        self._ask("what is 41 + 1?")
        self.assertEqual(set(self.session.memory.fact_keys()), before)


class GateVerdictTests(unittest.TestCase):
    """The PASS and the two failure kinds, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import arithmetic_gate

        doc = " ".join(arithmetic_gate.__doc__.split())
        self.assertIn("+0.557 at 3.69x seed sd", doc)
        self.assertIn("not the answer **45 times**", doc)

    def test_the_wrong_branch_is_on_the_record(self) -> None:
        from ultraquant.experiments import arithmetic_gate

        doc = " ".join(arithmetic_gate.__doc__.split())
        self.assertIn("126100789566373888", doc)
        self.assertIn("wrong BRANCH", doc)


if __name__ == "__main__":
    unittest.main()
