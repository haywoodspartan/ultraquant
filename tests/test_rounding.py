"""Rounding is a request - §11.84, pinned.

Exactness stays the default; a rounded answer says it was rounded
and names the exact value; a request that cannot be served is
refused by name.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from ultraquant.interpreter.thoughts import build_session, run_pipeline
from ultraquant.reason.calculate import _round_to


class SurfaceTests(unittest.TestCase):
    """The live pipeline over one small store."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="uq_round_t_"))
        self.session = build_session(self.root, seed=0)
        run_pipeline("the tower height is 300 meters", self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _ask(self, text: str) -> str:
        response, _t = run_pipeline(text, self.session)
        return response

    def test_a_rounded_answer_says_so_and_names_the_exact(self) -> None:
        answer = self._ask("what is 100 / 3 to 2 decimal places?")
        self.assertIn("= 33.33", answer)
        self.assertIn("rounded to 2 decimal places", answer)
        self.assertIn("the exact value is 100/3", answer)

    def test_a_value_needing_no_rounding_says_that(self) -> None:
        answer = self._ask("what is 1 / 2 to 2 decimal places?")
        self.assertIn("= 0.50", answer)
        self.assertIn("nothing was rounded", answer)

    def test_the_requested_width_is_the_answered_width(self) -> None:
        """"to 2 decimal places" is answered in 2 decimal places -
        the padding says how far the claim goes."""
        self.assertIn("= 10.00", self._ask("what is 110 / 11 to 2 "
                                           "decimal places?"))

    def test_the_word_form_reads(self) -> None:
        self.assertIn("= 0.6667", self._ask("what is 2 / 3 to four "
                                            "decimal places?"))

    def test_the_nearest_whole_number(self) -> None:
        answer = self._ask("what is 100 / 3 to the nearest whole "
                           "number?")
        self.assertIn("= 33", answer)
        self.assertIn("nearest whole number", answer)

    def test_an_irrational_root_rounds_when_asked(self) -> None:
        answer = self._ask("what is the square root of 2 to 3 "
                           "decimal places?")
        self.assertIn("= 1.414", answer)
        self.assertIn("the exact value is irrational", answer)

    def test_a_belief_keeps_its_unit_through_rounding(self) -> None:
        answer = self._ask("what is the tower height / 7 to 1 "
                           "decimal place?")
        self.assertIn("= 42.9 meters", answer)
        self.assertIn("the exact value is 300/7", answer)

    def test_nothing_rounds_unasked(self) -> None:
        self.assertIn("100/3", self._ask("what is 100 / 3?"))
        self.assertIn("is between",
                      self._ask("what is the square root of 2?"))

    def test_an_unservable_request_is_refused_by_name(self) -> None:
        """Not read as part of the expression and refused about the
        word "to" - the gap is admitted in its own words."""
        answer = self._ask("what is 100 / 3 to 3 significant "
                           "figures?")
        self.assertIn("a different rule and I do not have it", answer)
        self.assertNotIn("'to'", answer)


class TieRuleTests(unittest.TestCase):
    """Half rounds away from zero, symmetrically, in exact arithmetic."""

    def test_the_tie_rule_is_symmetric(self) -> None:
        self.assertEqual(_round_to(Fraction(1, 8), 2),
                         Fraction(13, 100))
        self.assertEqual(_round_to(Fraction(-1, 8), 2),
                         Fraction(-13, 100))

    def test_ordinary_rounding_is_exact(self) -> None:
        self.assertEqual(_round_to(Fraction(100, 3), 2),
                         Fraction(3333, 100))
        self.assertEqual(_round_to(Fraction(2, 3), 4),
                         Fraction(6667, 10000))


class GateVerdictTests(unittest.TestCase):
    """The PASS, and the two harness lessons behind it."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import rounding_gate

        doc = " ".join(rounding_gate.__doc__.split())
        self.assertIn("+0.802 at 14.51x seed sd", doc)
        self.assertIn("zero unlabelled approximations", doc)

    def test_the_harness_lessons_are_recorded(self) -> None:
        from ultraquant.experiments import rounding_gate

        doc = " ".join(rounding_gate.__doc__.split())
        self.assertIn("Whether a case rounds is arithmetic, not a "
                      "label the gate gets to choose", doc)
        self.assertIn("it is 1.4142135623730951 wearing a shorter "
                      "coat", doc)


if __name__ == "__main__":
    unittest.main()
