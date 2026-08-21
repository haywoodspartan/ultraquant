"""Asking whether the arithmetic holds - §11.83, pinned.

Equality as a comparison, an expression as a comparison side, and
the composition of three rungs in one sentence.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.interpreter.thoughts import build_session, run_pipeline


class SurfaceTests(unittest.TestCase):
    """The live pipeline over one small store."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="uq_polar_t_"))
        self.session = build_session(self.root, seed=0)
        for line in ("the tower height is 300 meters",
                     "the tower material is iron"):
            run_pipeline(line, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _ask(self, text: str) -> str:
        response, _t = run_pipeline(text, self.session)
        return response

    def test_literal_equality_both_ways(self) -> None:
        self.assertIn("Yes - 2 + 2 is 4", self._ask("is 2 + 2 = 4?"))
        self.assertIn("No - 2 + 2 is 4, not 5",
                      self._ask("is 2 + 2 = 5?"))

    def test_the_equals_sign_need_not_be_spaced(self) -> None:
        self.assertTrue(self._ask("is 2+2=4?").startswith("Yes"))
        self.assertTrue(self._ask("is 2+2=5?").startswith("No"))

    def test_the_spoken_equality_reads_the_same(self) -> None:
        self.assertTrue(self._ask("is 2 + 2 equal to "
                                  "4?").startswith("Yes"))
        self.assertTrue(self._ask("is 2 + 2 equals "
                                  "5?").startswith("No"))

    def test_an_expression_may_be_a_comparison_side(self) -> None:
        answer = self._ask("is 3 * 4 greater than 10?")
        self.assertTrue(answer.startswith("Yes"))
        self.assertIn("3 * 4 is 12", answer)

    def test_equality_against_a_belief_keeps_its_unit(self) -> None:
        self.assertTrue(self._ask("is the tower height equal to 300 "
                                  "meters?").startswith("Yes"))
        self.assertTrue(self._ask("is the tower height equal to 200 "
                                  "meters?").startswith("No"))

    def test_three_rungs_compose_in_one_sentence(self) -> None:
        """A belief multiplied (§11.78), keeping its unit, compared
        against a quantity nobody stored (§11.82), read as an
        expression (§11.83)."""
        answer = self._ask("is the tower height * 2 greater than 500 "
                           "meters?")
        self.assertTrue(answer.startswith("Yes"))
        self.assertIn("tower height * 2 is 600 meters", answer)

    def test_an_unheld_side_refuses(self) -> None:
        answer = self._ask("is the north gate height * 2 greater "
                           "than 5 meters?")
        self.assertFalse(answer.startswith(("Yes", "No")))

    def test_a_polar_lead_is_not_an_expression(self) -> None:
        """This rung's own bug on the way in: the arithmetic reader
        was taking polar questions and refusing about 'is'."""
        self.assertNotIn("I hold nothing for 'is'",
                         self._ask("is 2 + 2 equal to 4?"))

    def test_plain_polar_questions_are_untouched(self) -> None:
        self.assertTrue(self._ask("is the tower material "
                                  "iron?").startswith("Yes"))
        self.assertTrue(self._ask("is the tower material "
                                  "steel?").startswith("No"))


class GateVerdictTests(unittest.TestCase):
    """The PASS and the composition it demonstrates."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import polararith_gate

        doc = " ".join(polararith_gate.__doc__.split())
        self.assertIn("+0.659 at 8.52x seed sd", doc)
        self.assertIn("zero unreadable sides answered", doc)

    def test_the_composition_is_on_the_record(self) -> None:
        from ultraquant.experiments import polararith_gate

        doc = " ".join(polararith_gate.__doc__.split())
        self.assertIn("each was built as a rule rather than a case",
                      doc)


if __name__ == "__main__":
    unittest.main()
