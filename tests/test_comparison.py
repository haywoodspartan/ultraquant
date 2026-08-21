"""A coin that always said No - §11.82, pinned.

The words people actually use are compared; a written quantity is a
legitimate right-hand side; and a relation nothing can compare by
refuses instead of returning a verdict.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.interpreter.thoughts import build_session, run_pipeline


class SurfaceTests(unittest.TestCase):
    """One world, and the questions a person would actually ask."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="uq_cmp_t_"))
        self.session = build_session(self.root, seed=0)
        for line in ("the tower height is 300 meters",
                     "the keep height is 100 meters",
                     "the tower material is iron"):
            run_pipeline(line, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _ask(self, text: str) -> str:
        response, _t = run_pipeline(text, self.session)
        return response

    def test_greater_than_a_written_quantity_both_ways(self) -> None:
        """The bug in one pair: both of these answered No."""
        self.assertTrue(self._ask("is the tower height greater than "
                                  "200 meters?").startswith("Yes"))
        self.assertTrue(self._ask("is the tower height greater than "
                                  "500 meters?").startswith("No"))

    def test_less_and_more_are_compared_too(self) -> None:
        self.assertTrue(self._ask("is the tower height less than 500 "
                                  "meters?").startswith("Yes"))
        self.assertTrue(self._ask("is the tower height more than 200 "
                                  "meters?").startswith("Yes"))

    def test_the_written_side_converts(self) -> None:
        answer = self._ask("is the tower height greater than 2 "
                           "kilometers?")
        self.assertTrue(answer.startswith("No"))
        self.assertIn("units converted", answer)

    def test_belief_against_belief(self) -> None:
        self.assertTrue(self._ask("is the tower height greater than "
                                  "the keep height?").startswith("Yes"))
        self.assertTrue(self._ask("is the keep height greater than "
                                  "the tower height?").startswith("No"))

    def test_an_unknown_relation_refuses(self) -> None:
        """The line that outlives the vocabulary: adding four words
        fixes four words."""
        answer = self._ask("is the tower height wider than the keep "
                           "height?")
        self.assertIn("I don't know how to compare by 'wider'",
                      answer)
        self.assertFalse(answer.startswith(("Yes", "No")))

    def test_an_unknown_unit_refuses_and_names_it(self) -> None:
        answer = self._ask("is the tower height greater than 200 "
                           "florins?")
        self.assertIn("I hold nothing for '200 florins'", answer)

    def test_the_old_vocabulary_still_answers(self) -> None:
        self.assertTrue(self._ask("is the tower height taller than "
                                  "the keep height?").startswith("Yes"))

    def test_plain_polar_questions_are_untouched(self) -> None:
        self.assertTrue(self._ask("is the tower material "
                                  "iron?").startswith("Yes"))
        self.assertTrue(self._ask("is the tower material "
                                  "steel?").startswith("No"))


class GateVerdictTests(unittest.TestCase):
    """The PASS, and what the baseline's score actually measures."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import comparison_gate

        doc = " ".join(comparison_gate.__doc__.split())
        self.assertIn("+0.617 at 3.23x seed sd", doc)
        self.assertIn("zero unknown relations answered", doc)

    def test_the_baseline_score_is_read_correctly(self) -> None:
        from ultraquant.experiments import comparison_gate

        doc = " ".join(comparison_gate.__doc__.split())
        self.assertIn("the number measures the worlds, not the "
                      "system", doc)


if __name__ == "__main__":
    unittest.main()
