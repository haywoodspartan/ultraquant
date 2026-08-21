"""A list of numbers is not a list of beliefs - §11.87, pinned.

Written lists get their own voice; belief lists and fact aggregates
keep theirs; units, exactness and refusals are inherited whole.
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
        self.root = Path(tempfile.mkdtemp(prefix="uq_list_t_"))
        self.session = build_session(self.root, seed=0)
        for line in ("the tower height is 300 meters",
                     "the keep height is 100 meters"):
            run_pipeline(line, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _ask(self, text: str) -> str:
        response, _t = run_pipeline(text, self.session)
        return response

    def test_the_four_operations(self) -> None:
        self.assertIn("= 18", self._ask("what is the sum of 3, 5 "
                                        "and 10?"))
        self.assertIn("= 6", self._ask("what is the average of 3, 5 "
                                       "and 10?"))
        self.assertIn("= 10", self._ask("what is the largest of 3, 5 "
                                        "and 10?"))
        self.assertIn("= 3", self._ask("what is the smallest of 3, 5 "
                                       "and 10?"))

    def test_units_ride_the_list(self) -> None:
        self.assertIn("= 6 meters", self._ask(
            "what is the average of 3 meters, 5 meters and 10 "
            "meters?"))
        self.assertIn("= 2.3 kilometers", self._ask(
            "what is the sum of 300 meters and 2 kilometers?"))

    def test_exactness_is_inherited(self) -> None:
        """The average of 1, 2 and 4 is 7/3, not 2.3333333333333335."""
        self.assertIn("= 7/3", self._ask("what is the average of 1, "
                                         "2 and 4?"))

    def test_the_two_refusals_are_inherited(self) -> None:
        self.assertIn("connects meters and kilograms", self._ask(
            "what is the sum of 3 meters and 5 kilograms?"))
        self.assertIn("do not agree on a unit", self._ask(
            "what is the sum of 3 meters and 5?"))

    def test_a_belief_list_keeps_the_older_voice(self) -> None:
        """The boundary: every item must be WRITTEN, so a list
        naming beliefs falls through to the machinery that owns
        beliefs."""
        answer = self._ask("what is the sum of the tower height and "
                           "the keep height?")
        self.assertIn("400 meters", answer)
        self.assertIn("inferred, not stored", answer)

    def test_fact_aggregates_are_untouched(self) -> None:
        self.assertIn("height facts I hold",
                      self._ask("what is the average height?"))
        self.assertIn("height facts I hold",
                      self._ask("what is the total height?"))

    def test_a_single_item_is_not_a_list(self) -> None:
        self.assertNotIn("= 5", self._ask("what is the average of 5?"))


class GateVerdictTests(unittest.TestCase):
    """The PASS and the boundary it was built to protect."""

    def setUp(self) -> None:
        from ultraquant.experiments import list_gate

        self.doc = " ".join(list_gate.__doc__.split())

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        self.assertIn("+0.804 at 11.27x seed sd", self.doc)
        self.assertIn("zero boundary answers moved", self.doc)

    def test_the_lesson_was_applied_before_the_bug(self) -> None:
        self.assertIn("which is the only real use of a lesson",
                      self.doc)

    def test_the_median_ceiling_is_recorded(self) -> None:
        self.assertIn("The median is this rung's ceiling", self.doc)


if __name__ == "__main__":
    unittest.main()
