"""The unit that fell off - §11.77, pinned.

A combined answer names its unit whether or not a conversion
happened; the number never moves; unitless operands never gain one.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.interpreter.thoughts import build_session, run_pipeline


class SurfaceTests(unittest.TestCase):
    """The live pipeline, in the shapes a reader would ask."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="uq_unitkeep_t_"))
        self.session = build_session(self.root, seed=0)
        for line in ("the tower height is 300 meters",
                     "the keep height is 200 meters",
                     "the mill height is 1 kilometers",
                     "the north count is 30",
                     "the south count is 20"):
            run_pipeline(line, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _ask(self, text: str) -> str:
        response, _t = run_pipeline(text, self.session)
        return response.split(" - inferred", 1)[0]

    def test_a_same_unit_sum_says_what_it_is_a_sum_of(self) -> None:
        self.assertIn("500 meters", self._ask(
            "what is the sum of the tower height and the keep "
            "height?"))

    def test_a_same_unit_difference_names_its_unit(self) -> None:
        self.assertIn("100 meters", self._ask(
            "what is the difference between the tower height and the "
            "keep height?"))

    def test_the_larger_verdict_names_its_unit(self) -> None:
        self.assertIn("300 meters", self._ask(
            "which is larger, the tower height or the keep height?"))

    def test_the_converted_case_is_unchanged(self) -> None:
        answer = self._ask("what is the sum of the tower height and "
                           "the mill height?")
        self.assertIn("1.3 kilometers", answer)
        self.assertIn("(units converted)", answer)

    def test_unitless_operands_stay_unitless(self) -> None:
        answer = self._ask("what is the sum of the north count and "
                           "the south count?")
        self.assertIn("50", answer)
        for unit in ("meter", "kilometer", "second"):
            self.assertNotIn(unit, answer)


class GateVerdictTests(unittest.TestCase):
    """The PASS, the never-moves line, and why a green suite held."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import unitkeep_gate

        doc = " ".join(unitkeep_gate.__doc__.split())
        self.assertIn("+0.472 at 1.87x seed sd", doc)
        self.assertIn("zero numeric values moved", doc)

    def test_the_reason_the_suite_stayed_green_is_recorded(self) -> None:
        from ultraquant.experiments import unitkeep_gate

        doc = " ".join(unitkeep_gate.__doc__.split())
        self.assertIn("asserted only what the answer must NOT say",
                      doc)
        self.assertIn("the fix broke no pin, which is the tell", doc)


if __name__ == "__main__":
    unittest.main()
