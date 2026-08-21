"""A hundredth part, taken - §11.80, pinned.

Percentages read structurally, composing with units and chains; the
dangling shape refuses rather than take the convention.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.interpreter.thoughts import build_session, run_pipeline


class SurfaceTests(unittest.TestCase):
    """The live pipeline, in the shapes people write."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="uq_pct_t_"))
        self.session = build_session(self.root, seed=0)
        for line in ("the tower height is 300 meters",
                     "the west tower kind is spirekind",
                     "spirekind height is 900 meters"):
            run_pipeline(line, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _ask(self, text: str) -> str:
        response, _t = run_pipeline(text, self.session)
        return response

    def test_a_literal_percentage(self) -> None:
        self.assertIn("20% of 300 = 60", self._ask("what is 20% of "
                                                   "300?"))

    def test_the_word_reads_as_the_sign(self) -> None:
        self.assertEqual(self._ask("what is 20 percent of 300?"),
                         self._ask("what is 20% of 300?"))

    def test_a_percentage_of_a_belief_keeps_the_unit(self) -> None:
        answer = self._ask("what is 15% of the tower height?")
        self.assertIn("= 45 meters", answer)
        self.assertIn("inferred, not stored", answer)

    def test_a_percentage_of_a_reached_belief(self) -> None:
        """§11.80 composing with §11.79: the operand is derived and
        the percentage is taken of what was derived."""
        answer = self._ask("what is 20% of the west tower height?")
        self.assertIn("= 180 meters", answer)
        self.assertIn("derived via spirekind", answer)

    def test_the_percentage_binds_before_the_addition(self) -> None:
        self.assertIn("= 65", self._ask("what is 20% of 300 + 5?"))

    def test_one_matrix_one_voice(self) -> None:
        """A percentage IS the arithmetic it stands for, so the two
        spellings cannot disagree."""
        self.assertIn("= 60", self._ask("what is 20% of 300?"))
        self.assertIn("= 60", self._ask("what is 300 * 20 / 100?"))

    def test_the_dangling_shape_refuses(self) -> None:
        """Every calculator answers 360 here, by convention. A branch
        built to be exact must not guess at a speaker's meaning."""
        answer = self._ask("what is 300 + 20%?")
        self.assertTrue(answer.startswith("I can't compute"))
        self.assertIn("needs something to be a percentage of", answer)
        self.assertNotIn("360", answer)

    def test_ordinary_questions_are_untouched(self) -> None:
        self.assertIn("300 meters",
                      self._ask("what is the tower height?"))
        self.assertIn("= 4", self._ask("what is 2 + 2?"))


class GateVerdictTests(unittest.TestCase):
    """The PASS, the zero-variance failure, and the ceiling."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import percent_gate

        doc = " ".join(percent_gate.__doc__.split())
        self.assertIn("+0.832 at 17.28x seed sd", doc)
        self.assertIn("zero conventions taken", doc)

    def test_the_first_run_measured_nothing_and_says_so(self) -> None:
        from ultraquant.experiments import percent_gate

        doc = " ".join(percent_gate.__doc__.split())
        self.assertIn("the first run measured nothing, because "
                      "nothing varied", doc)
        self.assertIn("the rule is not softened when it is "
                      "inconvenient", doc)

    def test_the_inverse_ceiling_is_on_the_record(self) -> None:
        from ultraquant.experiments import percent_gate

        doc = " ".join(percent_gate.__doc__.split())
        self.assertIn("what percent of 300 is 60?", doc)


if __name__ == "__main__":
    unittest.main()
