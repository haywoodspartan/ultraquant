"""Arithmetic over what is believed - §11.78, pinned.

Quantities and beliefs as operands, units riding the operators, five
different refusals for five different absences, and the branch
taking nothing that another branch could answer.
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
        self.root = Path(tempfile.mkdtemp(prefix="uq_quant_t_"))
        self.session = build_session(self.root, seed=0)
        for line in ("the tower height is 300 meters",
                     "the bridge height is 2 kilometers",
                     "the keep weight is 5 tonnes",
                     "the mill material is oak",
                     "the gate height is not 50 meters",
                     "the plus size is large"):
            run_pipeline(line, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _ask(self, text: str) -> str:
        response, _t = run_pipeline(text, self.session)
        return response

    def test_a_belief_scaled_by_a_number(self) -> None:
        answer = self._ask("what is the tower height times 3?")
        self.assertIn("tower height * 3 = 900 meters", answer)
        self.assertIn("inferred, not stored", answer)
        self.assertIn("tower height is 300 meters", answer)

    def test_written_quantities_convert(self) -> None:
        self.assertIn("= 2.3 kilometers",
                      self._ask("what is 300 meters + 2 kilometers?"))

    def test_two_beliefs_read_in_the_larger_unit(self) -> None:
        self.assertIn("= 2.3 kilometers", self._ask(
            "what is the tower height + the bridge height?"))

    def test_one_matrix_one_voice(self) -> None:
        """§11.75's lesson as a pin: the same arithmetic asked two
        ways must agree on the number and the unit."""
        mine = self._ask("what is the tower height + the bridge "
                         "height?")
        theirs = self._ask("what is the sum of the tower height and "
                           "the bridge height?")
        self.assertIn("2.3 kilometers", mine)
        self.assertIn("2.3 kilometers", theirs)

    def test_a_quantity_over_a_quantity_is_a_ratio(self) -> None:
        answer = self._ask("what is the bridge height / the tower "
                           "height?")
        self.assertIn("= 20/3", answer)

    def test_five_absences_get_five_refusals(self) -> None:
        for question, phrase in (
                ("what is the tower height + the keep weight?",
                 "connects meters and tonnes"),
                ("what is 300 meters * 2 meters?",
                 "would name a unit no definition I hold covers"),
                ("what is the north gate height * 2?",
                 "I hold nothing for 'north gate height'"),
                ("what is the mill material * 2?",
                 "holds no number (oak)"),
                ("what is the gate height * 2?",
                 "I hold only a denial for 'gate height'")):
            answer = self._ask(question)
            self.assertTrue(answer.startswith("I can't compute"),
                            f"{question!r} was not refused: {answer}")
            self.assertIn(phrase, answer)

    def test_a_plain_number_beside_a_quantity_refuses(self) -> None:
        """Assuming the missing unit would be invention - the same
        line §11.42's combine path holds."""
        answer = self._ask("what is the tower height + 5?")
        self.assertIn("do not agree on a unit", answer)

    def test_an_expression_is_never_answered_by_one_operand(self) -> None:
        """The failure this rung was built on: "the mill material *
        2" was answered "mill material is oak"."""
        self.assertNotIn("is oak",
                         self._ask("what is the mill material * 2?"))

    def test_a_run_that_resolves_to_nothing_falls_through(self) -> None:
        """"plus" reads as an operator, and refusing there would eat
        an ordinary question."""
        self.assertIn("large", self._ask("what is the plus size?"))

    def test_ordinary_questions_are_untouched(self) -> None:
        self.assertIn("300 meters",
                      self._ask("what is the tower height?"))
        self.assertIn("oak", self._ask("what is the mill material?"))

    def test_literal_arithmetic_still_says_computed(self) -> None:
        answer = self._ask("what is 2 + 2?")
        self.assertIn("computed, not stored", answer)
        self.assertNotIn("inferred", answer)


class GateVerdictTests(unittest.TestCase):
    """The PASS, the harness lesson, and the recorded ceiling."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import quantity_gate

        doc = " ".join(quantity_gate.__doc__.split())
        self.assertIn("+0.800 at 18.46x seed sd", doc)
        self.assertIn("zero voice splits against the combine path",
                      doc)

    def test_the_harness_lesson_is_recorded(self) -> None:
        from ultraquant.experiments import quantity_gate

        doc = " ".join(quantity_gate.__doc__.split())
        self.assertIn("An attribute is not a family", doc)

    def test_the_derive_ceiling_is_on_the_record(self) -> None:
        from ultraquant.experiments import quantity_gate

        doc = " ".join(quantity_gate.__doc__.split())
        self.assertIn("this rung does not", doc)
        self.assertIn("instead of in a promise", doc)


if __name__ == "__main__":
    unittest.main()
