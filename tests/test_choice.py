"""Choices: the held disjunct answers by name.

§11.66's registered claim, and a shipped wrong answer closed: the
compound claim "steel or iron" matched nothing and the matrix said No
to a question whose true answer was offered. The gate passed at
+0.667, 5.92x seed sd with zero ungrounded elections; these tests pin
the matrix.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path


class ChoiceTests(unittest.TestCase):
    """Picks, neithers, denials, and the never-elected side."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import (build_session,
                                                     run_pipeline)

        self.dir = Path(tempfile.mkdtemp(prefix="uq_choice_"))
        self.session = build_session(self.dir, seed=0)
        self._run = run_pipeline
        for statement in ("the tower material is iron",
                          "the dome material is not steel"):
            run_pipeline(statement, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _ask(self, question: str) -> str:
        response, _trace = self._run(question, self.session)
        return response

    def test_the_held_disjunct_answers_by_name(self) -> None:
        """The shipped defect: this exact question answered No."""
        response = self._ask("is the tower material steel or iron?")
        self.assertTrue(response.startswith("Iron"))
        self.assertIn("tower material is iron", response)

    def test_order_does_not_matter(self) -> None:
        response = self._ask("is the tower material iron or steel?")
        self.assertTrue(response.startswith("Iron"))

    def test_neither_names_the_actual(self) -> None:
        response = self._ask("is the tower material oak or granite?")
        self.assertTrue(response.startswith("Neither"))
        self.assertIn("iron", response)

    def test_a_denial_rules_out_without_electing(self) -> None:
        """Ruling out is not picking: a denial of one value affirms
        nothing about another."""
        response = self._ask("is the dome material steel or iron?")
        self.assertTrue(response.startswith("Not steel, at least"))
        self.assertFalse(response.lower().startswith("iron"))

    def test_a_denial_of_neither_alternative_knows_nothing(self) -> None:
        response = self._ask("is the dome material oak or granite?")
        self.assertTrue(response.startswith("I don't know"))

    def test_an_unheld_subject_never_picks_a_side(self) -> None:
        response = self._ask("is the keep material steel or iron?")
        lowered = response.strip().lower()
        self.assertFalse(lowered.startswith("steel"))
        self.assertFalse(lowered.startswith("iron"))
        self.assertFalse(lowered.startswith("neither"))

    def test_plain_polar_is_untouched(self) -> None:
        response = self._ask("is the tower material iron?")
        self.assertTrue(response.startswith("Yes"))


class GateVerdictTests(unittest.TestCase):
    """The PASS and the completed polar family, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import choice_gate

        doc = " ".join(choice_gate.__doc__.split())
        self.assertIn("+0.667 at 5.92x seed sd", doc)
        self.assertIn("zero sides elected without grounds", doc)

    def test_the_wrong_no_is_recorded(self) -> None:
        from ultraquant.experiments import choice_gate

        doc = " ".join(choice_gate.__doc__.split())
        self.assertIn("No — to a question whose true answer was on "
                      "the table", doc)

    def test_the_family_close_is_recorded(self) -> None:
        from ultraquant.experiments import choice_gate

        doc = " ".join(choice_gate.__doc__.split())
        self.assertIn("absence never says yes, no, or picks a side",
                      doc)


if __name__ == "__main__":
    unittest.main()
