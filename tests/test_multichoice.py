"""Compound disjuncts compare whole.

§11.67's registered claim, cashing §11.66's ceiling: multi-word
alternatives fold-compare whole or not at all, the lead speaks the
belief as held, and absence never picks a side however many words the
side has. The gate passed at +0.825, 7.66x seed sd; these tests pin
the lift.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path


class MultiwordChoiceTests(unittest.TestCase):
    """Compound picks, qualified neithers, and the unmoved lines."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import (build_session,
                                                     run_pipeline)

        self.dir = Path(tempfile.mkdtemp(prefix="uq_multichoice_"))
        self.session = build_session(self.dir, seed=0)
        self._run = run_pipeline
        for statement in ("the spire architect is wren the younger",
                          "the tower material is iron"):
            run_pipeline(statement, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _ask(self, question: str) -> str:
        response, _trace = self._run(question, self.session)
        return response

    def test_a_compound_pick_speaks_the_belief_as_held(self) -> None:
        """The first probe caught the lead speaking the mangled parse
        ("Wren younger"); the lead names the stored value."""
        response = self._ask(
            "is the spire architect wren or wren the younger?")
        self.assertTrue(response.startswith("Wren the younger"))

    def test_order_does_not_matter(self) -> None:
        response = self._ask(
            "is the spire architect wren the younger or wren?")
        self.assertTrue(response.startswith("Wren the younger"))

    def test_a_qualified_value_is_a_different_value(self) -> None:
        """§11.47's lesson in question form: "reinforced steel" does
        not match a held "steel"."""
        response = self._ask(
            "is the tower material reinforced steel or oak?")
        self.assertTrue(response.startswith("Neither"))
        self.assertIn("iron", response)

    def test_the_bare_base_does_not_match_the_compound(self) -> None:
        response = self._ask(
            "is the spire architect wren or aldous?")
        self.assertTrue(response.startswith("Neither"))
        self.assertIn("wren the younger", response)

    def test_absence_never_picks_however_many_words(self) -> None:
        response = self._ask(
            "is the keep architect wren or wren the elder?")
        lowered = response.strip().lower()
        self.assertFalse(lowered.startswith("wren"))
        self.assertFalse(lowered.startswith("neither"))

    def test_single_word_choices_are_untouched(self) -> None:
        response = self._ask("is the tower material steel or iron?")
        self.assertTrue(response.startswith("Iron"))


class GateVerdictTests(unittest.TestCase):
    """The PASS and the cashed ceiling, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import multichoice_gate

        doc = " ".join(multichoice_gate.__doc__.split())
        self.assertIn("+0.825 at 7.66x seed sd", doc)

    def test_the_lead_fix_is_recorded(self) -> None:
        from ultraquant.experiments import multichoice_gate

        doc = " ".join(multichoice_gate.__doc__.split())
        self.assertIn("the answer now leads with the stored value",
                      doc)

    def test_the_momentum_line_is_recorded(self) -> None:
        from ultraquant.experiments import multichoice_gate

        doc = " ".join(multichoice_gate.__doc__.split())
        self.assertIn("the family's method carries its own momentum",
                      doc)


if __name__ == "__main__":
    unittest.main()
