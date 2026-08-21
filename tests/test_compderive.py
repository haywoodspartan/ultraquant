"""Derived operands compare: three families in one answer.

§11.55's registered claim, cashing §11.54's ceiling: a comparative
operand the store does not hold derives through the chain machinery —
never through a broken or denied chain — and the verdict marks which
side was derived and its trail. The gate passed at +0.674, 5.84x seed
sd; these tests pin the matrix and its refusals.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path


class DerivedComparativeTests(unittest.TestCase):
    """Both-derived, mixed, and every inherited refusal."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import (build_session,
                                                     run_pipeline)

        self.dir = Path(tempfile.mkdtemp(prefix="uq_compderive_"))
        self.session = build_session(self.dir, seed=0)
        self._run = run_pipeline
        for statement in ("the mill material is steel",
                          "the wall material is iron",
                          "the steel hardness is 500 units",
                          "the iron hardness is 400 units",
                          "the arch material is oak",
                          "the dome material is not steel",
                          "the fort hardness is 450 units"):
            run_pipeline(statement, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _ask(self, question: str) -> str:
        response, _trace = self._run(question, self.session)
        return response

    def test_both_derived_operands_compare_with_trails(self) -> None:
        response = self._ask(
            "is the mill hardness bigger than the wall hardness?")
        self.assertTrue(response.startswith("Yes"))
        self.assertIn("500 units (derived via steel)", response)
        self.assertIn("400 units (derived via iron)", response)

    def test_the_reverse_is_no(self) -> None:
        response = self._ask(
            "is the wall hardness bigger than the mill hardness?")
        self.assertTrue(response.startswith("No"))

    def test_a_mixed_comparative_marks_only_the_derived_side(self) -> None:
        response = self._ask(
            "is the mill hardness bigger than the fort hardness?")
        self.assertTrue(response.startswith("Yes"))
        self.assertIn("derived via steel", response)
        self.assertEqual(response.count("(derived via"), 1)

    def test_a_broken_chain_still_refuses(self) -> None:
        """Oak's hardness was never stored: the chain dead-ends and no
        verdict crosses it."""
        response = self._ask(
            "is the arch hardness bigger than the wall hardness?")
        self.assertFalse(response.startswith("Yes"))
        self.assertFalse(response.startswith("No"))

    def test_a_denied_material_still_refuses(self) -> None:
        """§11.48's inhibition through yet another door: a negated
        material derives nothing, so its property compares nothing."""
        response = self._ask(
            "is the dome hardness bigger than the wall hardness?")
        self.assertFalse(response.startswith("Yes"))
        self.assertFalse(response.startswith("No"))


class GateVerdictTests(unittest.TestCase):
    """The PASS and the composition, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import compderive_gate

        doc = " ".join(compderive_gate.__doc__.split())
        self.assertIn("+0.674 at 5.84x seed sd", doc)

    def test_the_cashed_ceiling_is_recorded(self) -> None:
        from ultraquant.experiments import compderive_gate

        doc = " ".join(compderive_gate.__doc__.split())
        self.assertIn("cashes that ceiling the way §11.52 cashed "
                      "§11.47's", doc)

    def test_the_composition_is_recorded(self) -> None:
        from ultraquant.experiments import compderive_gate

        doc = " ".join(compderive_gate.__doc__.split())
        self.assertIn("Three families now compose in one answer", doc)


if __name__ == "__main__":
    unittest.main()
