"""Comparatives: a verdict that was computed, refusals that are named.

§11.54's registered claim, and two shipped defects closed: the polar
derive path answered comparatives with a coin that always said No, and
a missing operand got a fabricated verdict through the direct matrix.
The gate passed at +0.893, 12.91x seed sd (coin arm: 38 fabrications,
measured); these tests pin the matrix, the four refusals, and the
derived-operand ceiling.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path


class ComparativeTests(unittest.TestCase):
    """Both directions, both units, every refusal."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import (build_session,
                                                     run_pipeline)

        self.dir = Path(tempfile.mkdtemp(prefix="uq_comparative_"))
        self.session = build_session(self.dir, seed=0)
        self._run = run_pipeline
        for statement in ("the tower height is 300 meters",
                          "the bridge height is 120 meters",
                          "the spire height is 2 kilometers",
                          "the dome height is not 500 meters",
                          "the keep weight is 40 tonnes",
                          "the gate material is oak"):
            run_pipeline(statement, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _ask(self, question: str) -> str:
        response, _trace = self._run(question, self.session)
        return response

    def test_a_true_comparative_is_yes_with_both_values(self) -> None:
        response = self._ask(
            "is the tower height taller than the bridge height?")
        self.assertTrue(response.startswith("Yes"))
        self.assertIn("300 meters", response)
        self.assertIn("120 meters", response)

    def test_the_reverse_is_no_not_a_coin(self) -> None:
        """The shipped derive path said No in BOTH directions."""
        yes = self._ask(
            "is the tower height taller than the bridge height?")
        no = self._ask(
            "is the bridge height taller than the tower height?")
        self.assertTrue(yes.startswith("Yes"))
        self.assertTrue(no.startswith("No"))

    def test_a_downward_comparative_inverts(self) -> None:
        response = self._ask(
            "is the bridge height shorter than the tower height?")
        self.assertTrue(response.startswith("Yes"))

    def test_units_convert_before_comparing(self) -> None:
        response = self._ask(
            "is the spire height taller than the tower height?")
        self.assertTrue(response.startswith("Yes"))
        self.assertIn("units converted", response)

    def test_a_missing_operand_gets_no_verdict(self) -> None:
        """The shipped matrix fabricated 'No - ... not taller than
        citadel height' over an operand it never held."""
        response = self._ask(
            "is the tower height taller than the citadel height?")
        self.assertFalse(response.startswith("Yes"))
        self.assertFalse(response.startswith("No"))
        self.assertIn("hold nothing for 'citadel height'", response)

    def test_a_negated_operand_refuses_with_the_denial(self) -> None:
        response = self._ask(
            "is the tower height taller than the dome height?")
        self.assertIn("only a denial", response)

    def test_cross_family_units_refuse(self) -> None:
        response = self._ask(
            "is the tower height taller than the keep weight?")
        self.assertIn("no definition connects them", response)

    def test_a_wordy_operand_refuses(self) -> None:
        response = self._ask(
            "is the tower height taller than the gate material?")
        self.assertIn("holds no number", response)

    def test_derived_operands_now_answer(self) -> None:
        """§11.54 pinned this refusing as its registered ceiling;
        §11.55 cashed it - derived operands compare, each marked with
        its trail (test_compderive pins the full matrix)."""
        self._run("the mill material is steel", self.session)
        self._run("the wall material is iron", self.session)
        self._run("the steel hardness is 500 units", self.session)
        self._run("the iron hardness is 400 units", self.session)
        response = self._ask(
            "is the mill hardness bigger than the wall hardness?")
        self.assertTrue(response.startswith("Yes"))
        self.assertIn("derived via steel", response)
        self.assertIn("derived via iron", response)


class GateVerdictTests(unittest.TestCase):
    """The PASS, the measured coin, and the ceiling, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import comparative_gate

        doc = " ".join(comparative_gate.__doc__.split())
        self.assertIn("+0.893 at 12.91x seed sd", doc)

    def test_the_coin_defect_is_measured_beside_the_fix(self) -> None:
        from ultraquant.experiments import comparative_gate

        doc = " ".join(comparative_gate.__doc__.split())
        self.assertIn("a coin that always says No", doc)
        self.assertIn("the coin arm fabricated 38", doc)

    def test_the_derived_operand_ceiling_is_recorded(self) -> None:
        from ultraquant.experiments import comparative_gate

        doc = " ".join(comparative_gate.__doc__.split())
        self.assertIn("comparing derived operands stays a registered "
                      "candidate", doc)


if __name__ == "__main__":
    unittest.main()
