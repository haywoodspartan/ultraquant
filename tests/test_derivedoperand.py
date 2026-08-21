"""An operand the store can reach - §11.79, pinned.

The chain machinery supplies an arithmetic operand the store does
not hold, under §11.55's three guards, marked where it is used - and
the two paths never disagree about what the store knows.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.interpreter.thoughts import build_session, run_pipeline


class SurfaceTests(unittest.TestCase):
    """One world with a chain, a denial, and a dead end."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="uq_deriv_t_"))
        self.session = build_session(self.root, seed=0)
        for line in ("the west tower kind is spirekind",
                     "spirekind height is 900 meters",
                     "the south tower height is 450 meters",
                     "the east keep kind is vaultkind",
                     "vaultkind height is oak",
                     "the old gate kind is nullkind",
                     "nullkind height is not 20 meters"):
            run_pipeline(line, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _ask(self, text: str) -> str:
        response, _t = run_pipeline(text, self.session)
        return response

    def test_a_reached_operand_computes_and_says_so(self) -> None:
        answer = self._ask("what is the west tower height * 2?")
        self.assertIn("= 1800 meters", answer)
        self.assertIn("derived via spirekind", answer)
        self.assertIn("inferred, not stored", answer)

    def test_reached_and_held_operands_compose(self) -> None:
        answer = self._ask("what is the west tower height + the "
                           "south tower height?")
        self.assertIn("= 1350 meters", answer)
        self.assertIn("derived via spirekind", answer)

    def test_one_matrix_one_voice(self) -> None:
        """The disagreement this rung removes: the comparative
        reached `west tower height` while the arithmetic said it did
        not exist."""
        comparative = self._ask("is the west tower taller than the "
                                "south tower?")
        arithmetic = self._ask("what is the west tower height * 2?")
        alone = self._ask("what is the west tower height?")
        for answer in (comparative, arithmetic, alone):
            self.assertIn("900 meters", answer)

    def test_a_reached_denial_is_never_a_number(self) -> None:
        answer = self._ask("what is the old gate height * 2?")
        self.assertTrue(answer.startswith("I can't compute"))
        self.assertIn("derive a denial for 'old gate height'", answer)
        self.assertNotIn("= ", answer)

    def test_a_reached_non_number_refuses(self) -> None:
        answer = self._ask("what is the east keep height * 2?")
        self.assertTrue(answer.startswith("I can't compute"))
        self.assertIn("holds no number", answer)

    def test_an_unreachable_operand_names_the_key(self) -> None:
        answer = self._ask("what is the north tower height * 2?")
        self.assertIn("I hold nothing for 'north tower height'",
                      answer)


class GateVerdictTests(unittest.TestCase):
    """The PASS and the line the rung exists for."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import derivedoperand_gate

        doc = " ".join(derivedoperand_gate.__doc__.split())
        self.assertIn("+0.338 at 3.64x seed sd", doc)
        self.assertIn("zero voice splits against the chain machinery",
                      doc)

    def test_the_closed_ceiling_is_recorded(self) -> None:
        from ultraquant.experiments import derivedoperand_gate

        doc = " ".join(derivedoperand_gate.__doc__.split())
        self.assertIn("removes a disagreement rather than adding a "
                      "capability", doc)


if __name__ == "__main__":
    unittest.main()
