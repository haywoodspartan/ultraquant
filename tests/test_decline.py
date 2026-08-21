"""Declining a derivation names its premises.

§11.70's registered claim, closing the testimony family: "no" after a
derived answer declines the consolidation and names the whole trail
for correction — nothing lowered, because a bare no cannot say which
premise it blames. The gate passed at +0.528, 3.07x seed sd; these
tests pin the cycle.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path


class DeclineTests(unittest.TestCase):
    """The decline, the named trail, and the full cycle."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import (build_session,
                                                     run_pipeline)

        self.dir = Path(tempfile.mkdtemp(prefix="uq_decline_"))
        self.session = build_session(self.dir, seed=0)
        self._run = run_pipeline
        for statement in ("the tower material is iron",
                          "the iron hardness is 490 units",
                          "the steel hardness is 800 units"):
            run_pipeline(statement, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _say(self, text: str) -> str:
        response, _trace = self._run(text, self.session)
        return response

    def test_a_decline_names_the_whole_trail(self) -> None:
        self._say("what is the tower hardness?")
        response = self._say("no")
        self.assertIn("not consolidating", response)
        self.assertIn("tower material is iron", response)
        self.assertIn("iron hardness is 490 units", response)

    def test_nothing_is_stored_and_nothing_lowers(self) -> None:
        """A bare no cannot say which premise it blames - blaming an
        unnamed premise would invent the accusation."""
        self._say("what is the tower hardness?")
        self._say("no")
        self.assertIsNone(
            self.session.memory.recall_fact("tower hardness"))
        for key in ("tower material", "iron hardness"):
            fact = self.session.memory.recall_fact(key)
            self.assertAlmostEqual(fact["confidence"], 0.6)

    def test_the_full_cycle_closes(self) -> None:
        """Contest, correct, re-derive, confirm - each step spoken."""
        self._say("what is the tower hardness?")
        self._say("no")
        revised = self._say("the tower material is steel")
        self.assertIn("revises", revised)
        rederived = self._say("what is the tower hardness?")
        self.assertIn("800", rederived)
        confirmed = self._say("yes")
        self.assertIn("Consolidated", confirmed)
        fact = self.session.memory.recall_fact("tower hardness")
        self.assertIn("800", str(fact["value"]))

    def test_yes_still_consolidates(self) -> None:
        self._say("what is the tower hardness?")
        response = self._say("yes")
        self.assertIn("Consolidated", response)


class GateVerdictTests(unittest.TestCase):
    """The PASS and the whole family, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import decline_gate

        doc = " ".join(decline_gate.__doc__.split())
        self.assertIn("+0.528 at 3.07x seed sd", doc)

    def test_the_accusation_line_is_recorded(self) -> None:
        from ultraquant.experiments import decline_gate

        doc = " ".join(decline_gate.__doc__.split())
        self.assertIn("blaming an unnamed premise would invent the "
                      "accusation", doc)

    def test_the_family_is_recorded_whole(self) -> None:
        from ultraquant.experiments import decline_gate

        doc = " ".join(decline_gate.__doc__.split())
        self.assertIn("four responses to two words, chosen by what "
                      "was actually pending", doc)


if __name__ == "__main__":
    unittest.main()
