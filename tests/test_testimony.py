"""Yes reaches the stored fact — at testimony strength, not gossip's.

§11.68's registered claim: an assertion of a held belief leaves a
one-turn pending confirmation, and "yes" raises the fact to 0.90 as
direct testimony — while a passing restatement stays at reinforcement's
0.1 nudge. The gate passed at +0.558, 3.99x seed sd with zero
conflations; these tests pin the strengths and their composition.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path


class TestimonyTests(unittest.TestCase):
    """Two strengths, one word, three pending kinds."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import (build_session,
                                                     run_pipeline)

        self.dir = Path(tempfile.mkdtemp(prefix="uq_testimony_"))
        self.session = build_session(self.dir, seed=0)
        self._run = run_pipeline

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _say(self, text: str) -> str:
        response, _trace = self._run(text, self.session)
        return response

    def test_yes_after_a_polar_yes_confirms_at_090(self) -> None:
        self._say("the tower material is iron")
        self._say("is the tower material iron?")
        response = self._say("yes")
        self.assertIn("Confirmed", response)
        self.assertIn("direct testimony", response)
        fact = self.session.memory.recall_fact("tower material")
        self.assertAlmostEqual(fact["confidence"], 0.9)

    def test_yes_after_a_choice_pick_confirms(self) -> None:
        self._say("the keep material is oak")
        self._say("is the keep material oak or granite?")
        response = self._say("yes")
        self.assertIn("Confirmed", response)

    def test_a_passing_restatement_is_gossip_not_testimony(self) -> None:
        """The docstring's eleven-section-old claim, measured: hearing
        it again nudges 0.1; being told "yes, correct" asserts 0.9."""
        self._say("the wall material is bronze")
        self._say("the wall material is bronze")
        fact = self.session.memory.recall_fact("wall material")
        self.assertAlmostEqual(fact["confidence"], 0.7)

    def test_the_strengths_compose(self) -> None:
        self._say("the mill material is granite")
        self._say("is the mill material granite?")
        self._say("yes")
        self._say("the mill material is granite")
        fact = self.session.memory.recall_fact("mill material")
        self.assertAlmostEqual(fact["confidence"], 1.0)

    def test_a_stray_yes_stays_inert(self) -> None:
        self._say("the gate material is copper")
        self._say("why is the gate material copper?")
        response = self._say("yes")
        self.assertNotIn("Confirmed", response)
        fact = self.session.memory.recall_fact("gate material")
        self.assertAlmostEqual(fact["confidence"], 0.6)

    def test_derivation_affirmation_still_consolidates(self) -> None:
        self._say("the dome material is iron")
        self._say("the iron hardness is 490 units")
        self._say("what is the dome hardness?")
        response = self._say("yes")
        self.assertIn("Consolidated", response)


class GateVerdictTests(unittest.TestCase):
    """The PASS and the three pending kinds, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import testimony_gate

        doc = " ".join(testimony_gate.__doc__.split())
        self.assertIn("+0.558 at 3.99x seed sd", doc)
        self.assertIn("zero strengths conflated", doc)

    def test_the_composition_is_recorded(self) -> None:
        from ultraquant.experiments import testimony_gate

        doc = " ".join(testimony_gate.__doc__.split())
        self.assertIn("each contribution at its own strength", doc)

    def test_the_three_pending_kinds_are_recorded(self) -> None:
        from ultraquant.experiments import testimony_gate

        doc = " ".join(testimony_gate.__doc__.split())
        self.assertIn("a derivation consolidates, an assertion "
                      "confirms, and nothing pending stays nothing",
                      doc)


if __name__ == "__main__":
    unittest.main()
