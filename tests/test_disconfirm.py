"""No is testimony too — doubt spoken, never absence.

§11.69's registered claim, §11.68's mirror: "no" after an asserted
belief drops it to 0.30 with the drop spoken and the correction asked,
a following statement closes into the narrated revision, and doubt
never deletes. The gate passed at +0.722, 3.33x seed sd; these tests
pin the pair.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path


class DisconfirmTests(unittest.TestCase):
    """The contest, the loop, and the lines that hold."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import (build_session,
                                                     run_pipeline)

        self.dir = Path(tempfile.mkdtemp(prefix="uq_disconfirm_"))
        self.session = build_session(self.dir, seed=0)
        self._run = run_pipeline

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _say(self, text: str) -> str:
        response, _trace = self._run(text, self.session)
        return response

    def test_no_after_an_assertion_drops_to_doubt(self) -> None:
        self._say("the tower material is iron")
        self._say("is the tower material iron?")
        response = self._say("no")
        self.assertIn("you say that is wrong", response)
        self.assertIn("0.30", response)
        fact = self.session.memory.recall_fact("tower material")
        self.assertIsNotNone(fact)
        self.assertAlmostEqual(fact["confidence"], 0.3)

    def test_doubt_never_deletes(self) -> None:
        """A bare no names no replacement; deleting on it would invent
        an absence."""
        self._say("the keep material is oak")
        self._say("is the keep material oak?")
        self._say("no")
        self.assertIsNotNone(
            self.session.memory.recall_fact("keep material"))

    def test_the_correction_closes_the_loop(self) -> None:
        self._say("the wall material is bronze")
        self._say("is the wall material bronze?")
        self._say("no")
        response = self._say("the wall material is copper")
        self.assertIn("revises", response)
        self.assertIn("bronze", response)
        fact = self.session.memory.recall_fact("wall material")
        self.assertEqual(fact["value"], "copper")

    def test_a_stray_no_stays_inert(self) -> None:
        self._say("the gate material is granite")
        self._say("what is the gate material?")
        response = self._say("no")
        self.assertNotIn("you say that is wrong", response)
        fact = self.session.memory.recall_fact("gate material")
        self.assertAlmostEqual(fact["confidence"], 0.6)

    def test_yes_still_confirms(self) -> None:
        self._say("the mill material is steel")
        self._say("is the mill material steel?")
        response = self._say("yes")
        self.assertIn("Confirmed", response)
        fact = self.session.memory.recall_fact("mill material")
        self.assertAlmostEqual(fact["confidence"], 0.9)


class GateVerdictTests(unittest.TestCase):
    """The PASS and the completed testimony pair, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import disconfirm_gate

        doc = " ".join(disconfirm_gate.__doc__.split())
        self.assertIn("+0.722 at 3.33x seed sd", doc)
        self.assertIn("zero deletions", doc)

    def test_the_doubt_line_is_recorded(self) -> None:
        from ultraquant.experiments import disconfirm_gate

        doc = " ".join(disconfirm_gate.__doc__.split())
        self.assertIn("doubt never became absence", doc)

    def test_the_pair_is_recorded_complete(self) -> None:
        from ultraquant.experiments import disconfirm_gate

        doc = " ".join(disconfirm_gate.__doc__.split())
        self.assertIn("yes raises to 0.90, no lowers to 0.30", doc)


if __name__ == "__main__":
    unittest.main()
