"""What happened: the retraction, askable — and never invented.

§11.65's registered claim, completing the tense family: a past-tense
question over a subject truth maintenance took down answers from the
retraction record, naming the premise that caused it. The gate passed
at +0.631, 7.82x seed sd with zero invented endings; these tests pin
the door and the tag fix that opened it.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.memory.systematic import SystematicMemory


class RetractionTagTests(unittest.TestCase):
    """The data fix: takedowns are findable by key."""

    def test_retraction_episodes_carry_the_keys_tag(self) -> None:
        memory = SystematicMemory()
        memory.remember_fact("tower material", "iron", confidence=0.6)
        memory.consolidate_fact(
            "tower hardness", "490 units", confidence=0.6,
            premises=[("tower material", "iron")])
        memory.remember_fact("tower material", "steel", confidence=0.6)
        episodes = memory.recall_episodes(kind="retraction",
                                          tags=["tower hardness"])
        self.assertEqual(len(episodes), 1)
        self.assertIn("tower material",
                      episodes[0]["content"]["because"])


class TakedownAnswerTests(unittest.TestCase):
    """The fourth tense: what happened."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import (build_session,
                                                     run_pipeline)

        self.dir = Path(tempfile.mkdtemp(prefix="uq_retraction_"))
        self.session = build_session(self.dir, seed=0)
        self._run = run_pipeline
        for statement in ("the tower material is iron",
                          "the iron hardness is 490 units"):
            run_pipeline(statement, self.session)
        run_pipeline("what is the tower hardness?", self.session)
        run_pipeline("yes", self.session)
        run_pipeline("the tower material is steel", self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _ask(self, question: str) -> str:
        response, _trace = self._run(question, self.session)
        return response

    def test_a_retracted_subject_names_its_takedown(self) -> None:
        response = self._ask("what was the tower hardness?")
        self.assertIn("I no longer hold tower hardness", response)
        self.assertIn("premise 'tower material' was revised", response)
        self.assertIn("does not outlive its premise", response)

    def test_a_never_existed_subject_gets_no_ending(self) -> None:
        """An ending invented is a history invented."""
        response = self._ask("what was the citadel hardness?")
        self.assertNotIn("no longer hold", response)

    def test_the_present_re_derives_forward(self) -> None:
        """The takedown is the PAST; the present honestly asks about
        the new material rather than mourning the old conclusion."""
        response = self._ask("what is the tower hardness?")
        self.assertNotIn("no longer hold", response)


class GateVerdictTests(unittest.TestCase):
    """The PASS and the completed tense family, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import retraction_gate

        doc = " ".join(retraction_gate.__doc__.split())
        self.assertIn("+0.631 at 7.82x seed sd", doc)
        self.assertIn("zero endings invented", doc)

    def test_the_tense_family_is_recorded_complete(self) -> None:
        from ultraquant.experiments import retraction_gate

        doc = " ".join(retraction_gate.__doc__.split())
        self.assertIn("one record, four doors, nothing invented "
                      "through any of them", doc)


if __name__ == "__main__":
    unittest.main()
