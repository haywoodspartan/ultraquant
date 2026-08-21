"""Belief history, askable — and never invented.

§11.64's registered claim: past-tense questions answer from the
revision record — every past belief in order, polarity included, the
no-record line for facts stated once, hedges for the unheld. The gate
passed at +0.873, 11.16x seed sd with zero invented histories; these
tests pin the matrix and the episode data-shape fix.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.memory.systematic import SystematicMemory


class EpisodeShapeTests(unittest.TestCase):
    """Revisions log spoken forms, polarity included."""

    def test_a_polarity_flip_logs_both_spoken_sides(self) -> None:
        """The §11.48 flip's history must read "not steel", not
        "steel" - the data-shape fix this unit required."""
        memory = SystematicMemory()
        memory.remember_fact("dome material", "steel", negated=True)
        memory.remember_fact("dome material", "steel")
        episode = memory.recall_episodes(kind="revision",
                                         tags=["dome material"])[0]
        self.assertEqual(episode["content"]["old_value"], "not steel")
        self.assertEqual(episode["content"]["new_value"], "steel")


class HistoryAnswerTests(unittest.TestCase):
    """Three tenses from one record."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import (build_session,
                                                     run_pipeline)

        self.dir = Path(tempfile.mkdtemp(prefix="uq_history_"))
        self.session = build_session(self.dir, seed=0)
        self._run = run_pipeline
        for statement in ("the tower material is iron",
                          "the tower material is steel",
                          "the tower material is bronze",
                          "the dome material is not steel",
                          "the dome material is steel",
                          "the keep material is oak"):
            run_pipeline(statement, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _ask(self, question: str) -> str:
        response, _trace = self._run(question, self.session)
        return response

    def test_the_full_history_in_order(self) -> None:
        response = self._ask("what was the tower material?")
        self.assertIn("It was iron, then steel", response)
        self.assertIn("it is bronze now", response)
        self.assertIn("2 revision(s) on record", response)

    def test_the_used_to_be_phrasing(self) -> None:
        response = self._ask(
            "what did the tower material used to be?")
        self.assertIn("It was iron", response)

    def test_a_polarity_flip_reads_its_denial(self) -> None:
        response = self._ask("what was the dome material?")
        self.assertIn("It was not steel", response)
        self.assertIn("it is steel now", response)

    def test_a_fact_stated_once_has_no_history(self) -> None:
        """The invention line: no record means saying so, never
        inventing a past."""
        response = self._ask("what was the keep material?")
        self.assertIn("I have no record of keep material ever being "
                      "different", response)
        self.assertIn("oak", response)

    def test_an_unheld_subject_gets_a_hedge_not_a_past(self) -> None:
        response = self._ask("what was the citadel material?")
        self.assertFalse(response.startswith("It was"))

    def test_the_present_tense_is_untouched(self) -> None:
        response = self._ask("what is the tower material?")
        self.assertIn("bronze", response)
        self.assertNotIn("It was", response)


class GateVerdictTests(unittest.TestCase):
    """The PASS, the shuffle lesson, and the three tenses, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import history_gate

        doc = " ".join(history_gate.__doc__.split())
        self.assertIn("+0.873 at 11.16x seed sd", doc)
        self.assertIn("zero histories invented", doc)

    def test_the_shuffle_lesson_is_recorded(self) -> None:
        from ultraquant.experiments import history_gate

        doc = " ".join(history_gate.__doc__.split())
        self.assertIn("the statement SEQUENCE is the semantics", doc)
        self.assertIn("built the way the tested thing actually "
                      "happens", doc)

    def test_the_three_tenses_are_recorded(self) -> None:
        from ultraquant.experiments import history_gate

        doc = " ".join(history_gate.__doc__.split())
        self.assertIn("answers in three tenses", doc)


if __name__ == "__main__":
    unittest.main()
