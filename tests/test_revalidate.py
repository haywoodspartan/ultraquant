"""A curiosity is revalidated before it is asked.

11.120: the learn survey read stored gaps back verbatim - asking for
premises it now held, and stating bridges it no longer believed. These
pins hold the revalidation, both polarity cases review found, the arm,
and the gate's record.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.interpreter import learning as L
from ultraquant.interpreter import thoughts as T


def _prompts(lines: list, on: bool = True) -> list:
    previous = L._CURIOSITY_REVALIDATE
    L._CURIOSITY_REVALIDATE = on
    root = Path(tempfile.mkdtemp(prefix="uq_revalidate_t_"))
    try:
        session = T.build_session(root, seed=0)
        for line in lines:
            T.run_pipeline(line, session)
        return [q.prompt for q in L.LearningSession(session).survey()
                if q.kind == "missing-premise"]
    finally:
        L._CURIOSITY_REVALIDATE = previous
        shutil.rmtree(root, ignore_errors=True)


_GAP = ["the tower material is steel", "what is the tower conductivity?"]
_ASKS = "What is the steel conductivity?"
_STALE = "I hold that tower material is steel;"


class RevalidationTests(unittest.TestCase):

    def test_a_live_gap_is_asked_once(self) -> None:
        prompts = _prompts(_GAP + ["what is the tower conductivity?"])
        self.assertEqual(sum(_ASKS in p for p in prompts), 1)

    def test_a_fulfilled_gap_is_not_asked(self) -> None:
        prompts = _prompts(_GAP + ["the steel conductivity is high"])
        self.assertFalse(any(_ASKS in p for p in prompts))

    def test_a_revised_bridge_is_not_stated(self) -> None:
        prompts = _prompts(_GAP + ["the tower material is iron"])
        self.assertFalse(any(_STALE in p for p in prompts))

    def test_a_bridge_flipped_to_a_denial_is_not_stated(self) -> None:
        """Review found this one: the value stays 'steel', negated."""
        prompts = _prompts(_GAP + ["the tower material is not steel"])
        self.assertFalse(any(_STALE in p for p in prompts))

    def test_a_premise_held_only_as_a_denial_is_still_asked(self) -> None:
        """A denial names no value (11.48): the chain still cannot close."""
        prompts = _prompts(_GAP + ["the steel conductivity is not high"])
        self.assertEqual(sum(_ASKS in p for p in prompts), 1)

    def test_the_flag_restores_the_old_survey(self) -> None:
        prompts = _prompts(_GAP + ["the steel conductivity is high"],
                           on=False)
        self.assertTrue(any(_ASKS in p for p in prompts))

    def test_the_fix_is_the_shipped_path(self) -> None:
        self.assertTrue(L._CURIOSITY_REVALIDATE)


class GateVerdictTests(unittest.TestCase):

    def setUp(self) -> None:
        from ultraquant.experiments import revalidate_gate
        self.gate = revalidate_gate
        self.doc = " ".join(revalidate_gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("No fulfilled prompts", "No stale-bridge prompts",
                       "Live gaps survive",
                       "The teaching loop still closes",
                       "Nothing else moves", "The full suite is green"):
            self.assertIn(phrase, self.doc)

    def test_the_amendment_is_recorded_with_its_reason(self) -> None:
        self.assertIn("before its fixed arm ever ran", self.doc)
        self.assertEqual(len(self.gate.FLIPS), 2)

    def test_the_result_is_recorded(self) -> None:
        self.assertIn("PASSED on all five measured criteria", self.doc)
        self.assertIn("the two polarity flips review found", self.doc)


if __name__ == "__main__":
    unittest.main()
