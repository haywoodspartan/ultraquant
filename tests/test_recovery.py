"""A recovered turn meets the evidence standard a stored fact does.

11.118: the conversation window's recovered turns were spoken as
answers on any shared word, from turns of any intent. These pins hold
the coverage rule, the intent rule, the arm, and the gate's record.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.interpreter import thoughts as T
from ultraquant.memory.context import ContextWindow

_FILLER = ["hello there", "thanks", "good morning", "hello again"]


def _ask(said: str, question: str, evidence_on: bool = True) -> str:
    """One turn, genuinely evicted by a 64-byte window, then a question."""
    previous = T._RECOVERY_EVIDENCE
    T._RECOVERY_EVIDENCE = evidence_on
    root = Path(tempfile.mkdtemp(prefix="uq_recovery_t_"))
    try:
        session = T.build_session(root, seed=0)
        session.context = ContextWindow(root / "context-small",
                                        budget_bytes=64)
        for line in [said] + _FILLER:
            T.run_pipeline(line, session)
        return T.run_pipeline(question, session)[0]
    finally:
        T._RECOVERY_EVIDENCE = previous
        shutil.rmtree(root, ignore_errors=True)


class RecoveryEvidenceTests(unittest.TestCase):

    def test_a_turn_about_another_subject_is_not_the_answer(self) -> None:
        reply = _ask("steel melting point = 1538",
                     "what is the tungsten melting point?")
        self.assertNotIn("1538", reply)

    def test_a_code_turn_is_not_testimony(self) -> None:
        reply = _ask("code: tower height = 300", "what is the tower height?")
        self.assertNotIn("Earlier in this conversation", reply)

    def test_a_genuine_recovery_still_answers(self) -> None:
        reply = _ask("steel melting point = 1538",
                     "what is the steel melting point?")
        self.assertEqual(reply, "Earlier in this conversation: "
                                "steel melting point is 1538.")

    def test_the_flag_restores_the_old_rung(self) -> None:
        reply = _ask("steel melting point = 1538",
                     "what is the tungsten melting point?",
                     evidence_on=False)
        self.assertIn("steel melting point is 1538", reply)

    def test_the_fix_is_the_shipped_path(self) -> None:
        self.assertTrue(T._RECOVERY_EVIDENCE)


class GateVerdictTests(unittest.TestCase):

    def setUp(self) -> None:
        from ultraquant.experiments import recovery_gate
        self.gate = recovery_gate
        self.doc = " ".join(recovery_gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("No wrong-subject assertion",
                       "No non-statement evidence",
                       "Genuine recoveries survive",
                       "Nothing else moves",
                       "The full suite is green"):
            self.assertIn(phrase, self.doc)

    def test_the_batteries_can_fail(self) -> None:
        self.assertIn("The OLD arm must assert all eight", self.doc)
        self.assertEqual(len(self.gate.PAIRS), 8)
        self.assertEqual(len(self.gate.CODE), 4)

    def test_the_result_is_recorded(self) -> None:
        self.assertIn("PASSED on all four measured criteria", self.doc)
        self.assertIn("nothing is injected", self.doc)


if __name__ == "__main__":
    unittest.main()
