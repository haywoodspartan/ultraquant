"""Curiosity from failed inference: the ask, the guards, the closed loop.

"Teach it so that it can learn": a refused convergence names the premise
it was missing, the learn queue asks for exactly that, and answering it
makes the original question answerable. Every guard has the spam it
prevents named beside it.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.memory.systematic import SystematicMemory
from ultraquant.reason.inference import missing_premise


def _memory(facts: dict) -> SystematicMemory:
    memory = SystematicMemory()
    for key, value in facts.items():
        memory.remember_fact(key, value, confidence=0.6)
    return memory


class MissingPremiseTests(unittest.TestCase):
    """The metacognitive half of the spread."""

    def test_the_gap_is_named_through_the_held_bridge(self) -> None:
        memory = _memory({"tower material": "steel"})
        gap = missing_premise("what is the tower conductivity?", memory)
        self.assertIsNotNone(gap)
        self.assertEqual(gap["premise_key"], "steel conductivity")
        self.assertEqual(gap["via_key"], "tower material")

    def test_a_numeric_bridge_generates_nothing(self) -> None:
        """'What is the obelisk melting point?' touches 'steel melting
        point' through attribute tokens, but 1370 degrees is a quantity,
        not a subject - asking about '1370 degrees obelisk' is the spam
        the guard exists to refuse."""
        memory = _memory({"steel melting point": "1370 degrees"})
        self.assertIsNone(missing_premise(
            "what is the obelisk melting point?", memory))

    def test_no_contact_means_no_curiosity(self) -> None:
        memory = _memory({"tower material": "steel"})
        self.assertIsNone(missing_premise(
            "what is the citadel hardness?", memory))

    def test_a_wordy_bridge_value_is_not_a_subject(self) -> None:
        """A value like a whole sentence names nothing askable."""
        memory = _memory({"tower material":
                          "a very complicated alloy of many disputed "
                          "provenances"})
        self.assertIsNone(missing_premise(
            "what is the tower conductivity?", memory))

    def test_remainder_words_keep_question_order(self) -> None:
        memory = _memory({"bridge material": "oak"})
        gap = missing_premise("what is the bridge melting point?", memory)
        self.assertIsNotNone(gap)
        self.assertEqual(gap["premise_key"], "oak melting point")


class ClosedLoopTests(unittest.TestCase):
    """Fail -> ask -> learn -> infer, on the live pipeline."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import build_session

        self.dir = Path(tempfile.mkdtemp(prefix="uq_curioloop_"))
        self.session = build_session(self.dir, seed=0)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_the_refusal_registers_and_hints(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        run_pipeline("the tower material is steel", self.session)
        response, _trace = run_pipeline("what is the tower conductivity?",
                                        self.session)
        self.assertIn("If I knew the steel conductivity", response)
        self.assertEqual(
            [c["premise_key"] for c in self.session.curiosities],
            ["steel conductivity"])

    def test_asking_twice_registers_once(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        run_pipeline("the tower material is steel", self.session)
        run_pipeline("what is the tower conductivity?", self.session)
        run_pipeline("what is the tower conductivity?", self.session)
        self.assertEqual(len(self.session.curiosities), 1)

    def test_the_learn_queue_asks_it_first_and_the_loop_closes(self) -> None:
        from ultraquant.interpreter.learning import LearningSession
        from ultraquant.interpreter.thoughts import run_pipeline

        run_pipeline("the tower material is steel", self.session)
        run_pipeline("what is the tower conductivity?", self.session)

        learning = LearningSession(self.session)
        learning.survey()
        question = learning.next_question()
        self.assertIsNotNone(question)
        self.assertEqual(question.kind, "missing-premise")
        self.assertEqual(question.subject, "steel conductivity")
        self.assertIn("what is the tower conductivity?", question.prompt)

        result = learning.answer(question, "4 megasiemens")
        self.assertTrue(result.accepted)
        self.assertEqual(self.session.curiosities, [])

        response, _trace = run_pipeline("what is the tower conductivity?",
                                        self.session)
        self.assertIn("4 megasiemens", response)
        self.assertIn("inferred", response)

    def test_a_ghost_question_registers_nothing(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        run_pipeline("the steel melting point is 1370 degrees", self.session)
        run_pipeline("what is the obelisk melting point?", self.session)
        self.assertEqual(self.session.curiosities, [])


class GateVerdictTests(unittest.TestCase):
    """The PASS, its ceiling, and the fourth zero-variance refusal."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import curiosity_gate

        doc = " ".join(curiosity_gate.__doc__.split())
        self.assertIn("PASSED", doc)
        self.assertIn("1.62x seed sd", doc)
        self.assertIn("precision 1.000", doc)

    def test_the_ceiling_is_groundedness(self) -> None:
        from ultraquant.experiments import curiosity_gate

        doc = " ".join(curiosity_gate.__doc__.split())
        self.assertIn("curiosity cannot outrun the library's groundedness",
                      doc.lower().replace("c", "c"))

    def test_the_zero_variance_pattern_is_recorded(self) -> None:
        """Fourth deterministic mechanism to ace an only-winnable protocol;
        the correction is always conservative - score the unwinnable."""
        from ultraquant.experiments import curiosity_gate

        doc = " ".join(curiosity_gate.__doc__.split())
        self.assertIn("fourth deterministic mechanism", doc)
        self.assertIn("conservative", doc)


if __name__ == "__main__":
    unittest.main()
