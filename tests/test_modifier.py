"""Adjective tolerance: one unknown token may decorate, never substitute.

§11.35's registered claim, closed: the inference path drops at most one
library-unknown token and retries the spread, safe because convergence
still demands a bridge — while the recall path never gets the drop,
because there it would assert steel's number for tungsten.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.memory.systematic import SystematicMemory
from ultraquant.reason.inference import infer, missing_premise


def _memory(facts: dict) -> SystematicMemory:
    memory = SystematicMemory()
    for key, value in facts.items():
        memory.remember_fact(key, value, confidence=0.6)
    return memory


class ModifierToleranceTests(unittest.TestCase):
    """The drop, its label, and its refusals."""

    def setUp(self) -> None:
        self.memory = _memory({
            "tower material": "steel",
            "steel melting point": "1370 degrees",
        })

    def test_one_modifier_reads_through_the_bridge(self) -> None:
        result = infer("what is the weathered tower melting point?",
                       self.memory)
        self.assertIsNotNone(result)
        self.assertIn("1370", result.answer)
        self.assertIn("reading 'weathered' as a modifier", result.answer)

    def test_an_unknown_entity_is_not_a_modifier(self) -> None:
        """Dropping 'tungsten' leaves all-direct coverage, which the
        spread refuses as plain recall's territory - the safety claim."""
        self.assertIsNone(infer("what is the melting point of tungsten?",
                                self.memory))

    def test_two_unknown_tokens_refuse(self) -> None:
        """Two unknowns is not decoration; it is a question the library
        does not understand, and saying so beats guessing."""
        self.assertIsNone(infer(
            "what is the weathered crumbling tower melting point?",
            self.memory))

    def test_a_clean_question_carries_no_modifier_label(self) -> None:
        result = infer("what is the tower melting point?", self.memory)
        self.assertIsNotNone(result)
        self.assertNotIn("modifier", result.answer)

    def test_curiosity_survives_a_modifier(self) -> None:
        """'the weathered tower conductivity' must still ask for the
        steel conductivity, not go silent on the unknown word."""
        gap = missing_premise("what is the weathered tower conductivity?",
                              self.memory)
        self.assertIsNotNone(gap)
        self.assertEqual(gap["premise_key"], "steel conductivity")


class RecallPathSafetyTests(unittest.TestCase):
    """The drop must never reach direct recall."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import build_session

        self.dir = Path(tempfile.mkdtemp(prefix="uq_modifier_"))
        self.session = build_session(self.dir, seed=0)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_tungsten_still_demotes_on_the_live_pipeline(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        run_pipeline("the steel melting point is 1370 degrees",
                     self.session)
        response, _trace = run_pipeline(
            "what is the melting point of tungsten?", self.session)
        self.assertTrue(response.startswith("I don't hold"),
                        f"the wrong metal asserted again: {response!r}")

    def test_the_modifier_answer_reaches_the_pipeline(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        run_pipeline("the tower material is steel", self.session)
        run_pipeline("the steel melting point is 1370 degrees",
                     self.session)
        response, _trace = run_pipeline(
            "what is the ancient tower melting point?", self.session)
        self.assertIn("1370", response)
        self.assertIn("as a modifier", response)


class GateVerdictTests(unittest.TestCase):
    """The PASS and the structural safety argument, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import modifier_gate

        doc = " ".join(modifier_gate.__doc__.split())
        self.assertIn("PASSED", doc)
        self.assertIn("1.35x seed sd", doc)

    def test_zero_fabrications_in_both_arms_is_recorded(self) -> None:
        from ultraquant.experiments import modifier_gate

        doc = " ".join(modifier_gate.__doc__.split())
        self.assertIn("zero unknown-entity fabrications", doc)

    def test_the_structural_argument_is_recorded(self) -> None:
        """Tolerance without a new hole, because the old guard guards."""
        from ultraquant.experiments import modifier_gate

        doc = " ".join(modifier_gate.__doc__.split())
        self.assertIn("the old guard does the guarding", doc)


if __name__ == "__main__":
    unittest.main()
