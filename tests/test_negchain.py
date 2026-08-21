"""Denials speak at terminals, and only at terminals.

§11.49's registered claim: a negated fact may be reached as a chain
terminal and answer "believed not X" with its trail — while §11.48's
inertness holds everywhere else, re-measured with the door open. The
gate passed at +0.694, 2.09x seed sd, with zero middle-bridging in
either arm; these tests pin the door's exact width.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.memory.systematic import SystematicMemory
from ultraquant.reason.inference import infer


def _memory(facts: dict) -> SystematicMemory:
    memory = SystematicMemory()
    for key, spec in facts.items():
        if isinstance(spec, tuple):
            value, negated = spec
        else:
            value, negated = spec, False
        memory.remember_fact(key, value, confidence=0.6, negated=negated)
    return memory


class TerminalVoiceTests(unittest.TestCase):
    """The one door: reached, spoken, never crossed."""

    def test_a_negated_terminal_answers_with_polarity(self) -> None:
        memory = _memory({
            "dome city": "york",
            "york climate": ("temperate", True),
        })
        result = infer("what is the dome city climate?", memory)
        self.assertIsNotNone(result)
        self.assertIn("believed not temperate", result.answer)
        self.assertIn("via york", result.answer)
        self.assertTrue(result.negated)

    def test_the_premise_trail_reads_the_denial_as_a_denial(self) -> None:
        memory = _memory({
            "dome city": "york",
            "york climate": ("temperate", True),
        })
        text = infer("what is the dome city climate?", memory).describe()
        self.assertIn("york climate is not temperate", text)

    def test_terminal_negation_composes_with_depth(self) -> None:
        """§11.47's two bridges carry §11.49's terminal."""
        memory = _memory({
            "tower architect": "wren",
            "wren birthplace": "york",
            "york climate": ("temperate", True),
        })
        result = infer(
            "what is the tower architect birthplace climate?", memory)
        self.assertIsNotNone(result)
        self.assertIn("believed not temperate", result.answer)
        self.assertIn("via wren, york", result.answer)

    def test_a_negated_middle_still_never_bridges(self) -> None:
        """The §11.48 control, re-asserted with the door open: opening
        the terminal must not have reopened the bridge."""
        memory = _memory({
            "dome material": ("steel", True),
            "steel melting point": "1370 degrees",
        })
        self.assertIsNone(infer("what is the dome melting point?",
                                memory))


class ConsolidationPolarityTests(unittest.TestCase):
    """An affirmed negative conclusion stays a denial forever."""

    def test_consolidated_denials_carry_the_flag(self) -> None:
        memory = SystematicMemory()
        memory.consolidate_fact(
            "dome city climate", "temperate", confidence=0.6,
            premises=[("dome city", "york"),
                      ("york climate", "not temperate")],
            negated=True)
        record = memory.recall_fact("dome city climate")
        self.assertTrue(record.get("negated"))

    def test_a_consolidated_denial_is_inert_as_a_bridge(self) -> None:
        memory = SystematicMemory()
        memory.consolidate_fact(
            "dome material", "steel", confidence=0.6,
            premises=[("dome alloy", "not steel")], negated=True)
        memory.remember_fact("steel melting point", "1370 degrees",
                             confidence=0.6)
        self.assertIsNone(infer("what is the dome melting point?",
                                memory))

    def test_the_pipeline_affirm_consolidates_the_polarity(self) -> None:
        from ultraquant.interpreter.thoughts import (build_session,
                                                     run_pipeline)

        root = Path(tempfile.mkdtemp(prefix="uq_negchain_"))
        try:
            session = build_session(root, seed=0)
            run_pipeline("the dome city is york", session)
            run_pipeline("the york climate is not temperate", session)
            response, _trace = run_pipeline(
                "what is the dome city climate?", session)
            self.assertIn("believed not temperate", response)
            response, _trace = run_pipeline("yes", session)
            self.assertIn("Consolidated", response)
            self.assertIn("not temperate", response)
            record = session.memory.recall_fact("dome city climate")
            self.assertIsNotNone(record)
            self.assertTrue(record.get("negated"))
        finally:
            shutil.rmtree(root, ignore_errors=True)


class GateVerdictTests(unittest.TestCase):
    """The PASS and the door's width, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import negchain_gate

        doc = " ".join(negchain_gate.__doc__.split())
        self.assertIn("+0.694 at 2.09x seed sd", doc)

    def test_the_door_width_is_recorded(self) -> None:
        from ultraquant.experiments import negchain_gate

        doc = " ".join(negchain_gate.__doc__.split())
        self.assertIn("zero times in EITHER arm", doc)
        self.assertIn("exactly its stated width", doc)

    def test_the_depth_composition_is_recorded(self) -> None:
        from ultraquant.experiments import negchain_gate

        doc = " ".join(negchain_gate.__doc__.split())
        self.assertIn("via wren, york", doc)


if __name__ == "__main__":
    unittest.main()
