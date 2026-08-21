"""The depth clock: parity absolute, then the frontier's speed.

§11.59's registered claim: frontier relaying and the per-spread probe
memo eliminate call volume without touching the convergence set. The
gate held parity on every battery question and read the clock at
270.3 -> 84.8 ms (+185.5 ms, 20.27x rep sd); the capstone re-read
726.5 -> 347.4 ms at 9,631 facts with every floor intact. These tests
pin the parity mechanism and the verdict.
"""

from __future__ import annotations

import unittest

from ultraquant.memory.systematic import SystematicMemory
from ultraquant.reason import inference as inference_module
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


_WORLD = {
    "tower architect": "wren",
    "wren hometown": "york",
    "york region": "norland",
    "norland climate": "temperate",
    "keep material": "bronzite",
    "bronzite base": "bronze",
    "bronze melting point": "913 degrees",
    "dome city": "delft",
    "delft climate": ("humid", True),
}

_QUESTIONS = (
    "what is the tower architect hometown region climate?",
    "what is the keep melting point?",
    "what is the weathered keep melting point?",
    "what is the dome city climate?",
)


class FrontierParityTests(unittest.TestCase):
    """Both arms, identical answers, every chain shape."""

    def test_the_arms_answer_identically(self) -> None:
        memory = _memory(_WORLD)
        real = inference_module._FRONTIER_SPREAD
        try:
            for question in _QUESTIONS:
                inference_module._FRONTIER_SPREAD = False
                slow = infer(question, memory)
                inference_module._FRONTIER_SPREAD = True
                fast = infer(question, memory)
                self.assertEqual(
                    slow.answer if slow else None,
                    fast.answer if fast else None,
                    f"arms diverged on {question!r}")
                if slow is not None:
                    self.assertEqual(slow.premises, fast.premises)
                    self.assertEqual(slow.confidence, fast.confidence)
        finally:
            inference_module._FRONTIER_SPREAD = real

    def test_refusals_are_shared_too(self) -> None:
        """A question neither arm may answer must refuse in both -
        parity covers silence. (The first probe here used an unknown
        ENTITY, which modifier tolerance legitimately read away in
        BOTH arms - parity held, the fixture was wrong. No hardness
        facts exist, so no residual can converge.)"""
        memory = _memory(_WORLD)
        real = inference_module._FRONTIER_SPREAD
        try:
            question = "what is the spire hardness?"
            inference_module._FRONTIER_SPREAD = False
            slow = infer(question, memory)
            inference_module._FRONTIER_SPREAD = True
            fast = infer(question, memory)
            self.assertIsNone(slow)
            self.assertIsNone(fast)
        finally:
            inference_module._FRONTIER_SPREAD = real


class GateVerdictTests(unittest.TestCase):
    """The PASS and the refreshed capstone price, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import clock_gate

        doc = " ".join(clock_gate.__doc__.split())
        self.assertIn("parity absolute, then a 3.2x clock", doc)
        self.assertIn("+185.5 ms per depth-4 query at 20.27x "
                      "repetition sd", doc)

    def test_the_licence_is_recorded(self) -> None:
        """Capability before optimization: the unit exists because
        §11.58 measured the price first."""
        from ultraquant.experiments import clock_gate

        doc = " ".join(clock_gate.__doc__.split())
        self.assertIn("§11.58 finally licensed this unit", doc)

    def test_rule_one_is_recorded(self) -> None:
        from ultraquant.experiments import clock_gate

        doc = " ".join(clock_gate.__doc__.split())
        self.assertIn("parity first, the clock second", doc)


if __name__ == "__main__":
    unittest.main()
