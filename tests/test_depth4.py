"""The fourth fact: the stated limit moved by measurement.

§11.52's registered claim: three bridge hops under the full rule stack
add reach without adding puns (+0.708 at 2.37x seed sd, zero decoy
assertions at depth), so `_MAX_HOPS = 3` and `_FLOOR = 0.1` shipped.
test_inference pins the new decay arithmetic; these tests pin the
capability, its decoys, and the verdict.
"""

from __future__ import annotations

import unittest

from ultraquant.memory.systematic import SystematicMemory
from ultraquant.reason.inference import infer


def _memory(facts: dict) -> SystematicMemory:
    memory = SystematicMemory()
    for key, value in facts.items():
        memory.remember_fact(key, value, confidence=0.6)
    return memory


_CHAIN = {
    "the tower architect": "wren",
    "wren hometown": "york",
    "york region": "norland",
    "the norland climate": "temperate",
}


class FourChainTests(unittest.TestCase):
    """Three bridges answer, with the whole trail."""

    def test_a_four_fact_chain_answers_with_all_middles(self) -> None:
        result = infer(
            "what is the tower architect hometown region climate?",
            _memory(_CHAIN))
        self.assertIsNotNone(result)
        self.assertIn("temperate", result.answer)
        self.assertIn("via wren, york, norland", result.answer)
        self.assertEqual(len(result.premises), 4)

    def test_a_broken_middle_refuses_at_depth(self) -> None:
        held = {k: v for k, v in _CHAIN.items() if "hometown" not in k}
        self.assertIsNone(infer(
            "what is the tower architect hometown region climate?",
            _memory(held)))

    def test_an_epithet_value_still_refuses_at_depth(self) -> None:
        """The §11.47 pun family, one level deeper: 'wren the younger'
        may not ride plain wren's hometown through three bridges."""
        memory = _memory(dict(_CHAIN, **{
            "the spire architect": "wren the younger",
        }))
        self.assertIsNone(infer(
            "what is the spire architect hometown region climate?",
            memory))

    def test_confidence_dilutes_to_the_weakest_of_four(self) -> None:
        memory = SystematicMemory()
        for spot, (key, value) in enumerate(_CHAIN.items()):
            memory.remember_fact(key, value,
                                 confidence=0.9 if spot != 2 else 0.3)
        result = infer(
            "what is the tower architect hometown region climate?",
            memory)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.confidence, 0.3)


class GateVerdictTests(unittest.TestCase):
    """The PASS and the re-measured limit, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import depth4_gate

        doc = " ".join(depth4_gate.__doc__.split())
        self.assertIn("+0.708 at 2.37x seed sd", doc)
        self.assertIn("ZERO decoy assertions at depth", doc)

    def test_the_warning_history_is_recorded(self) -> None:
        """§11.31's warning was right when written, and the record says
        both that it was right and why it no longer binds."""
        from ultraquant.experiments import depth4_gate

        doc = " ".join(depth4_gate.__doc__.split())
        self.assertIn("right when it was written", doc)
        self.assertIn("It lives in the structural rules now", doc)

    def test_the_next_limit_is_recorded_as_measurable(self) -> None:
        from ultraquant.experiments import depth4_gate

        doc = " ".join(depth4_gate.__doc__.split())
        self.assertIn("a measurement away from moving, never a promise",
                      doc)


if __name__ == "__main__":
    unittest.main()
