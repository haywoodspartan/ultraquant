"""Three adjectives may decorate, behind the same unmoved floor.

§11.62's registered claim, the modifier family's third cap move: at
most three library-unknown tokens read as decoration, the residual
floor untouched through all three moves. The gate passed at +0.583,
1.13x seed sd — narrowly, on the same dealt hand §11.46 drew — with
zero decoy assertions; these tests pin the rung.
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


class ThreeModifierTests(unittest.TestCase):
    """The third drop, its naming, and its refusals."""

    def setUp(self) -> None:
        self.memory = _memory({
            "tower material": "steel",
            "steel melting point": "1370 degrees",
        })

    def test_three_modifiers_read_through_the_bridge(self) -> None:
        result = infer(
            "what is the ancient weathered crumbling tower melting "
            "point?", self.memory)
        self.assertIsNotNone(result)
        self.assertIn("1370", result.answer)
        self.assertIn("reading 'ancient' and 'weathered' and "
                      "'crumbling' as modifiers", result.answer)

    def test_the_floor_survives_the_third_move(self) -> None:
        """Three drops into a one-token residual still refuse - the
        floor has held through caps one, two, and three without once
        being touched."""
        self.assertIsNone(infer(
            "what is the ancient weathered obelisk density?",
            self.memory))

    def test_four_unknowns_refuse(self) -> None:
        self.assertIsNone(infer(
            "what is the ancient weathered crumbling gleaming tower "
            "melting point?", self.memory))


class GateVerdictTests(unittest.TestCase):
    """The narrow pass and the cap-policy line, pinned."""

    def test_the_recorded_verdict_is_a_narrow_pass(self) -> None:
        from ultraquant.experiments import triplemodifier_gate

        doc = " ".join(triplemodifier_gate.__doc__.split())
        self.assertIn("PASSED — narrowly", doc)
        self.assertIn("+0.583 at 1.13x seed sd", doc)

    def test_the_dealt_hand_is_recorded(self) -> None:
        from ultraquant.experiments import triplemodifier_gate

        doc = " ".join(triplemodifier_gate.__doc__.split())
        self.assertIn("the same dealt hand §11.46 drew", doc)

    def test_the_cap_policy_line_is_recorded(self) -> None:
        from ultraquant.experiments import triplemodifier_gate

        doc = " ".join(triplemodifier_gate.__doc__.split())
        self.assertIn("the cap is policy, the floor is safety", doc)


if __name__ == "__main__":
    unittest.main()
