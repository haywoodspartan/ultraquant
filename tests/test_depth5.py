"""The fifth fact: the second limit moved by measurement.

§11.61's registered claim: four bridge hops under the structural rules
add reach without adding puns (+0.896 at 6.96x seed sd, zero decoy
assertions), so `_MAX_HOPS = 4` and `_FLOOR = 0.05` shipped.
test_inference pins the new decay arithmetic and test_ladder the moved
rung; these tests pin the capability and the verdict.
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
    "tower architect": "wren",
    "wren hometown": "york",
    "york region": "norland",
    "norland realm": "aurelia",
    "aurelia anthem": "golden",
}


class FiveChainTests(unittest.TestCase):
    """Four bridges answer, with the whole trail."""

    def test_a_five_fact_chain_answers_with_all_middles(self) -> None:
        result = infer(
            "what is the tower architect hometown region realm anthem?",
            _memory(_CHAIN))
        self.assertIsNotNone(result)
        self.assertIn("golden", result.answer)
        self.assertIn("via wren, york, norland, aurelia", result.answer)
        self.assertEqual(len(result.premises), 5)

    def test_a_broken_middle_refuses_at_depth(self) -> None:
        held = {k: v for k, v in _CHAIN.items() if "region" not in k}
        self.assertIsNone(infer(
            "what is the tower architect hometown region realm anthem?",
            _memory(held)))

    def test_an_epithet_value_still_refuses_at_depth(self) -> None:
        memory = _memory(dict(_CHAIN, **{
            "spire architect": "wren the younger",
        }))
        self.assertIsNone(infer(
            "what is the spire architect hometown region realm anthem?",
            memory))

    def test_confidence_dilutes_to_the_weakest_of_five(self) -> None:
        memory = SystematicMemory()
        for spot, (key, value) in enumerate(_CHAIN.items()):
            memory.remember_fact(key, value,
                                 confidence=0.9 if spot != 3 else 0.2)
        result = infer(
            "what is the tower architect hometown region realm anthem?",
            memory)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.confidence, 0.2)


class GateVerdictTests(unittest.TestCase):
    """The PASS and the method-over-constant close, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import depth5_gate

        doc = " ".join(depth5_gate.__doc__.split())
        self.assertIn("+0.896 at 6.96x seed sd", doc)
        self.assertIn("ZERO decoy assertions at four bridges", doc)

    def test_the_ladder_composition_is_recorded(self) -> None:
        from ultraquant.experiments import depth5_gate

        doc = " ".join(depth5_gate.__doc__.split())
        self.assertIn("moved up with the capability a second time", doc)

    def test_the_method_is_the_record(self) -> None:
        from ultraquant.experiments import depth5_gate

        doc = " ".join(depth5_gate.__doc__.split())
        self.assertIn("the § of record is not the constant but the "
                      "method", doc)


if __name__ == "__main__":
    unittest.main()
