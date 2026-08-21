"""Three-fact chains, and the three defects the depth gate surfaced.

§11.47's registered claim: chains of two bridges are a measured
capability (+0.688 at 2.85x seed sd against a true one-hop baseline),
and getting the measurement honest took three mechanism fixes — the
exact branch's coverage rule, synchronous spreading rounds, and
whole-value bridging. These tests pin each fix and the verdict.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.memory.systematic import SystematicMemory
from ultraquant.reason import inference as inference_module
from ultraquant.reason.inference import infer


def _memory(facts: dict) -> SystematicMemory:
    memory = SystematicMemory()
    for key, value in facts.items():
        memory.remember_fact(key, value, confidence=0.6)
    return memory


_CHAIN = {
    "tower architect": "wren",
    "wren birthplace": "york",
    "york climate": "temperate",
}


class ThreeChainTests(unittest.TestCase):
    """Two bridges answer, with the full trail."""

    def test_a_three_fact_chain_answers_with_both_middles(self) -> None:
        result = infer("what is the tower architect birthplace climate?",
                       _memory(_CHAIN))
        self.assertIsNotNone(result)
        self.assertIn("temperate", result.answer)
        self.assertIn("via wren, york", result.answer)
        self.assertEqual(len(result.premises), 3)

    def test_a_broken_middle_refuses(self) -> None:
        held = {k: v for k, v in _CHAIN.items() if "birthplace" not in k}
        self.assertIsNone(infer(
            "what is the tower architect birthplace climate?",
            _memory(held)))


class SynchronousRoundTests(unittest.TestCase):
    """Hop depth must not depend on iteration order."""

    def test_one_hop_refuses_three_chains_in_either_order(self) -> None:
        """The gate's 'one-hop' arm answered 3-chains at 0.667 because
        in-place origin updates cascaded through a lucky dict order.
        Rounds are frontiers now: one hop is one bridge, both ways
        round."""
        real = inference_module._MAX_HOPS
        try:
            inference_module._MAX_HOPS = 1
            question = "what is the tower architect birthplace climate?"
            forward = _memory(_CHAIN)
            self.assertIsNone(infer(question, forward))
            backward = SystematicMemory()
            for key in reversed(list(_CHAIN)):
                backward.remember_fact(key, _CHAIN[key], confidence=0.6)
            self.assertIsNone(infer(question, backward))
        finally:
            inference_module._MAX_HOPS = real

    def test_two_hops_reach_what_one_cannot(self) -> None:
        result = infer("what is the tower architect birthplace climate?",
                       _memory(_CHAIN))
        self.assertIsNotNone(result)


class WholeValueBridgeTests(unittest.TestCase):
    """A qualifier never drops, in either direction of a hop."""

    def test_an_epithet_value_does_not_borrow_the_plain_chain(self) -> None:
        """'spire architect' is wren THE YOUNGER, and only plain wren
        has a birthplace - the depth pun asserted york's climate for
        an entity whose birthplace was never stored, 72/72 the first
        time it was measured."""
        memory = _memory(dict(_CHAIN, **{
            "spire architect": "wren the younger",
        }))
        self.assertIsNone(infer(
            "what is the spire architect birthplace climate?", memory))

    def test_a_plain_value_does_not_read_the_epithet_fact(self) -> None:
        """The mirror: plain wren must not flow through 'wren the
        younger birthplace' - the unaccounted qualifier sits INSIDE
        the subject, so the hop refuses."""
        memory = _memory({
            "tower architect": "wren",
            "wren the younger birthplace": "delft",
            "delft climate": "wet",
        })
        self.assertIsNone(infer(
            "what is the tower architect birthplace climate?", memory))

    def test_a_terminal_relation_slot_still_traverses(self) -> None:
        """'bronzite BASE' carries one unaccounted token in terminal
        position - the relation slot, origin coverage's own allowance
        applied to every hop. test_inference pins the same chain; this
        pins it beside its refusing siblings."""
        memory = _memory({"tower material": "bronzite",
                          "bronzite base": "bronze",
                          "bronze melting point": "913 degrees"})
        result = infer("what is the tower melting point?", memory)
        self.assertIsNotNone(result)
        self.assertIn("913", result.answer)


class ExactBranchCoverageTests(unittest.TestCase):
    """A sub-key hit answers its own question, never a longer one."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import build_session

        self.dir = Path(tempfile.mkdtemp(prefix="uq_depth_"))
        self.session = build_session(self.dir, seed=0)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_depth_question_answers_the_chain_not_the_subkey(self) -> None:
        """Run one of the gate: 'what is the high mill sculptor
        workshop population?' was told who the sculptor IS, while the
        chain that could answer what was ASKED sat unreached."""
        from ultraquant.interpreter.thoughts import run_pipeline

        for key, value in {
            "the high mill sculptor": "aldous",
            "aldous workshop": "york",
            "the york population": "3230 units",
        }.items():
            self.session.memory.remember_fact(key, value, confidence=0.6)
        response, _trace = run_pipeline(
            "what is the high mill sculptor workshop population?",
            self.session)
        self.assertIn("3230", response)
        self.assertIn("inferred", response)

    def test_an_exact_question_still_answers_from_the_branch(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        self.session.memory.remember_fact("tower material", "steel",
                                          confidence=0.6)
        response, _trace = run_pipeline("what is the tower material?",
                                        self.session)
        self.assertIn("steel", response)
        self.assertNotIn("inferred", response)


class GateVerdictTests(unittest.TestCase):
    """The PASS and all three indicted layers, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import depth_gate

        doc = " ".join(depth_gate.__doc__.split())
        self.assertIn("+0.688 at 2.85x seed sd", doc)

    def test_the_intercept_is_recorded(self) -> None:
        from ultraquant.experiments import depth_gate

        doc = " ".join(depth_gate.__doc__.split())
        self.assertIn("the branch believed \"hit\" meant \"exact\"", doc)

    def test_the_cascade_is_recorded(self) -> None:
        from ultraquant.experiments import depth_gate

        doc = " ".join(depth_gate.__doc__.split())
        self.assertIn("Hop depth depended on dict order", doc)
        self.assertIn("synchronous frontiers", doc)

    def test_the_pun_and_its_rule_are_recorded(self) -> None:
        from ultraquant.experiments import depth_gate

        doc = " ".join(depth_gate.__doc__.split())
        self.assertIn("72/72 asserted through inference", doc)
        self.assertIn("a qualifier being dropped", doc)


if __name__ == "__main__":
    unittest.main()
