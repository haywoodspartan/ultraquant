"""Two adjectives may decorate, behind the residual floor.

§11.46's registered claim: the inference path's modifier cap moves from
one to two, and the safety line moves WITH it — at least two known
informative tokens must survive the drop, and convergence still demands
a bridge. The gate measured the move (+0.583 at 1.13x seed sd, zero
decoy assertions of either form); these tests pin the mechanism and the
verdict.
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
    for key, value in facts.items():
        memory.remember_fact(key, value, confidence=0.6)
    return memory


class TwoModifierTests(unittest.TestCase):
    """The second drop, its naming, and its refusals."""

    def setUp(self) -> None:
        self.memory = _memory({
            "tower material": "steel",
            "steel melting point": "1370 degrees",
        })

    def test_two_modifiers_read_through_the_bridge(self) -> None:
        result = infer(
            "what is the weathered crumbling tower melting point?",
            self.memory)
        self.assertIsNotNone(result)
        self.assertIn("1370", result.answer)
        self.assertIn("reading 'weathered' and 'crumbling' as modifiers",
                      result.answer)

    def test_modifiers_are_named_in_question_order(self) -> None:
        """'crumbling weathered' must not come back alphabetised - the
        reader vetoes a reading, and the reading is of THIS question."""
        result = infer(
            "what is the crumbling weathered tower melting point?",
            self.memory)
        self.assertIsNotNone(result)
        self.assertIn("reading 'crumbling' and 'weathered' as modifiers",
                      result.answer)

    def test_three_unknowns_refuse(self) -> None:
        """The cap. Three unknown tokens is a question the library does
        not understand - §11.46 records this ceiling as §11.36 recorded
        its own."""
        self.assertIsNone(infer(
            "what is the ancient weathered crumbling tower melting point?",
            self.memory))

    def test_the_residual_floor_refuses_a_thin_question(self) -> None:
        """Two drops into a one-token residual is a question read away,
        not a question answered."""
        self.assertIsNone(infer(
            "what is the ancient obelisk density?", self.memory))

    def test_an_unknown_entity_hides_behind_no_adjective(self) -> None:
        """'the melting point of glowing tungsten' drops both unknowns
        and leaves all-direct coverage, which the spread refuses as
        recall's territory - the §11.29 guard surviving the second
        drop."""
        self.assertIsNone(infer(
            "what is the melting point of glowing tungsten?",
            self.memory))

    def test_one_modifier_wording_is_unchanged(self) -> None:
        """The single drop still says 'as a modifier' - §11.36's
        surface, verbatim."""
        result = infer("what is the weathered tower melting point?",
                       self.memory)
        self.assertIsNotNone(result)
        self.assertIn("reading 'weathered' as a modifier", result.answer)
        self.assertNotIn("as modifiers", result.answer)


class LivePipelineTests(unittest.TestCase):
    """The second drop reaches the chat surface."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import build_session

        self.dir = Path(tempfile.mkdtemp(prefix="uq_multimod_"))
        self.session = build_session(self.dir, seed=0)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_double_modifier_answer_reaches_the_pipeline(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        run_pipeline("the tower material is steel", self.session)
        run_pipeline("the steel melting point is 1370 degrees",
                     self.session)
        response, _trace = run_pipeline(
            "what is the ancient gleaming tower melting point?",
            self.session)
        self.assertIn("1370", response)
        self.assertIn("reading 'ancient' and 'gleaming' as modifiers",
                      response)

    def test_the_thin_decoy_demotes_and_never_asserts(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        run_pipeline("the steel density is 900 units", self.session)
        response, _trace = run_pipeline(
            "what is the ancient obelisk density?", self.session)
        self.assertTrue(response.startswith("I don't hold"),
                        f"the ghost got an answer: {response!r}")


class GateVerdictTests(unittest.TestCase):
    """The narrow pass, the harness indictment, and the floor, pinned."""

    def test_the_recorded_verdict_is_a_narrow_pass(self) -> None:
        from ultraquant.experiments import multimodifier_gate

        doc = " ".join(multimodifier_gate.__doc__.split())
        self.assertIn("PASSED — narrowly", doc)
        self.assertIn("1.13x seed sd", doc)

    def test_the_harness_indictment_is_recorded(self) -> None:
        """The first run counted §11.29's demote hedge as fabrication -
        24/24 in BOTH arms said harness, not mechanism."""
        from ultraquant.experiments import multimodifier_gate

        doc = " ".join(multimodifier_gate.__doc__.split())
        self.assertIn("the first run indicted the harness", doc)
        self.assertIn("corrected to the registered word", doc)

    def test_the_residual_floor_is_recorded(self) -> None:
        from ultraquant.experiments import multimodifier_gate

        doc = " ".join(multimodifier_gate.__doc__.split())
        self.assertIn("the residual floor held", doc)

    def test_the_next_ceiling_is_recorded(self) -> None:
        from ultraquant.experiments import multimodifier_gate

        doc = " ".join(multimodifier_gate.__doc__.split())
        self.assertIn("a registered candidate, not a promise", doc)


if __name__ == "__main__":
    unittest.main()
