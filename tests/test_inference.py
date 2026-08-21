"""Spreading-activation inference: convergence, refusal, and the trail.

The mechanism that makes the library answer questions it was never told
the answer to — and the guards that keep it from answering questions it
should not. Every guard here has a fabrication ancestor in the book, so
every test names the wrong answer it prevents.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.memory.systematic import SystematicMemory
from ultraquant.reason.inference import Inference, infer


def _memory(facts: dict) -> SystematicMemory:
    memory = SystematicMemory()
    for key, value in facts.items():
        memory.remember_fact(key, value, confidence=0.6)
    return memory


class ChainInferenceTests(unittest.TestCase):
    """Convergence over one bridge, and the refusals around it."""

    def test_a_two_fact_chain_converges(self) -> None:
        memory = _memory({"tower material": "steel",
                          "steel melting point": "1370 degrees"})
        result = infer("what is the tower melting point?", memory)
        self.assertIsNotNone(result)
        self.assertIn("1370", result.answer)
        self.assertEqual(result.kind, "chain")
        self.assertEqual([k for k, _v in result.premises],
                         ["tower material", "steel melting point"])

    def test_a_distractor_with_partial_coverage_is_ignored(self) -> None:
        """Iron's melting point shares {melting, point} with the question;
        answering from it is the wrong-metal fabrication the keyword
        fallback committed. Convergence requires the tower half too."""
        memory = _memory({"tower material": "steel",
                          "steel melting point": "1370 degrees",
                          "iron melting point": "1538 degrees"})
        result = infer("what is the tower melting point?", memory)
        self.assertIsNotNone(result)
        self.assertIn("1370", result.answer)
        self.assertNotIn("1538", result.answer)

    def test_no_bridge_means_no_answer(self) -> None:
        """A held entity and a held property with nothing connecting them
        must stay silent - shared tokens are not a proof."""
        memory = _memory({"tower material": "steel",
                          "iron melting point": "1538 degrees"})
        self.assertIsNone(infer("what is the tower melting point?", memory))

    def test_a_ghost_entity_gets_nothing(self) -> None:
        memory = _memory({"tower material": "steel",
                          "steel melting point": "1370 degrees"})
        self.assertIsNone(infer("what is the obelisk melting point?",
                                memory))

    def test_confidence_is_the_weakest_premise(self) -> None:
        memory = SystematicMemory()
        memory.remember_fact("tower material", "steel", confidence=0.9)
        memory.remember_fact("steel melting point", "1370 degrees",
                             confidence=0.3)
        result = infer("what is the tower melting point?", memory)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.confidence, 0.3)

    def test_the_reply_is_marked_inferred_with_the_trail(self) -> None:
        memory = _memory({"tower material": "steel",
                          "steel melting point": "1370 degrees"})
        text = infer("what is the tower melting point?", memory).describe()
        self.assertIn("inferred, not stored", text)
        self.assertIn("tower material is steel", text)

    def test_depth_two_reaches_and_depth_three_stops(self) -> None:
        """Activation decays 0.5 per bridge over a 0.2 floor: two bridges
        arrive at 0.25 and answer; three arrive at 0.125 and honestly
        refuse. The floor is the architecture's stated limit, not a bug."""
        deep = _memory({"tower material": "bronzite",
                        "bronzite base": "bronze",
                        "bronze melting point": "913 degrees"})
        result = infer("what is the tower melting point?", deep)
        self.assertIsNotNone(result)
        self.assertIn("913", result.answer)

        deeper = _memory({"tower material": "bronzite",
                          "bronzite base": "bronzoid",
                          "bronzoid base": "bronze",
                          "bronze melting point": "913 degrees"})
        self.assertIsNone(infer("what is the tower melting point?", deeper))


class CombineInferenceTests(unittest.TestCase):
    """Arithmetic over exactly the facts the question names."""

    def test_a_sum_over_two_named_facts(self) -> None:
        memory = _memory({"tower height": "300 meters",
                          "bridge height": "120 meters"})
        result = infer("what is the sum of the tower height and the "
                       "bridge height?", memory)
        self.assertIsNotNone(result)
        self.assertIn("420", result.answer)
        self.assertEqual(result.kind, "combine")

    def test_unnamed_sharers_do_not_break_the_selection(self) -> None:
        """Four facts share 'height'; only the two the question names may
        participate - this exact shape made the first gate run return
        nothing at all."""
        memory = _memory({"tower height": "300 meters",
                          "bridge height": "120 meters",
                          "spire height": "80 meters",
                          "dome height": "44 meters"})
        result = infer("what is the sum of the tower height and the "
                       "bridge height?", memory)
        self.assertIsNotNone(result)
        self.assertIn("420", result.answer)

    def test_related_units_convert_and_unrelated_refuse(self) -> None:
        """300 meters + 2 kilometers is not 302 anything - it is 2.3
        kilometers, through SS11.42's definition table; meters plus
        degrees still refuses, because no definition connects them."""
        memory = _memory({"tower height": "300 meters",
                          "bridge length": "2 kilometers",
                          "oven heat": "200 degrees"})
        result = infer("what is the sum of the tower height and "
                       "the bridge length?", memory)
        self.assertIsNotNone(result)
        self.assertIn("2.3 kilometers", result.answer)
        self.assertIn("units converted", result.answer)
        self.assertNotIn("302", result.answer)
        self.assertIsNone(infer("what is the sum of the tower height and "
                                "the oven heat?", memory))

    def test_larger_names_the_winner(self) -> None:
        memory = _memory({"tower height": "300 meters",
                          "bridge height": "120 meters"})
        result = infer("which is larger, the tower height or the bridge "
                       "height?", memory)
        self.assertIsNotNone(result)
        self.assertIn("tower height", result.answer)
        self.assertIn("larger", result.answer)


class PipelineWiringTests(unittest.TestCase):
    """The live question path: derive before conceding, never fabricate."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import build_session

        self.dir = Path(tempfile.mkdtemp(prefix="uq_infer_"))
        self.session = build_session(self.dir, seed=0)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_the_pipeline_derives_and_says_so(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        run_pipeline("the tower material is steel", self.session)
        run_pipeline("the steel melting point is 1370 degrees", self.session)
        response, _trace = run_pipeline("what is the tower melting point?",
                                        self.session)
        self.assertIn("1370", response)
        self.assertIn("inferred", response)

    def test_the_wrong_metal_is_never_asserted(self) -> None:
        """The fabrication the battery caught: 'melting point of tungsten'
        answered with steel's number as if it were tungsten's. Nearest-held
        may be OFFERED; it must not be asserted as the answer."""
        from ultraquant.interpreter.thoughts import run_pipeline

        run_pipeline("the steel melting point is 1370 degrees", self.session)
        response, _trace = run_pipeline(
            "what is the melting point of tungsten?", self.session)
        self.assertNotIn("inferred", response)
        self.assertTrue(
            response.startswith("I don't hold"),
            f"a non-covering key must demote, got: {response!r}")


class ControlCanFailTests(unittest.TestCase):
    """§11.11's discipline: prove the decoy control is able to fail."""

    def test_removing_the_convergence_rule_fells_the_decoys(self) -> None:
        """An eager spread (answer on ANY activated fact) must fall on the
        gate's decoys - if it did not, the control would be measuring
        nothing and the PASS would be §11.11's defect wearing a verdict."""
        memory = _memory({"tower material": "steel",
                          "steel melting point": "1370 degrees",
                          "iron melting point": "1538 degrees"})

        from ultraquant.reason import inference as mod

        eager_hits = 0
        for question in ("what is the obelisk melting point?",
                         "what is the tower conductivity?"):
            tokens = mod._fold(question)
            facts = mod._reachable_facts(memory,
                                         [" ".join(sorted(tokens))])
            for key in facts:
                if mod._fold(key) & tokens:
                    eager_hits += 1
                    break
        self.assertGreater(
            eager_hits, 0,
            "the decoys must be reachable by an eager matcher, or the "
            "decoy control cannot fail and proves nothing")
        for question in ("what is the obelisk melting point?",
                         "what is the tower conductivity?"):
            self.assertIsNone(infer(question, memory))


class GateVerdictTests(unittest.TestCase):
    """The PASS, the two refused runs, and the floor, pinned."""

    def test_the_recorded_verdict_is_a_pass_at_8x(self) -> None:
        from ultraquant.experiments import inference_gate

        doc = " ".join(inference_gate.__doc__.split())
        self.assertIn("Then it PASSED", doc)
        self.assertIn("8.10x seed sd", doc)

    def test_the_two_saturated_runs_are_recorded(self) -> None:
        """The gate refused its own too-easy protocol twice; hardening the
        worlds - not softening the rule - is the story worth keeping."""
        from ultraquant.experiments import inference_gate

        doc = " ".join(inference_gate.__doc__.split())
        self.assertIn("zero-variance rule", doc)
        self.assertIn("hardened, not the rule", doc)

    def test_the_depth_floor_is_named_as_the_miss(self) -> None:
        from ultraquant.experiments import inference_gate

        doc = " ".join(inference_gate.__doc__.split())
        self.assertIn("depth-3", doc)
        self.assertIn("0.2 floor", doc.replace("under the 0.2 floor",
                                               "under the 0.2 floor"))


if __name__ == "__main__":
    unittest.main()
