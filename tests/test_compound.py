"""Compound questions: the splitter, the composition, and the hygiene.

A conjunction decomposes into parts that each get the full single-question
machinery; an arithmetic "and" does not; and no compound may ever again
mint a clause as a curiosity key. Every guard names the junk it prevents.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.interpreter.thoughts import Reason
from ultraquant.memory.systematic import SystematicMemory
from ultraquant.reason.inference import missing_premise


class SplitterTests(unittest.TestCase):
    """What decomposes, and what must not."""

    def test_a_real_conjunction_splits(self) -> None:
        parts = Reason._compound_parts(
            "what is the tower melting point and the bridge length?")
        self.assertEqual(parts, ["the tower melting point",
                                 "the bridge length"])

    def test_a_combine_word_vetoes_the_split(self) -> None:
        """'The sum of A and B' is ONE question about two facts;
        decomposing it would break the arithmetic that answers it."""
        self.assertIsNone(Reason._compound_parts(
            "what is the sum of the tower height and the bridge height?"))

    def test_a_degenerate_fragment_vetoes_the_split(self) -> None:
        """'the tower height and more' has no second question in it."""
        self.assertIsNone(Reason._compound_parts(
            "what is the tower height and more?"))

    def test_no_and_means_no_split(self) -> None:
        self.assertIsNone(Reason._compound_parts(
            "what is the tower height?"))

    def test_three_parts_split(self) -> None:
        parts = Reason._compound_parts(
            "what is the tower height and the bridge length and "
            "the spire width?")
        self.assertEqual(len(parts), 3)


class CompoundAnsweringTests(unittest.TestCase):
    """The live composition: recall beside inference beside honesty."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import build_session

        self.dir = Path(tempfile.mkdtemp(prefix="uq_compound_"))
        self.session = build_session(self.dir, seed=0)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_recall_and_inference_compose_in_one_reply(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        run_pipeline("the tower material is steel", self.session)
        run_pipeline("the steel melting point is 1370 degrees", self.session)
        run_pipeline("the bridge length is 2 kilometers", self.session)
        response, _trace = run_pipeline(
            "what is the tower melting point and the bridge length?",
            self.session)
        self.assertIn("1370", response)
        self.assertIn("2 kilometers", response)
        self.assertIn("inferred", response)

    def test_a_half_known_compound_names_its_gap(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        run_pipeline("the bridge length is 2 kilometers", self.session)
        run_pipeline("the tower material is steel", self.session)
        response, _trace = run_pipeline(
            "what is the tower conductivity and the bridge length?",
            self.session)
        self.assertIn("2 kilometers", response)
        self.assertIn("missing", response.lower())
        self.assertIn("steel conductivity", response)

    def test_the_part_curiosity_is_well_formed(self) -> None:
        """The compound's unknown part asks for 'steel conductivity' -
        never the old clause-key 'steel melting point bridge length'."""
        from ultraquant.interpreter.thoughts import run_pipeline

        run_pipeline("the bridge length is 2 kilometers", self.session)
        run_pipeline("the tower material is steel", self.session)
        run_pipeline(
            "what is the tower conductivity and the bridge length?",
            self.session)
        keys = [c["premise_key"] for c in self.session.curiosities]
        self.assertEqual(keys, ["steel conductivity"])


class CuriosityHygieneTests(unittest.TestCase):
    """The two minting guards the compound probe exposed."""

    def test_a_combine_question_mints_nothing(self) -> None:
        """'the sum of ...' once minted 'steel sum height bridge height';
        a combination's gaps belong to its parts."""
        memory = SystematicMemory()
        memory.remember_fact("tower material", "steel", confidence=0.6)
        self.assertIsNone(missing_premise(
            "what is the sum of the tower height and the bridge height?",
            memory))

    def test_a_clause_wide_remainder_mints_nothing(self) -> None:
        """A remainder spanning a second clause is not a fact key."""
        memory = SystematicMemory()
        memory.remember_fact("tower material", "steel", confidence=0.6)
        self.assertIsNone(missing_premise(
            "what is the tower melting point and the bridge length?",
            memory))

    def test_remainder_words_do_not_duplicate(self) -> None:
        memory = SystematicMemory()
        memory.remember_fact("tower material", "steel", confidence=0.6)
        gap = missing_premise("what is the tower point point?", memory)
        if gap is not None:
            words = gap["premise_key"].split()
            self.assertEqual(len(words), len(set(words)))


class GateVerdictTests(unittest.TestCase):
    """The PASS, the brittleness, and the three corrections, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import compound_gate

        doc = " ".join(compound_gate.__doc__.split())
        self.assertIn("PASSED", doc)
        self.assertIn("1.76x seed sd", doc)

    def test_the_adjective_brittleness_is_recorded(self) -> None:
        from ultraquant.experiments import compound_gate

        doc = " ".join(compound_gate.__doc__.split())
        self.assertIn("weathered tower", doc)
        self.assertIn("real limit", doc)

    def test_the_broken_decoy_correction_is_recorded(self) -> None:
        """A decoy failing in both arms is a broken decoy, not a held one
        - the distinction §11.11 exists for."""
        from ultraquant.experiments import compound_gate

        doc = " ".join(compound_gate.__doc__.split())
        self.assertIn("a broken decoy, not a held one", doc)

    def test_the_power_arithmetic_is_recorded(self) -> None:
        from ultraquant.experiments import compound_gate

        doc = " ".join(compound_gate.__doc__.split())
        self.assertIn("mean 0.5 against sd 0.5", doc)


if __name__ == "__main__":
    unittest.main()
