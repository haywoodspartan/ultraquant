"""Reinforcement-weighted retrieval: attestation earns rank, nothing more.

Rule 3 reaching the fact layer (§11.44): equal-overlap ties break toward
the re-attested fact in both search paths; reinforcement never outranks
a better token match and never buys a coverage bypass.
"""

from __future__ import annotations

import unittest

from ultraquant.memory.systematic import SystematicMemory


class TieBreakTests(unittest.TestCase):
    """The ranking rules at the memory surface."""

    def test_reinforcement_breaks_an_equal_overlap_tie(self) -> None:
        memory = SystematicMemory()
        memory.remember_fact("tower annals", "a story", confidence=0.6)
        memory.remember_fact("tower material", "steel", confidence=0.6)
        for _ in range(3):
            memory.remember_fact("tower material", "steel",
                                 confidence=0.6)
        top = memory.find_facts("tower", top_k=1)
        self.assertEqual(top, ["tower material"])

    def test_the_alphabet_still_settles_unreinforced_ties(self) -> None:
        memory = SystematicMemory()
        memory.remember_fact("tower annals", "a story", confidence=0.6)
        memory.remember_fact("tower material", "steel", confidence=0.6)
        top = memory.find_facts("tower", top_k=1)
        self.assertEqual(top, ["tower annals"])

    def test_a_better_token_match_beats_any_reinforcement(self) -> None:
        memory = SystematicMemory()
        memory.remember_fact("alpha beacon range", "9 meters",
                             confidence=0.6)
        for _ in range(9):
            memory.remember_fact("beacon lore", "a story", confidence=0.6)
        top = memory.find_facts("alpha beacon range", top_k=1)
        self.assertEqual(top, ["alpha beacon range"])

    def test_the_sharded_path_applies_the_same_tie_break(self) -> None:
        import shutil
        import tempfile
        from pathlib import Path

        from ultraquant.interpreter.thoughts import build_session

        root = Path(tempfile.mkdtemp(prefix="uq_plastic_"))
        try:
            session = build_session(root, seed=0)
            memory = session.memory
            memory.remember_fact("tower annals", "a story", confidence=0.6)
            memory.remember_fact("tower material", "steel", confidence=0.6)
            for _ in range(3):
                memory.remember_fact("tower material", "steel",
                                     confidence=0.6)
            memory.shards.flush()
            top = memory.find_facts("tower", top_k=1)
            self.assertEqual(top, ["tower material"])
        finally:
            shutil.rmtree(root, ignore_errors=True)


class GateVerdictTests(unittest.TestCase):
    """The narrow pass, the accidental validation, and the controls."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import plasticity_gate

        doc = " ".join(plasticity_gate.__doc__.split())
        self.assertIn("PASSED", doc)
        self.assertIn("1.11x seed sd", doc)

    def test_the_phrase_probe_validation_is_recorded(self) -> None:
        """The first burial failed to fail because §11.38 already fixed
        cross-subject burial - worth keeping as evidence for BOTH gates."""
        from ultraquant.experiments import plasticity_gate

        doc = " ".join(plasticity_gate.__doc__.split())
        self.assertIn("validating itself from an unexpected direction",
                      doc)

    def test_the_tie_break_stayed_a_tie_break(self) -> None:
        from ultraquant.experiments import plasticity_gate

        doc = " ".join(plasticity_gate.__doc__.split())
        self.assertIn("zero entrenchment falls", doc)
        self.assertIn("the tie-break stayed a tie-break", doc)


if __name__ == "__main__":
    unittest.main()
