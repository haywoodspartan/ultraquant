"""Polarity: belief-of-absence, inhibitory everywhere, honest aloud.

§11.48's registered claim. A denial stored as a value once derived the
dome's melting point "via not steel"; negation is now a flag on the
record, part of the fact's identity, excluded from every excitatory
path, and surfaced in speech. Polar questions answer yes, no, or an
honest don't-know — and absence is never no.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.memory.systematic import SystematicMemory
from ultraquant.reason.inference import infer, missing_premise


class StorageIdentityTests(unittest.TestCase):
    """Polarity is part of what a fact IS."""

    def test_same_polarity_reinforces(self) -> None:
        memory = SystematicMemory()
        memory.remember_fact("dome material", "steel", confidence=0.6,
                             negated=True)
        memory.remember_fact("dome material", "steel", confidence=0.6,
                             negated=True)
        record = memory.recall_fact("dome material")
        self.assertEqual(record["reinforcements"], 1)
        self.assertTrue(record.get("negated"))

    def test_a_polarity_flip_is_a_revision(self) -> None:
        """"It IS steel" after "it is NOT steel" is a change of mind,
        and the record must say so - same value, opposite belief."""
        memory = SystematicMemory()
        memory.remember_fact("dome material", "steel", confidence=0.6,
                             negated=True)
        memory.remember_fact("dome material", "steel", confidence=0.6)
        record = memory.recall_fact("dome material")
        self.assertFalse(record.get("negated"))
        self.assertEqual(record["reinforcements"], 0)
        revisions = [e for e in memory.recall_episodes(limit=5)
                     if e["kind"] == "revision"]
        self.assertEqual(len(revisions), 1)


class InhibitionTests(unittest.TestCase):
    """A denial never excites the network."""

    def test_a_denial_never_bridges(self) -> None:
        """The opening fabrication: "dome material is not steel" plus
        steel's melting point derived the dome's melting point "via
        not steel"."""
        memory = SystematicMemory()
        memory.remember_fact("dome material", "steel", confidence=0.6,
                             negated=True)
        memory.remember_fact("steel melting point", "1370 degrees",
                             confidence=0.6)
        self.assertIsNone(infer("what is the dome melting point?",
                                memory))

    def test_a_denial_never_joins_arithmetic(self) -> None:
        memory = SystematicMemory()
        memory.remember_fact("tower height", "300 meters",
                             confidence=0.6)
        memory.remember_fact("bridge height", "120 meters",
                             confidence=0.6, negated=True)
        self.assertIsNone(infer(
            "what is the sum of the tower height and the bridge "
            "height?", memory))

    def test_a_denial_never_mints_a_premise(self) -> None:
        """"not steel" once minted the curiosity key "not steel
        steel"."""
        memory = SystematicMemory()
        memory.remember_fact("dome material", "steel", confidence=0.6,
                             negated=True)
        gap = missing_premise("what is the dome conductivity?", memory)
        if gap is not None:
            self.assertNotIn("steel steel", gap["premise_key"])


class PolarQuestionTests(unittest.TestCase):
    """Yes against belief, no against contrary belief, open otherwise."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import (build_session,
                                                     run_pipeline)

        self.dir = Path(tempfile.mkdtemp(prefix="uq_negation_"))
        self.session = build_session(self.dir, seed=0)
        self._run = run_pipeline
        for statement in ("the tower material is iron",
                          "the dome material is not steel"):
            run_pipeline(statement, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _ask(self, question: str) -> str:
        response, _trace = self._run(question, self.session)
        return response

    def test_a_matching_claim_is_yes(self) -> None:
        self.assertTrue(self._ask("is the tower material iron?")
                        .startswith("Yes"))

    def test_a_contrary_claim_is_no_with_the_actual(self) -> None:
        response = self._ask("is the tower material steel?")
        self.assertTrue(response.startswith("No"))
        self.assertIn("iron", response)

    def test_a_negated_claim_inverts(self) -> None:
        self.assertTrue(self._ask("is the tower material not steel?")
                        .startswith("Yes"))

    def test_a_stored_negation_answers_no(self) -> None:
        response = self._ask("is the dome material steel?")
        self.assertTrue(response.startswith("No"))
        self.assertIn("believed not steel", response)

    def test_a_negation_says_nothing_about_other_values(self) -> None:
        response = self._ask("is the dome material iron?")
        self.assertTrue(response.startswith("I don't know"))
        self.assertIn("not steel", response)

    def test_absence_is_never_no(self) -> None:
        """The line the gate holds at zero tolerance: an unheld subject
        hedges - it is not denied."""
        response = self._ask("is the citadel material steel?")
        self.assertFalse(response.startswith("No"))
        self.assertFalse(response.startswith("Yes"))

    def test_direct_answers_speak_the_polarity(self) -> None:
        response = self._ask("what is the dome material?")
        self.assertIn("not steel", response)


class GateVerdictTests(unittest.TestCase):
    """The PASS, the measured fabrication, and the line, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import negation_gate

        doc = " ".join(negation_gate.__doc__.split())
        self.assertIn("+0.727 at 28.33x seed sd", doc)

    def test_the_old_fabrication_is_measured_beside_the_fix(self) -> None:
        from ultraquant.experiments import negation_gate

        doc = " ".join(negation_gate.__doc__.split())
        self.assertIn("straight through a denial 36 times", doc)

    def test_the_line_is_recorded(self) -> None:
        from ultraquant.experiments import negation_gate

        doc = " ".join(negation_gate.__doc__.split())
        self.assertIn("absence is never no", doc.lower())

    def test_the_next_rung_is_recorded(self) -> None:
        from ultraquant.experiments import negation_gate

        doc = " ".join(negation_gate.__doc__.split())
        self.assertIn("its own claim or none", doc)


if __name__ == "__main__":
    unittest.main()
