"""Polar questions answered through chains, split bugs pinned.

§11.50's registered claim, closing the §11.48 family: a polar question
whose subject is not stored derives it and meets the claim under the
stored-fact matrix. The gate passed at +0.312, 5.11x seed sd, perfect
on the claim and zero on every control; these tests pin the matrix and
the two split bugs building it caught.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path


class DerivedMatrixTests(unittest.TestCase):
    """Yes, no, negated-no, and open - all derived, all trailed."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import (build_session,
                                                     run_pipeline)

        self.dir = Path(tempfile.mkdtemp(prefix="uq_polard_"))
        self.session = build_session(self.dir, seed=0)
        self._run = run_pipeline
        for statement in ("the dome city is york",
                          "the york climate is not temperate",
                          "the tower material is iron",
                          "the iron hardness is 490 units"):
            run_pipeline(statement, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _ask(self, question: str) -> str:
        response, _trace = self._run(question, self.session)
        return response

    def test_a_true_claim_derives_yes_with_the_trail(self) -> None:
        response = self._ask("is the tower hardness 490 units?")
        self.assertTrue(response.startswith("Yes"))
        self.assertIn("via iron", response)
        self.assertIn("derived", response)

    def test_a_false_claim_derives_no_with_the_actual(self) -> None:
        response = self._ask("is the tower hardness 500 units?")
        self.assertTrue(response.startswith("No"))
        self.assertIn("490 units", response)

    def test_a_negated_terminal_derives_no(self) -> None:
        response = self._ask("is the dome city climate temperate?")
        self.assertTrue(response.startswith("No"))
        self.assertIn("believed not temperate", response)
        self.assertIn("via york", response)

    def test_an_agreeing_negated_claim_derives_yes(self) -> None:
        """The trailing-not bug: "dome city climate not" once absorbed
        the polarity word into the subject and answered No where the
        claim agreed. A split may never end a subject with a negator."""
        response = self._ask("is the dome city climate not temperate?")
        self.assertTrue(response.startswith("Yes"))

    def test_a_derived_negation_stays_silent_on_other_values(self) -> None:
        response = self._ask("is the dome city climate humid?")
        self.assertTrue(response.startswith("I don't know"))

    def test_absence_is_still_never_no(self) -> None:
        response = self._ask("is the citadel city climate humid?")
        self.assertFalse(response.startswith("No"))
        self.assertFalse(response.startswith("Yes"))

    def test_an_affirmed_derived_verdict_consolidates(self) -> None:
        self._ask("is the tower hardness 490 units?")
        response = self._ask("yes")
        self.assertIn("Consolidated", response)
        record = self.session.memory.recall_fact("tower hardness")
        self.assertIsNotNone(record)


class SplitBugTests(unittest.TestCase):
    """The two ladder bugs the build caught, pinned."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import (build_session,
                                                     run_pipeline)

        self.dir = Path(tempfile.mkdtemp(prefix="uq_polard_"))
        self.session = build_session(self.dir, seed=0)
        self._run = run_pipeline

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_longer_derivable_subject_beats_a_stored_prefix(self) -> None:
        """"dome city climate temperate" once matched the STORED "dome
        city" and answered about the city ("No - dome city is york,
        not climate temperate"). The ladder now tries deriving the
        longer subject first."""
        for statement in ("the dome city is york",
                          "the york climate is humid"):
            self._run(statement, self.session)
        response, _trace = self._run(
            "is the dome city climate humid?", self.session)
        self.assertTrue(response.startswith("Yes"))
        self.assertIn("via york", response)

    def test_a_modifier_rescued_split_refuses(self) -> None:
        """"tower hardness 490" once derived with '490' read as a
        modifier - a derivation that reads part of its own subject
        away means the split was wrong, and the claim's number must
        not decide the verdict from inside the subject."""
        for statement in ("the tower material is iron",
                          "the iron hardness is 490 units"):
            self._run(statement, self.session)
        response, _trace = self._run(
            "is the tower hardness 490 units?", self.session)
        self.assertTrue(response.startswith("Yes"))
        self.assertNotIn("as a modifier", response)


class GateVerdictTests(unittest.TestCase):
    """The PASS and the closed family, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import polarderive_gate

        doc = " ".join(polarderive_gate.__doc__.split())
        self.assertIn("+0.312 at 5.11x seed sd", doc)

    def test_the_split_bugs_are_recorded(self) -> None:
        from ultraquant.experiments import polarderive_gate

        doc = " ".join(polarderive_gate.__doc__.split())
        self.assertIn("answering about the city", doc)
        self.assertIn("reads part of its own subject away", doc)

    def test_the_family_closure_is_recorded(self) -> None:
        from ultraquant.experiments import polarderive_gate

        doc = " ".join(polarderive_gate.__doc__.split())
        self.assertIn("The §11.48 family is closed", doc)


if __name__ == "__main__":
    unittest.main()
