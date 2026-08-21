"""A change of mind, said out loud — and only a real one.

§11.53's registered claim: conflicting statements narrate the revision
(old belief named, retractions counted), agreements never do, and the
notice is commentary, not semantics. The gate passed at +0.803, 5.03x
seed sd with the exact-key ceiling scored in; these tests pin the
outcome contract and the narration.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.memory.systematic import SystematicMemory


class OutcomeContractTests(unittest.TestCase):
    """remember_fact reports what it did."""

    def test_new_reinforced_revised(self) -> None:
        memory = SystematicMemory()
        self.assertEqual(
            memory.remember_fact("tower material", "iron")["outcome"],
            "new")
        self.assertEqual(
            memory.remember_fact("tower material", "iron")["outcome"],
            "reinforced")
        result = memory.remember_fact("tower material", "steel")
        self.assertEqual(result["outcome"], "revised")
        self.assertEqual(result["was"], "iron")

    def test_a_polarity_flip_reports_the_old_polarity(self) -> None:
        memory = SystematicMemory()
        memory.remember_fact("dome material", "steel", negated=True)
        result = memory.remember_fact("dome material", "steel")
        self.assertEqual(result["outcome"], "revised")
        self.assertEqual(result["was"], "not steel")

    def test_a_revision_names_its_retractions(self) -> None:
        memory = SystematicMemory()
        memory.remember_fact("tower material", "iron", confidence=0.6)
        memory.remember_fact("iron hardness", "490 units",
                             confidence=0.6)
        memory.consolidate_fact(
            "tower hardness", "490 units", confidence=0.6,
            premises=[("tower material", "iron"),
                      ("iron hardness", "490 units")])
        result = memory.remember_fact("tower material", "steel")
        self.assertEqual(result["retracted"], ["tower hardness"])


class NarrationTests(unittest.TestCase):
    """The reply names the change - and only a change."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import (build_session,
                                                     run_pipeline)

        self.dir = Path(tempfile.mkdtemp(prefix="uq_revision_"))
        self.session = build_session(self.dir, seed=0)
        self._run = run_pipeline

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _say(self, text: str) -> str:
        response, _trace = self._run(text, self.session)
        return response

    def test_a_value_revision_names_the_old_belief(self) -> None:
        self._say("the tower material is iron")
        response = self._say("the tower material is steel")
        self.assertIn("revises", response)
        self.assertIn("tower material was iron", response)
        self.assertNotIn("retracted", response)

    def test_a_revision_under_a_derivative_counts_the_cost(self) -> None:
        self._say("the tower material is iron")
        self._say("the iron hardness is 490 units")
        self._say("what is the tower hardness?")
        self._say("yes")
        response = self._say("the tower material is steel")
        self.assertIn("revises", response)
        self.assertIn("1 derived fact(s)", response)
        self.assertIn("tower hardness", response)

    def test_a_polarity_flip_speaks_both_sides(self) -> None:
        self._say("the dome material is not steel")
        response = self._say("the dome material is steel")
        self.assertIn("revises", response)
        self.assertIn("was not steel", response)

    def test_a_reinforcement_is_never_a_revision(self) -> None:
        self._say("the tower material is iron")
        response = self._say("the tower material is iron")
        self.assertNotIn("revises", response)

    def test_a_near_key_statement_is_news_not_a_revision(self) -> None:
        """The exact-key ceiling: a more specific subject stores a new
        fact, and narrating a revision for it would be a false
        notice."""
        self._say("the tower material is iron")
        response = self._say("the old tower material is steel")
        self.assertNotIn("revises", response)

    def test_the_notice_is_commentary_not_semantics(self) -> None:
        self._say("the tower material is iron")
        self._say("the tower material is steel")
        response = self._say("what is the tower material?")
        self.assertIn("steel", response)
        self.assertNotIn("iron", response)


class GateVerdictTests(unittest.TestCase):
    """The PASS, the hardening, and the ceiling, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import revision_gate

        doc = " ".join(revision_gate.__doc__.split())
        self.assertIn("+0.803 at 5.03x seed sd", doc)

    def test_the_variance_hardening_is_recorded(self) -> None:
        """The seventh 'nothing varied' in the book, resolved the
        standing way: score the unwinnable, never soften the rule."""
        from ultraquant.experiments import revision_gate

        doc = " ".join(revision_gate.__doc__.split())
        self.assertIn("seventh occurrence in the book", doc)
        self.assertIn("never soften the rule", doc)

    def test_the_accountability_line_is_recorded(self) -> None:
        from ultraquant.experiments import revision_gate

        doc = " ".join(revision_gate.__doc__.split())
        self.assertIn("as unaccountable as one that revises silently",
                      doc)


if __name__ == "__main__":
    unittest.main()
