"""The trail is the answer, askable — and never invented.

§11.51's registered claim: "why is X Y?" answers from provenance —
statement, consolidation premises, or a fresh derivation's trail — and
holds two zero-tolerance lines: a false premise is corrected, never
rationalised, and an unheld subject is never explained. The gate
passed at +0.830, 19.88x seed sd, zero on both lines.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path


class WhyProvenanceTests(unittest.TestCase):
    """Each provenance kind answers in its own voice."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import (build_session,
                                                     run_pipeline)

        self.dir = Path(tempfile.mkdtemp(prefix="uq_why_"))
        self.session = build_session(self.dir, seed=0)
        self._run = run_pipeline
        for statement in ("the tower material is iron",
                          "the iron hardness is 490 units",
                          "the dome city is york",
                          "the york climate is not temperate"):
            run_pipeline(statement, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _ask(self, question: str) -> str:
        response, _trace = self._run(question, self.session)
        return response

    def test_a_stated_fact_cites_its_statement(self) -> None:
        response = self._ask("why is the tower material iron?")
        self.assertTrue(response.startswith("Because"))
        self.assertIn("stated directly", response)

    def test_a_consolidated_fact_cites_its_premises(self) -> None:
        self._ask("what is the tower hardness?")
        self._ask("yes")
        response = self._ask("why is the tower hardness 490 units?")
        self.assertTrue(response.startswith("Because"))
        self.assertIn("tower material is iron", response)
        self.assertIn("iron hardness is 490 units", response)
        self.assertIn("consolidated", response)

    def test_a_derivable_fact_derives_and_says_so(self) -> None:
        response = self._ask("why is the tower hardness 490 units?")
        self.assertTrue(response.startswith("Because"))
        self.assertIn("derived just now", response)

    def test_a_derived_denial_explains_with_its_polarity(self) -> None:
        response = self._ask(
            "why is the dome city climate not temperate?")
        self.assertTrue(response.startswith("Because"))
        self.assertIn("york climate is not temperate", response)

    def test_a_false_premise_is_corrected_never_rationalised(self) -> None:
        """The anti-rationalisation line: an explanation machine that
        will explain anything explains nothing."""
        response = self._ask("why is the tower material steel?")
        self.assertTrue(response.startswith("It isn't"))
        self.assertIn("iron", response)

    def test_a_false_claim_against_a_derivation_corrects(self) -> None:
        response = self._ask("why is the dome city climate temperate?")
        self.assertTrue(response.startswith("It isn't"))
        self.assertIn("believed not temperate", response)

    def test_an_unheld_subject_is_never_explained(self) -> None:
        response = self._ask("why is the citadel hardness 300 units?")
        self.assertFalse(response.startswith("Because"))
        self.assertFalse(response.startswith("It isn't"))

    def test_a_claim_free_why_explains_the_stored_fact(self) -> None:
        response = self._ask("why is the tower material?")
        self.assertTrue(response.startswith("Because"))


class GateVerdictTests(unittest.TestCase):
    """The PASS and both zero-tolerance lines, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import why_gate

        doc = " ".join(why_gate.__doc__.split())
        self.assertIn("+0.830 at 19.88x seed sd", doc)

    def test_the_rationalisation_line_is_recorded(self) -> None:
        from ultraquant.experiments import why_gate

        doc = " ".join(why_gate.__doc__.split())
        self.assertIn("ZERO rationalised", doc)
        self.assertIn("explains nothing", doc)

    def test_the_thesis_is_recorded(self) -> None:
        from ultraquant.experiments import why_gate

        doc = " ".join(why_gate.__doc__.split())
        self.assertIn("the trail IS the answer", doc)


if __name__ == "__main__":
    unittest.main()
