"""Superlatives: a claim about a set, with the set named.

§11.56's registered claim, completing the comparative family: "which
height is the tallest?" enumerates the whole attribute family, names
its scope, excludes by polarity and by name, and never breaks a tie
silently. The gate passed at +0.794, 7.45x seed sd with zero denials
crowned; these tests pin the matrix.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path


class SuperlativeTests(unittest.TestCase):
    """Winners, ties, exclusions, and the empty family."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import (build_session,
                                                     run_pipeline)

        self.dir = Path(tempfile.mkdtemp(prefix="uq_superlative_"))
        self.session = build_session(self.dir, seed=0)
        self._run = run_pipeline
        for statement in ("the tower height is 300 meters",
                          "the bridge height is 120 meters",
                          "the spire height is 2 kilometers",
                          "the keep height is not 900 meters",
                          "the mill height is unknown"):
            run_pipeline(statement, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _ask(self, question: str) -> str:
        response, _trace = self._run(question, self.session)
        return response

    def test_the_winner_names_its_scope(self) -> None:
        """A superlative is a claim about a set - the count says which
        set, so the claim can be checked."""
        response = self._ask("which height is the tallest?")
        self.assertTrue(response.startswith("The spire height"))
        self.assertIn("of the 3 height facts I hold", response)
        self.assertIn("2 kilometers", response)

    def test_units_convert_before_ranking(self) -> None:
        """2 kilometers out-ranks 300 meters only because §11.42's
        table sits under the ranking."""
        response = self._ask("which height is the tallest?")
        self.assertIn("spire", response)

    def test_the_downward_superlative_inverts(self) -> None:
        response = self._ask("which height is the shortest?")
        self.assertTrue(response.startswith("The bridge height"))

    def test_a_denial_is_excluded_and_accounted(self) -> None:
        """The keep's negated 900 meters would out-rank everything -
        §11.48's line holds at set scale, and the accounting is
        spoken."""
        response = self._ask("which height is the tallest?")
        self.assertNotIn("keep", response.split(":")[0])
        self.assertIn("denial(s) not counted", response)

    def test_a_wordy_candidate_is_excluded_by_name(self) -> None:
        response = self._ask("which height is the tallest?")
        self.assertIn("mill height (no number)", response)

    def test_a_tie_is_named_never_broken(self) -> None:
        self._run("the barn height is 2 kilometers", self.session)
        response = self._ask("which height is the tallest?")
        self.assertTrue(response.startswith("Tied"))
        self.assertIn("barn height", response)
        self.assertIn("spire height", response)

    def test_an_empty_family_refuses(self) -> None:
        response = self._ask("which depth is the lowest?")
        self.assertIn("I hold no depth facts", response)

    def test_ordinary_questions_are_untouched(self) -> None:
        response = self._ask("what is the tower height?")
        self.assertIn("300 meters", response)
        self.assertNotIn("facts I hold", response)


class GateVerdictTests(unittest.TestCase):
    """The PASS and the completed family, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import superlative_gate

        doc = " ".join(superlative_gate.__doc__.split())
        self.assertIn("+0.794 at 7.45x seed sd", doc)

    def test_the_denial_line_is_recorded(self) -> None:
        from ultraquant.experiments import superlative_gate

        doc = " ".join(superlative_gate.__doc__.split())
        self.assertIn("NEVER crowned", doc)
        self.assertIn("holding at set scale", doc)

    def test_the_family_completion_is_recorded(self) -> None:
        from ultraquant.experiments import superlative_gate

        doc = " ".join(superlative_gate.__doc__.split())
        self.assertIn("The comparative family is complete", doc)


if __name__ == "__main__":
    unittest.main()
