"""One grammar for teaching and asking.

§11.73's registered claim, §11.72's question-side mirror: the final
part's stated relation completes single-token question parts, the
same ceilings hold, and the compound formatter — caught asserting a
denied value positively — speaks polarity. The gate passed at +0.722,
4.04x seed sd; these tests pin the mirror.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path


class QuestionEllipsisTests(unittest.TestCase):
    """Asking in the shape the teaching came in."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import (build_session,
                                                     run_pipeline)

        self.dir = Path(tempfile.mkdtemp(prefix="uq_qellipsis_"))
        self.session = build_session(self.dir, seed=0)
        self._run = run_pipeline
        for statement in ("the tower material is iron",
                          "the bridge material is steel",
                          "the dome material is not steel",
                          "the keep height is 300 meters",
                          "the wall height is 120 meters"):
            run_pipeline(statement, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _ask(self, question: str) -> str:
        response, _trace = self._run(question, self.session)
        return response

    def test_the_stated_relation_completes_the_question(self) -> None:
        response = self._ask(
            "what is the tower and the bridge material?")
        self.assertIn("tower material is iron", response)
        self.assertIn("bridge material is steel", response)

    def test_a_denied_part_speaks_its_polarity(self) -> None:
        """The §11.48 gap the build caught: the compound formatter
        asserted 'dome material is steel' for a stored denial."""
        response = self._ask(
            "what is the tower and the dome material?")
        self.assertIn("dome material is not steel", response)

    def test_no_relation_anywhere_hedges(self) -> None:
        response = self._ask("what is the tower and the bridge?")
        self.assertNotIn("tower material", response)

    def test_full_subject_compounds_are_untouched(self) -> None:
        response = self._ask(
            "what is the tower material and the bridge material?")
        self.assertIn("tower material is iron", response)
        self.assertIn("bridge material is steel", response)

    def test_the_combine_keeps_its_door(self) -> None:
        response = self._ask(
            "what is the sum of the keep height and the wall height?")
        self.assertIn("420", response)


class GateVerdictTests(unittest.TestCase):
    """The PASS and the one-grammar line, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import qellipsis_gate

        doc = " ".join(qellipsis_gate.__doc__.split())
        self.assertIn("+0.722 at 4.04x seed sd", doc)
        self.assertIn("zero denials rendered bare", doc)

    def test_the_one_grammar_line_is_recorded(self) -> None:
        from ultraquant.experiments import qellipsis_gate

        doc = " ".join(qellipsis_gate.__doc__.split())
        self.assertIn("a sentence that stores two facts can be asked "
                      "back in the same shape", doc)


if __name__ == "__main__":
    unittest.main()
