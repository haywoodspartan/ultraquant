"""The relation stated once distributes.

§11.72's registered claim, cashing §11.71's ceiling for the shape it
is honest in: single-token conjuncts complete from the final part's
stated relation, negation rides the distribution, and nothing
distributes where nobody named a relation. The gate passed at +0.667,
3.83x seed sd; these tests pin the rule and its sharpened ceiling.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path


class EllipsisTests(unittest.TestCase):
    """The distribution, its polarity, and its refusals."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import (build_session,
                                                     run_pipeline)

        self.dir = Path(tempfile.mkdtemp(prefix="uq_ellipsis_"))
        self.session = build_session(self.dir, seed=0)
        self._run = run_pipeline

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _say(self, text: str) -> str:
        response, _trace = self._run(text, self.session)
        return response

    def test_the_stated_relation_completes_every_part(self) -> None:
        response = self._say(
            "the tower and the bridge material are iron")
        self.assertIn("tower material is iron", response)
        self.assertIn("bridge material is iron", response)
        for key in ("tower material", "bridge material"):
            self.assertIsNotNone(self.session.memory.recall_fact(key))

    def test_three_parts_distribute(self) -> None:
        self._say("the keep and the wall and the mill height are "
                  "300 meters")
        for key in ("keep height", "wall height", "mill height"):
            self.assertIsNotNone(self.session.memory.recall_fact(key))

    def test_negation_rides_the_distribution(self) -> None:
        self._say("the dome and the gate material are not steel")
        for key in ("dome material", "gate material"):
            fact = self.session.memory.recall_fact(key)
            self.assertTrue(fact.get("negated"))

    def test_no_relation_anywhere_refuses(self) -> None:
        response = self._say("the spire and the arch are oak")
        self.assertIn("couldn't find", response)
        self.assertIsNone(self.session.memory.recall_fact("spire"))

    def test_the_multiword_entity_ceiling(self) -> None:
        """With multi-word entities, token count cannot say whether
        'keep' is name or relation - distribution stays single-token,
        and the full-subject path stores parts as their singular
        equivalents would."""
        self._say("the north tower and the south keep material are "
                  "granite")
        self.assertIsNotNone(
            self.session.memory.recall_fact("north tower"))
        self.assertIsNotNone(
            self.session.memory.recall_fact("south keep material"))


class GateVerdictTests(unittest.TestCase):
    """The PASS and the two harness lessons, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import ellipsis_gate

        doc = " ".join(ellipsis_gate.__doc__.split())
        self.assertIn("+0.667 at 3.83x seed sd", doc)

    def test_the_unmade_claim_lesson_is_recorded(self) -> None:
        from ultraquant.experiments import ellipsis_gate

        doc = " ".join(ellipsis_gate.__doc__.split())
        self.assertIn("measuring a claim the mechanism never made",
                      doc)

    def test_the_slot_lesson_is_recorded(self) -> None:
        from ultraquant.experiments import ellipsis_gate

        doc = " ".join(ellipsis_gate.__doc__.split())
        self.assertIn("the third slicing-overflow in the book", doc)


if __name__ == "__main__":
    unittest.main()
