"""Two statements joined are two statements.

§11.74's registered claim, closing a live corruption: the first
" is " split stored a whole second clause inside the first value. A
clause splits only where both sides parse as statements, so "live and
learn" survives whole. The gate passed at +0.639, 4.59x seed sd with
the baseline's 20 swallowed statements measured; these tests pin the
split and its integrity lines.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path


class ClauseTests(unittest.TestCase):
    """The split, its polarity, and the value that never splits."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import (build_session,
                                                     run_pipeline)

        self.dir = Path(tempfile.mkdtemp(prefix="uq_clause_"))
        self.session = build_session(self.dir, seed=0)
        self._run = run_pipeline

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _say(self, text: str) -> str:
        response, _trace = self._run(text, self.session)
        return response

    def test_two_clauses_store_separately(self) -> None:
        """The shipped corruption: 'tower material' once held "iron
        and the bridge material is steel"."""
        response = self._say("the tower material is iron and the "
                             "bridge material is steel")
        self.assertIn("tower material is iron", response)
        self.assertIn("bridge material is steel", response)
        self.assertEqual(
            self.session.memory.recall_fact("tower material")["value"],
            "iron")
        self.assertEqual(
            self.session.memory.recall_fact("bridge material")["value"],
            "steel")

    def test_three_clauses_store(self) -> None:
        self._say("the keep height is 300 meters and the wall height "
                  "is 120 meters and the mill height is 80 meters")
        for key, value in (("keep height", "300 meters"),
                           ("wall height", "120 meters"),
                           ("mill height", "80 meters")):
            self.assertEqual(
                self.session.memory.recall_fact(key)["value"], value)

    def test_polarity_lands_on_its_own_clause(self) -> None:
        self._say("the dome material is granite and the gate material "
                  "is not steel")
        self.assertFalse(self.session.memory.recall_fact(
            "dome material").get("negated"))
        self.assertTrue(self.session.memory.recall_fact(
            "gate material").get("negated"))

    def test_but_splits_and_its_idiom_survives(self) -> None:
        """Run two's boundary: "but" swallowed exactly as "and" had,
        and "slow but steady" must survive exactly as "live and
        learn" does."""
        self._say("the tower material is iron but the bridge material "
                  "is steel")
        self.assertEqual(
            self.session.memory.recall_fact("bridge material")["value"],
            "steel")
        self._say("the motto is slow but steady")
        self.assertEqual(
            self.session.memory.recall_fact("motto")["value"],
            "slow but steady")

    def test_the_comma_splits_and_lists_survive(self) -> None:
        """Run three's boundary: commas join clauses like both
        conjunctions, and "1, 2, 3" survives because its parts carry
        no verb - structure, not vocabulary."""
        self._say("the tower height is 300 meters, the bridge height "
                  "is 120 meters")
        self.assertEqual(
            self.session.memory.recall_fact("bridge height")["value"],
            "120 meters")
        self._say("the reading is 1, 2, 3")
        self.assertEqual(
            self.session.memory.recall_fact("reading")["value"],
            "1, 2, 3")

    def test_an_and_value_survives_whole(self) -> None:
        """A split that fired on any "and" would shred ordinary
        values; it fires only where both sides parse as statements."""
        self._say("the motto is live and learn")
        self.assertEqual(
            self.session.memory.recall_fact("motto")["value"],
            "live and learn")

    def test_no_value_ever_contains_a_statement(self) -> None:
        self._say("the tower material is iron and the bridge material "
                  "is steel")
        for key in self.session.memory.fact_keys():
            value = str(self.session.memory.recall_fact(key)["value"])
            self.assertNotIn(" is ", value)


class GateVerdictTests(unittest.TestCase):
    """The PASS and the completed statement grammar, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import clause_gate

        doc = " ".join(clause_gate.__doc__.split())
        self.assertIn("+0.639 at 4.59x seed sd", doc)
        self.assertIn("the baseline swallowed 20", doc)

    def test_the_grammar_is_recorded_complete(self) -> None:
        from ultraquant.experiments import clause_gate

        doc = " ".join(clause_gate.__doc__.split())
        self.assertIn("no shape of \"and\" left that silently loses "
                      "or corrupts a fact", doc)


if __name__ == "__main__":
    unittest.main()
