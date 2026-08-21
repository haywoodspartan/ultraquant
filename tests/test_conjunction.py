"""Conjunctive statements teach all their facts.

§11.71's registered claim, closing a silent teaching failure: "the A
material and the B material are iron" taught zero facts before, and
now teaches every part through the full statement machinery. The gate
passed at +0.836, 6.45x seed sd with zero junk keys; these tests pin
the split, its honesty, and its ceiling.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path


class ConjunctionTests(unittest.TestCase):
    """Every part lands, each through the machinery it would meet alone."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import (build_session,
                                                     run_pipeline)

        self.dir = Path(tempfile.mkdtemp(prefix="uq_conjunction_"))
        self.session = build_session(self.dir, seed=0)
        self._run = run_pipeline

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _say(self, text: str) -> str:
        response, _trace = self._run(text, self.session)
        return response

    def test_two_parts_teach_and_narrate_together(self) -> None:
        response = self._say(
            "the tower material and the bridge material are iron")
        self.assertIn("tower material is iron", response)
        self.assertIn("bridge material is iron", response)
        for key in ("tower material", "bridge material"):
            fact = self.session.memory.recall_fact(key)
            self.assertEqual(fact["value"], "iron")

    def test_three_parts_teach(self) -> None:
        self._say("the keep height and the wall height and the mill "
                  "height are 300 meters")
        for key in ("keep height", "wall height", "mill height"):
            self.assertIsNotNone(self.session.memory.recall_fact(key))

    def test_negation_distributes_to_every_part(self) -> None:
        self._say("the dome material and the gate material are not "
                  "steel")
        for key in ("dome material", "gate material"):
            fact = self.session.memory.recall_fact(key)
            self.assertTrue(fact.get("negated"))

    def test_a_revising_part_speaks_its_notice(self) -> None:
        """§11.53 per part: a conjunction may not revise silently."""
        self._say("the tower material is steel")
        response = self._say(
            "the tower material and the bridge material are iron")
        self.assertIn("revises", response)
        self.assertIn("steel", response)

    def test_the_elided_form_refuses_rather_than_guess(self) -> None:
        response = self._say("the tower and the bridge are iron")
        self.assertIn("couldn't find", response)

    def test_no_junk_and_key_is_ever_stored(self) -> None:
        self._say("the tower material and the bridge material are iron")
        self.assertFalse([k for k in self.session.memory.fact_keys()
                          if " and " in k])

    def test_singular_statements_are_untouched(self) -> None:
        response = self._say("the arch material is oak")
        self.assertIn("Noted: arch material is oak", response)


class GateVerdictTests(unittest.TestCase):
    """The PASS and the one-sentence line, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import conjunction_gate

        doc = " ".join(conjunction_gate.__doc__.split())
        self.assertIn("+0.836 at 6.45x seed sd", doc)
        self.assertIn("zero junk keys ever stored", doc)

    def test_the_teaching_line_is_recorded(self) -> None:
        from ultraquant.experiments import conjunction_gate

        doc = " ".join(conjunction_gate.__doc__.split())
        self.assertIn("say two things in one sentence and have both "
                      "of them land", doc)


if __name__ == "__main__":
    unittest.main()
