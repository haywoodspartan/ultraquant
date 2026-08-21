"""The neighbor named: adjacency spoken, never merged.

§11.63's registered claim, closing §11.53's ceiling the visible way: a
new fact whose leading-stripped base is held answers with the neighbor
spoken, polarity included, and the three silences stay silent. The
gate passed at +0.912, 8.37x seed sd with zero false notes; these
tests pin the note and its lines.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path


class NearKeyNoteTests(unittest.TestCase):
    """The note, its polarity, and the three silences."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import (build_session,
                                                     run_pipeline)

        self.dir = Path(tempfile.mkdtemp(prefix="uq_nearkey_"))
        self.session = build_session(self.dir, seed=0)
        self._run = run_pipeline
        for statement in ("the tower material is iron",
                          "the dome material is not steel"):
            run_pipeline(statement, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _say(self, text: str) -> str:
        response, _trace = self._run(text, self.session)
        return response

    def test_a_near_key_names_the_held_base(self) -> None:
        response = self._say("the old tower material is steel")
        self.assertIn("I separately hold", response)
        self.assertIn("tower material is iron", response)

    def test_the_note_speaks_the_bases_polarity(self) -> None:
        response = self._say("the old dome material is oak")
        self.assertIn("I separately hold", response)
        self.assertIn("dome material is not steel", response)

    def test_a_fresh_fact_stays_silent(self) -> None:
        response = self._say("the keep material is bronze")
        self.assertNotIn("I separately hold", response)

    def test_a_reinforcement_stays_silent(self) -> None:
        response = self._say("the tower material is iron")
        self.assertNotIn("I separately hold", response)

    def test_a_revision_keeps_its_own_notice_and_nothing_else(self) -> None:
        response = self._say("the tower material is steel")
        self.assertIn("revises", response)
        self.assertNotIn("I separately hold", response)

    def test_the_note_is_information_never_a_merge(self) -> None:
        self._say("the old tower material is steel")
        answer = self._say("what is the old tower material?")
        self.assertIn("steel", answer)
        answer = self._say("what is the tower material?")
        self.assertIn("iron", answer)

    def test_the_single_strip_ceiling(self) -> None:
        """The base two strips away stays unnamed - the mechanism's
        registered ceiling, scored into the gate at zero."""
        response = self._say("the ruined old tower material is oak")
        self.assertNotIn("I separately hold", response)


class GateVerdictTests(unittest.TestCase):
    """The three-run story, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import nearkey_gate

        doc = " ".join(nearkey_gate.__doc__.split())
        self.assertIn("+0.912 at 8.37x seed sd", doc)

    def test_the_slot_collision_is_recorded(self) -> None:
        from ultraquant.experiments import nearkey_gate

        doc = " ".join(nearkey_gate.__doc__.split())
        self.assertIn("world-builder slot collision", doc)
        self.assertIn("caught by the same kind of floor", doc)

    def test_the_information_line_is_recorded(self) -> None:
        from ultraquant.experiments import nearkey_gate

        doc = " ".join(nearkey_gate.__doc__.split())
        self.assertIn("the note is information, never a merge", doc)


if __name__ == "__main__":
    unittest.main()
