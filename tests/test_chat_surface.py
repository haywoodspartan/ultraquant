"""The chat surface defects a scripted battery found, pinned.

A full battery through the real CLI surfaced five behavioral defects: a
pasted glyph answered row by row, the pipeline pleading total ignorance
while quarantined claims sat on the topic, a plural question missing a
just-stored fact, the planner failing without naming the missing part, and
the TUI persisting a temp-dir session as the standing home. Each fix gets
the regression that would have caught it.
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.interpreter.chat import ChatCLI
from ultraquant.interpreter.thoughts import build_session, run_pipeline


class _Capture(io.StringIO):
    pass


class GlyphPasteTests(unittest.TestCase):
    """A bare pasted glyph must be collected, not answered row by row."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_chatsurf_"))
        self.session = build_session(self.dir, seed=0)
        self.out = _Capture()
        self.cli = ChatCLI(self.session, out=self.out)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_five_rows_become_one_glyph(self) -> None:
        rows = ["..#..", "..#..", "#####", "..#..", "..#.."]
        stream = iter(rows[1:])
        self.cli.handle(rows[0], stream)
        text = self.out.getvalue()
        self.assertEqual(text.count("\n"), 1,
                         "five rows must produce one reply, not five")
        self.assertNotIn("I have nothing on that", text)

    def test_a_blank_line_is_the_interactive_escape(self) -> None:
        """One stray row plus enter must not trap the user in collection."""
        stream = iter([""])
        self.cli.handle("..#..", stream)
        self.assertTrue(self.out.getvalue().strip(),
                        "the stray row still gets a reply")

    def test_a_non_glyph_line_is_not_swallowed(self) -> None:
        stream = iter(["hello there"])
        self.cli.handle("..#..", stream)
        replies = [ln for ln in self.out.getvalue().splitlines() if ln]
        self.assertEqual(len(replies), 2,
                         "the glyph row and the text line both answer")


class StashAwareUnknownTests(unittest.TestCase):
    """'I hold nothing' must not be said while staged claims mention it."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_chatsurf_"))
        self.session = build_session(self.dir, seed=0)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_staged_claims_are_surfaced_not_believed(self) -> None:
        self.session.stash.add_page(
            "https://example.invalid/x", "about code",
            "Code is a set of instructions executed by a computer.")
        response, _trace = run_pipeline("what is code?", self.session)
        self.assertIn("staged claim", response)
        self.assertIn(":analyze", response)
        # Quarantine holds: the claim's content is pointed at, never quoted
        # as an answer.
        self.assertNotIn("set of instructions", response)

    def test_the_plain_unknown_survives_when_the_stash_is_silent(self) -> None:
        response, _trace = run_pipeline("what is the melting point of "
                                        "tungsten?", self.session)
        self.assertIn("I don't hold anything on that yet", response)


class PluralQuestionTests(unittest.TestCase):
    """'what are towers?' must find the fact stored as 'tower height'."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_chatsurf_"))
        self.session = build_session(self.dir, seed=0)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_the_plural_folds_into_the_stored_key(self) -> None:
        run_pipeline("the tower height is 300 meters", self.session)
        response, _trace = run_pipeline("what are towers?", self.session)
        self.assertIn("300 meters", response)


class GoalDiagnosticsTests(unittest.TestCase):
    """A failed plan names what is missing; a partial plan admits it."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_chatsurf_"))
        self.session = build_session(self.dir, seed=0)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_the_missing_part_is_named_with_both_remedies(self) -> None:
        run_pipeline("the tower height is 300 meters", self.session)
        response, _trace = run_pipeline(
            "goal: the tower height and the bridge length", self.session)
        self.assertIn("bridge length", response)
        self.assertIn("tower height", response)
        self.assertIn(":learn", response)
        self.assertNotIn("depth", response,
                         "the search-internals message is not an explanation")

    def test_a_complete_goal_still_plans(self) -> None:
        run_pipeline("the tower height is 300 meters", self.session)
        run_pipeline("the bridge length is 2 kilometers", self.session)
        response, _trace = run_pipeline(
            "goal: the tower height and the bridge length", self.session)
        self.assertIn("Plan (", response)
        self.assertNotIn("hold nothing", response)


class TempHomePersistenceTests(unittest.TestCase):
    """A temp-dir session home must never become the standing default."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_chatsurf_"))
        self._old = os.environ.get("ULTRAQUANT_CONFIG")
        os.environ["ULTRAQUANT_CONFIG"] = str(self.dir / "settings.json")

    def tearDown(self) -> None:
        if self._old is None:
            os.environ.pop("ULTRAQUANT_CONFIG", None)
        else:
            os.environ["ULTRAQUANT_CONFIG"] = self._old
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_the_tui_refuses_to_persist_a_temp_home(self) -> None:
        from ultraquant.tui import UltraQuantTUI

        out = io.StringIO()
        tui = UltraQuantTUI(home=self.dir / "session", stream=out, color=False)
        tui._save_settings()
        self.assertEqual(tui.settings.get("home", ""), "",
                         "a home under tempdir must not be written")

    def test_a_real_home_still_persists(self) -> None:
        from ultraquant.tui import UltraQuantTUI

        out = io.StringIO()
        real = Path("H:/AI Model AGI/uq_home")
        tui = UltraQuantTUI(home=real, stream=out, color=False)
        tui._save_settings()
        self.assertEqual(tui.settings.get("home", ""), str(real))


class EmptyTrainingReportTests(unittest.TestCase):
    """No teacher, no zero-table: the empty run says nothing was measured."""

    def test_the_measurement_block_is_omitted(self) -> None:
        from ultraquant.forge.train_from_llm import TrainingReport

        report = TrainingReport()
        report.skipped["(all)"] = "every independent voice has already taught"
        text = report.as_text()
        self.assertNotIn("held-out routing", text)
        self.assertNotIn("0.000", text)


class HelpFormsTests(unittest.TestCase):
    """The question forms the pipeline understands are discoverable.

    §11.45-§11.57 added ten forms; a capability the ':help' text does
    not name is half-shipped. These pins keep the help current - a new
    form without its line fails here."""

    def test_every_question_family_is_named(self) -> None:
        from ultraquant.interpreter.chat import HELP

        for marker in ("is not steel",
                       "revised aloud",
                       "hometown region climate",
                       "absence is never no",
                       "why is the tower hardness",
                       "corrected, never",
                       "taller than the bridge height",
                       "derived and marked",
                       "which height is the tallest",
                       "how many height facts",
                       "average height",
                       "consolidates it",
                       "what was the tower material",
                       "none is invented",
                       "names its takedown",
                       "I separately hold"):
            self.assertIn(marker, HELP,
                          f"help text lost the {marker!r} form")


if __name__ == "__main__":
    unittest.main()
