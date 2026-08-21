"""The comparative word carries its attribute - §11.75, pinned.

"is A taller than B?" and "which is the tallest?" read height out of
the word itself; the polysemous words refuse rather than guess; exact
bare keys still win first; the tried key is named in refusals.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.interpreter.thoughts import build_session, run_pipeline


class ImpliedAttrTests(unittest.TestCase):
    """The natural comparative shapes, over one small world."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="uq_implt_"))
        self.session = build_session(self.root, seed=0)
        for line in (
            "the north tower height is 300 meters",
            "the south tower height is 450 meters",
            "the west tower kind is spirekind",
            "spirekind height is 900 meters",
            "the old mill is 40",
            "the royal hall is 60",
            "the east gate material is granite",
        ):
            run_pipeline(line, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _ask(self, question: str) -> str:
        response, _t = run_pipeline(question, self.session)
        return response

    def test_taller_reads_height_from_the_word(self) -> None:
        """The shape nobody used to be able to speak: bare subjects."""
        response = self._ask(
            "is the north tower taller than the south tower?")
        self.assertTrue(response.startswith("No"))
        self.assertIn("north tower height is 300 meters", response)
        self.assertIn("south tower height is 450 meters", response)

    def test_the_derive_rides_the_implication(self) -> None:
        response = self._ask(
            "is the west tower taller than the south tower?")
        self.assertTrue(response.startswith("Yes"))
        self.assertIn("derived via spirekind", response)

    def test_which_is_the_tallest_answers_the_family(self) -> None:
        response = self._ask("which is the tallest?")
        self.assertIn("900 meters", response)
        self.assertIn("height facts I hold", response)

    def test_polysemous_words_imply_nothing(self) -> None:
        """"biggest" names no family; "shortest" spans two (height
        and length) - the same test excluded one of our own words."""
        for word in ("biggest", "shortest"):
            response = self._ask(f"which is the {word}?")
            self.assertNotIn(" is the ", response)

    def test_refusal_names_the_tried_key(self) -> None:
        """The appended key is what was sought, so it is what the
        refusal names - never a bare 'nothing for east gate'."""
        response = self._ask(
            "is the east gate taller than the north tower?")
        self.assertIn("east gate height", response)

    def test_exact_bare_keys_still_win_first(self) -> None:
        """A bare-key numeric fact compares by its own value, never
        shadowed by an appended attribute."""
        response = self._ask(
            "is the old mill taller than the royal hall?")
        self.assertTrue(response.startswith("No"))
        self.assertIn("old mill is 40", response)

    def test_named_and_natural_forms_agree(self) -> None:
        named = self._ask("which height is the tallest?")
        natural = self._ask("which is the tallest?")
        self.assertEqual(named, natural)


class GateVerdictTests(unittest.TestCase):
    """The PASS and the polysemy line, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import impliedattr_gate

        doc = " ".join(impliedattr_gate.__doc__.split())
        self.assertIn("+0.700 at 9.48x seed sd", doc)
        self.assertIn("zero bare pairs shadowed", doc)

    def test_the_polysemy_exclusions_are_on_the_record(self) -> None:
        from ultraquant.experiments import impliedattr_gate
        from ultraquant.interpreter import thoughts

        doc = " ".join(impliedattr_gate.__doc__.split())
        self.assertIn('"shorter" was DROPPED', doc)
        for word in ("bigger", "larger", "smaller", "highest",
                     "lowest", "shorter", "shortest"):
            self.assertNotIn(word, thoughts._IMPLIED_ATTRS)


if __name__ == "__main__":
    unittest.main()
