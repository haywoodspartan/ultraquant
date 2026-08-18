"""Model-drawn glyph variants: parsing, validation, and the metric's history.

The validator went through two wrong metrics before the right one, and both
failures are pinned so they cannot return:

* Plain cell agreement rejected offset horizontal bars — unambiguously stripes
  to a reader — scoring them 0.000 against ``stripes_h`` (perfectly
  anti-aligned) and 0.4 against ``plus``.
* Fixing that with shift-invariant *agreement* over-corrected, because sparse
  glyphs agree on empty cells almost everywhere; Jaccard over the set pixels
  under shift is what discriminates.
"""

from __future__ import annotations

import unittest

from ultraquant.forge.glyph_distill import (
    contradiction,
    parse_glyphs,
    _agreement,
)
from ultraquant.pattern.recognition import PATTERNS


class ParseTests(unittest.TestCase):
    """Model output is prose-wrapped, numbered, and token-littered."""

    def test_a_clean_glyph_parses(self):
        text = "\n".join(PATTERNS["plus"])
        self.assertEqual(parse_glyphs(text), [PATTERNS["plus"]])

    def test_prose_and_numbering_are_ignored(self):
        text = ("1st Variant:\n\n" + "\n".join(PATTERNS["plus"])
                + "\n\nsecond one:\n" + "\n".join(PATTERNS["square"]))
        self.assertEqual(len(parse_glyphs(text)), 2)

    def test_template_tokens_do_not_break_a_row(self):
        rows = list(PATTERNS["plus"])
        rows[-1] = rows[-1] + "<|END_OF_TURN_TOKEN|>"
        self.assertEqual(parse_glyphs("\n".join(rows)), [PATTERNS["plus"]])

    def test_a_broken_run_is_discarded_not_stitched(self):
        """Four valid rows, an interruption, then more rows must not merge."""
        rows = PATTERNS["plus"]
        text = "\n".join(rows[:4]) + "\noops\n" + "\n".join(rows)
        self.assertEqual(parse_glyphs(text), [rows])

    def test_wrong_width_rows_never_parse(self):
        self.assertEqual(parse_glyphs("######\n#....\n#....\n#....\n#...."),
                         [])


class MetricTests(unittest.TestCase):
    """Both measured failures of earlier metrics, held down."""

    def test_translated_stripes_match_their_own_family(self):
        """Positional agreement scored this 0.000 vs stripes_h and rejected it."""
        offset = [".....", "#####", ".....", "#####", "....."]
        self.assertIsNone(contradiction(offset, "stripes_h"))

    def test_a_shifted_plus_is_still_a_plus(self):
        shifted = [".#...", ".#...", "#####", ".#...", ".#..."]
        self.assertIsNone(contradiction(shifted, "plus"))

    def test_a_diagonal_labeled_plus_is_rejected(self):
        """The 7B's actual failure mode: a diagonal drawn under a plus label."""
        diagonal = ["#....", ".#...", "..#..", "...#.", "....#"]
        self.assertEqual(contradiction(diagonal, "plus"), "cross")

    def test_jaccard_not_dot_agreement(self):
        """Sparse glyphs agree on dots everywhere; only set pixels count."""
        sparse = ["#....", ".....", ".....", ".....", "....."]
        # Under dot-counting this would score >0.6 against most anchors; under
        # Jaccard it is a poor match for everything but still nearest small
        # shapes. What matters: no near-perfect scores from emptiness.
        for label in PATTERNS:
            self.assertLess(_agreement(sparse, PATTERNS[label]), 0.5)

    def test_identity_scores_one(self):
        self.assertEqual(_agreement(PATTERNS["plus"], PATTERNS["plus"]), 1.0)


class GateShapeTests(unittest.TestCase):
    """The gate's bookkeeping, without a variants file or a model."""

    def test_no_variants_file_is_a_skip_never_a_pass(self):
        from ultraquant.experiments.glyph_gate import run_gate

        report = run_gate("Z:/definitely/not/here.json")
        self.assertTrue(report.skipped)
        self.assertFalse(report.passes)

    def test_the_recorded_verdict_is_a_fail(self):
        """The run failed at 0.23x seed sd and the docstring says so.

        Pinned so a silent re-run cannot upgrade the story without the text
        being revisited: the features are position-bound, and that diagnosis
        is the section's finding.
        """
        from ultraquant.experiments import glyph_gate

        self.assertIn("FAILED", glyph_gate.__doc__)
        self.assertIn("position-bound", glyph_gate.__doc__)


if __name__ == "__main__":
    unittest.main()
