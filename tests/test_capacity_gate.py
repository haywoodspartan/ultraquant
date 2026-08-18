"""Scale normalisation, the capacity result, and the closed diagnosis chain.

Three gates ran against the same 53 variants, each correcting the last:
§11.20 said the features were position-bound; §11.21 built centering and
measured that wrong (62% of variants already centred); this gate measured the
scale half wrong too. The winner was capacity — the 30→24→8 ternary net was
under-fitting everything — and the verdicts are pinned so a silent re-run
cannot rewrite the story.
"""

from __future__ import annotations

import unittest

from ultraquant.pattern.recognition import PATTERNS, scale_normalize


class ScaleNormalizeTests(unittest.TestCase):
    """The mechanism: correct at what it does, even though it did not help."""

    def test_a_small_glyph_fills_the_grid(self):
        small_plus = [".....", "..#..", ".###.", "..#..", "....."]
        scaled = scale_normalize(small_plus)
        set_pixels = sum(row.count("#") for row in scaled)
        self.assertGreater(set_pixels,
                           sum(row.count("#") for row in small_plus))
        # Mass reaches the grid's edges after scaling.
        self.assertTrue(any("#" in scaled[i] for i in (0, 4)) or
                        any(row[0] == "#" or row[4] == "#" for row in scaled))

    def test_aspect_is_preserved_a_bar_stays_a_bar(self):
        """Independent-axis stretching would turn a bar into a full block,
        destroying the stripes-versus-square distinction."""
        bar = [".....", ".....", "#####", ".....", "....."]
        self.assertEqual(scale_normalize(bar), bar)

    def test_canonicals_are_unchanged(self):
        for label, rows in PATTERNS.items():
            with self.subTest(label):
                self.assertEqual(scale_normalize(rows), rows)

    def test_an_empty_glyph_is_unchanged(self):
        blank = ["....."] * 5
        self.assertEqual(scale_normalize(blank), blank)

    def test_scaling_is_idempotent(self):
        small = [".....", ".##..", ".##..", ".....", "....."]
        once = scale_normalize(small)
        self.assertEqual(scale_normalize(once), once)

    def test_scaled_features_share_the_shape(self):
        from ultraquant.forge.corpus import features_of, scaled_features

        rows = [".....", "..#..", ".###.", "..#..", "....."]
        self.assertEqual(len(scaled_features(rows)), len(features_of(rows)))


class GateVerdictTests(unittest.TestCase):
    """The pass, its winner, and its limits, pinned."""

    def test_no_variants_file_is_a_skip(self):
        from ultraquant.experiments.capacity_gate import run_gate

        report = run_gate("Z:/nowhere/variants.json")
        self.assertTrue(report.skipped)
        self.assertFalse(report.passes)

    def test_the_recorded_verdict_names_capacity_not_scale(self):
        """The winner was the arm neither prior diagnosis predicted.

        If "positional/64" or the under-fitting sentence leaves the docstring,
        the story changed and the numbers need re-measuring.
        """
        from ultraquant.experiments import capacity_gate

        doc = capacity_gate.__doc__
        self.assertIn("PASSED", doc)
        self.assertIn("positional/64", doc)
        self.assertIn("under-fitting", doc)
        self.assertIn("does nothing", doc)

    def test_the_limits_are_recorded_beside_the_pass(self):
        """Capacity is the binding constraint, not a solution - and the forge
        default is not changed silently, because a wider expert is a larger
        shard in a library whose size ratios are load-bearing."""
        from ultraquant.experiments import capacity_gate

        doc = capacity_gate.__doc__
        self.assertIn("46%", doc)
        self.assertIn("sizing decision", doc)

    def test_the_reference_arm_is_the_failed_one(self):
        """The gate measures against §11.20's arm, not a fresh baseline."""
        from ultraquant.experiments.capacity_gate import _ARMS, _REFERENCE

        self.assertEqual(_REFERENCE, "positional/24")
        self.assertEqual(len(_ARMS), 4)


if __name__ == "__main__":
    unittest.main()
