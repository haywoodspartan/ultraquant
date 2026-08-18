"""Centroid centering, and the double failure of translation mechanisms.

The mechanism works as specified — a corner plus centres to a centred plus,
already-centred glyphs are untouched — and the gate it was built for failed,
in a way that redefined the problem: the variants' difficulty is structure and
scale, not translation. Both halves are pinned.
"""

from __future__ import annotations

import unittest

from ultraquant.forge.corpus import centered_features, features_of
from ultraquant.pattern.recognition import PATTERNS, center_glyph


class CenterGlyphTests(unittest.TestCase):
    """The primitive itself: deterministic, bounded, identity when centred."""

    def test_a_corner_glyph_moves_to_the_centre(self):
        corner = [".#...", "###..", ".#...", ".....", "....."]
        centred = center_glyph(corner)
        self.assertEqual(centred, [".....", "..#..", ".###.", "..#..", "....."])

    def test_canonicals_are_already_centred(self):
        for label, rows in PATTERNS.items():
            with self.subTest(label):
                self.assertEqual(center_glyph(rows), rows)

    def test_an_empty_glyph_is_returned_unchanged(self):
        blank = ["....."] * 5
        self.assertEqual(center_glyph(blank), blank)

    def test_centering_is_idempotent(self):
        corner = ["#....", "#....", "##...", ".....", "....."]
        once = center_glyph(corner)
        self.assertEqual(center_glyph(once), once)

    def test_centering_is_deterministic_across_calls(self):
        rows = [".#...", "###..", ".#...", ".....", "....."]
        self.assertEqual(center_glyph(rows), center_glyph(list(rows)))

    def test_centred_features_have_the_same_shape(self):
        """Same 30 dims - the downstream machinery must not notice."""
        rows = [".#...", "###..", ".#...", ".....", "....."]
        self.assertEqual(len(centered_features(rows)), len(features_of(rows)))

    def test_a_translation_produces_identical_centred_features(self):
        """The property the representation exists for."""
        small_plus = [".....", "..#..", ".###.", "..#..", "....."]
        corner = [".#...", "###..", ".#...", ".....", "....."]
        self.assertEqual(centered_features(corner),
                         centered_features(small_plus))

    def test_features_of_is_untouched(self):
        """Deployed experts were trained on positional features; changing the
        definition under them would silently break existing libraries."""
        rows = PATTERNS["plus"]
        pixels = features_of(rows)
        self.assertEqual(len(pixels), 30)
        self.assertEqual(pixels[:25].count(1.0),
                         sum(row.count("#") for row in rows))


class GateVerdictTests(unittest.TestCase):
    """The measured double failure, pinned so re-runs cannot rewrite it."""

    def test_no_variants_file_is_a_skip(self):
        from ultraquant.experiments.representation_gate import run_gate

        report = run_gate("Z:/nowhere/variants.json")
        self.assertTrue(report.skipped)
        self.assertFalse(report.passes)

    def test_the_recorded_verdict_names_both_failures(self):
        """Centering hurt its own target subset; augmentation collapsed the
        control. If either sentence leaves the docstring, the story changed
        and these numbers need re-measuring."""
        from ultraquant.experiments import representation_gate

        doc = representation_gate.__doc__
        self.assertIn("FAILED", doc)
        self.assertIn("0.200", doc, "the off-centre subset regression")
        self.assertIn("0.348", doc, "the augmentation collapse")
        self.assertIn("structure and scale, not translation", doc)

    def test_the_gate_is_factorial(self):
        """Four arms, so gains are attributable - not two."""
        from ultraquant.experiments.representation_gate import _ARMS

        self.assertEqual(len(_ARMS), 4)
        names = {name for name, _f, _v in _ARMS}
        self.assertIn("positional+variants", names)
        self.assertIn("centered+variants", names)


if __name__ == "__main__":
    unittest.main()
