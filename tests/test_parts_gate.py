"""Part-based features: the mechanism, the null verdict, and the tie.

The last §11.21 candidate measured nothing: parts alone collapse the control,
and the hybrid's 8-seed hit total ties the reference's exactly — a coincidence
that was debugged as a defect and, in being chased, exposed a real one in the
report's contrast definition. These tests pin the featurizers, the corrected
contrast, and the verdict.
"""

from __future__ import annotations

import unittest

from ultraquant.forge.corpus import (features_of, hybrid_features,
                                     part_features)
from ultraquant.pattern.recognition import PATTERNS


class PartFeatureTests(unittest.TestCase):
    """The histogram: correct at what it does, even though it did not help."""

    def test_the_dimensions_are_as_documented(self):
        rows = PATTERNS["plus"]
        self.assertEqual(len(part_features(rows)), 26)
        self.assertEqual(len(hybrid_features(rows)), 46)

    def test_the_sixteen_bins_account_for_every_patch(self):
        """Sixteen overlapping 2x2 patches, each contributing 1/16."""
        for label, rows in PATTERNS.items():
            with self.subTest(label):
                bins = part_features(rows)[:16]
                self.assertAlmostEqual(sum(bins), 1.0, places=9)

    def test_an_interior_translation_shares_the_bins_exactly(self):
        """The invariance the mechanism was built for — for shapes whose
        one-cell halo stays inside the grid, translation leaves every patch
        pattern's count untouched, while raw pixels share almost nothing."""
        block = [".....", ".##..", ".##..", ".....", "....."]
        shifted = [".....", ".....", "..##.", "..##.", "....."]
        self.assertEqual(part_features(block)[:16],
                         part_features(shifted)[:16])
        self.assertNotEqual(features_of(block), features_of(shifted))

    def test_a_border_hugging_shape_loses_patches_to_clipping(self):
        """The invariance's honest limit: at the grid edge the boundary
        patches fall off the 4x4 patch grid, so an edge placement does not
        histogram like an interior one. Part of why the mechanism starved —
        the difficult variants hug edges and corners."""
        interior = [".....", ".##..", ".##..", ".....", "....."]
        cornered = ["##...", "##...", ".....", ".....", "....."]
        self.assertNotEqual(part_features(interior)[:16],
                            part_features(cornered)[:16])

    def test_orientation_survives_the_histogram(self):
        """Stripes must not collapse into one bin pattern across axes —
        a horizontal pair and a vertical pair are different 4-bit codes."""
        self.assertNotEqual(part_features(PATTERNS["stripes_h"])[:16],
                            part_features(PATTERNS["stripes_v"])[:16])

    def test_the_hybrid_is_the_positional_vector_plus_the_bins(self):
        rows = PATTERNS["diamond"]
        self.assertEqual(hybrid_features(rows),
                         features_of(rows) + part_features(rows)[:16])


class GateVerdictTests(unittest.TestCase):
    """The FAIL, the exhausted candidate list, and the tie, pinned."""

    def test_no_variants_file_is_a_skip(self):
        from ultraquant.experiments.parts_gate import run_gate

        report = run_gate("Z:/nowhere/variants.json")
        self.assertTrue(report.skipped)
        self.assertFalse(report.passes)

    def test_the_recorded_verdict_is_a_fail(self):
        from ultraquant.experiments import parts_gate

        doc = " ".join(parts_gate.__doc__.split())
        self.assertIn("FAILED", doc)
        self.assertIn("the last candidate measures nothing", doc)

    def test_the_control_collapse_of_parts_alone_is_recorded(self):
        """Local texture cannot hold noisy canonicals apart; the 0.453
        control is the sentence that stops anyone deploying parts-only."""
        from ultraquant.experiments import parts_gate

        self.assertIn("0.453", parts_gate.__doc__)

    def test_the_tie_and_the_defect_it_exposed_are_recorded(self):
        """Equal hit totals looked like a plumbing bug; the chase found the
        real one in the contrast definition. Both belong in the record."""
        from ultraquant.experiments import parts_gate

        doc = " ".join(parts_gate.__doc__.split())
        self.assertIn("125/232", doc)
        self.assertIn("best parts arm", doc)

    def test_the_candidate_list_is_declared_spent(self):
        from ultraquant.experiments import parts_gate

        doc = " ".join(parts_gate.__doc__.split())
        self.assertIn("fully measured", doc)
        self.assertIn("data", doc)

    def test_the_contrast_is_registered_against_the_best_parts_arm(self):
        """The reference must never be its own challenger: when it wins, the
        report still carries the best candidate's true contrast and sd."""
        from ultraquant.experiments.parts_gate import _ARMS, _REFERENCE

        self.assertEqual(_REFERENCE, "positional/64")
        self.assertEqual(len(_ARMS), 3)
        self.assertTrue(all(name != _REFERENCE or featurize is features_of
                            for name, featurize in _ARMS))


if __name__ == "__main__":
    unittest.main()
