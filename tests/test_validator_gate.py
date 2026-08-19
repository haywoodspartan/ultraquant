"""The validator gate: a clean negative, and the mechanism shelved.

The margin-and-floor tightening of the shift-Jaccard validator measured
5% recall against the blind reader's rejections at an unstable operating
point: the reader rejects on shape gestalt, which agreement thresholds do
not encode. These tests pin the mechanism's behavior at the historical
point, the calibration honesty, and the negative verdict that makes blind
reading the permanent admission protocol.
"""

from __future__ import annotations

import unittest

from ultraquant.forge.glyph_distill import contradiction, strict_contradiction
from ultraquant.pattern.recognition import PATTERNS


class StrictContradictionTests(unittest.TestCase):
    """Correct at what it does, even though it measured out of a job."""

    def test_zero_parameters_reproduce_the_historical_rule(self) -> None:
        for label, rows in PATTERNS.items():
            with self.subTest(label):
                self.assertEqual(
                    strict_contradiction(rows, label, margin=0.0, floor=0.0),
                    contradiction(rows, label))

    def test_the_floor_rejects_a_weak_resemblance(self) -> None:
        """The _agreement docstring's own honest limit: a bare 5-cell
        vertical line labeled 'plus' passes, because among these anchors
        it genuinely is nearest the plus arm. A floor names it ambiguous
        instead of passing it on 'nearest'."""
        bare_line = ["..#..", "..#..", "..#..", "..#..", "..#.."]
        self.assertIsNone(strict_contradiction(bare_line, "plus",
                                               floor=0.0))
        self.assertEqual(strict_contradiction(bare_line, "plus", floor=0.9),
                         "(ambiguous)")

    def test_the_margin_rejects_a_whisker_win(self) -> None:
        """A glyph that beats its nearest rival by less than the margin is
        returned as contradicted by that rival."""
        rows = PATTERNS["plus"]
        self.assertIsNone(strict_contradiction(rows, "plus", margin=0.0))
        verdict = strict_contradiction(rows, "plus", margin=1.0)
        self.assertIsNotNone(verdict)
        self.assertNotEqual(verdict, "(ambiguous)")


class GateMachineryTests(unittest.TestCase):
    """Calibration honesty and the guards."""

    def test_a_missing_curation_file_is_a_skip(self) -> None:
        from ultraquant.experiments.validator_gate import run_gate

        report = run_gate("Z:/nowhere/curation.json")
        self.assertTrue(report.skipped)
        self.assertFalse(report.passes)

    def test_scoring_separates_keep_rate_from_recall(self) -> None:
        from ultraquant.experiments.validator_gate import _score

        items = [
            {"rows": PATTERNS["plus"], "label": "plus", "keep": True},
            {"rows": PATTERNS["cross"], "label": "plus", "keep": False},
        ]
        keep_rate, recall = _score(items, margin=0.0, floor=0.0)
        self.assertEqual(keep_rate, 1.0)
        self.assertEqual(recall, 1.0)


class GateVerdictTests(unittest.TestCase):
    """The negative, its cause, and the permanent protocol, pinned."""

    def test_the_recorded_verdict_is_a_fail(self) -> None:
        from ultraquant.experiments import validator_gate

        doc = " ".join(validator_gate.__doc__.split())
        self.assertIn("FAILED", doc)
        self.assertIn("cannot see what the reader sees", doc)
        self.assertIn("0.050", doc)

    def test_the_gestalt_diagnosis_is_recorded(self) -> None:
        from ultraquant.experiments import validator_gate

        doc = " ".join(validator_gate.__doc__.split())
        self.assertIn("shape gestalt", doc)

    def test_the_blind_reading_protocol_is_made_permanent(self) -> None:
        from ultraquant.experiments import validator_gate

        doc = " ".join(validator_gate.__doc__.split())
        self.assertIn("admission to the training bank goes through a "
                      "blind reading", doc)

    def test_the_shelf_now_has_three_residents(self) -> None:
        """center_glyph (§11.21), scale_normalize (§11.22),
        strict_contradiction (§11.32): built, tested, not adopted."""
        from ultraquant.experiments import validator_gate
        from ultraquant.forge import glyph_distill

        gate_doc = " ".join(validator_gate.__doc__.split())
        mech_doc = " ".join(glyph_distill.strict_contradiction.__doc__
                            .split())
        for doc in (gate_doc, mech_doc):
            self.assertIn("center_glyph", doc)
            self.assertIn("scale_normalize", doc)


if __name__ == "__main__":
    unittest.main()
