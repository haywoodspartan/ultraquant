"""The distribution gate: the hypothesis test that killed its own premise.

§11.25 blamed off-distribution interference; this gate made the claim
falsifiable and could not reject one-distribution — transfer holds within
noise in both directions. These tests pin the test's machinery (a hypothesis
test, not a capability gate) and the verdict that redirects the chain from
style to load.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from ultraquant.experiments.distribution_gate import (DistributionReport,
                                                      run_gate)


class GateMachineryTests(unittest.TestCase):
    """A hypothesis test's outcomes are REJECTED / NOT REJECTED, never PASS."""

    def test_a_missing_bank_is_a_skip(self):
        report = run_gate(variants_path="Z:/nowhere/original.json")
        self.assertTrue(report.skipped)
        self.assertFalse(report.rejected)

    def test_the_report_never_says_pass(self):
        """Rejecting or keeping a hypothesis is not a capability; the word
        PASS in this report would misfile it in the book."""
        report = DistributionReport(
            accuracy={n: 0.5 for n in ("within-original", "within-new",
                                       "cross original->new",
                                       "cross new->original")},
            canon={n: 0.8 for n in ("within-original", "within-new",
                                    "cross original->new",
                                    "cross new->original")},
            gaps={"original->new": (-0.01, 0.1)},
            reason="test")
        self.assertNotIn("PASS", report.as_text())
        self.assertIn("NOT REJECTED", report.as_text())

    def test_a_failed_control_yields_no_verdict(self):
        """run_gate marks invalid after the tables are filled; the text must
        carry INVALID and neither hypothesis verdict."""
        arms = ("within-original", "within-new",
                "cross original->new", "cross new->original")
        report = DistributionReport(
            valid=False, reason="control fell",
            accuracy={n: 0.5 for n in arms},
            canon={n: 0.4 for n in arms})
        text = report.as_text()
        self.assertIn("INVALID", text)
        self.assertNotIn("NOT REJECTED", text)

    def test_the_cross_arms_train_on_the_full_banks(self):
        """The bias must favour transfer: rejection is only conservative if
        the cross arms see twice the within arms' data."""
        import tempfile

        from ultraquant.pattern.recognition import PATTERNS

        with tempfile.TemporaryDirectory() as tmp:
            a = {"plus": [PATTERNS["cross"], PATTERNS["square"]]}
            b = {"plus": [PATTERNS["diamond"], PATTERNS["arrow_up"]]}
            apath = Path(tmp) / "a.json"
            bpath = Path(tmp) / "b.json"
            apath.write_text(json.dumps(a), encoding="utf-8")
            bpath.write_text(json.dumps(b), encoding="utf-8")
            report = run_gate(variants_path=apath, new_path=bpath, seeds=2)
            self.assertFalse(report.skipped)
            self.assertEqual(set(report.gaps),
                             {"original->new", "new->original"})


class GateVerdictTests(unittest.TestCase):
    """NOT REJECTED, the dead diagnosis, and the load hypothesis, pinned."""

    def test_the_recorded_verdict_is_not_rejected(self):
        from ultraquant.experiments import distribution_gate

        doc = " ".join(distribution_gate.__doc__.split())
        self.assertIn("NOT rejected", doc)
        self.assertIn("§11.25's diagnosis dies", doc)

    def test_the_chain_correction_is_recorded(self):
        """Twice now a section's parting diagnosis died in the next gate;
        the record must say so, or the pattern is invisible."""
        from ultraquant.experiments import distribution_gate

        doc = " ".join(distribution_gate.__doc__.split())
        self.assertIn("second time", doc)

    def test_the_load_hypothesis_is_labelled_a_hypothesis(self):
        from ultraquant.experiments import distribution_gate

        doc = " ".join(distribution_gate.__doc__.split())
        self.assertIn("stated as the next hypothesis, not a finding", doc)
        self.assertIn("load", doc)

    def test_the_drawing_moratorium_is_recorded(self):
        """Transfer works, so volume is not the bottleneck: more drawing
        tokens are wasted until the load question is answered."""
        from ultraquant.experiments import distribution_gate

        doc = " ".join(distribution_gate.__doc__.split())
        self.assertIn("no more drawing tokens", doc)


if __name__ == "__main__":
    unittest.main()
