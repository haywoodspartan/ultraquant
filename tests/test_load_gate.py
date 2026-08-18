"""The load gate: the last suspect measured, and the ceiling characterised.

Six times the variant mass produced a flat curve on a pooled target, killing
the load hypothesis; anchor scaling preserved the control but bought no gain.
With style (§11.26) and load both dead, the variant chain closes on the data
itself. These tests pin the factorial's machinery and the double-negative
verdict.
"""

from __future__ import annotations

import unittest

from ultraquant.experiments.load_gate import (_SIZES, LoadReport, run_gate)


class GateMachineryTests(unittest.TestCase):
    """The factorial and its guards."""

    def test_a_missing_bank_is_a_skip(self):
        report = run_gate(variants_path="Z:/nowhere/original.json")
        self.assertTrue(report.skipped)
        self.assertFalse(report.load_supported)
        self.assertFalse(report.remedy_supported)

    def test_the_sizes_span_the_measured_regime(self):
        """N=90 must reach §11.25's failing share (~46% of training), and
        N=15 must sit below the historical light load."""
        self.assertEqual(_SIZES, (15, 30, 60, 90))

    def test_the_report_carries_both_contrasts(self):
        arms = {(p, s): 0.5 for p in ("fixed", "scaled") for s in _SIZES}
        report = LoadReport(held=dict(arms), canon=dict(arms),
                            best_fixed=60, load_delta=0.01, load_sd=0.02,
                            remedy_delta=0.01, remedy_sd=0.02, reason="test")
        text = report.as_text()
        self.assertIn("load contrast", text)
        self.assertIn("remedy contrast", text)
        self.assertIn("NOT SUPPORTED", text)

    def test_support_requires_beating_sd_not_just_winning(self):
        """A best-N that wins within noise must not support the hypothesis —
        that is exactly the §11.20 trap (+0.013 at 0.23x sd) re-armed."""
        report = LoadReport(best_fixed=60, load_delta=0.01, load_sd=0.04)
        self.assertFalse(report.load_supported)


class GateVerdictTests(unittest.TestCase):
    """The double negative and the closed chain, pinned."""

    def test_neither_claim_survived(self):
        from ultraquant.experiments import load_gate

        doc = " ".join(load_gate.__doc__.split())
        self.assertIn("neither claim survived", doc)
        self.assertIn("NOT SUPPORTED, twice", doc)

    def test_the_flat_curve_is_recorded(self):
        """Six times the variant mass, no damage: the sentence that kills
        load must survive verbatim intent."""
        from ultraquant.experiments import load_gate

        doc = " ".join(load_gate.__doc__.split())
        self.assertIn("flat", doc)
        self.assertIn("+0.000 at sd 0.040", doc)

    def test_the_suspect_list_is_closed(self):
        from ultraquant.experiments import load_gate

        doc = " ".join(load_gate.__doc__.split())
        self.assertIn("suspects are now all dead", doc)

    def test_the_11_25_regression_is_reclassified(self):
        """Three gates triangulate it as borderline noise on the hardest
        sub-target, not a mechanism — the record must say so."""
        from ultraquant.experiments import load_gate

        doc = " ".join(load_gate.__doc__.split())
        self.assertIn("borderline-noise event", doc)

    def test_the_remaining_question_needs_an_oracle(self):
        """No training arrangement learns past ambiguous targets; the next
        step is human labels through the learn queue, not another gate."""
        from ultraquant.experiments import load_gate

        doc = " ".join(load_gate.__doc__.split())
        self.assertIn("ask the human second", doc)


if __name__ == "__main__":
    unittest.main()
