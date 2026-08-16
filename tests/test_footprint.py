"""The disk-footprint claim, pinned so it cannot go stale.

A summary of this codebase claimed ternary weights are "64x smaller than fp32".
It is unreachable — three states need 2 bits, so 32/2 = 16x is the ceiling — and
this project does not pack bits anyway, storing shards as JSON inside zlib.
These tests hold the measured number in place so nobody plans hardware on the
wrong one.
"""

from __future__ import annotations

import unittest

from ultraquant.experiments.footprint import measure


class CeilingTests(unittest.TestCase):
    """What ternary can and cannot reach, as arithmetic rather than opinion."""

    def test_the_theoretical_ceiling_is_sixteen_not_sixty_four(self):
        report = measure()
        ceiling = report.fp32_packed / report.ternary_packed
        self.assertAlmostEqual(ceiling, 16.0, places=1)
        self.assertLess(ceiling, 64.0,
                        "no arrangement of three states reaches 64x vs fp32")

    def test_what_is_actually_stored_is_below_the_ceiling(self):
        """JSON+zlib is the real path; the packed ideal is not available."""
        report = measure()
        self.assertGreater(report.ternary_stored, report.ternary_packed)

    def test_the_honest_headline_is_about_eleven_times(self):
        report = measure()
        self.assertGreater(report.ratio_vs_packed_fp32, 8.0)
        self.assertLess(report.ratio_vs_packed_fp32, 14.0)


class InterpretationTests(unittest.TestCase):
    """The parts that make the number mean less than it looks."""

    def test_sparsity_does_much_of_the_work(self):
        """So the ratio moves with the weight distribution, not by design."""
        report = measure()
        self.assertGreater(report.sparsity, 0.3,
                           "the threshold should zero a large minority")
        self.assertLess(report.sparsity, 0.7)

    def test_the_flattering_comparison_is_labelled_as_such(self):
        report = measure()
        self.assertGreater(report.ratio_vs_stored_fp32,
                           report.ratio_vs_packed_fp32,
                           "comparing against JSON floats flatters ternary")
        self.assertIn("FLATTERING", report.as_text())

    def test_the_report_names_the_fair_baseline_first(self):
        text = measure().as_text()
        self.assertIn("fair baseline", text)
        self.assertLess(text.index("fair baseline"), text.index("FLATTERING"))

    def test_the_ratio_is_not_a_constant_of_the_design(self):
        """Different weights, different ratio - it must not be quoted as fixed."""
        ratios = {round(measure(seed=s).ratio_vs_packed_fp32, 3)
                  for s in range(4)}
        self.assertGreater(len(ratios), 1,
                           "if this were constant the caveat would be wrong")


if __name__ == "__main__":
    unittest.main()
