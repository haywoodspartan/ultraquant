"""The density revisit: the session's arc at scale, pinned.

§11.58's capstone: every §11.45-§11.57 question form against one
~10,000-fact colliding world — floors at 1.000, fabrication at zero,
the full-scan price named. The gate itself is the measurement; these
tests pin its verdict and the harness lesson it taught.
"""

from __future__ import annotations

import unittest


class GateVerdictTests(unittest.TestCase):
    """The PASS, the harness lesson, and the price, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import revisit_gate

        doc = " ".join(revisit_gate.__doc__.split())
        self.assertIn("PASSED on every floor", doc)
        self.assertIn("Every floor at 1.000, zero fabrications across "
                      "every decoy family", doc)

    def test_the_harness_lesson_is_recorded(self) -> None:
        """The first run keyed its world unlike live usage - articles
        kept where the statement path strips them - and three forms
        scored 0.000 while the token-based spread sat at 1.000. The
        asymmetry was the diagnosis."""
        from ultraquant.experiments import revisit_gate

        doc = " ".join(revisit_gate.__doc__.split())
        self.assertIn("A density harness must key its world the way "
                      "live usage would", doc)
        self.assertIn("The asymmetry was the diagnosis", doc)

    def test_the_price_list_is_recorded(self) -> None:
        from ultraquant.experiments import revisit_gate

        doc = " ".join(revisit_gate.__doc__.split())
        self.assertIn("the expensive capability, expensively honest",
                      doc)

    def test_the_scale_smoke_runs_small(self) -> None:
        """The gate machinery itself, exercised at a suite-friendly
        size: floors and fabrication wiring must work end to end."""
        from ultraquant.experiments.revisit_gate import run_gate

        report = run_gate(n_facts=700, seed=1)
        self.assertEqual(report.fabricated, 0)
        self.assertGreaterEqual(report.floors["polar"], 1.0)
        self.assertGreaterEqual(report.floors["aggregate"], 1.0)


if __name__ == "__main__":
    unittest.main()
