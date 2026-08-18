"""The data gate: more validated variants measured negative, and why.

Six voices drew 67 fresh validated variants; training on them made both the
held set and the control worse. The post-hoc probes cleared class imbalance
and capacity, leaving off-distribution interference — five drawing styles
pulling features away from the frozen evaluation's style. These tests pin the
frozen-held-set design and the verdict.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from ultraquant.experiments.data_gate import DataReport, run_gate


class GateMachineryTests(unittest.TestCase):
    """The frozen-evaluation design, pinned."""

    def test_a_missing_original_bank_is_a_skip(self):
        report = run_gate(variants_path="Z:/nowhere/original.json")
        self.assertTrue(report.skipped)
        self.assertFalse(report.passes)

    def test_a_missing_new_file_is_a_skip_not_a_crash(self):
        """The gate must not invent an empty enlargement and 'measure' it."""
        report = run_gate(new_path="Z:/nowhere/new.json")
        self.assertTrue(report.skipped)
        self.assertFalse(report.passes)

    def test_the_report_text_carries_the_new_count(self):
        report = DataReport(held={"baseline": 0.5, "enlarged": 0.5},
                            canon={"baseline": 0.8, "enlarged": 0.8},
                            new_count=67, reason="test")
        self.assertIn("new variants trained on: 67", report.as_text())

    def test_new_variants_never_enter_the_held_set(self):
        """The design constraint the whole gate rests on: a tiny run with a
        sentinel new variant must never evaluate on it."""
        import tempfile

        from ultraquant.pattern.recognition import PATTERNS

        with tempfile.TemporaryDirectory() as tmp:
            original = {"plus": [PATTERNS["cross"], PATTERNS["square"]]}
            sentinel = {"plus": [["#####"] * 5]}
            opath = Path(tmp) / "original.json"
            npath = Path(tmp) / "new.json"
            opath.write_text(json.dumps(original), encoding="utf-8")
            npath.write_text(json.dumps(sentinel), encoding="utf-8")
            report = run_gate(variants_path=opath, new_path=npath, seeds=2)
            self.assertFalse(report.skipped)
            self.assertEqual(report.new_count, 1)


class GateVerdictTests(unittest.TestCase):
    """The FAIL, the cleared suspects, and the successor question, pinned."""

    def test_the_recorded_verdict_is_a_fail(self):
        from ultraquant.experiments import data_gate

        doc = " ".join(data_gate.__doc__.split())
        self.assertIn("FAILED", doc)
        self.assertIn("pulled naively, is negative", doc)

    def test_the_cleared_suspects_are_recorded(self):
        """Balance and capacity were measured out post-hoc — and labelled
        post-hoc; losing either sentence un-aims the successor gate."""
        from ultraquant.experiments import data_gate

        doc = " ".join(data_gate.__doc__.split())
        self.assertIn("Class balance is not the cause", doc)
        self.assertIn("Capacity is not the cause", doc)
        self.assertIn("none was pre-registered", doc)

    def test_the_diagnosis_is_off_distribution_interference(self):
        from ultraquant.experiments import data_gate

        doc = " ".join(data_gate.__doc__.split())
        self.assertIn("off-distribution interference", doc)

    def test_the_claim_boundary_is_stated(self):
        """The gate must not be citable as 'the new data is worthless' —
        it measured old-style recognition only."""
        from ultraquant.experiments import data_gate

        doc = " ".join(data_gate.__doc__.split())
        self.assertIn("not that the new data is worthless", doc)

    def test_the_runaway_reasoner_is_recorded(self):
        """qwen3-48b burned every budget in hidden tokens; the exception to
        the reasoning_effort remedy belongs beside §11.13's measurement."""
        from ultraquant.experiments import data_gate

        doc = " ".join(data_gate.__doc__.split())
        self.assertIn("0/0", doc)
        self.assertIn("runaway reasoner", doc)


if __name__ == "__main__":
    unittest.main()
