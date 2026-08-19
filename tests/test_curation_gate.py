"""The curation gate: the blind reader, the confirmed suspect, the record.

A quarter of the validated banks failed a blind independent reading, and
removing those items lifted endorsed-held recognition at exactly 2.00x seed
sd while the error mass concentrated in the rejected pile. These tests pin
the blind protocol's integrity guards, the verdict, and the honesty notes —
including that the oracle was Claude, not a human.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from ultraquant.experiments.curation_gate import CurationReport, run_gate


class GateMachineryTests(unittest.TestCase):
    """The guards that keep the curation honest."""

    def test_missing_files_are_skips(self):
        report = run_gate(curation_path="Z:/nowhere/curation.json")
        self.assertTrue(report.skipped)
        self.assertFalse(report.passes)

    def test_incomplete_curation_refuses_to_run(self):
        """A verdict for only part of the banks would let selection bias
        pick which items get judged; the gate must demand full coverage."""
        import tempfile

        from ultraquant.pattern.recognition import PATTERNS

        with tempfile.TemporaryDirectory() as tmp:
            bank = {"plus": [PATTERNS["cross"], PATTERNS["square"]]}
            curation = {"oracle": "test", "verdicts": [
                {"label": "plus", "rows": PATTERNS["cross"],
                 "keep": True, "read_as": "plus"}]}
            opath = Path(tmp) / "o.json"
            npath = Path(tmp) / "n.json"
            cpath = Path(tmp) / "c.json"
            opath.write_text(json.dumps(bank), encoding="utf-8")
            npath.write_text(json.dumps({"plus": []}), encoding="utf-8")
            cpath.write_text(json.dumps(curation), encoding="utf-8")
            report = run_gate(variants_path=opath, new_path=npath,
                              curation_path=cpath)
            self.assertTrue(report.skipped)
            self.assertIn("lack a verdict", report.reason)

    def test_an_all_keep_curation_cannot_run_the_contrast(self):
        """If the reader rejects nothing there is no treatment arm, and the
        gate must say so rather than compare an arm to itself."""
        import tempfile

        from ultraquant.pattern.recognition import PATTERNS

        with tempfile.TemporaryDirectory() as tmp:
            bank = {"plus": [PATTERNS["cross"]]}
            curation = {"oracle": "test", "verdicts": [
                {"label": "plus", "rows": PATTERNS["cross"],
                 "keep": True, "read_as": "plus"}]}
            opath = Path(tmp) / "o.json"
            npath = Path(tmp) / "n.json"
            cpath = Path(tmp) / "c.json"
            opath.write_text(json.dumps(bank), encoding="utf-8")
            npath.write_text(json.dumps({"plus": []}), encoding="utf-8")
            cpath.write_text(json.dumps(curation), encoding="utf-8")
            report = run_gate(variants_path=opath, new_path=npath,
                              curation_path=cpath)
            self.assertTrue(report.skipped)
            self.assertIn("contrast", report.reason)

    def test_the_report_states_the_rejection_share(self):
        report = CurationReport(kept={"plus": [["#"]] * 3},
                                rejected={"plus": [["#"]]},
                                reason="test")
        self.assertIn("25% of the banks", report.as_text())


class GateVerdictTests(unittest.TestCase):
    """The PASS, the concentration, and the oracle's name, pinned."""

    def test_the_recorded_verdict_is_a_pass(self):
        from ultraquant.experiments import curation_gate

        doc = " ".join(curation_gate.__doc__.split())
        self.assertIn("PASSED", doc)
        self.assertIn("confirmed, not buried", doc)
        self.assertIn("2.00x seed sd", doc)

    def test_the_concentration_numbers_are_recorded(self):
        from ultraquant.experiments import curation_gate

        doc = " ".join(curation_gate.__doc__.split())
        self.assertIn("0.580", doc)
        self.assertIn("0.406", doc)

    def test_the_oracle_is_named_as_claude_not_a_human(self):
        """The single most important honesty note: the reader was Claude
        Fable 5 under a blind protocol, overturnable by any human."""
        from ultraquant.experiments import curation_gate

        doc = " ".join(curation_gate.__doc__.split())
        self.assertIn("Claude Fable 5's readings, not a human's", doc)
        self.assertIn("blind", doc)

    def test_the_partial_explanation_boundary_is_stated(self):
        """0.633 on endorsed data: ambiguity was part of the ceiling, and
        the earlier capacity/feature limits stay real."""
        from ultraquant.experiments import curation_gate

        doc = " ".join(curation_gate.__doc__.split())
        self.assertIn("part* of the ceiling, not all of it".replace("*", ""),
                      doc.replace("*", ""))

    def test_the_validator_consequence_is_scoped(self):
        """Tightening shift-Jaccard is a new mechanism for a new gate; the
        standing instruction is to train on the curated bank."""
        from ultraquant.experiments import curation_gate

        doc = " ".join(curation_gate.__doc__.split())
        self.assertIn("its own gate", doc)
        self.assertIn("curated bank", doc)


class CurationArtifactTests(unittest.TestCase):
    """The live curation file, when present, must be complete and blind."""

    def test_the_live_curation_file_is_well_formed(self):
        path = Path("uq_home/vault/glyph_curation.json")
        if not path.exists():
            self.skipTest("no live curation file in this checkout")
        curation = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(curation["oracle"], "claude-fable-5")
        self.assertIn("blind", curation["protocol"])
        for verdict in curation["verdicts"]:
            self.assertIn(verdict["keep"],
                          (True, False))
            if verdict["keep"]:
                self.assertEqual(verdict["read_as"], verdict["label"])
            else:
                self.assertNotEqual(verdict["read_as"], verdict["label"])


if __name__ == "__main__":
    unittest.main()
