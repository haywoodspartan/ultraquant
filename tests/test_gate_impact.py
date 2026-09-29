"""The exams a planned change could flip (tools/gate_impact.py)."""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("gate_impact", ROOT / "tools" / "gate_impact.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GateImpactTests(unittest.TestCase):
    def test_the_filer_reaches_the_forms_exam_and_flags_its_whole_record_comparison(self):
        rows = {row["gate"]: row for row in _tool().impact(["ultraquant/distill/file.py"])}
        self.assertIn("forms_gate", rows)
        self.assertTrue(rows["forms_gate"]["whole-record comparison"])

    def test_a_module_no_exam_imports_reaches_none(self):
        self.assertEqual(_tool().impact(["tools/gate_impact.py"]), [])

    def test_imports_through_another_experiments_module_count(self):
        # second_gate reaches the filer only through the exams it imports.
        rows = {row["gate"] for row in _tool().impact(["ultraquant/distill/file.py"])}
        self.assertIn("second_gate", rows)


if __name__ == "__main__":
    unittest.main()
