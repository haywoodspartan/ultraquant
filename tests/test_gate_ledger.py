"""§11.151: the gate ledger - every gate's verdict at a named commit."""

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "ultraquant" / "experiments" / "records" / "gate_ledger.json"
# §11.151 and its Amendment A: "needs the environment" gates are listed as skipped.
CLASSES = {"on its own success", "superseded by design", "historical", "open",
           "failed when written"}
HYPOTHESES = {"REJECTED", "NOT REJECTED", "SUPPORTED", "NOT SUPPORTED"}
VERDICTS = {"PASS", "FAIL", "VOID", "QUESTIONABLE"} | HYPOTHESES


def _sweep():
    spec = importlib.util.spec_from_file_location("gate_sweep",
                                                  ROOT / "tools" / "gate_sweep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VerdictParserTests(unittest.TestCase):
    def test_printed_verdict_wins_over_exit_code(self):
        verdict = _sweep().verdict
        self.assertEqual(verdict("FAIL - the ladder closes nothing (+0.000)\n", 0)[0],
                         "FAIL")
        self.assertEqual(verdict("PASS: 4 cases; 3 of 3 plants caught\n", 1)[0], "PASS")

    def test_measurement_lines_before_the_verdict(self):
        verdict = _sweep().verdict
        text = ("arm   correct\nliterals 0.000\n"
                "FAIL - quantity arithmetic at 1.000 against 0.000\n")
        self.assertEqual(verdict(text, 0)[0], "FAIL")
        self.assertEqual(verdict("delta +1.000\nthe verdict: VOID here\n", 0)[0], "VOID")
        # Hypothesis tests report their own outcome words (distribution).
        self.assertEqual(verdict("gap -0.059\nNOT REJECTED - transfer holds\n", 0)[0],
                         "NOT REJECTED")

    def test_no_verdict_falls_back_to_the_exit_code(self):
        verdict = _sweep().verdict
        self.assertEqual(verdict("turns compared: 1430\n", 0)[0], "EXIT0")
        self.assertEqual(verdict("", 2)[0], "EXIT2")
        self.assertEqual(verdict("partial\n", None)[0], "TIMEOUT")


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.ledger = json.loads(LEDGER.read_text(encoding="utf-8"))

    def test_every_gate_module_is_recorded_or_skipped(self):
        gates = {p.stem for p in (ROOT / "ultraquant" / "experiments").glob("*gate*.py")}
        known = set(self.ledger["gates"]) | set(self.ledger["skipped"])
        self.assertEqual(sorted(gates - known), [])

    def test_every_failure_has_a_class_and_a_cause(self):
        for name, entry in self.ledger["gates"].items():
            with self.subTest(gate=name):
                self.assertIn(entry["verdict"], VERDICTS)
                if entry["verdict"] in {"FAIL", "VOID", "QUESTIONABLE"}:
                    self.assertIn(entry.get("class"), CLASSES)
                    self.assertTrue(entry.get("cause", "").strip())
                elif entry["verdict"] in HYPOTHESES:
                    # A hypothesis test's outcome is read against what it recorded.
                    self.assertTrue(entry.get("recorded", "").strip())

    def test_every_skip_has_a_reason(self):
        for name, reason in self.ledger["skipped"].items():
            with self.subTest(gate=name):
                self.assertTrue(str(reason).strip())


if __name__ == "__main__":
    unittest.main()
