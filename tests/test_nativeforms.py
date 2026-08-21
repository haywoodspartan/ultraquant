"""A claim about a set, twice - §11.93, pinned.

The trailing clauses are the answer here, not decoration: scope,
exclusions by name, ties named, denials accounted. And the summation
is pinned, because that is where the two tiers actually parted.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from ultraquant.experiments import nativeforms_gate

_PROBE = nativeforms_gate.probe_path()
_BUILT = _PROBE.exists()

_WORLD = [
    "the north tower weight is 625 tonnes",
    "the south tower weight is 725 kilograms",
    "the west keep weight is 875 kilograms",
    "the east mill weight is 350 kilograms",
    "the old barn weight is 875 tonnes",
    "the royal hall weight is 825 kilograms",
]


def _native(lines: list[str]) -> list[str]:
    got = subprocess.run([str(_PROBE), "--plain"],
                         input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True)
    return got.stdout.splitlines()


def _python(lines: list[str]) -> list[str]:
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    root = Path(tempfile.mkdtemp(prefix="uq_formspin_"))
    try:
        session = build_session(root, seed=0)
        return [run_pipeline(line, session)[0] for line in lines]
    finally:
        shutil.rmtree(root, ignore_errors=True)


@unittest.skipUnless(_BUILT, "native pipeline not built")
class ParityTests(unittest.TestCase):
    """Set-claims, with their trailing clauses."""

    def _same(self, lines: list[str]) -> list[str]:
        native, python = _native(lines), _python(lines)
        self.assertEqual(native, python)
        return native

    def test_the_average_that_found_the_summation(self) -> None:
        """One digit in the last place, and it was CPython's
        compensated sum rather than the arithmetic."""
        answers = self._same(_WORLD + ["what is the average weight?"])
        self.assertIn("250.463 tonnes", answers[-1])

    def test_ranking_scope_and_exclusions(self) -> None:
        self._same([
            "the north tower height is 300 meters",
            "the south tower height is 450 meters",
            "the west keep height is 2 kilometers",
            "the mill height is not 900 meters",
            "the barn height is tall",
            "which height is the tallest?",
            "which height is the shortest?",
            "what is the total height?",
            "how many height facts do you hold?",
        ])

    def test_a_tie_is_named_not_broken(self) -> None:
        answers = self._same([
            "the north tower height is 300 meters",
            "the south tower height is 300 meters",
            "which height is the tallest?",
        ])
        self.assertIn("Tied for tallest", answers[-1])

    def test_an_empty_family_and_a_polysemous_word(self) -> None:
        self._same(["the tower height is 300 meters",
                    "what is the total weight?",
                    "which weight is the tallest?",
                    "which is the biggest?"])

    def test_generated_worlds_agree(self) -> None:
        report = nativeforms_gate.run_gate(worlds=10, seed=4)
        self.assertEqual(report.mismatches, 0, report.reason)


class GateVerdictTests(unittest.TestCase):
    """The PASS, and the summation finding."""

    def setUp(self) -> None:
        self.doc = " ".join(nativeforms_gate.__doc__.split())

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        self.assertIn("| answers that differed | **0** |", self.doc)

    def test_the_summation_finding_is_recorded(self) -> None:
        self.assertIn("CPython's `sum()` over floats has carried a "
                      "Neumaier compensation term since 3.12", self.doc)

    def test_widening_the_comparison_was_refused(self) -> None:
        self.assertIn('widening the comparison to "close enough" would '
                      "have turned this gate into one that cannot see "
                      "the next real difference either", self.doc)


if __name__ == "__main__":
    unittest.main()
