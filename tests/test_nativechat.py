"""One conversation, two tiers, the same words - §11.91, pinned.

The pipeline's product is a sentence, so these pins compare
sentences. They skip without the built binary; the recorded verdict
and the named gap are pinned unconditionally.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from ultraquant.experiments import nativechat_gate

_PROBE = nativechat_gate.probe_path()
_BUILT = _PROBE.exists()


def _native(lines: list[str]) -> list[str]:
    got = subprocess.run([str(_PROBE), "--plain"],
                         input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True)
    return got.stdout.splitlines()


def _python(lines: list[str]) -> list[str]:
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    root = Path(tempfile.mkdtemp(prefix="uq_chatpin_"))
    try:
        session = build_session(root, seed=0)
        return [run_pipeline(line, session)[0] for line in lines]
    finally:
        shutil.rmtree(root, ignore_errors=True)


@unittest.skipUnless(_BUILT, "native pipeline not built")
class ParityTests(unittest.TestCase):
    """A conversation, walked in both tiers."""

    def test_a_session_reads_the_same(self) -> None:
        lines = [
            "the tower material is iron",
            "what is the tower material?",
            "the tower height is 300 meters",
            "what is the tower height?",
            "the tower material is steel",
            "the old tower material is oak",
            "the dome material is not steel",
            "what is the dome material?",
            "what is 2 + 2?",
            "what is 100 / 3 to 2 decimal places?",
            "what is the square root of 2?",
            "what is the average of 3, 5 and 10?",
            "hello there",
        ]
        self.assertEqual(_native(lines), _python(lines))

    def test_a_revision_is_named_aloud_identically(self) -> None:
        lines = ["the keep material is oak",
                 "the keep material is bronze"]
        self.assertEqual(_native(lines)[1], _python(lines)[1])

    def test_the_near_key_note_matches(self) -> None:
        lines = ["the tower material is iron",
                 "the old tower material is oak"]
        self.assertEqual(_native(lines)[1], _python(lines)[1])

    def test_a_generated_session_agrees(self) -> None:
        report = nativechat_gate.run_gate(turns=25, conversations=3,
                                          seed=5)
        self.assertEqual(report.real_differences, 0, report.reason)
        self.assertEqual(report.intent_differences, 0, report.reason)
        self.assertEqual(report.store_differences, 0, report.reason)


class GateVerdictTests(unittest.TestCase):
    """The recorded PASS, and the debt it names."""

    def setUp(self) -> None:
        self.doc = " ".join(nativechat_gate.__doc__.split())

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        self.assertIn("| unexplained differences | **0** |", self.doc)

    def test_the_quiet_zeros_are_explained(self) -> None:
        self.assertIn("a right answer reached the wrong way diverges "
                      "later, not now", self.doc)
        self.assertIn("also remembered the right thing", self.doc)

    def test_the_gap_is_named_rather_than_avoided(self) -> None:
        self.assertIn("a gate quietly avoiding what the tier cannot "
                      "do", self.doc)
        self.assertIn("a measurement of what is missing rather than a "
                      "silence about it", self.doc)

    def test_the_unported_forms_are_listed(self) -> None:
        """A scope claimed is a scope that can be checked."""
        for form in ("polar questions", "why", "comparatives",
                     "superlatives", "aggregates", "history",
                     "choices", "testimony"):
            self.assertIn(form, self.doc)

    def test_the_closed_debt_is_recorded_as_closed(self) -> None:
        """A gap that was paid should read as paid, not as still
        owed - a stale verdict is the same lie as an unmeasured
        one."""
        self.assertIn("| differing only by the curiosity hint | **0** |",
                      self.doc)
        self.assertIn("The 71 was never an excuse", self.doc)


if __name__ == "__main__":
    unittest.main()
