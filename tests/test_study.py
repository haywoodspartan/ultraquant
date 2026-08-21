"""The study cycle: apply only what corroborates, report everything.

Loop logic under a fake panel — no LM Studio in the suite. The gate
measures the live hallucination risk; these pin the plumbing: budget,
formatting, acceptance-checking, and the honest report.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.interpreter.study import StudyCycle


class _FakeConsensus:
    def __init__(self, corroborated, split, voices):
        self.corroborated = corroborated
        self.split = split
        self.voices = voices


class _FakePanel:
    """Corroborates a scripted answer for prompts containing a keyword."""

    def __init__(self, script: dict) -> None:
        self.script = script
        self.asked: list[str] = []

    def teach(self, prompt, stash, usages=None):
        self.asked.append(prompt)
        for keyword, answer in self.script.items():
            if keyword in prompt:
                consensus = _FakeConsensus(
                    True, {answer: ["m1", "m2"]}, 2)
                return {"consensus": consensus, "entry_ids": [1],
                        "filed": 1}
        return {"consensus": _FakeConsensus(False, {}, 1),
                "entry_ids": [], "filed": 1}


def _session_with_gap(root: Path, term: str):
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    session = build_session(root, seed=0)
    for phrase in (f"tell me about the {term}",
                   f"i keep thinking about the {term}",
                   f"the {term} again",
                   f"what about the {term} then"):
        run_pipeline(phrase, session)
    return session


class StudyCycleTests(unittest.TestCase):
    """The loop's rules, under a scripted panel."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_study_"))
        self.session = _session_with_gap(self.dir, "gravity")

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _cycle(self, script: dict) -> StudyCycle:
        cycle = StudyCycle(self.session, panel_models=["fake"])
        cycle._panel = _FakePanel(script)
        return cycle

    def test_a_corroborated_answer_is_applied_and_recalls(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        report = self._cycle(
            {"gravity": "attraction between masses"}).run()
        subjects = [subject for subject, _a, _v in report.applied]
        self.assertIn("gravity", subjects)
        response, _trace = run_pipeline("what is gravity?", self.session)
        self.assertIn("attraction between masses", response)

    def test_uncorroborated_stays_open_and_is_reported(self) -> None:
        report = self._cycle({}).run()
        self.assertEqual(report.applied, [])
        self.assertTrue(any("not corroborated" in why
                            for _s, why in report.open))
        self.assertGreater(report.quarantined, 0)

    def test_the_budget_bounds_the_asking(self) -> None:
        cycle = self._cycle({})
        cycle.max_questions = 1
        report = cycle.run()
        self.assertEqual(report.asked, 1)

    def test_no_panel_means_everything_stays_open(self) -> None:
        cycle = StudyCycle(self.session, panel_models=None)
        report = cycle.run()
        self.assertEqual(report.applied, [])

    def test_the_report_names_voices(self) -> None:
        report = self._cycle(
            {"gravity": "attraction between masses"}).run()
        _subject, _answer, voices = report.applied[0]
        self.assertEqual(voices, 2)
        self.assertIn("2 voices", report.as_text())


class GateMachineryTests(unittest.TestCase):
    """Skip semantics: no LM Studio, no verdict."""

    def test_the_skip_never_passes(self) -> None:
        from ultraquant.experiments.study_gate import StudyGateReport

        report = StudyGateReport(skipped=True, reason="test")
        self.assertFalse(report.passes)
        self.assertIn("SKIPPED", report.as_text())


if __name__ == "__main__":
    unittest.main()
