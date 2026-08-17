"""Reinforcement gated on confirmation, and the two ways this went wrong first.

§11.16 measured the harm: reinforcing the router's own top route does not reduce
accuracy but grows the wrong answer's margin 0.60 -> 2.33, making every error
3.9x harder to overturn. This pins the fix, its control, and the fact that the
control had to be rebuilt before it meant anything.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path


class GateTests(unittest.TestCase):
    """The pre-registered measurement."""

    @classmethod
    def setUpClass(cls) -> None:
        from ultraquant.experiments.reinforcement_gate import run_gate

        cls.report = run_gate(seeds=6, rounds=40)

    def test_the_gate_passes(self):
        self.assertTrue(self.report.passes, self.report.reason)
        self.assertGreater(self.report.margin_ratio, 1.0)

    def test_entrenchment_is_removed(self):
        """The margin should fall back toward the un-reinforced baseline."""
        self.assertLess(self.report.margin_fixed,
                        self.report.margin_baseline / 2)

    def test_learning_is_retained(self):
        """The control: a fix that stops learning is an amputation."""
        self.assertGreaterEqual(self.report.transfer_fixed,
                                self.report.transfer_baseline)
        self.assertGreater(self.report.weight_fixed, 0.0,
                           "it must still learn something")

    def test_the_control_can_actually_fail(self):
        """A router that never reinforces must score 0.000 on the paraphrases.

        The first version of this control could not fail: its paraphrases
        contained `mark` and `square`, both *registered* keywords, so they
        routed correctly with no learning at all and the control read 1.000 for
        a mechanism that had learned nothing. Verified here rather than assumed.
        """
        from ultraquant.experiments.reinforcement_gate import (
            _PARAPHRASES,
            _build,
            _transfer,
        )

        untaught = _build()
        self.assertEqual(_transfer(untaught), 0.0,
                         "the paraphrases must be unroutable without learning")
        for query, truth in _PARAPHRASES:
            with self.subTest(query):
                shared = set(query.lower().split()) & untaught._base.get(truth, set())
                self.assertEqual(shared, set(),
                                 "a paraphrase must share no base keyword")


class ConfirmationTests(unittest.TestCase):
    """Which signals may confirm a route, and which had to be removed."""

    def _ctx(self, **data):
        class _Ctx:
            def __init__(self, payload):
                self.data = payload

        return _Ctx(data)

    def test_an_expert_placing_the_input_confirms(self):
        from ultraquant.interpreter.thoughts import _route_confirmation

        self.assertEqual(
            _route_confirmation(self._ctx(prediction=("plus", 0.9))),
            "expert placed it")

    def test_an_unfamiliar_prediction_does_not_confirm(self):
        from ultraquant.interpreter.thoughts import _route_confirmation

        self.assertIsNone(_route_confirmation(
            self._ctx(prediction=("plus", 0.2), unfamiliar=True)))

    def test_a_composed_reading_confirms(self):
        from ultraquant.interpreter.thoughts import _route_confirmation

        self.assertEqual(_route_confirmation(self._ctx(composed={"shape": "x"})),
                         "blackboard composed")

    def test_a_recalled_fact_does_not_confirm_a_route(self):
        """Recall runs BEFORE Route, so a fact is found regardless of routing.

        Accepting it reinforced `crosses` for "what shape is this glyph" purely
        because a fact happened to be recalled - the original defect wearing a
        confirmation signal that confirmed the wrong thing.
        """
        from ultraquant.interpreter.thoughts import _route_confirmation

        self.assertIsNone(_route_confirmation(self._ctx(facts=["a fact"])))

    def test_code_and_plans_do_not_confirm_a_route(self):
        """Both are driven by intent, not by which category won."""
        from ultraquant.interpreter.thoughts import _route_confirmation

        self.assertIsNone(_route_confirmation(self._ctx(code_result=42)))
        self.assertIsNone(_route_confirmation(self._ctx(plan="step one")))

    def test_recall_really_does_precede_route(self):
        """The reason the fact signal is invalid, asserted so it cannot rot."""
        from ultraquant.interpreter.thoughts import PIPELINE

        names = [type(step).__name__ for step in PIPELINE]
        self.assertLess(names.index("Recall"), names.index("Route"))

    def test_nothing_confirms_an_empty_turn(self):
        from ultraquant.interpreter.thoughts import _route_confirmation

        self.assertIsNone(_route_confirmation(self._ctx()))


class PipelineTests(unittest.TestCase):
    """The change reaching a real session, on a copy of the deployed library."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_rgp_"))
        source = Path("uq_home")
        if not source.exists():
            self.skipTest("no deployed library to copy")
        self.home = self.dir / "home"
        shutil.copytree(source, self.home)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _run(self, text: str):
        from ultraquant.interpreter.thoughts import build_session, run_pipeline

        session = build_session(self.home, seed=0)
        _response, trace = run_pipeline(text, session)
        return [step["summary"] for step in trace if step["thought"] == "Learn"]

    def test_a_placed_glyph_is_reinforced(self):
        from ultraquant.pattern.recognition import PATTERNS

        summaries = " ".join(self._run("\n".join(PATTERNS["plus"])))
        self.assertIn("reinforced", summaries)
        self.assertIn("expert placed it", summaries)

    def test_an_unconfirmed_route_is_not_reinforced_and_says_so(self):
        summaries = " ".join(self._run("what shape is this glyph"))
        self.assertIn("not reinforced", summaries)
        self.assertIn("unconfirmed route", summaries)


if __name__ == "__main__":
    unittest.main()
