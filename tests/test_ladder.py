"""The ladder: hints guide the climb, confirmations compose, revision
takes the whole ladder down.

§11.31 measured one confirmation buying one step; this pins the
composition — a cooperative teacher following only the system's own
hints closes depth-4 and depth-5 questions at exactly the minimal
confirmation count, and a bottom revision retracts every rung.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.experiments.ladder_gate import _HINT_RE, _climb


class ClimbDriverTests(unittest.TestCase):
    """The mechanical hint-follower: no world knowledge, no wandering."""

    def test_the_hint_regex_matches_the_live_phrasing(self) -> None:
        response = ("I don't hold that exactly. Nearest I hold: x is y "
                    "(confidence 0.60). If I knew the steel conductivity, "
                    "I could work this out - ':learn' will ask.")
        match = _HINT_RE.search(response)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1).strip(), "steel conductivity")

    def test_a_cyclic_hint_terminates(self) -> None:
        """A hint that points back at a question already on the stack must
        end the climb, not loop until the step cap."""
        script = {
            "what is the a b?": "If I knew the c b, I could work this out",
            "what is the c b?": "If I knew the a b, I could work this out",
        }
        calls = []

        def fake_pipeline(text, session):
            calls.append(text)
            return script.get(text, "I don't hold anything on that yet."), []

        ok, used, _resp = _climb(fake_pipeline, None, "what is the a b?")
        self.assertFalse(ok)
        self.assertEqual(used, 0)
        self.assertLess(len(calls), 6)

    def test_only_rungs_earn_the_yes(self) -> None:
        """The deep question deriving is the ANSWER; affirming it would pad
        the economy count the gate reports."""
        responses = iter([
            ("the a b is 5, via c - inferred, not stored: ...", []),
            ("the a b is 5, via c - inferred, not stored: ...", []),
        ])
        calls = []

        def fake_pipeline(text, session):
            calls.append(text)
            if text == "yes":
                return "Consolidated", []
            return next(responses)

        ok, used, _resp = _climb(fake_pipeline, None, "what is the a b?")
        self.assertTrue(ok)
        self.assertEqual(used, 0)
        self.assertNotIn("yes", calls)


class LiveLadderTests(unittest.TestCase):
    """Depth-4 end to end on the real pipeline: climb, close, retract."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import build_session

        self.dir = Path(tempfile.mkdtemp(prefix="uq_ladderlive_"))
        self.session = build_session(self.dir, seed=0)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _world(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        for statement in ("the tower material is bronzite",
                          "the bronzite base is bronzoid",
                          "the bronzoid base is bronze",
                          "the bronze melting point is 913 degrees"):
            run_pipeline(statement, self.session)

    def test_the_climb_closes_depth_four(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        self._world()
        ok, used, response = _climb(run_pipeline, self.session,
                                    "what is the tower melting point?")
        self.assertTrue(ok)
        self.assertIn("913", response)
        self.assertEqual(used, 1, "depth-4 needs exactly one confirmation")

    def test_bottom_revision_takes_the_ladder_down(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        self._world()
        ok, _used, _resp = _climb(run_pipeline, self.session,
                                  "what is the tower melting point?")
        self.assertTrue(ok)
        run_pipeline("the bronze melting point is 1 degree", self.session)
        response, _trace = run_pipeline("what is the tower melting point?",
                                        self.session)
        self.assertFalse(
            response.startswith("tower melting point is 913"),
            f"stale ladder survived a bottom revision: {response!r}")


class GateVerdictTests(unittest.TestCase):
    """The narrow pass, the exact economy, and the corrections, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import ladder_gate

        doc = " ".join(ladder_gate.__doc__.split())
        self.assertIn("PASSED", doc)
        self.assertIn("1.21x seed sd", doc)

    def test_the_exact_economy_is_the_headline(self) -> None:
        from ultraquant.experiments import ladder_gate

        doc = " ".join(ladder_gate.__doc__.split())
        self.assertIn("1.80 confirmations per closed world", doc)
        self.assertIn("exactly minimal", doc)

    def test_both_protocol_corrections_are_recorded(self) -> None:
        """Depth-3 dilution and the padded yes: a gate that hides its own
        corrections is §11.11's defect wearing a verdict."""
        from ultraquant.experiments import ladder_gate

        doc = " ".join(ladder_gate.__doc__.split())
        self.assertIn("pure dilution", doc)
        self.assertIn("padding the economy count", doc)


if __name__ == "__main__":
    unittest.main()
