"""One call, and the bill: the on-demand retrieval engine.

§11.113 composed five retrieval mechanisms that already existed
separately into one call that reports what a question cost. These
pins hold the composition's contract — nothing dropped, the expensive
route ordered last, coverage reported rather than decided — and the
constant width that §11.104 measured into place.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.interpreter.thoughts import build_session, run_pipeline
from ultraquant.reason.retrieval import (DEFAULT_WIDTH, RetrievalEngine,
                                         Retrieved, _fold)


class _Counting:
    def __init__(self) -> None:
        self.calls = 0

    def suggest(self, question, memory):
        self.calls += 1
        return None


class EngineTests(unittest.TestCase):
    """The contract, on a world small enough to reason about."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_retrieval_"))
        self.session = build_session(self.dir, seed=0)
        for fact in ("the tower height is 300 meters",
                     "the tower material is steel",
                     "steel hardness is high",
                     "the bridge length is 2 kilometers"):
            run_pipeline(fact, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_held_fact_is_found_and_covered(self) -> None:
        engine = RetrievalEngine(self.session.memory)
        result = engine.retrieve("what is the tower height?")
        self.assertIn("tower height", result.keys)
        self.assertTrue(result.covered)

    def test_an_easy_question_stops_at_the_first_route(self) -> None:
        """The cheapest route answers, and nothing else is paid for."""
        engine = RetrievalEngine(self.session.memory)
        result = engine.retrieve("what is the tower height?")
        self.assertEqual(result.stopped_after, "lexical")
        self.assertNotIn("semantic", result.routes)

    def test_a_chain_question_surfaces_both_premises(self) -> None:
        """Neither key covers it; both are needed to derive it."""
        engine = RetrievalEngine(self.session.memory)
        result = engine.retrieve("what is the tower hardness?")
        self.assertIn("tower material", result.keys)
        self.assertIn("steel hardness", result.keys)
        self.assertFalse(result.covered,
                         "no single key covers a chain question")

    def test_the_semantic_route_is_never_paid_when_covered(self) -> None:
        counter = _Counting()
        engine = RetrievalEngine(self.session.memory, suggester=counter)
        engine.retrieve("what is the tower height?")
        self.assertEqual(counter.calls, 0)

    def test_the_semantic_route_runs_when_nothing_covered(self) -> None:
        counter = _Counting()
        engine = RetrievalEngine(self.session.memory, suggester=counter)
        engine.retrieve("what is the obelisk height?")
        self.assertEqual(counter.calls, 1)

    def test_absence_of_a_suggester_costs_nothing(self) -> None:
        """The optional tier stays optional."""
        with_one = RetrievalEngine(self.session.memory,
                                   suggester=_Counting())
        without = RetrievalEngine(self.session.memory)
        for question in ("what is the tower height?",
                         "what is the tower hardness?"):
            self.assertEqual(
                [k for k in without.retrieve(question).keys],
                [item.key for item in with_one.retrieve(question).facts
                 if item.route != "semantic"])

    def test_coverage_is_reported_not_decided(self) -> None:
        """The engine has no refusal machinery and must not seem to."""
        engine = RetrievalEngine(self.session.memory)
        result = engine.retrieve("what is the obelisk height?")
        self.assertFalse(result.covered)
        # It still returns what it found - deciding is the caller's job.
        self.assertIsInstance(result.facts, list)

    def test_an_empty_question_retrieves_nothing(self) -> None:
        engine = RetrievalEngine(self.session.memory)
        result = engine.retrieve("the of and")
        self.assertEqual(result.facts, [])
        self.assertFalse(result.covered)

    def test_every_fact_names_the_route_that_found_it(self) -> None:
        engine = RetrievalEngine(self.session.memory)
        for item in engine.retrieve("what is the tower hardness?").facts:
            self.assertIsInstance(item, Retrieved)
            self.assertIn(item.route, ("lexical", "reach", "semantic"))


class WidthTests(unittest.TestCase):
    """A constant, because an adaptive one was measured and lost."""

    def test_the_default_width_is_the_measured_constant(self) -> None:
        self.assertEqual(DEFAULT_WIDTH, 3)

    def test_the_reason_is_recorded_where_the_constant_is(self) -> None:
        from ultraquant.reason import retrieval

        note = " ".join(retrieval.__doc__.split())
        self.assertIn("lost to a plain constant three separate ways", note)


class GateVerdictTests(unittest.TestCase):
    """The pass, and its size."""

    def setUp(self) -> None:
        from ultraquant.experiments import retrieval_gate
        self.doc = " ".join(retrieval_gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("Nothing is lost",
                       "Cost scales with the question",
                       "The expensive route is not paid when it is not "
                       "needed",
                       "Coverage is reported, never decided",
                       "Absence costs nothing"):
            self.assertIn(phrase, self.doc)

    def test_the_effect_is_reported_as_small(self) -> None:
        self.assertIn("33% difference on a four-fact world", self.doc)


if __name__ == "__main__":
    unittest.main()
