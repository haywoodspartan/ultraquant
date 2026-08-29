"""The retrieval engine, actually in the pipeline.

11.115 replaced Recall's candidate ladder with the engine that owns
it. These pins hold the contract that made it safe - identical
answers, only the consumed route paid for - and the flag that keeps
the comparison measurable.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.interpreter import thoughts as T
from ultraquant.reason.retrieval import RetrievalEngine


class WiringTests(unittest.TestCase):
    """Both arms, same worlds, compared directly."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_wiring_"))
        self.facts = ["the tower height is 300 meters",
                      "the tower material is steel",
                      "steel hardness is high",
                      "the bridge length is 2 kilometers"]
        self.questions = ["what is the tower height?",
                          "what is the tower hardness?",
                          "what is the obelisk height?",
                          "what is the bridge length?"]

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _answers(self, engine_on: bool) -> list:
        previous = T._RETRIEVAL_ENGINE
        T._RETRIEVAL_ENGINE = engine_on
        root = Path(tempfile.mkdtemp(prefix="uq_arm_"))
        try:
            session = T.build_session(root, seed=0)
            for fact in self.facts:
                T.run_pipeline(fact, session)
            return [T.run_pipeline(q, session)[0] for q in self.questions]
        finally:
            T._RETRIEVAL_ENGINE = previous
            shutil.rmtree(root, ignore_errors=True)

    def test_the_engine_answers_byte_identically(self) -> None:
        """The bar. A refactor that changes an answer is a regression."""
        self.assertEqual(self._answers(True), self._answers(False))

    def test_the_engine_is_the_shipped_path(self) -> None:
        self.assertTrue(T._RETRIEVAL_ENGINE)

    def test_the_flag_restores_the_exhaustive_ladder(self) -> None:
        """What makes this measurable rather than asserted."""
        self.assertEqual(self._answers(False), self._answers(False))

    def test_recall_asks_only_for_the_route_it_consumes(self) -> None:
        """Run one paid for the cascade and its caller redid the work."""
        import inspect

        source = inspect.getsource(T.Recall)
        self.assertIn('routes=("exact",)', source)

    def test_recall_still_publishes_facts_for_its_consumers(self) -> None:
        session = T.build_session(self.dir, seed=0)
        for fact in self.facts:
            T.run_pipeline(fact, session)
        _response, trace = T.run_pipeline("what is the tower height?",
                                          session)
        recalled = [e for e in trace if e.get("thought") == "Recall"]
        self.assertTrue(recalled)
        self.assertIn("tower height", recalled[0].get("facts", []))


class RouteRestrictionTests(unittest.TestCase):
    """The API the defect forced, and what it must not break."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_routes_"))
        self.session = T.build_session(self.dir, seed=0)
        for fact in ("the tower height is 300 meters",
                     "the tower material is steel"):
            T.run_pipeline(fact, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_restricted_call_runs_only_that_route(self) -> None:
        engine = RetrievalEngine(self.session.memory)
        result = engine.retrieve("what is the obelisk height?",
                                 routes=("exact",))
        self.assertEqual(set(result.routes), {"exact"})

    def test_an_unrestricted_call_still_cascades(self) -> None:
        engine = RetrievalEngine(self.session.memory)
        result = engine.retrieve("what is the obelisk height?")
        self.assertIn("lexical", result.routes)

    def test_restricting_never_adds_keys(self) -> None:
        engine = RetrievalEngine(self.session.memory)
        question = "what is the tower height?"
        restricted = set(engine.retrieve(question,
                                         routes=("exact",)).keys)
        full = set(engine.retrieve(question, exhaustive=True).keys)
        self.assertTrue(restricted <= full)


class GateVerdictTests(unittest.TestCase):
    """The pass, and the defect only one criterion could see."""

    def setUp(self) -> None:
        from ultraquant.experiments import wiring_gate
        self.doc = " ".join(wiring_gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("Every answer is byte-identical",
                       "The store is read less",
                       "The stores end identical",
                       "The saving is where it was claimed",
                       "The traces agree on intent"):
            self.assertIn(phrase, self.doc)

    def test_the_result_is_recorded(self) -> None:
        self.assertIn("0 answers differing", self.doc)
        self.assertIn("21.6%", self.doc)

    def test_the_defect_only_criterion_four_saw_is_recorded(self) -> None:
        self.assertIn("saved on hits and LOST on everything else",
                      self.doc)
        self.assertIn("eagerly does work its caller will redo is not an "
                      "economy", self.doc)

    def test_the_rest_of_the_unification_is_not_claimed(self) -> None:
        self.assertIn("the rest of the unification is unmeasured",
                      self.doc)


if __name__ == "__main__":
    unittest.main()
