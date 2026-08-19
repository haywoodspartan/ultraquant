"""Consolidation and truth maintenance: promote on confirmation, retract on
revision, and never trust a flush that skips an emptied bucket.

The episodic-to-semantic mechanism §11.30 registered: a derivation the
user confirms becomes a stored, provenance-carrying fact — and a premise
revision retracts every conclusion resting on it, recursively. The
staleness control caught a real resurrection before the gate existed
(factshards.flush skipped emptied buckets); that regression is pinned
here beside the behavior it protects.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.memory.systematic import SystematicMemory


class MemoryConsolidationTests(unittest.TestCase):
    """The memory layer: provenance in, retraction out."""

    def test_a_consolidated_fact_carries_its_premises(self) -> None:
        memory = SystematicMemory()
        memory.consolidate_fact("tower melting point", "1370 degrees", 0.6,
                                [("tower material", "steel"),
                                 ("steel melting point", "1370 degrees")])
        record = memory.recall_fact("tower melting point")
        self.assertEqual(record["value"], "1370 degrees")
        self.assertIn(["tower material", "steel"], record["derived_from"])

    def test_revising_a_premise_retracts_the_derivative(self) -> None:
        memory = SystematicMemory()
        memory.remember_fact("tower material", "steel", confidence=0.6)
        memory.consolidate_fact("tower melting point", "1370 degrees", 0.6,
                                [("tower material", "steel")])
        memory.remember_fact("tower material", "oak", confidence=0.6)
        self.assertIsNone(memory.recall_fact("tower melting point"))

    def test_retraction_is_recursive(self) -> None:
        """A conclusion built on a conclusion falls with its ancestor."""
        memory = SystematicMemory()
        memory.remember_fact("a", "b", confidence=0.6)
        memory.consolidate_fact("derived one", "x", 0.6, [("a", "b")])
        memory.consolidate_fact("derived two", "y", 0.6,
                                [("derived one", "x")])
        memory.remember_fact("a", "c", confidence=0.6)
        self.assertIsNone(memory.recall_fact("derived one"))
        self.assertIsNone(memory.recall_fact("derived two"))

    def test_retraction_is_logged_as_an_episode(self) -> None:
        memory = SystematicMemory()
        memory.remember_fact("a", "b", confidence=0.6)
        memory.consolidate_fact("derived", "x", 0.6, [("a", "b")])
        memory.remember_fact("a", "c", confidence=0.6)
        kinds = [ep["kind"] for ep in memory.recall_episodes(limit=5)]
        self.assertIn("retraction", kinds)

    def test_reinforcing_a_premise_does_not_retract(self) -> None:
        """Same value again is reinforcement, not revision - conclusions
        rest on the value, and the value did not change."""
        memory = SystematicMemory()
        memory.remember_fact("a", "b", confidence=0.6)
        memory.consolidate_fact("derived", "x", 0.6, [("a", "b")])
        memory.remember_fact("a", "b", confidence=0.6)
        self.assertIsNotNone(memory.recall_fact("derived"))


class EmptiedBucketFlushTests(unittest.TestCase):
    """The resurrection regression: an emptied bucket must persist empty."""

    def test_deleting_a_buckets_last_fact_survives_the_flush(self) -> None:
        """factshards.flush skipped empty staged buckets, so the vault kept
        the old record, the cache served it, and recall-reinforcement
        re-persisted the ghost. The truth-maintenance control caught it
        live: a retracted consolidated fact answered again one turn later.
        """
        from ultraquant.interpreter.thoughts import build_session

        root = Path(tempfile.mkdtemp(prefix="uq_flushbug_"))
        try:
            session = build_session(root, seed=0)
            memory = session.memory
            self.assertIsNotNone(memory.shards,
                                 "this regression lives in the shard store")
            memory.remember_fact("lonely fact", "42 things", confidence=0.6)
            memory.shards.flush()
            self.assertIsNotNone(memory.recall_fact("lonely fact"))
            memory.shards.delete("lonely fact")
            memory.shards.flush()
            self.assertIsNone(memory.recall_fact("lonely fact"),
                              "the flush resurrected a deleted fact")
        finally:
            shutil.rmtree(root, ignore_errors=True)


class PipelineConsolidationTests(unittest.TestCase):
    """The dialogue surface: affirmation promotes, nothing else does."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import build_session

        self.dir = Path(tempfile.mkdtemp(prefix="uq_consolpipe_"))
        self.session = build_session(self.dir, seed=0)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _teach_chain(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        run_pipeline("the tower material is steel", self.session)
        run_pipeline("the steel melting point is 1370 degrees", self.session)

    def test_yes_after_a_derivation_consolidates(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        self._teach_chain()
        run_pipeline("what is the tower melting point?", self.session)
        response, _trace = run_pipeline("yes", self.session)
        self.assertIn("Consolidated", response)
        record = self.session.memory.recall_fact("tower melting point")
        self.assertIsNotNone(record)
        self.assertIn("derived_from", record)

    def test_pending_survives_exactly_one_turn(self) -> None:
        """An affirmation after an intervening turn confirms nothing -
        the user is agreeing with something else by then."""
        from ultraquant.interpreter.thoughts import run_pipeline

        self._teach_chain()
        run_pipeline("what is the tower melting point?", self.session)
        run_pipeline("the bridge length is 2 kilometers", self.session)
        response, _trace = run_pipeline("yes", self.session)
        self.assertNotIn("Consolidated", response)
        record = self.session.memory.recall_fact("tower melting point")
        self.assertIsNone(record)

    def test_a_bare_yes_with_nothing_pending_is_not_an_error(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        response, _trace = run_pipeline("yes", self.session)
        self.assertTrue(response.strip())
        self.assertNotIn("Consolidated", response)

    def test_repetition_never_consolidates(self) -> None:
        """§11.16's trap, pinned at the surface: asking twice is not
        confirmation, and nothing may be stored by it."""
        from ultraquant.interpreter.thoughts import run_pipeline

        self._teach_chain()
        run_pipeline("what is the tower melting point?", self.session)
        run_pipeline("what is the tower melting point?", self.session)
        record = self.session.memory.recall_fact("tower melting point")
        self.assertIsNone(record)

    def test_revision_retracts_and_the_answer_demotes(self) -> None:
        """The full cycle: derive, confirm, recall, revise - and the old
        conclusion must stop being asserted the moment its premise goes."""
        from ultraquant.interpreter.thoughts import run_pipeline

        self._teach_chain()
        run_pipeline("what is the tower melting point?", self.session)
        run_pipeline("yes", self.session)
        run_pipeline("the tower material is oak", self.session)
        response, _trace = run_pipeline("what is the tower melting point?",
                                        self.session)
        self.assertFalse(response.startswith("tower melting point is"),
                         f"stale assertion survived revision: {response!r}")


class ControlCanFailTests(unittest.TestCase):
    """§11.11's discipline: the staleness control must be able to fail."""

    def test_disabling_retraction_produces_the_stale_answer(self) -> None:
        from ultraquant.interpreter.thoughts import build_session, run_pipeline
        from ultraquant.memory.systematic import SystematicMemory

        root = Path(tempfile.mkdtemp(prefix="uq_consolsab_"))
        real = SystematicMemory._retract_derivatives
        try:
            SystematicMemory._retract_derivatives = lambda self, key: []
            session = build_session(root, seed=0)
            run_pipeline("the tower material is steel", session)
            run_pipeline("the steel melting point is 1370 degrees", session)
            run_pipeline("what is the tower melting point?", session)
            run_pipeline("yes", session)
            run_pipeline("the tower material is oak", session)
            response, _trace = run_pipeline(
                "what is the tower melting point?", session)
            self.assertIn("1370", response)
            self.assertTrue(
                response.startswith("tower melting point is"),
                "without retraction the stale answer must assert - if it "
                "does not, the staleness control measures nothing")
        finally:
            SystematicMemory._retract_derivatives = real
            shutil.rmtree(root, ignore_errors=True)


class GateVerdictTests(unittest.TestCase):
    """The narrow pass and its two forced corrections, pinned."""

    def test_the_recorded_verdict_is_a_narrow_pass(self) -> None:
        from ultraquant.experiments import consolidation_gate

        doc = " ".join(consolidation_gate.__doc__.split())
        self.assertIn("PASSED — narrowly", doc.replace("PASSED - narrowly",
                                                       "PASSED — narrowly"))
        self.assertIn("1.21x seed sd", doc)

    def test_one_confirmation_buys_one_step_is_recorded(self) -> None:
        from ultraquant.experiments import consolidation_gate

        doc = " ".join(consolidation_gate.__doc__.split())
        self.assertIn("exactly one step of reach", doc)

    def test_the_claiming_vs_mentioning_correction_is_recorded(self) -> None:
        from ultraquant.experiments import consolidation_gate

        doc = " ".join(consolidation_gate.__doc__.split())
        self.assertIn("claiming from mentioning", doc)


if __name__ == "__main__":
    unittest.main()
