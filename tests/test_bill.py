"""The bill describes the work.

11.122: RetrievalEngine reported `examined` as the keys it KEPT, so a
retrieval that read fifteen keys and found nothing reported zero - in
the one engine whose stated purpose is to say what a retrieval cost.
This time the roles were reversed: GPT-6 Astra wrote and froze the exam
(experiments/bill_gate.py) before the fix existed, and Claude wrote the
fix. These pins hold the counters, the arm, and the exam as frozen.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.memory.systematic import SystematicMemory
from ultraquant.reason import retrieval as R

#: sha256 of experiments/bill_gate.py as GPT-6 Astra froze it, before the
#: fix existed - computed over LF line endings so a checkout's CRLF
#: conversion cannot change it.
_EXAM_SHA256 = ("2ddb73209845c15baa7b4b2b7b23cedc"
                "31ff0bdf7148e5d6de47da45fdb4ac32")


class _Counted:
    """The memory, with every real call counted."""

    def __init__(self, memory) -> None:
        self.memory = memory
        self.lookups = self.index = 0

    def recall_fact(self, *args, **kwargs):
        self.lookups += 1
        return self.memory.recall_fact(*args, **kwargs)

    def find_facts(self, *args, **kwargs):
        self.index += 1
        return self.memory.find_facts(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.memory, name)


class HonestBillTests(unittest.TestCase):

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_bill_t_"))
        self.memory = SystematicMemory(self.dir / "memory.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _retrieve(self, question: str, honest: bool = True, **kw):
        previous = R._HONEST_BILL
        R._HONEST_BILL = honest
        counted = _Counted(self.memory)
        try:
            found = R.RetrievalEngine(counted).retrieve(question, **kw)
        finally:
            R._HONEST_BILL = previous
        return found, counted

    def test_a_miss_reports_the_lookups_it_made(self) -> None:
        """The defect: fifteen reads, reported as zero."""
        found, counted = self._retrieve("what is the tower height?",
                                        routes=("exact",))
        self.assertGreater(counted.lookups, 0)
        self.assertEqual(found.lookup_attempts, counted.lookups)
        self.assertEqual(found.examined, counted.lookups)

    def test_every_counter_matches_the_real_calls(self) -> None:
        self.memory.remember_fact("tower height", "300 meters")
        for question in ("what is the tower height?",
                         "what is the obelisk colour?"):
            with self.subTest(question=question):
                found, counted = self._retrieve(question)
                self.assertEqual(found.lookup_attempts, counted.lookups)
                self.assertEqual(found.index_probes, counted.index)
                self.assertEqual(found.unique_facts_returned,
                                 len(set(found.keys)))

    def test_the_bill_is_per_call_not_cumulative(self) -> None:
        engine = R.RetrievalEngine(self.memory)
        first = engine.retrieve("what is the tower height?")
        second = engine.retrieve("what is the tower height?")
        self.assertEqual(first.lookup_attempts, second.lookup_attempts)

    def test_the_answer_does_not_change(self) -> None:
        self.memory.remember_fact("tower height", "300 meters")
        new, _ = self._retrieve("what is the tower height?", honest=True)
        old, _ = self._retrieve("what is the tower height?", honest=False)
        self.assertEqual((new.keys, new.routes, new.covered,
                          new.stopped_after),
                         (old.keys, old.routes, old.covered,
                          old.stopped_after))

    def test_the_flag_restores_the_old_count(self) -> None:
        found, counted = self._retrieve("what is the tower height?",
                                        honest=False, routes=("exact",))
        self.assertEqual(found.examined, 0)
        self.assertGreater(counted.lookups, 0)

    def test_the_memory_is_restored_after_the_call(self) -> None:
        engine = R.RetrievalEngine(self.memory)
        engine.retrieve("what is the tower height?")
        self.assertIs(engine.memory, self.memory)

    def test_the_fix_is_the_shipped_path(self) -> None:
        self.assertTrue(R._HONEST_BILL)


class ExamTests(unittest.TestCase):
    """The exam belongs to the examiner."""

    def test_the_exam_is_the_one_frozen_before_the_fix(self) -> None:
        path = (Path(__file__).resolve().parents[1] / "ultraquant"
                / "experiments" / "bill_gate.py")
        text = path.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(hashlib.sha256(text).hexdigest(), _EXAM_SHA256)

    def test_the_exam_passes(self) -> None:
        from ultraquant.experiments import bill_gate
        report = bill_gate.run_gate()
        self.assertTrue(report.valid)
        self.assertTrue(report.passes)


if __name__ == "__main__":
    unittest.main()


class ReviewRegressionTests(unittest.TestCase):
    """GPT-6 Astra's adversarial review of the first fix, as tests.

    Each reproduces a defect the frozen exam's battery could not see:
    the exam passed, and the first implementation still shipped these.
    """

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_bill_r_"))
        self.memory = SystematicMemory(self.dir / "memory.json")
        self.memory.remember_fact("tower height", "300 meters")

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_overlapping_calls_keep_separate_bills(self) -> None:
        """[high] Two threads on one engine: each bill is its own."""
        import threading

        engine = R.RetrievalEngine(self.memory)
        gate = threading.Barrier(2)
        original = self.memory.find_facts
        bills = {}

        def slowed(*args, **kwargs):
            gate.wait(timeout=5)      # force the two calls to overlap
            return original(*args, **kwargs)

        self.memory.find_facts = slowed
        try:
            def run(name, routes):
                bills[name] = engine.retrieve("what is the obelisk colour?",
                                              routes=routes)
            threads = [threading.Thread(target=run, args=("a", ("lexical",))),
                       threading.Thread(target=run, args=("b", ("lexical",)))]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
        finally:
            del self.memory.find_facts
        self.assertEqual(bills["a"].index_probes, 1)
        self.assertEqual(bills["b"].index_probes, 1)
        self.assertIs(engine.memory, self.memory)

    def test_the_suggester_sees_the_real_memory(self) -> None:
        """[medium] A type-checking suggester must not lose its answer."""

        class Strict:
            def suggest(inner, question, memory):
                assert isinstance(memory, SystematicMemory), type(memory)
                from ultraquant.reason.semantic import Suggestion
                return Suggestion(key="tower height", value="300 meters",
                                  confidence=0.6, similarity=0.9)

        found = R.RetrievalEngine(self.memory, suggester=Strict()).retrieve(
            "how tall is the spire?", routes=("semantic",))
        self.assertEqual(found.keys, ["tower height"])

    def test_a_direct_semantic_query_is_not_a_phrase_probe(self) -> None:
        """[medium] Only _reach and _reachable_facts make phrase probes."""

        class Direct:
            def suggest(inner, question, memory):
                memory.find_facts(question, top_k=3)
                return None

        found = R.RetrievalEngine(self.memory, suggester=Direct()).retrieve(
            "how tall is the spire?", routes=("semantic",))
        self.assertEqual(found.index_probes, 1)
        self.assertEqual(found.phrase_probes, 0)
        self.assertEqual(found.semantic_calls, 1)

    def test_nested_work_is_billed_to_every_enclosing_call(self) -> None:
        """[medium] One policy for every counter, semantic calls too."""
        engine = R.RetrievalEngine(self.memory)
        seen = {"depth": 0}

        class Recursive:
            def suggest(inner, question, memory):
                if seen["depth"] == 0:
                    seen["depth"] += 1
                    engine.retrieve("what is the obelisk colour?",
                                    routes=("semantic",))
                return None

        engine.suggester = Recursive()
        outer = engine.retrieve("how tall is the spire?",
                                routes=("semantic",))
        self.assertEqual(outer.semantic_calls, 2)
