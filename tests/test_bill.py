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


class SecondReviewRegressionTests(unittest.TestCase):
    """GPT-6 Astra's second adversarial review: work that went unbilled."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_bill_r2_"))
        self.memory = SystematicMemory(self.dir / "memory.json")
        self.memory.remember_fact("tower height", "300 meters")

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _executor_suggester(self, carried: bool):
        from concurrent.futures import ThreadPoolExecutor

        from ultraquant.memory.metering import carry

        class Pooled:
            def suggest(inner, question, memory):
                def work():
                    memory.find_facts(question, top_k=3)
                    memory.recall_fact("tower height")
                task = carry(work) if carried else work
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(task).result()
                return None

        return Pooled()

    def test_carried_executor_work_is_billed(self) -> None:
        engine = R.RetrievalEngine(self.memory,
                                   suggester=self._executor_suggester(True))
        found = engine.retrieve("how tall is the spire?",
                                routes=("semantic",))
        self.assertEqual(found.lookup_attempts, 1)
        self.assertEqual(found.index_probes, 1)
        self.assertEqual(found.unattributed_memory_calls, 0)

    def test_uncarried_executor_work_is_flagged_not_hidden(self) -> None:
        """It cannot be attributed - so the bill must say it is short."""
        engine = R.RetrievalEngine(self.memory,
                                   suggester=self._executor_suggester(False))
        found = engine.retrieve("how tall is the spire?",
                                routes=("semantic",))
        self.assertEqual(found.lookup_attempts, 0)
        self.assertEqual(found.unattributed_memory_calls, 2)

    def test_an_unmetered_memory_reports_unknown_not_zero(self) -> None:
        class Stub:
            def recall_fact(inner, key):
                return ({"value": "300 meters", "confidence": 0.6}
                        if key == "tower height" else None)

            def find_facts(inner, text, top_k=5):
                return ["tower height"]

        found = R.RetrievalEngine(Stub()).retrieve(
            "what is the tower height?")
        self.assertEqual(found.keys, ["tower height"])
        self.assertFalse(found.metered)
        self.assertIsNone(found.lookup_attempts)
        self.assertIsNone(found.examined)
        self.assertIsNone(found.index_probes)

    def test_systematic_memory_declares_itself_metered(self) -> None:
        found = R.RetrievalEngine(self.memory).retrieve(
            "what is the tower height?")
        self.assertTrue(found.metered)
        self.assertIsInstance(found.lookup_attempts, int)


class ThirdReviewRegressionTests(unittest.TestCase):
    """GPT-6 Astra's third review: bills that called themselves complete."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_bill_r3_"))
        self.memory = SystematicMemory(self.dir / "memory.json")
        self.memory.remember_fact("tower height", "300 meters")

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_synchronous_bill_is_complete(self) -> None:
        found = R.RetrievalEngine(self.memory).retrieve(
            "what is the tower height?")
        self.assertTrue(found.bill_complete)
        self.assertEqual(found.unattributed_memory_calls, 0)

    def test_a_workers_own_bill_does_not_hide_the_parents_gap(self) -> None:
        """An uncarried task that opens a bill of its own."""
        from concurrent.futures import ThreadPoolExecutor

        memory = self.memory

        class Delegating:
            def suggest(inner, question, mem):
                child = R.RetrievalEngine(memory)
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(child.retrieve,
                                "what is the tower height?").result()
                return None

        found = R.RetrievalEngine(self.memory, suggester=Delegating()) \
            .retrieve("how tall is the spire?", routes=("semantic",))
        self.assertFalse(found.bill_complete)
        self.assertGreater(found.unattributed_memory_calls, 0)

    def test_an_unmetered_nested_call_taints_the_enclosing_bill(self) -> None:
        class Stub:
            def recall_fact(inner, key):
                return None

            def find_facts(inner, text, top_k=5):
                return []

        class Nesting:
            def suggest(inner, question, mem):
                R.RetrievalEngine(Stub()).retrieve("what is the tower height?")
                return None

        found = R.RetrievalEngine(self.memory, suggester=Nesting()) \
            .retrieve("how tall is the spire?", routes=("semantic",))
        self.assertTrue(found.metered)
        self.assertFalse(found.bill_complete)

    def test_carried_work_still_running_makes_the_bill_incomplete(self) -> None:
        """The suggester stops waiting; the worker is still going."""
        import threading
        from concurrent.futures import ThreadPoolExecutor, TimeoutError

        from ultraquant.memory.metering import carry

        release = threading.Event()
        started = threading.Event()
        pool = ThreadPoolExecutor(max_workers=1)

        class Impatient:
            def suggest(inner, question, mem):
                def work():
                    started.set()
                    release.wait(timeout=5)
                    mem.recall_fact("tower height")
                future = pool.submit(carry(work))
                started.wait(timeout=5)
                try:
                    future.result(timeout=0.05)
                except TimeoutError:
                    pass
                return None

        try:
            found = R.RetrievalEngine(self.memory, suggester=Impatient()) \
                .retrieve("how tall is the spire?", routes=("semantic",))
        finally:
            release.set()
            pool.shutdown(wait=True)
        self.assertFalse(found.bill_complete)

    def test_unscoped_reads_cost_the_same_with_many_bills_open(self) -> None:
        """Was O(open bills) under one lock: 2.6 ms -> 664 ms at 1,000.

        The bills are held open on ANOTHER thread, so the reads here are
        outside every bill - the review's case. (Reads nested inside a
        thousand enclosing bills are charged to all of them, by policy.)
        """
        import threading
        import time
        from contextlib import ExitStack

        from ultraquant.memory.metering import Bill, metering

        def reads() -> float:
            start = time.perf_counter()
            for _ in range(5000):
                self.memory.recall_fact("no such key")
            return time.perf_counter() - start

        def hold(count: int, opened, release) -> None:
            with ExitStack() as stack:
                for _ in range(count):
                    stack.enter_context(metering(Bill()))
                opened.set()
                release.wait(timeout=30)

        timings = {}
        for count in (1, 1000):
            opened, release = threading.Event(), threading.Event()
            holder = threading.Thread(target=hold,
                                      args=(count, opened, release))
            holder.start()
            try:
                opened.wait(timeout=30)
                timings[count] = min(reads(), reads())
            finally:
                release.set()
                holder.join(timeout=30)
        self.assertLess(timings[1000], timings[1] * 5 + 0.02)
