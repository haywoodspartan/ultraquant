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
