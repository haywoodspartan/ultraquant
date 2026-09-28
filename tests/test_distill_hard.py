"""The filter on harder facts.

11.134: US-state, Canadian and Australian capitals, heavy-element
numbers, rare symbols, and less famous authors - certain answers the
teachers were less likely to share. These pins hold the benchmark's
shape and the verdict, recomputed from the 3,765 samples on record and
their manifest; nothing here starts a model.
"""

from __future__ import annotations

import unittest

from ultraquant.experiments import distill_hard_gate as G
from ultraquant.experiments import knowledge_bench as K
from ultraquant.experiments import knowledge_bench_hard as H


class BenchmarkTests(unittest.TestCase):

    def test_its_shape(self) -> None:
        self.assertEqual((len(H.KNOWN), len(H.FICTITIOUS)), (211, 40))
        keys = [(i.category, i.subject) for i in H.KNOWN + H.FICTITIOUS]
        self.assertEqual(len(keys), len(set(keys)))

    def test_states_are_asked_unambiguously(self) -> None:
        washington = next(i for i in H.KNOWN if "Washington" in i.subject)
        self.assertEqual(washington.question,
                         "What is the capital of the US state of Washington?")
        self.assertTrue(K.is_correct(washington, "Olympia"))


class RecordedRunTests(unittest.TestCase):

    def test_the_recorded_run_passes(self) -> None:
        report = G.run_gate(elicit=False)
        self.assertTrue(report.passes, report.reason)
        self.assertEqual(report.records, 3765)
        held = report.score["held_out"]
        self.assertEqual((held["right"], held["promoted"]), (101, 101))
        self.assertEqual(report.score["invented_promoted"],
                         [("symbol", "varnadium", "v")])

    def test_the_verdict_is_recorded(self) -> None:
        doc = " ".join(G.__doc__.split())
        for phrase in ("PASSED", "101 of 101", "Varnadium",
                       "The plain reading"):
            self.assertIn(phrase, doc)


if __name__ == "__main__":
    unittest.main()
