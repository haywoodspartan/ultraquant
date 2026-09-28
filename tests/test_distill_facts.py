"""Facts distilled from local teachers, and what they are worth.

11.130: three local teachers from three families, filtered by a rule
fixed before any run, scored against answers that are known and
subjects that were invented. These pins hold the filter's pieces, the
benchmark's scoring, the Wilson bound, and the verdict itself,
recomputed from the 3,900 raw samples on record. Nothing here starts a
model: the recorded run is re-scored, not re-run.
"""

from __future__ import annotations

import math
import unittest

from ultraquant.distill import elicit as E
from ultraquant.experiments import distill_facts_gate as G
from ultraquant.experiments import knowledge_bench as K


class ExtractTests(unittest.TestCase):

    def test_reasoning_markers_and_markdown_come_off(self) -> None:
        for raw, want in (("H<|END_OF_TURN_TOKEN|>", "H"),
                          ("<think>gold is Au</think>\nAu", "Au"),
                          ("<think>still thinking", ""),
                          ("**Answer:** \"Paris.\"", "Paris"),
                          ("\n\n  Canberra.\nIt is in the ACT.", "Canberra")):
            with self.subTest(raw=raw):
                self.assertEqual(E.extract(raw), want)

    def test_normalize_folds_what_does_not_matter(self) -> None:
        self.assertEqual(E.normalize("Brasília"), "brasilia")
        self.assertEqual(E.normalize("The Hague"), "hague")
        self.assertEqual(E.normalize("F. Scott Fitzgerald"), "f scott fitzgerald")


class AbstentionTests(unittest.TestCase):

    def test_a_refusal_anywhere_in_a_reply_is_no_position(self) -> None:
        for raw in ("UNKNOWN", "I don't know.", "", "None",
                    "Veltra\n(This country is fictional.)",
                    "There is no such element."):
            with self.subTest(raw=raw):
                self.assertTrue(E.is_abstention(raw))
        for raw in ("Paris", "Unknownium", "Au", "Leo Tolstoy"):
            with self.subTest(raw=raw):
                self.assertFalse(E.is_abstention(raw))

    def test_held_judges_whole_replies(self) -> None:
        """Claude's review: held() once saw only each reply's first line."""
        self.assertIsNone(E.held(["Veltra\n(This country is fictional.)"] * 5))
        self.assertEqual(E.held(["Paris"] * 3 + ["UNKNOWN"] * 2), "paris")
        self.assertIsNone(E.held(["Paris", "Paris", "Lyon", "Nice", "UNKNOWN"]))


class FilterTests(unittest.TestCase):

    def test_agreement_is_equality_or_a_matching_ending(self) -> None:
        self.assertTrue(E.agree("tolstoy", "leo tolstoy"))
        self.assertTrue(E.agree("paris", "paris"))
        self.assertFalse(E.agree("leo tolstoy", "lev tolstoy"))
        self.assertFalse(E.agree("paris", "lyon"))

    def test_promotion_needs_two_lineages_and_no_dissent(self) -> None:
        self.assertEqual(E.promote({"a": "paris", "b": "paris", "c": None}),
                         "paris")
        self.assertIsNone(E.promote({"a": "paris", "b": "paris", "c": "lyon"}))
        self.assertIsNone(E.promote({"a": "paris", "b": None, "c": None}))
        self.assertEqual(E.promote({"a": "tolstoy", "b": "leo tolstoy"}),
                         "leo tolstoy")

    def test_decide_counts_lineages_not_teachers(self) -> None:
        item = K.Item("capital", "France", "What is the capital of France?",
                      ("paris",))

        def samples(teacher, lineage, answer):
            return [E.Record(teacher, lineage, "x.gguf", 1,
                             E.question_id(item), item.question, seed,
                             answer, E.normalize(E.extract(answer)))
                    for seed in E.SEEDS]

        one_family = samples("t1", "L0", "Paris") + samples("t2", "L0", "Paris")
        self.assertIsNone(E.decide(one_family, [item])[E.question_id(item)])
        two_families = one_family + samples("t3", "L1", "Paris")
        self.assertEqual(E.decide(two_families, [item])[E.question_id(item)],
                         "paris")


class BenchmarkTests(unittest.TestCase):

    def test_its_shape(self) -> None:
        self.assertEqual((len(K.KNOWN), len(K.FICTITIOUS)), (180, 80))
        self.assertEqual((len(K.split(K.KNOWN, 0)), len(K.split(K.KNOWN, 1))),
                         (90, 90))

    def test_scoring(self) -> None:
        swift = next(i for i in K.KNOWN if i.subject == "Gulliver's Travels")
        self.assertTrue(K.is_correct(swift, "Jonathan Swift"))
        self.assertFalse(K.is_correct(swift, "Taylor Swift"))
        gold = next(i for i in K.KNOWN
                    if i.subject == "gold" and i.category == "symbol")
        self.assertTrue(K.is_correct(gold, "The symbol is Au."))
        self.assertFalse(any(K.is_abstention(a) for i in K.KNOWN
                             for a in i.answers))

    def test_the_wilson_bound(self) -> None:
        self.assertAlmostEqual(G.wilson_lower(90, 90), 0.959, places=3)
        self.assertEqual(G.wilson_lower(0, 0), 0.0)
        self.assertLess(G.wilson_lower(85, 90), G.wilson_lower(90, 90))


class ReviewFourTests(unittest.TestCase):
    """11.133: Astra's counterexamples, each one a regression pin."""

    def test_every_counterexample_is_refused(self) -> None:
        cases = G.review4_cases()
        self.assertTrue(all(cases.values()), cases)

    def test_the_replay_needs_no_model_files(self) -> None:
        self.assertTrue(G.replay_without_models())

    def test_a_family_in_conflict_vetoes(self) -> None:
        self.assertIs(E.family_position({"t1": "paris", "t2": "lyon"}),
                      E.CONFLICT)
        self.assertIsNone(E.promote({"a": "lyon", "b": "lyon",
                                     "c": E.CONFLICT}))


class RecordedRunTests(unittest.TestCase):
    """The verdict, recomputed from the samples on record."""

    def test_the_recorded_run_passes(self) -> None:
        if not all(path.exists() for _name, path in G.TEACHERS):
            self.skipTest("the teachers' GGUFs are not on this machine")
        report = G.run_gate(elicit=False)
        self.assertTrue(report.passes, report.reason)
        self.assertEqual(report.records, 3900)
        self.assertEqual(report.score["held_out"]["right"], 90)
        self.assertEqual(report.score["invented_promoted"], [])
        self.assertTrue(math.isclose(report.score["confidence"], 0.959,
                                     abs_tol=0.001))
        self.assertTrue(all(report.planted.values()))

    def test_the_verdict_is_recorded(self) -> None:
        doc = " ".join(G.__doc__.split())
        for phrase in ("Precise on the held-out half", "Silent about what "
                       "does not exist", "The confidence is honest",
                       "PASSED", "What this does not show"):
            self.assertIn(phrase, doc)


if __name__ == "__main__":
    unittest.main()
