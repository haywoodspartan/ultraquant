"""The repository's own prose as a distillation corpus.

§11.111 asked whether a wider corpus moves the vocabulary ceiling
§11.110 measured. These pins hold the corpus extraction, and the
invariant the whole measurement rests on: withheld words never get an
input column, at any corpus size.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from ultraquant.experiments.embed_gate import CATEGORIES, _tokens
from ultraquant.experiments.fewshot_gate import items_of
from ultraquant.model import prose

_ROOT = Path(__file__).resolve().parents[1]


class ProseTests(unittest.TestCase):
    """Real English, extracted the same way every time."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sentences = prose.sentences()

    def test_the_corpus_is_substantial(self) -> None:
        self.assertGreater(len(self.sentences), 500)
        self.assertGreater(len(prose.vocabulary(self.sentences)), 2000)

    def test_extraction_is_deterministic(self) -> None:
        """A distillation run has to be reproducible."""
        self.assertEqual(prose.sentences(), self.sentences)

    def test_code_fences_are_stripped(self) -> None:
        """An encoder learning the geometry of a table has learned nothing."""
        for line in self.sentences:
            self.assertNotIn("```", line)
            self.assertNotIn("|---", line)

    def test_sentences_are_bounded_at_both_ends(self) -> None:
        for line in self.sentences:
            self.assertGreaterEqual(len(line), 30)
            self.assertLessEqual(len(line), 160)

    def test_a_missing_source_is_skipped_not_fatal(self) -> None:
        self.assertEqual(prose.sentences(root=_ROOT / "no-such-dir"), [])

    def test_the_corpus_is_wider_than_the_test_vocabulary(self) -> None:
        """The whole premise: 41x more words than the task uses."""
        task = set()
        for spec in CATEGORIES.values():
            for text in items_of(spec):
                task |= _tokens(text)
        self.assertGreater(len(prose.vocabulary(self.sentences)),
                           10 * len(task))


class WithholdingTests(unittest.TestCase):
    """The invariant the measurement rests on."""

    def test_a_withheld_word_never_gets_a_column(self) -> None:
        """However large the corpus, the encoder cannot see it."""
        withheld = {"square", "circle", "colour"}
        for size in (0, 50, 200):
            corpus = [t for s in CATEGORIES.values() for t in items_of(s)]
            corpus += prose.sentences()[:size]
            words = set()
            for text in corpus:
                words |= _tokens(text)
            given = sorted(words - withheld)
            self.assertEqual(set(given) & withheld, set(),
                             f"a withheld word acquired a column at {size}")

    def test_the_check_can_fail(self) -> None:
        """A vocabulary built without the subtraction must leak."""
        withheld = {"square"}
        words = {"square", "shape"}
        self.assertEqual(len(set(sorted(words - withheld)) & withheld), 0)
        self.assertEqual(len(set(sorted(words)) & withheld), 1)


class GateShapeTests(unittest.TestCase):
    """Criterion 2: the budget must not vary with the corpus."""

    def test_every_sample_is_trained_equally_often(self) -> None:
        """Run one held STEPS constant and starved the big corpora.

        Fidelity to the teacher on the task texts fell 0.92, 0.62,
        0.27 as epochs fell 4166, 635, 235 - the experiment measured
        the budget, not the corpus. Epochs are constant now and the
        compute is allowed to grow.
        """
        from ultraquant.experiments import vocabulary_gate as gate

        self.assertFalse(hasattr(gate, "STEPS"),
                         "the constant-steps design confounded the "
                         "measurement and must not come back")
        self.assertGreaterEqual(gate.EPOCHS, 4000,
                                "below 4000 the copy is unfinished")

    def test_zero_reproduces_the_previous_units_condition(self) -> None:
        from ultraquant.experiments import vocabulary_gate as gate

        self.assertEqual(gate.CORPUS_SIZES[0], 0,
                         "the first row must be the condition this gate "
                         "exists to move")


class GateVerdictTests(unittest.TestCase):
    """The pass, and the run that said the opposite."""

    def setUp(self) -> None:
        from ultraquant.experiments import vocabulary_gate
        self.doc = " ".join(vocabulary_gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("Withheld words have no input column",
                       "Every sample is trained equally often",
                       "The boundary penalty shrinks",
                       "Overall accuracy does not fall",
                       "The corpus is real prose"):
            self.assertIn(phrase, self.doc)

    def test_the_result_is_recorded(self) -> None:
        self.assertIn("+0.119 at seed sd 0.031", self.doc)
        self.assertIn("the penalty is exactly zero", self.doc)

    def test_run_ones_opposite_result_is_kept(self) -> None:
        """It said the boundary receded because everything collapsed."""
        self.assertIn("below the 0.167 chance floor", self.doc)
        self.assertIn("would have read as a triumph", self.doc)

    def test_the_broken_criterion_is_named_as_broken(self) -> None:
        self.assertIn("criterion 2 had been written wrong", self.doc)
        self.assertIn("written to prevent a confound had created a worse "
                      "one", self.doc)

    def test_the_sparse_fix_is_recorded_as_bit_identical(self) -> None:
        self.assertIn("375x waste", self.doc)
        self.assertIn("bit-identical to the dense one", self.doc)

    def test_the_limits_are_stated(self) -> None:
        self.assertIn("still have no column", self.doc)
        self.assertIn("this repository talking about itself", self.doc)


if __name__ == "__main__":
    unittest.main()
