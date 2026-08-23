"""Stage 1's criterion, met with a borrowed representation.

§11.109 ran Stage 1's gate verbatim in the domain §11.5 named, and it
passed. These pins hold the invariants the criteria caught defects
with — the bag arms being literally the same computation, five
examples meaning five — and the exact scope of what passed.
"""

from __future__ import annotations

import unittest

from ultraquant.experiments import fewshot_gate as gate
from ultraquant.experiments.embed_gate import CATEGORIES
from ultraquant.experiments.permutation_gate import (build_permutation,
                                                     permute)


class CorpusTests(unittest.TestCase):
    """Five examples means five, from a corpus nobody wrote for this."""

    def test_every_category_has_exactly_six_texts(self) -> None:
        """Six is what makes 5-train/1-test leave-one-out exact."""
        for name, spec in CATEGORIES.items():
            self.assertEqual(len(gate.items_of(spec)), 6, name)

    def test_the_corpus_predates_this_gate(self) -> None:
        """Nothing was written for the measurement it is measured by."""
        import inspect

        source = inspect.getsource(gate)
        self.assertNotIn("CATEGORIES = ", source,
                         "this gate defines its own corpus, which is how "
                         "a deck gets stacked without anybody deciding to")


class InvariantTests(unittest.TestCase):
    """What criterion 3 caught, as invariants rather than hopes."""

    def setUp(self) -> None:
        self.mapping = build_permutation(seed=0)
        self.real_vocab = gate._vocabulary(CATEGORIES)
        self.perm_vocab = [self.mapping[w] for w in self.real_vocab]

    def test_the_permuted_vocabulary_is_the_image_in_the_same_order(self):
        """Sorting it independently made the two arms different sums."""
        self.assertEqual(len(self.perm_vocab), len(self.real_vocab))
        for real, image in zip(self.real_vocab, self.perm_vocab):
            self.assertEqual(self.mapping[real], image)

    def test_bag_of_words_is_identical_under_the_permutation(self) -> None:
        """Criterion 3, and a bijection makes it provable."""
        for spec in CATEGORIES.values():
            for text in gate.items_of(spec):
                self.assertEqual(
                    gate.bag_of_words(text, self.real_vocab),
                    gate.bag_of_words(permute(text, self.mapping),
                                      self.perm_vocab),
                    f"{text!r} became a different feature vector")

    def test_bag_of_words_is_binary_presence(self) -> None:
        vector = gate.bag_of_words("a square shape", self.real_vocab)
        self.assertTrue(set(vector) <= {0.0, 1.0})
        self.assertEqual(sum(vector), 2.0, "square and shape, not more")

    def test_the_arms_share_architecture_and_hyperparameters(self) -> None:
        """Paired: only the representation may differ."""
        import inspect

        source = inspect.getsource(gate._fold_accuracy)
        self.assertIn("_HIDDEN", source)
        self.assertIn("_EPOCHS", source)
        self.assertIn("_LR", source)


class GateVerdictTests(unittest.TestCase):
    """The pass, and exactly how far it reaches."""

    def setUp(self) -> None:
        self.doc = " ".join(gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("The arms are paired",
                       "The embedding arm wins at five examples",
                       "The control is faithful",
                       "The advantage does not survive the permutation",
                       "Both arms beat chance"):
            self.assertIn(phrase, self.doc)

    def test_stage_ones_criterion_is_quoted_not_paraphrased(self) -> None:
        self.assertIn("beats the same category trained on 5 raw-pixel "
                      "examples, by a margin larger than seed variance",
                      self.doc)

    def test_the_result_is_recorded_with_its_thinness(self) -> None:
        self.assertIn("+0.167 at seed sd 0.099", self.doc)
        self.assertIn("they clear them thinly", self.doc)

    def test_both_harness_defects_are_kept(self) -> None:
        self.assertIn("Criterion 3 failed on the first run", self.doc)
        self.assertIn("passed through untranslated", self.doc)

    def test_the_borrowed_representation_is_named_as_borrowed(self) -> None:
        """The gate's criterion is met; the stage's mechanism is not."""
        self.assertIn("met with a borrowed representation; the stage's "
                      "mechanism is not built", self.doc)


if __name__ == "__main__":
    unittest.main()
