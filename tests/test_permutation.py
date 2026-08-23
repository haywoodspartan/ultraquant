"""Is the embedding's edge semantic, or just capacity? The permutation control.

§11.108 answered §11.11's open question with a control that proves its
own faithfulness: a bijection over the vocabulary leaves lexical
routing mathematically unchanged while destroying meaning. These pins
hold that invariant — which is the whole design — and the recorded
result, including what it does NOT establish.
"""

from __future__ import annotations

import unittest

from ultraquant.experiments import permutation_gate as gate
from ultraquant.experiments.embed_gate import (CATEGORIES, _lexical_route,
                                               _tokens)


class PermutationTests(unittest.TestCase):
    """The control, and the invariant that makes it a control."""

    def test_the_map_is_injective(self) -> None:
        """A merge would fuse two tokens and change lexical routing."""
        for seed in range(4):
            mapping = gate.build_permutation(seed=seed)
            self.assertEqual(len(set(mapping.values())), len(mapping))

    def test_every_corpus_word_is_mapped(self) -> None:
        mapping = gate.build_permutation(seed=0)
        for word in gate.corpus_vocabulary():
            self.assertIn(word, mapping)

    def test_no_corpus_word_maps_to_itself_or_another(self) -> None:
        """The image set must be disjoint, or meaning partly survives."""
        mapping = gate.build_permutation(seed=0)
        corpus = set(gate.corpus_vocabulary())
        self.assertEqual(corpus & set(mapping.values()), set())

    def test_lexical_routing_is_identical_under_the_permutation(self):
        """Criterion 1, as an invariant rather than a hope.

        Token overlap survives a bijection exactly, so this cannot
        fail unless the permutation is broken — which is precisely
        why it is worth asserting.
        """
        for seed in range(4):
            mapping = gate.build_permutation(seed=seed)
            spec = gate.permuted_categories(mapping)
            real = {n: _tokens(s["canonical"])
                    for n, s in CATEGORIES.items()}
            perm = {n: _tokens(s["canonical"]) for n, s in spec.items()}
            for name, entry in CATEGORIES.items():
                for text in entry["paraphrases"] + entry["partial"]:
                    self.assertEqual(
                        _lexical_route(text, real),
                        _lexical_route(gate.permute(text, mapping), perm),
                        f"{text!r} routed differently after permutation")

    def test_the_permuted_text_is_real_words_not_nonsense(self) -> None:
        """Nonsense would degrade the embedding and break criterion 4."""
        mapping = gate.build_permutation(seed=0)
        for image in mapping.values():
            self.assertIn(image, gate._IMAGE_WORDS)
            self.assertTrue(image.isalpha())

    def test_too_small_an_image_set_is_refused(self) -> None:
        """A non-injective map must raise, not silently merge."""
        saved = gate._IMAGE_WORDS
        try:
            gate._IMAGE_WORDS = ["one", "two", "three"]
            with self.assertRaises(ValueError):
                gate.build_permutation(seed=0)
        finally:
            gate._IMAGE_WORDS = saved


class GateVerdictTests(unittest.TestCase):
    """The pass, and the claims it deliberately does not make."""

    def setUp(self) -> None:
        self.doc = " ".join(gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("The control is faithful",
                       "The embedding wins on paraphrases",
                       "The advantage is semantic",
                       "The permutation must not simply break the "
                       "embedding"):
            self.assertIn(phrase, self.doc)

    def test_the_declined_rescue_is_registered_in_advance(self) -> None:
        """§11.11 named this test and refused to use it after the fact."""
        self.assertIn("registered here, before this run", self.doc)
        self.assertIn("difference-in-differences", self.doc)

    def test_the_result_is_recorded(self) -> None:
        self.assertIn("+0.472 at seed sd 0.127", self.doc)
        self.assertIn("85% of the embedding's advantage disappears",
                      self.doc)

    def test_the_leak_and_its_correction_are_kept(self) -> None:
        """Run one's control let real meaning through untranslated."""
        self.assertIn("passed through untranslated", self.doc)
        self.assertIn("run one (leaky)", self.doc)
        self.assertIn("+0.514 at 4.8x before", self.doc)

    def test_why_criterion_one_missed_the_leak_is_explained(self):
        """A symmetric leak is invisible to a faithfulness check."""
        self.assertIn("the leak was CONSISTENT", self.doc)
        self.assertIn("cannot see a leak that is symmetric", self.doc)

    def test_stage_one_is_not_claimed(self) -> None:
        """Routing is not few-shot transfer, and the doc says so."""
        self.assertIn("does **not** pass Stage 1", self.doc)
        self.assertIn("This is about **routing**", self.doc)

    def test_the_representation_is_admitted_to_be_external(self) -> None:
        self.assertIn("the representation is not ours", self.doc)
        self.assertIn("It is a measurement, not a dependency", self.doc)


if __name__ == "__main__":
    unittest.main()
