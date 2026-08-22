"""A prediction layer that says how much it knows.

§11.102 gave the prediction layer entropy, evidence and margin, and
a floor below which it declines. These pins hold the definitions,
the refusal, and the measured finding that entropy and margin catch
different failures - a finding that came from a FAILED criterion and
would otherwise be lost.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.model import uncertainty
from ultraquant.model.network import UltraQuantNet
from ultraquant.pattern.recognition import PATTERNS, PatternRecognizer


class MeasureTests(unittest.TestCase):
    """The definitions, including the cases that make them differ."""

    def test_knowing_nothing_is_the_full_entropy(self) -> None:
        uniform = [0.125] * 8
        self.assertAlmostEqual(uncertainty.entropy(uniform), 3.0, places=12)
        self.assertAlmostEqual(uncertainty.evidence(uniform), 0.0, places=12)

    def test_knowing_everything_is_none(self) -> None:
        certain = [1.0] + [0.0] * 7
        self.assertAlmostEqual(uncertainty.entropy(certain), 0.0, places=12)
        self.assertAlmostEqual(uncertainty.evidence(certain), 3.0, places=12)

    def test_entropy_matches_its_definition(self) -> None:
        probs = [0.4, 0.3, 0.2, 0.1]
        want = -sum(p * math.log2(p) for p in probs)
        self.assertAlmostEqual(uncertainty.entropy(probs), want, places=12)

    def test_zero_log_zero_is_taken_as_zero(self) -> None:
        self.assertAlmostEqual(uncertainty.entropy([0.5, 0.5, 0.0, 0.0]),
                               1.0, places=12)

    def test_margin_sees_what_entropy_does_not(self) -> None:
        """A two-way tie has decent evidence and no margin at all."""
        tie = [0.45, 0.44, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01]
        self.assertGreater(uncertainty.evidence(tie), 1.0)
        self.assertLess(uncertainty.margin(tie), 0.02)


class DecisionTests(unittest.TestCase):
    """Two floors, because they catch different failures."""

    def test_an_evidence_floor_declines_a_flat_distribution(self) -> None:
        self.assertIsNone(uncertainty.decide([0.125] * 8, floor=1.0))
        self.assertEqual(uncertainty.decide([0.9] + [0.1 / 7] * 7,
                                            floor=1.0), 0)

    def test_a_margin_floor_declines_a_two_way_tie(self) -> None:
        tie = [0.45, 0.44] + [0.11 / 6] * 6
        self.assertEqual(uncertainty.decide(tie, floor=0.0), 0)
        self.assertIsNone(uncertainty.decide(tie, floor=0.0,
                                             margin_floor=0.1))

    def test_the_two_floors_are_independent(self) -> None:
        """One can decline where the other accepts, both ways."""
        wide_margin_low_evidence = [0.5, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.0]
        self.assertIsNone(uncertainty.decide(wide_margin_low_evidence,
                                             floor=2.0))
        self.assertEqual(uncertainty.decide(wide_margin_low_evidence,
                                            floor=0.0, margin_floor=0.3), 0)

    def test_a_declined_prediction_still_carries_its_guess(self) -> None:
        result = uncertainty.describe([0.125] * 8, floor=1.0)
        self.assertTrue(result.declined)
        self.assertEqual(result.label, 0)
        self.assertEqual(len(result.distribution), 8)

    def test_bits_of_names_the_question_size(self) -> None:
        self.assertIn("of 3.00 bits",
                      uncertainty.describe([0.125] * 8).bits_of)


class NetworkTests(unittest.TestCase):
    """The layer, on a network that has learned nothing."""

    def test_an_untrained_network_says_it_knows_nothing(self) -> None:
        net = UltraQuantNet(30, [32], 8, seed=0)
        worst = max(net.predict_with_uncertainty(
            [0.1 * i for i in range(30)]).evidence for _ in range(1))
        self.assertLess(worst, 1.0,
                        "random weights reported real knowledge")

    def test_the_distribution_is_a_distribution(self) -> None:
        net = UltraQuantNet(30, [16], 8, seed=1)
        probs = net.predict_distribution([0.05 * i for i in range(30)])
        self.assertAlmostEqual(sum(probs), 1.0, places=12)
        self.assertEqual(len(probs), 8)

    def test_predict_still_agrees_with_the_new_path(self) -> None:
        """The old signature must not have changed meaning."""
        net = UltraQuantNet(30, [16], 8, seed=2)
        x = [0.03 * i for i in range(30)]
        label, confidence = net.predict(x)
        result = net.predict_with_uncertainty(x)
        self.assertEqual(label, result.label)
        self.assertAlmostEqual(confidence, result.probability, places=15)


class RecognizerTests(unittest.TestCase):
    """The refusal, end to end on a trained recognizer."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = Path(tempfile.mkdtemp(prefix="uq_uncertainty_"))
        cls.recognizer = PatternRecognizer(mode="classical", seed=0,
                                           memory_path=cls.dir / "m")
        cls.recognizer.train(epochs=30, n_per_class=30, flips=2, seed=1)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_a_real_glyph_is_named(self) -> None:
        label, result = self.recognizer.recognize_with_uncertainty(
            PATTERNS["cross"], floor=1.774, margin_floor=0.3)
        self.assertEqual(label, "cross")
        self.assertGreater(result.evidence, 1.774)

    def test_something_that_is_not_a_glyph_is_declined(self) -> None:
        label, result = self.recognizer.recognize_with_uncertainty(
            [0.5] * 25, floor=1.774, margin_floor=0.3)
        self.assertIsNone(label, f"named a class on flat input: {result}")

    def test_the_floors_default_to_naming_everything(self) -> None:
        """Zero floors must behave exactly as recognize always did."""
        self.assertEqual(self.recognizer.evidence_floor, 0.0)
        self.assertEqual(self.recognizer.margin_floor, 0.0)
        label, _r = self.recognizer.recognize_with_uncertainty([0.5] * 25)
        self.assertIsNotNone(label)


class GateVerdictTests(unittest.TestCase):
    """A failed criterion, and the design it produced."""

    def setUp(self) -> None:
        from ultraquant.experiments import entropy_gate
        self.doc = " ".join(entropy_gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("The measures are their definitions",
                       "Entropy rejects errors better than max-softmax",
                       "A network that has learned nothing says so",
                       "Things that are not glyphs are declined",
                       "The floor is fitted on one split"):
            self.assertIn(phrase, self.doc)

    def test_the_failure_is_recorded_as_a_failure(self) -> None:
        self.assertIn("FAILED criterion 2", self.doc)
        self.assertIn("Entropy is the WORST of the three measures",
                      self.doc)

    def test_the_measured_split_is_recorded(self) -> None:
        self.assertIn("margin` catches wrong predictions", self.doc)
        self.assertIn("evidence` catches inputs the", self.doc)

    def test_the_rejected_combination_is_recorded(self) -> None:
        """So nobody retries it."""
        self.assertIn("does not compose", self.doc)

    def test_the_follow_up_is_marked_as_not_gated(self) -> None:
        self.assertIn("reported rather than gated", self.doc)
        self.assertIn("scoring the coin after the throw", self.doc)


if __name__ == "__main__":
    unittest.main()
