"""Tests for ultraquant.pattern.recognition (SPEC section 10/12).

Uses SimulatorBackend explicitly — never a QPU — and stays small enough to
run fast.  Trained recognizers are shared per test class via setUpClass.
"""

from __future__ import annotations

import unittest

from ultraquant.pattern.recognition import (
    LABELS,
    PATTERNS,
    PatternRecognizer,
    make_dataset,
    render,
    row_means,
)
from ultraquant.quantum.backend import SimulatorBackend


class TestDatasetHelpers(unittest.TestCase):
    """render / make_dataset / row_means basics."""

    def test_labels_order(self) -> None:
        self.assertEqual(len(LABELS), 8)
        self.assertEqual(LABELS[0], "plus")
        self.assertEqual(LABELS[-1], "stripes_v")
        self.assertEqual(LABELS, list(PATTERNS))

    def test_render_plus(self) -> None:
        flat = render(PATTERNS["plus"])
        self.assertEqual(len(flat), 25)
        self.assertTrue(all(v in (0.0, 1.0) for v in flat))
        # Row 2 of "plus" is "#####".
        self.assertEqual(flat[10:15], [1.0] * 5)
        self.assertEqual(flat[0], 0.0)
        self.assertEqual(flat[2], 1.0)

    def test_make_dataset_shape_and_determinism(self) -> None:
        xs1, ys1 = make_dataset(n_per_class=3, flips=2, seed=5)
        xs2, ys2 = make_dataset(n_per_class=3, flips=2, seed=5)
        self.assertEqual(xs1, xs2)
        self.assertEqual(ys1, ys2)
        self.assertEqual(len(xs1), 3 * len(LABELS))
        self.assertEqual(ys1[:3], [0, 0, 0])
        self.assertEqual(ys1[-1], len(LABELS) - 1)

    def test_make_dataset_flips_distinct_pixels(self) -> None:
        xs, ys = make_dataset(n_per_class=2, flips=3, seed=11)
        for x, y in zip(xs, ys):
            base = render(PATTERNS[LABELS[y]])
            differing = sum(1 for a, b in zip(x, base) if a != b)
            self.assertEqual(differing, 3)

    def test_row_means(self) -> None:
        means = row_means(render(PATTERNS["stripes_h"]))
        self.assertEqual(means, [1.0, 0.0, 1.0, 0.0, 1.0])


class TestClassicalRecognizer(unittest.TestCase):
    """Classical mode: training, recognition, memory logging, snapshots."""

    rec: PatternRecognizer

    @classmethod
    def setUpClass(cls) -> None:
        cls.rec = PatternRecognizer(mode="classical", seed=0)
        cls.train_stats = cls.rec.train(epochs=30, n_per_class=15, seed=1)

    def test_trains_to_eval_accuracy(self) -> None:
        self.assertGreaterEqual(self.train_stats["train_accuracy"], 0.8)
        acc = self.rec.evaluate(n_per_class=15, flips=3, seed=99)
        self.assertGreaterEqual(acc, 0.8)

    def test_train_stats_shape(self) -> None:
        self.assertEqual(self.train_stats["epochs"], 30)
        self.assertIsInstance(self.train_stats["final_loss"], float)

    def test_recognize_logs_episode_and_signature(self) -> None:
        episodic_before = self.rec.memory.stats()["episodic"]
        signatures_before = self.rec.memory.stats()["signatures"]
        label, confidence = self.rec.recognize(PATTERNS["square"])
        self.assertIn(label, LABELS)
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        stats = self.rec.memory.stats()
        self.assertEqual(stats["episodic"], episodic_before + 1)
        self.assertEqual(stats["signatures"], signatures_before + 1)
        episodes = self.rec.memory.recall_episodes(kind="recognition", limit=1)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["content"]["label"], label)
        self.assertEqual(episodes[0]["content"]["mode"], "classical")

    def test_recognize_accepts_flat_floats(self) -> None:
        label, confidence = self.rec.recognize(render(PATTERNS["stripes_v"]))
        self.assertIn(label, LABELS)
        self.assertGreater(confidence, 0.0)

    def test_snapshot_restore_identical_predictions(self) -> None:
        payload = self.rec.snapshot()
        self.assertEqual(payload["config"]["mode"], "classical")
        other = PatternRecognizer(mode="classical", seed=123)
        other.restore_snapshot(payload)
        xs, _ = make_dataset(n_per_class=2, flips=2, seed=77)
        for x in xs:
            feats = self.rec.features(x)
            self.assertEqual(self.rec.net.predict(feats), other.net.predict(feats))

    def test_snapshot_is_json_safe(self) -> None:
        import json

        payload = self.rec.snapshot()
        round_tripped = json.loads(json.dumps(payload))
        other = PatternRecognizer(mode="classical", seed=5)
        other.restore_snapshot(round_tripped)
        x = render(PATTERNS["cross"])
        feats = self.rec.features(x)
        self.assertEqual(self.rec.net.predict(feats), other.net.predict(feats))


class TestHybridRecognizer(unittest.TestCase):
    """Hybrid exact mode with an explicit SimulatorBackend (never a QPU)."""

    rec: PatternRecognizer

    @classmethod
    def setUpClass(cls) -> None:
        cls.rec = PatternRecognizer(
            mode="hybrid", backend=SimulatorBackend(seed=7), shots=None, seed=0
        )
        cls.train_stats = cls.rec.train(epochs=30, n_per_class=15, seed=1)

    def test_backend_is_simulator(self) -> None:
        assert self.rec.backend is not None
        self.assertEqual(self.rec.backend.name, "ultraquant-statevector")

    def test_features_dimension(self) -> None:
        feats = self.rec.features(render(PATTERNS["diamond"]))
        self.assertEqual(len(feats), 30)
        for v in feats[25:]:
            self.assertGreaterEqual(v, -1.0)
            self.assertLessEqual(v, 1.0)

    def test_trains_to_eval_accuracy(self) -> None:
        self.assertGreaterEqual(self.train_stats["train_accuracy"], 0.8)
        acc = self.rec.evaluate(n_per_class=15, flips=3, seed=99)
        self.assertGreaterEqual(acc, 0.8)

    def test_recognize_returns_known_label(self) -> None:
        label, confidence = self.rec.recognize(PATTERNS["plus"])
        self.assertIn(label, LABELS)
        self.assertGreater(confidence, 0.0)
        episodes = self.rec.memory.recall_episodes(kind="recognition", limit=1)
        self.assertEqual(episodes[0]["content"]["mode"], "hybrid")

    def test_snapshot_restore_identical_predictions(self) -> None:
        payload = self.rec.snapshot()
        other = PatternRecognizer(
            mode="hybrid", backend=SimulatorBackend(seed=7), shots=None, seed=42
        )
        other.restore_snapshot(payload)
        xs, _ = make_dataset(n_per_class=2, flips=2, seed=55)
        for x in xs:
            # Exact mode: features are deterministic across instances.
            feats = self.rec.features(x)
            self.assertEqual(self.rec.net.predict(feats), other.net.predict(feats))

    def test_invalid_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PatternRecognizer(mode="quantum-ai")


if __name__ == "__main__":
    unittest.main()
