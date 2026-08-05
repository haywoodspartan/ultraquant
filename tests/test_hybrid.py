"""Tests for the hybrid quantum/classical expert library."""

from __future__ import annotations

import json
import random
import shutil
import tempfile
import unittest
import zlib
from pathlib import Path

from ultraquant.experts.moe import ExpertPool
from ultraquant.hybrid.expert import QuantumExpert, quantum_expert_payload
from ultraquant.hybrid.pool import HybridExpertPool
from ultraquant.pattern.recognition import PATTERNS, render
from ultraquant.shards.budget import ShardCache
from ultraquant.shards.vault import ShardVault


class _Corpus:
    """Minimal stand-in for a forge corpus."""

    def __init__(self, name, labels, xs, ys, keywords=None) -> None:
        self.name = name
        self.labels = labels
        self.xs = xs
        self.ys = ys
        self.keywords = keywords or [name, *labels]


def _dataset(labels, per_class=12, flips=1, seed=0):
    """Noisy glyph samples for the given labels."""
    rng = random.Random(seed)
    xs, ys = [], []
    for index, name in enumerate(labels):
        base = render(PATTERNS[name])
        for _ in range(per_class):
            variant = list(base)
            for pixel in rng.sample(range(25), flips):
                variant[pixel] = 1.0 - variant[pixel]
            xs.append(variant)
            ys.append(index)
    return xs, ys


class QuantumExpertTests(unittest.TestCase):
    """The quantum expert trains, serialises and reloads exactly."""

    def test_trains_on_glyphs(self) -> None:
        xs, ys = _dataset(["plus", "square"])
        expert = QuantumExpert(labels=["plus", "square"], seed=0)
        report = expert.fit(xs, ys, epochs=6, seed=0)
        self.assertGreaterEqual(report["accuracy"], 0.8)
        label, confidence = expert.predict(render(PATTERNS["plus"]))
        self.assertIn(label, ("plus", "square"))
        self.assertGreaterEqual(confidence, 0.0)

    def test_round_trip_preserves_predictions(self) -> None:
        xs, ys = _dataset(["plus", "square"])
        expert = QuantumExpert(labels=["plus", "square"], seed=1)
        expert.fit(xs, ys, epochs=4, seed=1)

        state = json.loads(json.dumps(expert.state_dict()))
        restored = QuantumExpert.load(state)
        for sample in xs[:6]:
            self.assertEqual(expert.predict(sample), restored.predict(sample))

    def test_load_rejects_a_classical_payload(self) -> None:
        with self.assertRaises(ValueError):
            QuantumExpert.load({"kind": "ternary-net"})

    def test_parameter_count_is_small(self) -> None:
        """The reason to bother: far fewer stored numbers."""
        expert = QuantumExpert(labels=["a", "b"], num_qubits=5, layers=2)
        self.assertEqual(expert.parameter_count, 20 + 2 * 5 + 2)
        classical = 25 * 16 + 16 + 16 * 2 + 2
        self.assertLess(expert.parameter_count, classical / 5)

    def test_payload_is_smaller_than_the_classical_equivalent(self) -> None:
        from ultraquant.model.network import UltraQuantNet

        xs, ys = _dataset(["plus", "square"])
        expert = QuantumExpert(labels=["plus", "square"], seed=0)
        expert.fit(xs, ys, epochs=2, seed=0)

        classical = UltraQuantNet(25, [16], 2, seed=0)
        quantum_bytes = _stored(quantum_expert_payload(expert))
        classical_bytes = _stored(
            {"labels": ["plus", "square"], "net": classical.state_dict(),
             "trained_examples": len(xs)}
        )
        self.assertLess(quantum_bytes * 3, classical_bytes)

    def test_describe_reports_circuit_and_cost(self) -> None:
        expert = QuantumExpert(labels=["a", "b", "c"])
        info = expert.describe()
        self.assertEqual(info["kind"], "quantum-vqc")
        self.assertIn("depth", info["circuit"])
        self.assertGreater(info["evaluations_per_training_step"], 1)


class HybridPoolTests(unittest.TestCase):
    """Selection is by measurement, and serving hides which kind won."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_hybrid_"))
        self.vault = ShardVault(self.dir / "vault")
        self.cache = ShardCache(max_bytes=1 << 20)
        self.pool = HybridExpertPool(
            self.vault, self.cache, input_dim=25, hidden=16,
            num_qubits=5, layers=2, seed=0,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _corpus(self, name="frames", labels=("square", "diamond")):
        xs, ys = _dataset(list(labels))
        return _Corpus(name, list(labels), xs, ys)

    def test_builds_and_stores_an_expert(self) -> None:
        corpus = self._corpus()
        outcome = self.pool.build_category(
            corpus.name, corpus.labels, corpus.xs, corpus.ys,
            quantum_epochs=4, classical_epochs=30,
        )
        self.assertIn(outcome.chosen, ("quantum", "classical"))
        self.assertTrue(self.vault.has(ExpertPool.shard_id("frames")))
        self.assertGreater(outcome.size_ratio, 1.0,
                           "the quantum expert should be the smaller one")

    def test_serving_is_agnostic_to_the_kind(self) -> None:
        corpus = self._corpus()
        self.pool.build_category(
            corpus.name, corpus.labels, corpus.xs, corpus.ys,
            quantum_epochs=4, classical_epochs=30,
        )
        label, confidence = self.pool.predict("frames", render(PATTERNS["square"]))
        self.assertIn(label, corpus.labels)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertIn(self.pool.kind_of("frames"), ("quantum", "classical"))

    def test_forced_rules_are_honoured(self) -> None:
        corpus = self._corpus()
        for rule in ("quantum", "classical"):
            with self.subTest(rule=rule):
                outcome = self.pool.build_category(
                    f"{rule}-cat", corpus.labels, corpus.xs, corpus.ys,
                    rule=rule, quantum_epochs=3, classical_epochs=20,
                )
                self.assertEqual(outcome.chosen, rule)
                self.assertEqual(self.pool.kind_of(f"{rule}-cat"), rule)

    def test_size_rule_prefers_the_smaller_expert_within_tolerance(self) -> None:
        self.assertEqual(
            HybridExpertPool._choose("size", 0.95, 1.0, 500, 5000, tolerance=0.1),
            "quantum", "a slightly worse but far smaller expert should win on size",
        )
        self.assertEqual(
            HybridExpertPool._choose("size", 0.60, 1.0, 500, 5000, tolerance=0.1),
            "classical", "a much worse expert must not win on size alone",
        )

    def test_accuracy_rule_ignores_size(self) -> None:
        self.assertEqual(
            HybridExpertPool._choose("accuracy", 0.90, 0.95, 500, 5000, tolerance=0.1),
            "classical",
        )
        self.assertEqual(
            HybridExpertPool._choose("accuracy", 0.99, 0.95, 500, 5000, tolerance=0.1),
            "quantum",
        )

    def test_accuracy_tie_breaks_on_size(self) -> None:
        self.assertEqual(
            HybridExpertPool._choose("accuracy", 0.9, 0.9, 400, 4000, tolerance=0.0),
            "quantum",
        )

    def test_unknown_rule_rejected(self) -> None:
        corpus = self._corpus()
        with self.assertRaises(ValueError):
            self.pool.build_category(
                "x", corpus.labels, corpus.xs, corpus.ys, rule="vibes",
            )

    def test_missing_category_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.pool.predict("never-built", [0.0] * 25)

    def test_build_reports_every_category(self) -> None:
        corpora = [
            self._corpus("frames", ("square", "diamond")),
            self._corpus("crosses", ("plus", "cross")),
        ]
        report = self.pool.build(corpora, rule="size", quantum_epochs=3,
                                 classical_epochs=20)
        summary = report.summary()
        self.assertEqual(summary["categories"], 2)
        self.assertEqual(
            summary["quantum_chosen"] + summary["classical_chosen"], 2
        )
        self.assertGreater(summary["mean_size_ratio"], 1.0)
        self.assertGreater(summary["classical_bytes_total"], summary["quantum_bytes_total"])

    def test_experts_page_through_the_cache(self) -> None:
        """A hybrid expert must be an ordinary shard to everything downstream."""
        corpus = self._corpus()
        self.pool.build_category(
            corpus.name, corpus.labels, corpus.xs, corpus.ys,
            quantum_epochs=3, classical_epochs=20,
        )
        self.pool.predict("frames", render(PATTERNS["square"]))
        self.assertGreater(self.cache.stats()["misses"], 0)
        self.pool.predict("frames", render(PATTERNS["diamond"]))
        self.assertGreater(self.cache.stats()["hits"], 0)

    def test_empty_summary(self) -> None:
        from ultraquant.hybrid.pool import HybridReport

        self.assertEqual(HybridReport().summary(), {"categories": 0})


def _stored(payload: dict) -> int:
    """Compressed size, as the vault would store it."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return len(zlib.compress(raw, 6))


if __name__ == "__main__":
    unittest.main()
