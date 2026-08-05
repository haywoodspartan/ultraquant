"""Tests for the model forge: corpora, multi-tier training, and the build pipeline."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.forge.corpus import (
    BUILTIN_FAMILIES,
    build_corpora,
    builtin_taxonomy,
    features_of,
    synthetic_taxonomy,
)
from ultraquant.forge.forge import ModelForge
from ultraquant.forge.trainer import (
    ExpertSpec,
    forge_tier_report,
    train_experts,
)
from ultraquant.pattern.recognition import PATTERNS

_TIERS = forge_tier_report()
_CPU_OK = _TIERS["cpu"]
_GPU_OK = _TIERS["cuda"]


def _specs(n_experts: int, n_samples: int = 40, in_dim: int = 12, n_classes: int = 3):
    """Small, deterministic, separable training problems."""
    import random

    rng = random.Random(7)
    out = []
    for e in range(n_experts):
        xs, ys = [], []
        for s in range(n_samples):
            c = s % n_classes
            xs.append([rng.gauss(c, 0.25) for _ in range(in_dim)])
            ys.append(c)
        out.append(ExpertSpec(f"cat{e}", [f"l{i}" for i in range(n_classes)], xs, ys))
    return out


class CorpusTests(unittest.TestCase):
    """Corpora are built from patterns and are well formed."""

    def test_builtin_taxonomy_covers_the_reference_glyphs(self) -> None:
        taxonomy = builtin_taxonomy()
        self.assertEqual(set(taxonomy), set(BUILTIN_FAMILIES))
        covered = {label for labels in taxonomy.values() for label in labels}
        self.assertEqual(covered, set(PATTERNS))

    def test_features_are_thirty_dimensional(self) -> None:
        self.assertEqual(len(features_of(PATTERNS["plus"])), 30)

    def test_augmentation_counts_and_shapes(self) -> None:
        corpora = build_corpora(builtin_taxonomy(), n_per_class=12, flips=2, seed=1)
        self.assertEqual(len(corpora), 4)
        for corpus in corpora:
            self.assertEqual(len(corpus), 12 * len(corpus.labels))
            self.assertEqual(len(corpus.ys), len(corpus.xs))
            self.assertTrue(all(len(x) == 30 for x in corpus.xs))
            self.assertEqual(set(corpus.ys), set(range(len(corpus.labels))))

    def test_augmentation_is_deterministic(self) -> None:
        a = build_corpora(builtin_taxonomy(), n_per_class=5, seed=3)
        b = build_corpora(builtin_taxonomy(), n_per_class=5, seed=3)
        self.assertEqual(a[0].xs, b[0].xs)

    def test_synthetic_taxonomy_is_separable(self) -> None:
        taxonomy = synthetic_taxonomy(6, n_labels=4, seed=2, min_distance=8)
        self.assertEqual(len(taxonomy), 6)
        for prototypes in taxonomy.values():
            self.assertEqual(len(prototypes), 4)
            rows = list(prototypes.values())
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    distance = sum(
                        1 for a, b in zip("".join(rows[i]), "".join(rows[j])) if a != b
                    )
                    self.assertGreaterEqual(distance, 4, "prototypes too similar")


class TrainerTests(unittest.TestCase):
    """Every tier trains the same model to the same place."""

    def test_python_tier_learns(self) -> None:
        report = train_experts(_specs(2), hidden=8, epochs=20, lr=0.1,
                               batch_size=10, seed=1, tier="python")
        self.assertEqual(report.tier, "python")
        self.assertGreaterEqual(report.mean_accuracy, 0.8)

    def test_rejects_ragged_specs(self) -> None:
        specs = _specs(2)
        specs[1].xs = specs[1].xs[:-1]
        with self.assertRaises(ValueError):
            train_experts(specs, hidden=8, epochs=2, tier="python")

    def test_unknown_tier_rejected(self) -> None:
        with self.assertRaises(ValueError):
            train_experts(_specs(1), tier="quantum-teapot")

    @unittest.skipUnless(_CPU_OK, "forge CPU tier not built")
    def test_cpu_matches_python_weights(self) -> None:
        """The C++ tier mirrors the reference's summation order, so weights match."""
        specs = _specs(3)
        kwargs = dict(hidden=8, epochs=15, lr=0.1, batch_size=10, seed=5)
        ref = train_experts(specs, tier="python", **kwargs)
        got = train_experts(specs, tier="cpu", **kwargs)
        self.assertEqual(got.tier, "cpu")
        for a, b in zip(ref.experts, got.experts):
            self.assertAlmostEqual(a.loss, b.loss, places=9)
            self.assertAlmostEqual(a.accuracy, b.accuracy, places=12)
            for ra, rb in zip(a.w1, b.w1):
                for va, vb in zip(ra, rb):
                    self.assertAlmostEqual(va, vb, places=12)

    @unittest.skipUnless(_GPU_OK, "forge CUDA tier not built")
    def test_cuda_matches_python_accuracy(self) -> None:
        """The GPU reduces in parallel, so it agrees statistically, not bitwise."""
        specs = _specs(4)
        kwargs = dict(hidden=8, epochs=15, lr=0.1, batch_size=10, seed=5)
        ref = train_experts(specs, tier="python", **kwargs)
        got = train_experts(specs, tier="cuda", **kwargs)
        self.assertEqual(got.tier, "cuda")
        self.assertAlmostEqual(ref.mean_accuracy, got.mean_accuracy, delta=0.05)
        for a, b in zip(ref.experts, got.experts):
            self.assertAlmostEqual(a.loss, b.loss, places=6)

    @unittest.skipUnless(_GPU_OK and _CPU_OK, "both native tiers required")
    def test_auto_prefers_a_native_tier(self) -> None:
        report = train_experts(_specs(2), hidden=8, epochs=5, tier="auto")
        self.assertIn(report.tier, ("cuda", "cpu"))

    def test_weights_have_the_declared_shape(self) -> None:
        report = train_experts(_specs(2, in_dim=12, n_classes=3), hidden=8,
                               epochs=3, tier="python")
        expert = report.experts[0]
        self.assertEqual(len(expert.w1), 8)
        self.assertEqual(len(expert.w1[0]), 12)
        self.assertEqual(len(expert.w2), 3)
        self.assertEqual(len(expert.w2[0]), 8)
        self.assertEqual(len(expert.b1), 8)
        self.assertEqual(len(expert.b2), 3)


class ForgeTests(unittest.TestCase):
    """The forge produces a usable, catalogued, packed model."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_forge_"))
        self.corpora = build_corpora(builtin_taxonomy(), n_per_class=20, flips=2, seed=1,
                                     keywords=None)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _build(self, tier: str = "python", **kwargs):
        forge = ModelForge(self.dir / tier, seed=0, hidden=16, tier=tier)
        report = forge.build(self.corpora, epochs=25, lr=0.1, batch_size=16, **kwargs)
        return forge, report

    def test_build_creates_shards_library_and_snapshot(self) -> None:
        forge, report = self._build()
        self.assertEqual(report.experts, 4)
        self.assertEqual(report.categories, 4)
        self.assertGreater(report.parameters, 0)
        self.assertGreaterEqual(report.mean_accuracy, 0.8)

        self.assertTrue(Path(report.library).exists())
        self.assertTrue(report.snapshot.startswith("T-"))

        catalog = forge.vault.catalog()
        self.assertEqual(len(catalog), 4)
        for entry in catalog:
            self.assertEqual(entry["location"], "library")
            self.assertGreater(entry["nbytes"], 0)

        restored = forge.archive.restore(report.snapshot)
        self.assertEqual(restored["experts"], 4)

    def test_forged_model_serves_from_the_library(self) -> None:
        """Experts must work when paged back in through the normal serving path."""
        forge, report = self._build()
        served = forge.verify(self.corpora, samples=20)
        self.assertEqual(set(served), {c.name for c in self.corpora})
        mean = sum(served.values()) / len(served)
        self.assertGreaterEqual(mean, 0.8)
        self.assertGreater(forge.cache.stats()["misses"], 0)

    def test_router_finds_the_forged_categories(self) -> None:
        forge, _report = self._build()
        ranked = forge.router.route("arrow pointing up", top_k=2)
        self.assertTrue(ranked)
        self.assertEqual(ranked[0][0], "arrows")

    def test_empty_corpora_rejected(self) -> None:
        forge = ModelForge(self.dir / "empty", tier="python")
        with self.assertRaises(ValueError):
            forge.build([])

    def test_build_without_packing_leaves_shards_loose(self) -> None:
        forge, report = self._build(pack=False, snapshot=False)
        self.assertIsNone(report.library)
        self.assertIsNone(report.snapshot)
        self.assertTrue(all(e["location"] == "loose" for e in forge.vault.catalog()))

    @unittest.skipUnless(_GPU_OK, "forge CUDA tier not built")
    def test_gpu_forged_model_is_as_accurate(self) -> None:
        _fp, rp = self._build("python")
        _fg, rg = self._build("cuda")
        self.assertEqual(rg.tier, "cuda")
        self.assertAlmostEqual(rp.mean_accuracy, rg.mean_accuracy, delta=0.05)


class LearnedDispatchWiringTests(unittest.TestCase):
    """tier="auto" now means: ask the machine's own experience.

    The static preference walk was measured at 73x regret on a real 28-expert
    build (CUDA first-touch 49 ms where cpp@half cost 0.67 ms). A cold start
    probes by training the first expert on every configuration; the full
    build runs on the winner; enough distinct workload shapes make later
    builds decide without probing. Dispatch may never break a build - any
    scheduler failure falls back to the old walk.
    """

    def setUp(self) -> None:
        import tempfile

        self.dir = Path(tempfile.mkdtemp(prefix="uq_dispatchwire_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _forge(self, hidden: int = 8, epochs: int = 6):
        from ultraquant.forge.corpus import build_corpora, synthetic_taxonomy

        corpora = build_corpora(
            synthetic_taxonomy(n_categories=2, n_labels=2, seed=5),
            n_per_class=8, seed=1,
        )
        forge = ModelForge(self.dir / "home", seed=0, hidden=hidden,
                           tier="auto")
        report = forge.build(corpora, epochs=epochs)
        return forge, report

    def test_a_cold_auto_build_probes_and_records(self) -> None:
        import json

        forge, report = self._forge()
        self.assertEqual(forge.last_dispatch["reason"], "probe")
        self.assertIn("timings_ms", forge.last_dispatch)
        payload = json.loads(
            (self.dir / "home" / "dispatch.json").read_text(encoding="utf-8")
        )
        self.assertTrue(payload["records"])

    def test_enough_shapes_make_decisions_learned(self) -> None:
        """Four distinct workload shapes are the policy's training minimum."""
        for hidden, epochs in ((8, 6), (12, 6), (8, 10), (16, 8)):
            self._forge(hidden=hidden, epochs=epochs)
        forge, _report = self._forge(hidden=8, epochs=6)
        self.assertEqual(forge.last_dispatch["reason"], "learned")

    def test_the_probe_winner_is_the_tier_actually_used(self) -> None:
        forge, report = self._forge()
        config = forge.last_dispatch["config"]
        expected = {"python": "python", "cuda": "cuda"}.get(
            config, "cpu" if config.startswith("cpp") else config
        )
        self.assertEqual(report.tier, expected)

    def test_an_explicit_tier_bypasses_the_scheduler(self) -> None:
        from ultraquant.forge.corpus import build_corpora, synthetic_taxonomy

        corpora = build_corpora(
            synthetic_taxonomy(n_categories=2, n_labels=2, seed=5),
            n_per_class=8, seed=1,
        )
        forge = ModelForge(self.dir / "home2", seed=0, hidden=8,
                           tier="python")
        report = forge.build(corpora, epochs=4)
        self.assertEqual(report.tier, "python")
        self.assertEqual(forge.last_dispatch, {})

    def test_the_build_cli_reports_the_decision(self) -> None:
        import io
        from contextlib import redirect_stdout

        from ultraquant.forge.build import main as forge_main

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            forge_main(["--root", str(self.dir / "cli"), "--synthetic", "2",
                        "--labels", "2", "--per-class", "8",
                        "--epochs", "4", "--hidden", "8", "--tier", "auto"])
        self.assertIn("dispatch:", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
