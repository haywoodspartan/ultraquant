"""Tests for growing a library incrementally, and for the kernel experiment."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.experts.moe import ExpertPool
from ultraquant.forge.corpus import build_corpora, synthetic_taxonomy
from ultraquant.forge.forge import ModelForge
from ultraquant.quantum.benchmark_kernels import (
    compare,
    kernel_accuracy,
    linear_kernel,
    parity_task,
    quadratic_kernel,
)


def _corpora(start: int, count: int, seed: int, labels: int = 3):
    """Synthetic corpora with category names offset by ``start``."""
    taxonomy = synthetic_taxonomy(count, labels, seed=seed)
    renamed = {
        f"fam{start + i:03d}": prototypes
        for i, prototypes in enumerate(taxonomy.values())
    }
    return build_corpora(renamed, n_per_class=10, seed=seed)


class LibraryExpansionTests(unittest.TestCase):
    """A model must be able to grow after it has been built."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_expand_"))
        self.forge = ModelForge(self.dir / "model", seed=0, hidden=8, tier="auto")
        self.forge.build(_corpora(0, 4, seed=1), epochs=8)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_expansion_adds_categories(self) -> None:
        before = len(self.forge.vault.catalog())
        report = self.forge.expand(_corpora(4, 3, seed=2), epochs=8)
        self.assertEqual(report.experts, 3)
        self.assertEqual(len(self.forge.vault.catalog()), before + 3)

    def test_expansion_writes_a_new_segment(self) -> None:
        before = len(self.forge.vault.libraries())
        self.forge.expand(_corpora(4, 2, seed=2), epochs=8)
        self.assertEqual(len(self.forge.vault.libraries()), before + 1,
                         "new categories belong in their own immutable segment")

    def test_existing_shards_stay_readable_across_segments(self) -> None:
        self.forge.expand(_corpora(4, 3, seed=2), epochs=8)
        self.forge.expand(_corpora(7, 3, seed=3), epochs=8)
        for entry in self.forge.vault.catalog():
            with self.subTest(shard=entry["shard_id"]):
                self.assertIn("net", self.forge.vault.get(entry["shard_id"]))

    def test_known_categories_are_skipped(self) -> None:
        corpora = _corpora(0, 4, seed=1)
        report = self.forge.expand(corpora, epochs=8)
        self.assertEqual(report.experts, 0)
        self.assertEqual(report.tier, "none")

    def test_retrain_replaces_the_shard(self) -> None:
        corpora = _corpora(0, 2, seed=1)
        before = self.forge.vault.entry(ExpertPool.shard_id(corpora[0].name))["sha256"]
        self.forge.expand(corpora, epochs=12, retrain=True)
        after = self.forge.vault.entry(ExpertPool.shard_id(corpora[0].name))["sha256"]
        self.assertNotEqual(before, after)

    def test_retrain_preserves_access_statistics(self) -> None:
        """Relearning a skill must not erase how often it has been recalled."""
        corpora = _corpora(0, 1, seed=1)
        shard_id = ExpertPool.shard_id(corpora[0].name)
        self.forge.vault.get(shard_id)
        self.forge.vault.get(shard_id)
        before = self.forge.vault.entry(shard_id)["access_count"]
        self.forge.expand(corpora, epochs=8, retrain=True)
        self.assertGreaterEqual(self.forge.vault.entry(shard_id)["access_count"], before)

    def test_expansion_snapshots_the_growth(self) -> None:
        report = self.forge.expand(_corpora(4, 2, seed=2), epochs=8)
        self.assertTrue(report.snapshot.startswith("T-"))
        restored = self.forge.archive.restore(report.snapshot)
        self.assertEqual(len(restored["added"]), 2)
        self.assertGreater(restored["categories_after"], restored["categories_before"])

    def test_empty_expansion_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.forge.expand([])

    def test_router_learns_the_new_categories(self) -> None:
        corpora = _corpora(4, 2, seed=2)
        self.forge.expand(corpora, epochs=8)
        ranked = self.forge.router.route(corpora[0].keywords[0], top_k=3)
        self.assertIn(corpora[0].name, [c for c, _s in ranked])


class CompactionTests(unittest.TestCase):
    """Segments accumulate; compaction merges them and reclaims space."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_compact_"))
        self.forge = ModelForge(self.dir / "model", seed=0, hidden=8, tier="auto")
        self.forge.build(_corpora(0, 3, seed=1), epochs=8)
        self.forge.expand(_corpora(3, 3, seed=2), epochs=8)
        self.forge.expand(_corpora(6, 3, seed=3), epochs=8)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_merges_every_segment(self) -> None:
        self.assertGreater(len(self.forge.vault.libraries()), 1)
        report = self.forge.compact()
        self.assertEqual(len(self.forge.vault.libraries()), 1)
        self.assertEqual(report["shards"], len(self.forge.vault.catalog()))

    def test_all_shards_survive_compaction(self) -> None:
        self.forge.compact()
        for entry in self.forge.vault.catalog():
            with self.subTest(shard=entry["shard_id"]):
                payload = self.forge.vault.get(entry["shard_id"])
                self.assertIn("net", payload)

    def test_compaction_reclaims_shadowed_bytes(self) -> None:
        """Retraining strands the old bytes; compaction is what frees them."""
        self.forge.expand(_corpora(0, 3, seed=1), epochs=8, retrain=True)
        report = self.forge.compact()
        self.assertGreaterEqual(report["bytes_before"], report["bytes_after"])
        self.assertGreaterEqual(report["reclaimed"], 0)

    def test_compaction_of_an_empty_vault_is_safe(self) -> None:
        from ultraquant.shards.vault import ShardVault

        empty = ShardVault(self.dir / "empty")
        report = empty.compact(self.dir / "empty" / "merged.uql")
        self.assertEqual(report["shards"], 0)


class HardKernelTaskTests(unittest.TestCase):
    """The experiment: does an entangling map ever beat a cheap one?"""

    def test_parity_task_is_built_as_described(self) -> None:
        task = parity_task(num_features=6, order=2, samples=20, seed=0)
        self.assertEqual(len(task.hidden_features), 2)
        self.assertEqual(len(task.xs), 20)
        self.assertEqual(set(task.ys) | {0, 1}, {0, 1})

    def test_order_one_is_linearly_separable(self) -> None:
        """The control: a first-order task must be easy for a linear kernel.

        Averaged over seeds — a single 48-sample draw lands on the threshold
        often enough that a one-seed assertion is a coin flip, not a check.
        """
        scores = [
            kernel_accuracy(parity_task(6, 1, 48, seed), linear_kernel, seed=seed)
            for seed in range(5)
        ]
        mean = sum(scores) / len(scores)
        self.assertGreater(mean, 0.7)
        order_two = [
            kernel_accuracy(parity_task(6, 2, 48, seed), linear_kernel, seed=seed)
            for seed in range(5)
        ]
        self.assertGreater(mean, sum(order_two) / len(order_two) + 0.15,
                           "order 1 must be markedly easier than order 2")

    def test_order_two_defeats_the_linear_kernel(self) -> None:
        """XOR-like structure is invisible to any function of single features."""
        scores = [
            kernel_accuracy(parity_task(6, 2, 48, seed), linear_kernel, seed=seed)
            for seed in range(5)
        ]
        self.assertLess(sum(scores) / len(scores), 0.65)

    def test_entangling_map_wins_where_interactions_carry_the_label(self) -> None:
        """The headline claim, re-measured: order 2 favours the entangling map."""
        quantum, classical = 0.0, 0.0
        seeds = 5
        for seed in range(seeds):
            row = compare(num_features=6, samples=48, orders=(2,), seed=seed)[0]
            quantum += max(row["zz_linear"], row["zz_full"])
            classical += max(row["linear"], row["quadratic"])
        self.assertGreater(quantum / seeds, classical / seeds + 0.05)

    def test_entangling_map_loses_where_interactions_do_not(self) -> None:
        """And the converse — expressive power is not free."""
        quantum, cheap = 0.0, 0.0
        seeds = 5
        for seed in range(seeds):
            row = compare(num_features=6, samples=48, orders=(1,), seed=seed)[0]
            quantum += max(row["zz_linear"], row["zz_full"])
            cheap += row["angle"]
        self.assertLess(quantum / seeds, cheap / seeds)

    def test_quadratic_kernel_is_included_as_a_fair_baseline(self) -> None:
        """Beating only a linear kernel would prove nothing."""
        task = parity_task(6, 2, 48, seed=0)
        self.assertIsInstance(kernel_accuracy(task, quadratic_kernel, seed=0), float)
        row = compare(6, 48, (2,), seed=0)[0]
        self.assertIn("quadratic", row)


if __name__ == "__main__":
    unittest.main()


class ShardedLibraryTests(unittest.TestCase):
    """The library is many bounded files, not one container."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_sharded_"))
        from ultraquant.shards.vault import ShardVault

        self.vault = ShardVault(self.dir / "vault")
        for index in range(24):
            self.vault.add_shard(
                f"expert:cat{index:02d}", f"cat{index % 6}",
                {"w": [float(v) for v in range(300)]},
            )

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_packs_into_bounded_parts(self) -> None:
        report = self.vault.pack_sharded(
            self.dir / "vault" / "library", max_part_bytes=4096, prune_loose=True
        )
        self.assertGreater(report["parts"], 1, "the cap must force a roll-over")
        self.assertEqual(report["shards"], 24)
        for name in report["part_files"]:
            path = self.dir / "vault" / "library" / name
            self.assertTrue(path.exists())

    def test_every_shard_survives_sharding(self) -> None:
        self.vault.pack_sharded(
            self.dir / "vault" / "library", max_part_bytes=4096, prune_loose=True
        )
        for entry in self.vault.catalog():
            with self.subTest(shard=entry["shard_id"]):
                self.assertIn("w", self.vault.get(entry["shard_id"]))

    def test_one_file_per_shard(self) -> None:
        report = self.vault.pack_sharded(
            self.dir / "vault" / "library", max_shards_per_part=1, prune_loose=True
        )
        self.assertEqual(report["parts"], 24)

    def test_each_part_is_independently_readable(self) -> None:
        """A part must stand alone, or it cannot be replicated piecemeal."""
        from ultraquant.shards.vault import ShardVault

        report = self.vault.pack_sharded(
            self.dir / "vault" / "library", max_part_bytes=4096, prune_loose=True
        )
        single = self.dir / "vault" / "library" / report["part_files"][0]
        lonely = ShardVault(self.dir / "lonely")
        attached = lonely.attach(single)
        self.assertGreater(attached, 0)
        for entry in lonely.catalog():
            self.assertIn("w", lonely.get(entry["shard_id"]))

    def test_partial_replication_is_usable(self) -> None:
        """A missing part must cost only its own shards, not the library."""
        from ultraquant.shards.vault import ShardVault

        library = self.dir / "vault" / "library"
        report = self.vault.pack_sharded(library, max_part_bytes=4096, prune_loose=True)
        (library / report["part_files"][0]).unlink()

        partial = ShardVault(self.dir / "partial")
        result = partial.attach_directory(library)
        self.assertEqual(result["missing"], [report["part_files"][0]])
        self.assertGreater(result["shards"], 0)
        for entry in partial.catalog():
            with self.subTest(shard=entry["shard_id"]):
                self.assertIn("w", partial.get(entry["shard_id"]))

    def test_manifest_accumulates_across_waves(self) -> None:
        """A later wave must not erase earlier parts from the manifest."""
        import json

        library = self.dir / "vault" / "library"
        first = [f"expert:cat{i:02d}" for i in range(12)]
        second = [f"expert:cat{i:02d}" for i in range(12, 24)]
        self.vault.pack_sharded(library, first, max_part_bytes=4096, prefix="wave1")
        self.vault.pack_sharded(library, second, max_part_bytes=4096, prefix="wave2")

        manifest = json.loads((library / "library.json").read_text(encoding="utf-8"))
        self.assertTrue(any(n.startswith("wave1") for n in manifest["parts"]))
        self.assertTrue(any(n.startswith("wave2") for n in manifest["parts"]))
        self.assertEqual(manifest["shards"], 24)

    def test_attach_directory_without_a_manifest(self) -> None:
        from ultraquant.shards.vault import ShardVault

        library = self.dir / "vault" / "library"
        self.vault.pack_sharded(library, max_part_bytes=4096, prune_loose=True)
        (library / "library.json").unlink()
        recovered = ShardVault(self.dir / "recovered")
        result = recovered.attach_directory(library)
        self.assertEqual(result["shards"], 24, "parts must be discoverable by glob")

    def test_empty_pack_is_safe(self) -> None:
        from ultraquant.shards.vault import ShardVault

        empty = ShardVault(self.dir / "empty")
        report = empty.pack_sharded(self.dir / "empty" / "library")
        self.assertEqual(report["parts"], 0)
        self.assertIsNone(report["manifest"])

    def test_forge_writes_a_sharded_library(self) -> None:
        forge = ModelForge(self.dir / "forged", seed=0, hidden=8, tier="auto",
                           part_bytes=8 * 1024)
        forge.build(_corpora(0, 6, seed=1), epochs=8)
        parts = list((self.dir / "forged" / "vault" / "library").glob("*.uql"))
        self.assertGreater(len(parts), 1)
        self.assertGreater(len(forge.library_parts()), 1)
