"""Tests for pluggable storage, the RAM tier, the paged index, and the working set."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.shards.vault import ShardVault
from ultraquant.shards.working_set import PatternWorkingSet
from ultraquant.storage import (
    BlockDeviceStorage,
    LocalStorage,
    RamStorage,
    ShardIndexReader,
    ShardIndexWriter,
    StorageError,
    index_footprint,
    open_storage,
)
from ultraquant.storage.blockdev import PROFILES, sector_size
from ultraquant.storage.registry import parse_size
from ultraquant.storage.tiered import TieredStorage


class StorageContractTests(unittest.TestCase):
    """Every backend honours the same contract."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_stor_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def backends(self):
        """One instance of each locally testable backend."""
        yield LocalStorage(self.dir / "local")
        yield RamStorage(name="unit")
        yield BlockDeviceStorage(self.dir / "blk", profile="generic")
        yield BlockDeviceStorage(self.dir / "nvme", profile="nvmeof")

    def test_round_trip_and_ranges(self) -> None:
        payload = bytes(range(256)) * 40      # 10 240 bytes
        for storage in self.backends():
            with self.subTest(backend=type(storage).__name__, uri=storage.uri):
                storage.write("a/b.uqs", payload)
                self.assertTrue(storage.exists("a/b.uqs"))
                self.assertEqual(storage.size("a/b.uqs"), len(payload))
                self.assertEqual(storage.read_all("a/b.uqs"), payload)
                self.assertEqual(storage.read_range("a/b.uqs", 0, 16), payload[:16])
                self.assertEqual(storage.read_range("a/b.uqs", 1000, 500),
                                 payload[1000:1500])
                self.assertEqual(storage.read_range("a/b.uqs", 5, 0), b"")
                self.assertIn("a/b.uqs", storage.list_keys())
                storage.delete("a/b.uqs")
                self.assertFalse(storage.exists("a/b.uqs"))
                storage.delete("a/b.uqs")   # deleting twice is not an error

    def test_unaligned_ranges_are_exact(self) -> None:
        """Direct I/O widens reads to sectors; the returned bytes must not widen."""
        payload = bytes((i * 7) % 251 for i in range(20_000))
        storage = BlockDeviceStorage(self.dir / "aligned", profile="nvmeof")
        storage.write("m.uql", payload)
        for offset, length in ((0, 1), (1, 1), (513, 1000), (4095, 4098), (19_999, 1)):
            with self.subTest(offset=offset, length=length):
                self.assertEqual(
                    storage.read_range("m.uql", offset, length),
                    payload[offset:offset + length],
                )

    def test_missing_object_raises(self) -> None:
        for storage in self.backends():
            with self.subTest(backend=type(storage).__name__):
                with self.assertRaises(StorageError):
                    storage.read_all("nope.uqs")

    def test_key_cannot_escape_the_root(self) -> None:
        storage = LocalStorage(self.dir / "jail")
        with self.assertRaises(StorageError):
            storage.write("../escape.uqs", b"x")

    def test_capabilities_are_declared(self) -> None:
        self.assertTrue(RamStorage().capabilities.volatile)
        self.assertFalse(LocalStorage(self.dir / "c").capabilities.volatile)
        self.assertTrue(BlockDeviceStorage(self.dir / "d", profile="pure").capabilities.remote)
        self.assertEqual(
            BlockDeviceStorage(self.dir / "e", profile="lightbits").capabilities.medium,
            "lightbits-nvme-tcp",
        )

    def test_every_profile_is_usable(self) -> None:
        for profile in PROFILES:
            with self.subTest(profile=profile):
                storage = BlockDeviceStorage(self.dir / profile, profile=profile)
                storage.write("x.uqs", b"hello")
                self.assertEqual(storage.read_range("x.uqs", 1, 3), b"ell")
                self.assertIn("profile", storage.describe())

    def test_unknown_profile_rejected(self) -> None:
        with self.assertRaises(StorageError):
            BlockDeviceStorage(self.dir / "bad", profile="not-a-vendor")

    def test_sector_size_is_sane(self) -> None:
        size = sector_size(self.dir)
        self.assertGreaterEqual(size, 512)
        self.assertEqual(size & (size - 1), 0, "sector size must be a power of two")

    def test_stats_track_amplification(self) -> None:
        storage = LocalStorage(self.dir / "stats")
        storage.write("k", b"x" * 1000)
        storage.read_range("k", 0, 100)
        self.assertEqual(storage.stats.reads, 1)
        self.assertEqual(storage.stats.bytes_requested, 100)
        self.assertAlmostEqual(storage.stats.amplification, 1.0)


class RegistryTests(unittest.TestCase):
    """URIs resolve to the right backend."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_uri_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_schemes(self) -> None:
        cases = {
            f"local://{(self.dir / 'l').as_posix()}": LocalStorage,
            "ram://scratch": RamStorage,
            f"blockdev://{(self.dir / 'b').as_posix()}": BlockDeviceStorage,
            f"nvmeof://{(self.dir / 'n').as_posix()}": BlockDeviceStorage,
            f"lightbits://{(self.dir / 'li').as_posix()}": BlockDeviceStorage,
            f"pure://{(self.dir / 'p').as_posix()}": BlockDeviceStorage,
            f"3par://{(self.dir / 't').as_posix()}": BlockDeviceStorage,
            f"cephfs://{(self.dir / 'c').as_posix()}": LocalStorage,   # CephFS subclasses it
        }
        for uri, expected in cases.items():
            with self.subTest(uri=uri):
                self.assertIsInstance(open_storage(uri), expected)

    def test_bare_path_is_local(self) -> None:
        self.assertIsInstance(open_storage(str(self.dir / "bare")), LocalStorage)

    def test_unknown_scheme_rejected(self) -> None:
        with self.assertRaises(StorageError):
            open_storage("floppy:///a/b")

    def test_cache_wraps_in_a_tier(self) -> None:
        storage = open_storage(f"local://{(self.dir / 'cached').as_posix()}", cache="1MB")
        self.assertIsInstance(storage, TieredStorage)
        self.assertEqual(storage.max_bytes, 1024 * 1024)
        self.assertFalse(storage.capabilities.volatile, "the library must stay durable")

    def test_query_options_apply(self) -> None:
        storage = open_storage(
            f"nvmeof://{(self.dir / 'q').as_posix()}?queue_depth=64&direct=0"
        )
        self.assertEqual(storage.queue_depth, 64)
        self.assertFalse(storage.direct)

    def test_parse_size(self) -> None:
        self.assertEqual(parse_size("2GB"), 2 * 1024 ** 3)
        self.assertEqual(parse_size("512MiB"), 512 * 1024 ** 2)
        self.assertEqual(parse_size(4096), 4096)
        self.assertIsNone(parse_size(None))
        self.assertIsNone(parse_size("off"))
        self.assertGreater(parse_size("auto"), 0)
        with self.assertRaises(ValueError):
            parse_size("banana")


class TieredStorageTests(unittest.TestCase):
    """RAM caches, but the library of record stays cold."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_tier_"))
        self.cold = LocalStorage(self.dir / "cold")

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_read_through_and_hit(self) -> None:
        self.cold.write("a", b"x" * 500)
        tier = TieredStorage(self.cold, max_bytes=64 * 1024)
        self.assertEqual(tier.read_all("a"), b"x" * 500)
        self.assertEqual((tier.hits, tier.misses), (0, 1))
        self.assertEqual(tier.read_all("a"), b"x" * 500)
        self.assertEqual((tier.hits, tier.misses), (1, 1))

    def test_writes_go_through_to_the_durable_tier(self) -> None:
        tier = TieredStorage(self.cold, max_bytes=64 * 1024)
        tier.write("b", b"durable")
        self.assertEqual(self.cold.read_all("b"), b"durable",
                         "RAM must never be the only copy")

    def test_lru_eviction_respects_the_budget(self) -> None:
        for name in "abcd":
            self.cold.write(name, bytes(400))
        tier = TieredStorage(self.cold, max_bytes=1000)
        for name in "abcd":
            tier.read_all(name)
        self.assertLessEqual(tier.resident_bytes, 1000)
        self.assertGreater(tier.evictions, 0)

    def test_pinned_entries_survive_pressure(self) -> None:
        for name in "abcdef":
            self.cold.write(name, bytes(400))
        tier = TieredStorage(self.cold, max_bytes=1200)
        tier.pin("a")
        for name in "bcdef":
            tier.read_all(name)
        self.assertIn("a", tier.pinned())
        self.assertTrue(tier.is_resident("a"))
        tier.unpin("a")
        self.assertEqual(tier.pinned(), [])

    def test_large_objects_are_cached_by_block_not_whole(self) -> None:
        """The point of the design: one shard must not drag in the library."""
        library = bytes((i * 13) % 251 for i in range(4096)) * 64     # 256 KB
        self.cold.write("model.uql", library)
        tier = TieredStorage(self.cold, max_bytes=1024 * 1024, block_size=16 * 1024)

        chunk = tier.read_range("model.uql", 70_000, 100)
        self.assertEqual(chunk, library[70_000:70_100])
        self.assertLessEqual(
            tier.cold_bytes_read, 32 * 1024,
            "reading 100 bytes must not pull the whole library into RAM",
        )
        self.assertLess(tier.resident_bytes, len(library))

    def test_block_reads_spanning_boundaries_are_exact(self) -> None:
        library = bytes((i * 7) % 251 for i in range(50_000))
        self.cold.write("m.uql", library)
        tier = TieredStorage(self.cold, max_bytes=1024 * 1024, block_size=4096)
        for offset, length in ((0, 10), (4090, 20), (4096, 4096), (49_990, 10), (100, 20_000)):
            with self.subTest(offset=offset, length=length):
                self.assertEqual(
                    tier.read_range("m.uql", offset, length),
                    library[offset:offset + length],
                )

    def test_pin_range_pins_only_the_span(self) -> None:
        library = bytes(200_000)
        self.cold.write("m.uql", library)
        tier = TieredStorage(self.cold, max_bytes=1024 * 1024, block_size=8192)
        pinned = tier.pin_range("m.uql", 100_000, 1000)
        self.assertLessEqual(pinned, 16 * 1024)
        self.assertLess(tier.resident_bytes, len(library))
        tier.unpin_range("m.uql", 100_000, 1000)
        self.assertEqual(tier.pinned(), [])

    def test_budget_shrink_evicts(self) -> None:
        for name in "abcd":
            self.cold.write(name, bytes(400))
        tier = TieredStorage(self.cold, max_bytes=4000)
        for name in "abcd":
            tier.read_all(name)
        tier.set_budget(500)
        self.assertLessEqual(tier.resident_bytes, 500)

    def test_tier_stats_shape(self) -> None:
        self.cold.write("a", b"x" * 100)
        tier = TieredStorage(self.cold, max_bytes=8192)
        tier.read_all("a")
        stats = tier.tier_stats()
        for key in ("hits", "misses", "hit_rate", "resident_bytes",
                    "library_bytes", "resident_fraction", "cold_medium"):
            self.assertIn(key, stats)
        self.assertEqual(stats["cold_medium"], "local-fs")


class PagedIndexTests(unittest.TestCase):
    """The index stays small in RAM however many shards it describes."""

    def _index(self, count: int, stride: int = 64):
        writer = ShardIndexWriter(stride=stride)
        for i in range(count):
            writer.add(f"slice:{i:06d}", i * 1000, 1000)
        store = RamStorage()
        store.write("ix", writer.build())
        return ShardIndexReader(store, "ix")

    def test_lookup_every_record(self) -> None:
        reader = self._index(500)
        for i in range(500):
            record = reader.lookup(f"slice:{i:06d}")
            self.assertIsNotNone(record, f"slice:{i:06d} missing")
            self.assertEqual(record.offset, i * 1000)
            self.assertEqual(record.length, 1000)

    def test_absent_key_returns_none(self) -> None:
        reader = self._index(200)
        self.assertIsNone(reader.lookup("slice:999999"))
        self.assertIsNone(reader.lookup(""))

    def test_empty_index(self) -> None:
        store = RamStorage()
        store.write("ix", ShardIndexWriter().build())
        self.assertIsNone(ShardIndexReader(store, "ix").lookup("anything"))

    def test_resident_cost_is_bounded_by_stride_not_count(self) -> None:
        small = self._index(500, stride=64)
        large = self._index(5000, stride=640)
        self.assertLessEqual(
            large.resident_bytes, small.resident_bytes * 1.5,
            "a 10x bigger index with a 10x stride must not cost 10x the RAM",
        )

    def test_probe_count_is_bounded(self) -> None:
        reader = self._index(4000, stride=128)
        reader.probes = 0
        for i in range(0, 4000, 97):
            reader.lookup(f"slice:{i:06d}")
        lookups = len(range(0, 4000, 97))
        self.assertLess(reader.probes / lookups, 10,
                        "fence must keep lookups to a handful of reads")

    def test_iter_records_streams_everything(self) -> None:
        reader = self._index(300)
        names = {r.name for r in reader.iter_records(batch=32)}
        self.assertEqual(len(names), 300)

    def test_name_too_long_is_rejected(self) -> None:
        writer = ShardIndexWriter()
        with self.assertRaises(ValueError):
            writer.add("x" * 200, 0, 1)

    def test_not_an_index(self) -> None:
        store = RamStorage()
        store.write("junk", b"not an index at all, really not")
        with self.assertRaises(StorageError):
            ShardIndexReader(store, "junk")

    def test_footprint_projection_matches_reality(self) -> None:
        reader = self._index(1000, stride=100)
        predicted = index_footprint(1000, 100)
        self.assertEqual(predicted["fence_entries"], reader.stats()["fence_entries"])
        self.assertEqual(predicted["disk_bytes"], reader.stats()["index_bytes"])

    def test_trillion_parameter_projection_stays_small(self) -> None:
        """1.2T parameters must not need gigabytes of resident index."""
        shards = 1_200_000
        footprint = index_footprint(shards, stride=1024)
        self.assertLess(footprint["ram_bytes"], 1024 * 1024,
                        "resident index for 1.2T params must stay under 1 MB")
        self.assertLess(footprint["probes"], 16)


class VaultOnStorageTests(unittest.TestCase):
    """The vault reads shard bytes through whatever storage it was given."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_vs_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _vault(self, storage=None):
        vault = ShardVault(self.dir / "vault", storage=storage)
        for i in range(8):
            vault.add_shard(f"expert:cat{i}", f"cat{i}",
                            {"labels": ["a"], "w": [float(x) for x in range(200)]})
        return vault

    def test_default_storage_unchanged(self) -> None:
        vault = self._vault()
        self.assertIn("w", vault.get("expert:cat0"))

    def test_reads_route_through_storage(self) -> None:
        cold = LocalStorage(self.dir / "vault")
        vault = self._vault(storage=cold)
        before = cold.stats.reads
        vault.get("expert:cat3")
        self.assertGreater(cold.stats.reads, before)

    def test_packed_shards_read_by_range_through_a_tier(self) -> None:
        cold = LocalStorage(self.dir / "vault")
        tier = TieredStorage(cold, max_bytes=256 * 1024, block_size=4096)
        vault = self._vault(storage=tier)
        vault.pack(self.dir / "vault" / "lib.uql", prune_loose=True)
        for i in range(8):
            self.assertIn("w", vault.get(f"expert:cat{i}"))
        self.assertGreater(tier.tier_stats()["hits"], 0)

    def test_storage_key_matches_location(self) -> None:
        vault = self._vault()
        self.assertTrue(vault.storage_key("expert:cat0").startswith("loose/"))
        vault.pack(self.dir / "vault" / "lib.uql", prune_loose=True)
        self.assertEqual(vault.storage_key("expert:cat0"), "lib.uql")


class WorkingSetTests(unittest.TestCase):
    """Pattern recognition decides what is worth holding in RAM."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_ws_"))
        from ultraquant.shards.router import CategoryRouter

        self.cold = LocalStorage(self.dir / "vault")
        self.tier = TieredStorage(self.cold, max_bytes=128 * 1024, block_size=8192)
        self.vault = ShardVault(self.dir / "vault", storage=self.tier)
        for name in ("arrows", "frames", "lines", "crosses"):
            self.vault.add_shard(
                f"expert:{name}", name,
                {"labels": [name], "w": [float(i) for i in range(300)]},
                associations={name: 1.0, name.rstrip("s"): 0.8},
            )
        self.router = CategoryRouter(self.vault)
        for name in ("arrows", "frames", "lines", "crosses"):
            self.router.register(name, [name, name.rstrip("s")])
        self.ws = PatternWorkingSet(self.vault, self.router, tier=self.tier)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_prediction_ranks_the_right_shard(self) -> None:
        predicted = self.ws.predict("show me an arrow")
        self.assertTrue(predicted)
        self.assertEqual(predicted[0].shard_id, "expert:arrows")

    def test_prefetch_makes_predictions_resident(self) -> None:
        report = self.ws.prefetch("an arrow pointing up", pin=True)
        self.assertGreater(report.loaded, 0)
        self.assertIn("expert:arrows", self.ws.pinned)
        self.assertTrue(self.tier.is_resident(self.vault.storage_key("expert:arrows")))
        self.assertIn("predicted", report.summary())

    def test_pinned_working_set_is_not_evicted(self) -> None:
        self.ws.pin_categories(["arrows"])
        for name in ("frames", "lines", "crosses"):
            self.vault.get(f"expert:{name}")
        self.assertTrue(self.tier.is_resident(self.vault.storage_key("expert:arrows")))

    def test_release_unpins(self) -> None:
        self.ws.pin_categories(["arrows", "frames"])
        self.assertTrue(self.ws.pinned)
        self.ws.release()
        self.assertEqual(self.ws.pinned, set())
        self.assertEqual(self.tier.pinned(), [])

    def test_labels_drive_prediction(self) -> None:
        predicted = self.ws.predict("", labels=["lines"])
        self.assertIn("expert:lines", [p.shard_id for p in predicted])

    def test_works_without_a_tier(self) -> None:
        plain = PatternWorkingSet(ShardVault(self.dir / "v2"), self.router, tier=None)
        report = plain.prefetch("arrow")
        self.assertEqual(report.loaded, 0)
        self.assertFalse(plain.stats()["tiered"])

    def test_stats_report_residency(self) -> None:
        self.ws.prefetch("arrow", pin=True)
        stats = self.ws.stats()
        self.assertTrue(stats["tiered"])
        self.assertGreaterEqual(stats["pinned_keys"], 1)
        self.assertIn("resident_fraction", stats)


if __name__ == "__main__":
    unittest.main()
