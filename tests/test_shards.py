"""Tests for the shard library layer: vault, byte-budget cache, and router.

Covers SPEC-INTERPRETER sections 1-3:

* :class:`ShardVault` — add/get round trips with sha256 integrity, loose-file
  tampering detection, ``.uql`` pack/attach with on-demand chunked reads,
  ``prune_loose`` verification, and re-add preserving learned state.
* :class:`ShardCache` — LRU eviction under a byte budget, the oversize-single
  case, immediate ``set_budget`` eviction, and accounting.
* :class:`CategoryRouter` — associative routing, learned tie-breaking, and
  vault-association contributions.

Deterministic and self-contained; every test uses a fresh temp directory.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.shards.budget import ShardCache
from ultraquant.shards.router import CategoryRouter
from ultraquant.shards.vault import ShardIntegrityError, ShardVault


# --------------------------------------------------------------------------- #
# ShardVault
# --------------------------------------------------------------------------- #


class TestShardVault(unittest.TestCase):
    """Container format, integrity, on-demand reads, and learned-state survival."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="uq_vault_"))
        self.vault = ShardVault(self.tmp / "vault")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _payload(self, seed: int) -> dict:
        return {"weights": [seed, seed + 1, seed + 2], "label": f"shard-{seed}"}

    def test_add_get_round_trip_and_sha(self) -> None:
        entry = self.vault.add_shard("s1", "math", self._payload(1))
        self.assertEqual(entry["location"], "loose")
        self.assertEqual(entry["codec"], "zlib")
        self.assertEqual(entry["access_count"], 0)
        # sha256 in the entry matches the actual stored bytes.
        import hashlib

        stored = self.vault.load_bytes("s1")
        self.assertEqual(hashlib.sha256(stored).hexdigest(), entry["sha256"])
        # Round trip returns the exact payload and counts the access.
        self.assertEqual(self.vault.get("s1"), self._payload(1))
        self.assertEqual(self.vault.entry("s1")["access_count"], 1)

    def test_catalog_persists_and_reloads(self) -> None:
        self.vault.add_shard("s1", "math", self._payload(1))
        self.vault.add_shard("s2", "bio", self._payload(2))
        reopened = ShardVault(self.tmp / "vault")  # auto-loads catalog.json
        self.assertEqual(len(reopened.catalog()), 2)
        self.assertEqual(reopened.get("s2"), self._payload(2))

    def test_tampered_loose_file_raises_integrity_error(self) -> None:
        self.vault.add_shard("s1", "math", self._payload(1))
        loose = self.vault._loose_path("s1")
        loose.write_bytes(loose.read_bytes() + b"\x00")  # corrupt the bytes
        with self.assertRaises(ShardIntegrityError):
            self.vault.get("s1")

    def test_pack_then_attach_fresh_vault_chunked_get(self) -> None:
        payloads = {sid: self._payload(i) for i, sid in enumerate(("a", "b", "c"))}
        for sid, payload in payloads.items():
            self.vault.add_shard(sid, "math", payload)
        lib = self.tmp / "libs" / "lib.uql"
        packed = self.vault.pack(lib)
        self.assertEqual(packed, 3)

        # Low-level container format: magic + 8-byte BE index length + JSON index.
        raw = lib.read_bytes()
        self.assertEqual(raw[:4], b"UQL1")
        index_len = int.from_bytes(raw[4:12], "big")
        index = json.loads(raw[12 : 12 + index_len].decode("utf-8"))
        self.assertEqual([e["shard_id"] for e in index], ["a", "b", "c"])
        # First payload sits immediately after magic+length+index (absolute offset).
        self.assertEqual(index[0]["offset"], 12 + index_len)
        for e in index:  # each recorded chunk hashes to its recorded sha
            import hashlib

            chunk = raw[e["offset"] : e["offset"] + e["length"]]
            self.assertEqual(hashlib.sha256(chunk).hexdigest(), e["sha256"])

        # A FRESH vault attaches only the index, then reads chunks on demand.
        fresh = ShardVault(self.tmp / "fresh")
        attached = fresh.attach(lib)
        self.assertEqual(attached, 3)
        for sid, payload in payloads.items():
            self.assertEqual(fresh.entry(sid)["location"], "library")
            self.assertEqual(fresh.get(sid), payload)
        # Freshly attached shards start with clean learned state.
        self.assertEqual(fresh.entry("a")["access_count"], 1)  # +1 from get above
        self.assertEqual(fresh.entry("a")["associations"], {})

    def test_prune_loose_leaves_library_reads_working(self) -> None:
        for i, sid in enumerate(("a", "b")):
            self.vault.add_shard(sid, "math", self._payload(i))
        loose_a = self.vault._loose_path("a")
        self.assertTrue(loose_a.exists())
        lib = self.tmp / "lib.uql"
        self.vault.pack(lib, prune_loose=True)
        self.assertFalse(loose_a.exists())  # loose file removed after verified pack
        # Library-only reads still work.
        self.assertEqual(self.vault.get("a"), self._payload(0))
        self.assertEqual(self.vault.get("b"), self._payload(1))

    def test_readd_preserves_associations_and_access_count(self) -> None:
        self.vault.add_shard("s1", "math", self._payload(1), associations={"alpha": 2.0})
        self.vault.reinforce("s1", ["beta"], 0.5)
        self.vault.get("s1")  # access_count -> 1
        self.vault.get("s1")  # access_count -> 2
        before = self.vault.entry("s1")
        self.assertEqual(before["access_count"], 2)
        self.assertEqual(before["associations"], {"alpha": 2.0, "beta": 0.5})

        # Re-add with brand-new bytes and no associations argument.
        self.vault.add_shard("s1", "math", {"weights": [999]})
        after = self.vault.entry("s1")
        self.assertEqual(after["access_count"], 2)  # preserved
        self.assertEqual(after["associations"], {"alpha": 2.0, "beta": 0.5})  # preserved
        self.assertEqual(self.vault.get("s1"), {"weights": [999]})  # new bytes

    def test_readd_preserves_the_routing_signature(self) -> None:
        """Re-training must not erase what the category looks like.

        ``associations`` were preserved from the start; ``signature`` was added
        later and was not, so every re-train silently dropped it. The category
        stayed trained and became unroutable — patterns could no longer find the
        expert that had just learned them.
        """
        self.vault.add_shard("s1", "math", self._payload(1))
        self.vault.set_signature("s1", [[0.5] * 4, [0.25] * 4])
        self.assertEqual(len(self.vault.entry("s1")["signature"]), 2)

        self.vault.add_shard("s1", "math", {"weights": [999]})
        after = self.vault.entry("s1")["signature"]
        self.assertEqual(after, [[0.5] * 4, [0.25] * 4])
        self.assertEqual(self.vault.signatures()["math"], after)

    def test_attaching_a_library_keeps_signatures_and_lookups(self) -> None:
        """Relocating a shard's bytes must not un-learn its category.

        ``attach`` rebuilds catalog entries from the library index, and the
        index did not carry signatures at all — so a packed library was
        unroutable by pattern on any machine that had not trained it: every
        expert present, correct, and unreachable. It also left the derived
        lookups stale, hiding newly attached shards from keyword routing too.
        """
        self.vault.add_shard("s1", "math", self._payload(1), associations={"a": 1.0})
        self.vault.set_signature("s1", [[0.5] * 4])
        library = self.tmp / "lib.uql"
        self.vault.pack(library, ["s1"], prune_loose=True)

        fresh = ShardVault(self.tmp / "other")
        fresh.attach(library)
        self.assertEqual(fresh.entry("s1")["location"], "library")
        self.assertEqual(fresh.signatures()["math"], [[0.5] * 4])
        self.assertEqual(fresh.shards_in("math"), ["s1"],
                         "an attached shard must appear in the category lookup")

    def test_signature_storage_is_compact_but_looks_like_lists(self) -> None:
        """The packed representation is internal and must not leak.

        Signatures are held as ``array('d')`` because Python float objects were
        46 MB of a 74 MB resident catalog at 20,000 categories. Callers still
        get plain lists, and the values are bit-identical — the array is a
        memory choice, not a precision one.
        """
        self.vault.add_shard("s1", "math", self._payload(1))
        raw = [0.1234, 0.5, 0.9999, 0.0]
        self.vault.set_signature("s1", [raw])

        self.assertEqual(self.vault.entry("s1")["signature"], [raw])
        self.assertEqual(self.vault.signatures()["math"], [raw])
        self.assertIsInstance(self.vault.entry("s1")["signature"][0], list)

        reopened = ShardVault(self.tmp / "vault")
        self.assertEqual(reopened.signatures()["math"], [raw],
                         "a round trip through JSON must not shift the values")

    def test_reinforce_caps_at_five(self) -> None:
        self.vault.add_shard("s1", "math", self._payload(1))
        for _ in range(100):
            self.vault.reinforce("s1", ["x"], 0.1)
        self.assertAlmostEqual(self.vault.entry("s1")["associations"]["x"], 5.0)

    def test_stats_by_location(self) -> None:
        for i, sid in enumerate(("a", "b", "c")):
            self.vault.add_shard(sid, "math", self._payload(i))
        loose_stats = self.vault.stats()
        self.assertEqual(loose_stats["shards"], 3)
        self.assertGreater(loose_stats["loose_bytes"], 0)
        self.assertEqual(loose_stats["library_bytes"], 0)
        self.assertEqual(loose_stats["total_bytes"], loose_stats["loose_bytes"])
        self.assertEqual(loose_stats["libraries"], [])

        lib = self.tmp / "lib.uql"
        self.vault.pack(lib)
        packed_stats = self.vault.stats()
        self.assertEqual(packed_stats["shards"], 3)
        self.assertEqual(packed_stats["loose_bytes"], 0)  # all now library-located
        self.assertEqual(packed_stats["library_bytes"], loose_stats["loose_bytes"])
        self.assertEqual(packed_stats["libraries"], [str(Path(lib))])

    def test_load_bytes_reads_only_the_chunk(self) -> None:
        # Two shards packed together; reading one must not depend on the other.
        self.vault.add_shard("a", "math", {"data": "A" * 500})
        self.vault.add_shard("b", "math", {"data": "B" * 500})
        lib = self.tmp / "lib.uql"
        self.vault.pack(lib)
        entry_b = self.vault.entry("b")
        chunk = self.vault.load_bytes("b")
        self.assertEqual(len(chunk), entry_b["length"])  # exactly length bytes
        import hashlib

        self.assertEqual(hashlib.sha256(chunk).hexdigest(), entry_b["sha256"])


# --------------------------------------------------------------------------- #
# ShardCache
# --------------------------------------------------------------------------- #


class TestShardCache(unittest.TestCase):
    """LRU eviction under a byte budget with full accounting."""

    def setUp(self) -> None:
        self.loads: dict[str, int] = {}

    def _loader(self, shard_id: str, nbytes: int):
        """Return a loader that records how many times it materialised the shard."""

        def load() -> tuple[dict, int]:
            self.loads[shard_id] = self.loads.get(shard_id, 0) + 1
            return {"id": shard_id}, nbytes

        return load

    def test_lru_eviction_order_and_counts(self) -> None:
        cache = ShardCache(max_bytes=200)  # holds two 100-byte shards
        cache.get("A", self._loader("A", 100))
        cache.get("B", self._loader("B", 100))
        cache.get("C", self._loader("C", 100))  # evicts LRU A
        self.assertEqual(cache.resident(), ["B", "C"])
        self.assertEqual(cache.stats()["evictions"], 1)
        self.assertEqual(cache.current_bytes, 200)

        cache.get("A", self._loader("A", 100))  # reload A -> evicts LRU B
        self.assertEqual(cache.resident(), ["C", "A"])
        self.assertEqual(cache.stats()["evictions"], 2)
        self.assertEqual(cache.current_bytes, 200)

        stats = cache.stats()
        self.assertEqual(stats["misses"], 4)
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["peak_bytes"], 200)
        self.assertEqual(stats["budget_bytes"], 200)
        # A was loaded twice (evicted then requested again).
        self.assertEqual(self.loads["A"], 2)

    def test_hit_promotes_and_skips_loader(self) -> None:
        cache = ShardCache(max_bytes=300)
        cache.get("A", self._loader("A", 100))
        cache.get("B", self._loader("B", 100))
        # Hit on A: promoted to MRU, loader NOT called again.
        cache.get("A", self._loader("A", 100))
        self.assertEqual(cache.resident(), ["B", "A"])
        self.assertEqual(cache.stats()["hits"], 1)
        self.assertEqual(self.loads["A"], 1)

    def test_oversize_single_shard_allowed_alone(self) -> None:
        cache = ShardCache(max_bytes=200)
        cache.get("BIG", self._loader("BIG", 500))  # exceeds budget, resident alone
        self.assertEqual(cache.resident(), ["BIG"])
        self.assertEqual(cache.current_bytes, 500)
        self.assertEqual(cache.stats()["evictions"], 0)
        self.assertEqual(cache.stats()["peak_bytes"], 500)
        # A later small load evicts the oversize shard (LRU) to make room.
        cache.get("small", self._loader("small", 100))
        self.assertEqual(cache.resident(), ["small"])
        self.assertEqual(cache.current_bytes, 100)
        self.assertEqual(cache.stats()["evictions"], 1)

    def test_set_budget_shrink_evicts_immediately(self) -> None:
        cache = ShardCache(max_bytes=300)
        for sid in ("A", "B", "C"):
            cache.get(sid, self._loader(sid, 100))
        self.assertEqual(cache.resident(), ["A", "B", "C"])
        cache.set_budget(150)  # must evict down to fit immediately
        self.assertEqual(cache.resident(), ["C"])
        self.assertEqual(cache.current_bytes, 100)
        self.assertEqual(cache.stats()["evictions"], 2)
        # Shrinking below one item keeps the sole (oversize) shard resident.
        cache.set_budget(50)
        self.assertEqual(cache.resident(), ["C"])

    def test_invalidate_frees_bytes_without_eviction_count(self) -> None:
        cache = ShardCache(max_bytes=300)
        cache.get("A", self._loader("A", 100))
        cache.get("B", self._loader("B", 100))
        cache.invalidate("A")
        self.assertEqual(cache.resident(), ["B"])
        self.assertEqual(cache.current_bytes, 100)
        self.assertEqual(cache.stats()["evictions"], 0)
        cache.invalidate("missing")  # no-op, no error
        self.assertEqual(cache.current_bytes, 100)


# --------------------------------------------------------------------------- #
# CategoryRouter
# --------------------------------------------------------------------------- #


class TestCategoryRouter(unittest.TestCase):
    """Associative routing, learned tie-breaking, and vault-association scores."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="uq_router_"))
        self.vault = ShardVault(self.tmp / "vault")
        self.router = CategoryRouter(self.vault, path=self.tmp / "vault" / "router.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_route_ranks_registered_category(self) -> None:
        self.router.register("math", ["calculus", "integral", "derivative"])
        self.router.register("biology", ["cell", "dna", "protein"])
        ranked = self.router.route("please solve this integral")
        self.assertTrue(ranked)
        self.assertEqual(ranked[0][0], "math")

    def test_empty_and_no_match_return_empty(self) -> None:
        self.router.register("math", ["integral"])
        self.assertEqual(self.router.route(""), [])
        self.assertEqual(self.router.route("!!! ??? ..."), [])
        self.assertEqual(self.router.route("nothing relevant here"), [])

    def test_learn_flips_a_tie(self) -> None:
        self.router.register("alpha", ["shared"])
        self.router.register("beta", ["shared"])
        # Both match "shared" equally -> alphabetical tie-break puts alpha first.
        ranked = self.router.route("shared thing")
        self.assertEqual(ranked[0][0], "alpha")
        self.assertEqual(ranked[0][1], ranked[1][1])  # genuinely tied scores
        # Teaching beta the phrase flips the ranking deterministically.
        self.router.learn("shared thing", "beta", delta=0.5)
        flipped = self.router.route("shared thing")
        self.assertEqual(flipped[0][0], "beta")
        self.assertGreater(flipped[0][1], flipped[1][1])

    def test_vault_associations_contribute_to_score(self) -> None:
        # "derivative" is NOT a base keyword; it only reaches math via a shard.
        self.router.register("math", ["calculus"])
        self.router.register("biology", ["cell"])
        self.vault.add_shard(
            "expert:math", "math", {"net": [1]}, associations={"derivative": 2.0}
        )
        ranked = self.router.route("a derivative problem")
        self.assertEqual(ranked[0][0], "math")
        self.assertAlmostEqual(dict(ranked)["math"], 2.0)

    def test_learn_reinforces_vault_shards(self) -> None:
        self.router.register("math", ["calculus"])
        self.vault.add_shard("expert:math", "math", {"net": [1]})
        self.router.learn("integration by parts", "math", delta=0.1)
        assoc = self.vault.entry("expert:math")["associations"]
        # Every distinct token of the taught text is now an association.
        for token in ("integration", "by", "parts"):
            self.assertAlmostEqual(assoc[token], 0.1)

    def test_state_save_load_round_trip(self) -> None:
        self.router.register("math", ["calculus", "integral"])
        self.router.learn("integral problem", "math", delta=0.3)
        before = self.router.route("integral problem")
        self.router.save()

        reloaded = CategoryRouter(self.vault, path=self.tmp / "vault" / "router.json")
        after = reloaded.route("integral problem")
        self.assertEqual(before, after)
        self.assertEqual(reloaded.state(), self.router.state())

    def test_top_k_limits_results(self) -> None:
        for cat in ("a", "b", "c", "d"):
            self.router.register(cat, ["common"])
        ranked = self.router.route("common", top_k=2)
        self.assertEqual(len(ranked), 2)


if __name__ == "__main__":
    unittest.main()
