"""RAM in front, library on storage: the tier that makes a huge model usable.

The premise of the whole design is that the shard library *is* the model, and it
stays where it is durable — NVMe, SAN, Ceph, plain disk. Loading it whole would
defeat the point. So a bounded slice of it lives in RAM, and everything else is a
byte-range read away.

:class:`TieredStorage` is a read-through cache at the *bytes* level, sitting
between the vault and the durable backend:

* a read that hits RAM never touches storage;
* a miss is served from the cold tier and admitted to RAM;
* admission is bounded by a byte budget, and eviction is LRU;
* shards can be **pinned**, which is how the pattern-recognition working set
  keeps what it needs resident regardless of eviction pressure;
* writes go through to the cold tier immediately — RAM is a cache, never the
  book of record. Nothing is ever only in memory.

The counters answer the question that matters: how much of the model did we
actually have to read, and how little of it was resident while doing so.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from ultraquant.storage.base import ShardStorage, StorageCapabilities, StorageError
from ultraquant.storage.ram import RamStorage, available_ram

__all__ = ["TieredStorage", "suggested_ram_budget"]


def suggested_ram_budget(fraction: float = 0.25, floor: int = 32 * 1024 * 1024) -> int:
    """A RAM budget derived from what the machine currently has free.

    Args:
        fraction: Share of *available* (not total) memory to claim.
        floor: Minimum budget, used when memory cannot be measured.

    Returns:
        A byte budget.
    """
    free = available_ram()
    if free <= 0:
        return floor
    return max(floor, int(free * fraction))


class TieredStorage(ShardStorage):
    """A RAM hot tier over a durable cold tier."""

    def __init__(
        self,
        cold: ShardStorage,
        max_bytes: int | None = None,
        name: str = "tiered",
        admit_larger_than_budget: bool = False,
        block_size: int = 1024 * 1024,
    ) -> None:
        """Wrap ``cold`` with a bounded RAM cache.

        Args:
            cold: The durable backend holding the library.
            max_bytes: RAM budget; defaults to :func:`suggested_ram_budget`.
            name: Label for the URI.
            admit_larger_than_budget: Whether a single object bigger than the
                whole budget may be cached (evicting everything else). Off by
                default, so one outsized shard cannot flush the working set.
            block_size: Objects larger than this are cached in blocks of this
                size rather than whole. This is what makes the tier work at all
                on a real library: a packed ``.uql`` can be hundreds of
                gigabytes, and caching it whole on first touch would be exactly
                the "load the model into RAM" the design exists to avoid. Only
                the blocks a shard actually spans are ever resident.
        """
        super().__init__()
        self.cold = cold
        self.hot = RamStorage(name=f"{name}-hot")
        self.max_bytes = suggested_ram_budget() if max_bytes is None else int(max_bytes)
        self.name = name
        self.admit_larger_than_budget = bool(admit_larger_than_budget)

        alignment = max(1, cold.capabilities.alignment)
        self.block_size = max(alignment, (int(block_size) // alignment) * alignment)

        self._order: OrderedDict[str, int] = OrderedDict()   # cache key -> bytes, LRU first
        self._pinned: dict[str, int] = {}
        self._sizes: dict[str, int] = {}
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.cold_bytes_read = 0

    def _size_of(self, key: str) -> int:
        """Object size, remembered so repeat reads need no extra stat."""
        with self._lock:
            size = self._sizes.get(key)
        if size is None:
            size = self.cold.size(key)
            with self._lock:
                self._sizes[key] = size
        return size

    def _fetch_block(self, key: str, index: int, size: int, pin: bool) -> bytes:
        """Return one block of ``key``, reading it from cold storage on a miss."""
        cache_key = f"{key}#{index}"
        with self._lock:
            resident = cache_key in self._order or cache_key in self._pinned
            if resident and cache_key in self._order:
                self._order.move_to_end(cache_key)
        if resident:
            self.hits += 1
            block = self.hot.read_all(cache_key)
            if pin:
                self._promote_to_pinned(cache_key, len(block))
            return block

        self.misses += 1
        start = index * self.block_size
        span = min(self.block_size, size - start)
        block = self.cold.read_range(key, start, span)
        self.cold_bytes_read += len(block)
        self._admit(cache_key, block, pin=pin)
        return block

    def _promote_to_pinned(self, cache_key: str, size: int) -> None:
        """Move an already-cached entry into the pinned set."""
        with self._lock:
            self._order.pop(cache_key, None)
            self._pinned[cache_key] = size

    # -- identity ----------------------------------------------------------

    @property
    def uri(self) -> str:
        """The URI this backend was opened from."""
        return f"tiered://{self.name}?cold={self.cold.uri}"

    @property
    def capabilities(self) -> StorageCapabilities:
        """The cold tier's capabilities; RAM only changes latency, not semantics."""
        cold = self.cold.capabilities
        return StorageCapabilities(
            ranged_reads=cold.ranged_reads,
            alignment=cold.alignment,
            queue_depth=cold.queue_depth,
            remote=cold.remote,
            volatile=False,            # the library of record is durable
            writable=cold.writable,
            medium=f"ram-over-{cold.medium}",
        )

    # -- residency ---------------------------------------------------------

    @property
    def resident_bytes(self) -> int:
        """Bytes currently held in RAM, pinned included."""
        with self._lock:
            return sum(self._order.values()) + sum(self._pinned.values())

    def resident(self) -> list[str]:
        """Cached keys, least recently used first; pinned keys last."""
        with self._lock:
            return list(self._order) + sorted(self._pinned)

    def is_resident(self, key: str) -> bool:
        """Whether ``key`` is served from RAM right now."""
        with self._lock:
            return key in self._order or key in self._pinned

    def _evict_to_budget(self) -> None:
        """Drop least-recently-used entries until inside the budget.

        Pinned entries are never evicted; if pins alone exceed the budget the
        cache simply holds them, because the caller has declared them essential.
        """
        with self._lock:
            pinned = sum(self._pinned.values())
            while self._order and pinned + sum(self._order.values()) > self.max_bytes:
                key, _size = self._order.popitem(last=False)
                self.hot.delete(key)
                self.evictions += 1

    def _admit(self, key: str, blob: bytes, pin: bool = False) -> None:
        """Insert ``blob`` into RAM, evicting as needed."""
        with self._lock:
            if key in self._pinned:
                return
            if not pin and not self.admit_larger_than_budget and len(blob) > self.max_bytes:
                return
            try:
                self.hot.write(key, blob)
            except StorageError:
                return
            if pin:
                self._pinned[key] = len(blob)
            else:
                self._order[key] = len(blob)
                self._order.move_to_end(key)
            self._evict_to_budget()

    # -- pinning -----------------------------------------------------------

    def pin(self, key: str) -> int:
        """Load ``key`` into RAM and hold it there.

        This is how a working set is made resident: the pattern-recognition
        prefetch pins the shards it expects to need, and ordinary traffic can no
        longer evict them.

        Returns:
            The number of bytes pinned.
        """
        size = self._size_of(key)
        if size > self.block_size:
            # Pinning a huge object means pinning every block it spans; callers
            # that only need one shard should use pin_range instead.
            return self.pin_range(key, 0, size)

        with self._lock:
            if key in self._pinned:
                return self._pinned[key]
            blob = self.hot.read_all(key) if key in self._order else None
        if blob is None:
            blob = self.cold.read_all(key)
            self.cold_bytes_read += len(blob)
            self.hot.write(key, blob)
        with self._lock:
            self._order.pop(key, None)
            self._pinned[key] = len(blob)
            self._evict_to_budget()
        return len(blob)

    def unpin(self, key: str) -> None:
        """Release a pin; the key becomes ordinary cache content."""
        with self._lock:
            size = self._pinned.pop(key, None)
            if size is None:
                return
            self._order[key] = size
            self._order.move_to_end(key)
            self._evict_to_budget()

    def unpin_all(self) -> None:
        """Release every pin."""
        with self._lock:
            for key in list(self._pinned):
                self.unpin(key)

    def pinned(self) -> list[str]:
        """Currently pinned keys, sorted."""
        with self._lock:
            return sorted(self._pinned)

    # -- reads -------------------------------------------------------------

    def read_range(self, key: str, offset: int, length: int) -> bytes:
        """Read a range, from RAM when possible.

        Objects up to ``block_size`` are cached whole; larger ones are cached in
        blocks, so reading one shard out of a huge packed library brings in only
        the blocks that shard spans.
        """
        if length <= 0:
            return b""
        size = self._size_of(key)

        if size <= self.block_size:
            with self._lock:
                resident = key in self._order or key in self._pinned
                if resident and key in self._order:
                    self._order.move_to_end(key)
            if resident:
                self.hits += 1
                self.stats.record_read(length, length)
                return self.hot.read_range(key, offset, length)
            self.misses += 1
            blob = self.cold.read_all(key)
            self.cold_bytes_read += len(blob)
            self._admit(key, blob)
            self.stats.record_read(length, len(blob))
            return blob[offset:offset + length]

        return self._read_blocks(key, offset, length, size, pin=False)

    def _read_blocks(
        self, key: str, offset: int, length: int, size: int, pin: bool
    ) -> bytes:
        """Assemble a range from block-sized cache entries."""
        end = min(offset + length, size)
        if end <= offset:
            return b""
        first = offset // self.block_size
        last = (end - 1) // self.block_size
        chunks = [
            self._fetch_block(key, index, size, pin) for index in range(first, last + 1)
        ]
        data = b"".join(chunks)
        head = offset - first * self.block_size
        self.stats.record_read(length, len(data))
        return data[head:head + (end - offset)]

    def read_all(self, key: str) -> bytes:
        """Read an object whole, from RAM when possible."""
        size = self._size_of(key)
        if size > self.block_size:
            return self._read_blocks(key, 0, size, size, pin=False)

        with self._lock:
            resident = key in self._order or key in self._pinned
            if resident and key in self._order:
                self._order.move_to_end(key)
        if resident:
            self.hits += 1
            blob = self.hot.read_all(key)
            self.stats.record_read(len(blob), len(blob))
            return blob

        self.misses += 1
        blob = self.cold.read_all(key)
        self.cold_bytes_read += len(blob)
        self._admit(key, blob)
        self.stats.record_read(len(blob), len(blob))
        return blob

    def pin_range(self, key: str, offset: int, length: int) -> int:
        """Pin exactly the blocks a byte range spans.

        This is what a working set actually needs: a shard is a range inside a
        library, so pinning "the shard" means pinning its blocks, not the
        library. Returns the bytes now pinned for this range.
        """
        size = self._size_of(key)
        if size <= self.block_size:
            return self.pin(key)
        end = min(offset + length, size)
        first, last = offset // self.block_size, (end - 1) // self.block_size
        total = 0
        for index in range(first, last + 1):
            total += len(self._fetch_block(key, index, size, pin=True))
        self._evict_to_budget()
        return total

    def unpin_range(self, key: str, offset: int, length: int) -> None:
        """Release pins on the blocks a byte range spans."""
        size = self._size_of(key)
        if size <= self.block_size:
            self.unpin(key)
            return
        end = min(offset + length, size)
        first, last = offset // self.block_size, (end - 1) // self.block_size
        for index in range(first, last + 1):
            self.unpin(f"{key}#{index}")

    def prefetch_ranges(
        self, ranges: list[tuple[str, int, int]], pin: bool = False
    ) -> dict:
        """Warm exactly the blocks a set of byte ranges spans.

        This is the prefetch the working set uses: each shard is ``(library,
        offset, length)``, so only its blocks are pulled — never the library.

        Returns:
            ``{"loaded": n, "already_resident": n, "bytes": n, "missing": [...]}``.
        """
        loaded = warm = total = 0
        missing: list[str] = []
        for key, offset, length in ranges:
            try:
                size = self._size_of(key)
            except StorageError:
                missing.append(key)
                continue
            first = offset // self.block_size
            last = (min(offset + length, size) - 1) // self.block_size
            was_resident = all(
                self.is_resident(f"{key}#{i}") if size > self.block_size
                else self.is_resident(key)
                for i in range(first, last + 1)
            )
            try:
                if pin:
                    total += self.pin_range(key, offset, length)
                else:
                    total += len(self.read_range(key, offset, length))
            except StorageError:
                missing.append(key)
                continue
            if was_resident:
                warm += 1
            else:
                loaded += 1
        return {
            "loaded": loaded,
            "already_resident": warm,
            "bytes": total,
            "missing": missing,
        }

    def prefetch(self, keys: list[str], pin: bool = False) -> dict:
        """Pull ``keys`` into RAM ahead of use.

        Args:
            keys: Keys to warm.
            pin: Hold them against eviction.

        Returns:
            ``{"loaded": n, "already_resident": n, "bytes": n, "missing": [...]}``.
        """
        loaded = warm = total = 0
        missing: list[str] = []
        for key in keys:
            if self.is_resident(key):
                warm += 1
                if pin:
                    self.pin(key)
                continue
            try:
                if pin:
                    total += self.pin(key)
                else:
                    total += len(self.read_all(key))
                loaded += 1
            except StorageError:
                missing.append(key)
        return {
            "loaded": loaded,
            "already_resident": warm,
            "bytes": total,
            "missing": missing,
        }

    # -- writes ------------------------------------------------------------

    def write(self, key: str, data: bytes) -> None:
        """Write through to the durable tier, then refresh RAM."""
        self.cold.write(key, data)
        self.stats.record_write(len(data))
        with self._lock:
            if key in self._pinned:
                self.hot.write(key, bytes(data))
                self._pinned[key] = len(data)
                return
            if key in self._order:
                self.hot.write(key, bytes(data))
                self._order[key] = len(data)
                self._order.move_to_end(key)
                self._evict_to_budget()

    def delete(self, key: str) -> None:
        """Delete from both tiers."""
        self.cold.delete(key)
        with self._lock:
            self._order.pop(key, None)
            self._pinned.pop(key, None)
        self.hot.delete(key)

    # -- inspection --------------------------------------------------------

    def exists(self, key: str) -> bool:
        """Whether ``key`` exists in either tier."""
        return self.is_resident(key) or self.cold.exists(key)

    def size(self, key: str) -> int:
        """Size of ``key`` in bytes."""
        with self._lock:
            if key in self._order:
                return self._order[key]
            if key in self._pinned:
                return self._pinned[key]
        return self.cold.size(key)

    def list_keys(self, prefix: str = "") -> list[str]:
        """Keys in the durable library (the book of record)."""
        return self.cold.list_keys(prefix)

    def set_budget(self, max_bytes: int) -> None:
        """Change the RAM budget, evicting immediately if it shrank."""
        with self._lock:
            self.max_bytes = int(max_bytes)
        self._evict_to_budget()

    def tier_stats(self) -> dict:
        """How well RAM is standing in for the library."""
        total = self.hits + self.misses
        library = 0
        try:
            library = self.cold.total_bytes()
        except StorageError:
            library = 0
        resident = self.resident_bytes
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "evictions": self.evictions,
            "budget_bytes": self.max_bytes,
            "resident_bytes": resident,
            "pinned_bytes": sum(self._pinned.values()),
            "pinned_keys": len(self._pinned),
            "cached_keys": len(self._order),
            "library_bytes": library,
            "resident_fraction": round(resident / library, 6) if library else 0.0,
            "cold_bytes_read": self.cold_bytes_read,
            "cold_medium": self.cold.capabilities.medium,
        }

    def describe(self) -> dict:
        """Summary of both tiers."""
        info = super().describe()
        info["tier"] = self.tier_stats()
        info["cold"] = self.cold.describe()
        return info

    def close(self) -> None:
        """Drop the RAM tier and close the cold one."""
        self.hot.clear()
        with self._lock:
            self._order.clear()
            self._pinned.clear()
        self.cold.close()
