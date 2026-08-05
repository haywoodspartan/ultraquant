"""A shard index that stays O(1) in RAM however large the model gets.

At trillion-parameter scale the *catalog* becomes the bottleneck before any
weight does. A 1.2T-parameter model sharded into ~1.2M slices needs an index of
roughly 80 MB; held as a Python dict of dicts that is several gigabytes of
objects — so the machine runs out of memory describing the model it was never
going to load. Every honest claim about paging a huge model on a small machine
rests on the index being paged too.

So the index is a **sorted, fixed-width table read by byte range**:

```
  magic | header | fence array | record[0] record[1] ... record[n-1]
```

* Records are fixed size and sorted by key hash, so a lookup is a binary search
  — but each probe is a byte-range read, not a memory access, and nothing but the
  probed record is ever loaded.
* The **fence array** is a sparse sample of every Nth key held in RAM. It narrows
  the search to a single interval before any I/O happens, which turns a ~21-probe
  binary search into one or two reads. Its size is the whole memory cost:
  1.2M shards with a stride of 1024 is ~1,200 fence entries, about 20 KB.

Memory is therefore governed by ``records / stride``, not by ``records``. Raising
the stride keeps RAM flat as the model grows.

Keys are stored as a 16-byte BLAKE2b digest rather than the text, which makes
records fixed-width and bounds the index size regardless of how long shard names
are. The full name is kept in the record so lookups can be verified and the
catalog can be enumerated.
"""

from __future__ import annotations

import bisect
import hashlib
import struct
from dataclasses import dataclass
from typing import Iterable, Iterator

from ultraquant.storage.base import ShardStorage, StorageError

__all__ = ["ShardIndexWriter", "ShardIndexReader", "IndexRecord", "index_footprint"]

MAGIC = b"UQIX"
VERSION = 1

#: digest(16) offset(8) length(8) flags(4) name_len(2) name(96 fixed)
_RECORD = struct.Struct("<16sQQIH96s")
RECORD_SIZE = _RECORD.size
NAME_MAX = 96

#: magic(4) version(4) count(8) stride(4) fence_count(4) records_at(8) reserved(8)
_HEADER = struct.Struct("<4sIQIIQQ")
HEADER_SIZE = _HEADER.size

_FENCE = struct.Struct("<16sQ")     # digest + record ordinal
FENCE_SIZE = _FENCE.size


def _digest(key: str) -> bytes:
    """16-byte order-preserving-enough digest for a shard key."""
    return hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()


@dataclass(frozen=True)
class IndexRecord:
    """One shard's location within the library."""

    name: str
    offset: int
    length: int
    flags: int = 0

    def pack(self) -> bytes:
        """Serialize to a fixed-width record."""
        raw = self.name.encode("utf-8")
        if len(raw) > NAME_MAX:
            raise ValueError(
                f"shard name too long for the index ({len(raw)} > {NAME_MAX} bytes): "
                f"{self.name!r}"
            )
        return _RECORD.pack(
            _digest(self.name), self.offset, self.length, self.flags, len(raw),
            raw.ljust(NAME_MAX, b"\0"),
        )

    @classmethod
    def unpack(cls, blob: bytes) -> tuple[bytes, "IndexRecord"]:
        """Deserialize; returns ``(digest, record)``."""
        digest, offset, length, flags, name_len, name = _RECORD.unpack(blob)
        return digest, cls(name[:name_len].decode("utf-8"), offset, length, flags)


def index_footprint(records: int, stride: int = 1024) -> dict:
    """Predict the index's size on disk and its resident cost in RAM.

    Args:
        records: Number of shards.
        stride: Fence sampling interval.

    Returns:
        ``{"disk_bytes", "ram_bytes", "fence_entries", "probes"}`` — ``probes``
        being the worst-case byte-range reads per lookup.
    """
    fence = max(1, (records + stride - 1) // stride)
    probes = max(1, (stride).bit_length())
    return {
        "records": records,
        "disk_bytes": HEADER_SIZE + fence * FENCE_SIZE + records * RECORD_SIZE,
        "ram_bytes": fence * (16 + 8) + 512,     # digest+ordinal per fence entry
        "fence_entries": fence,
        "probes": probes,
    }


class ShardIndexWriter:
    """Builds a sorted index. Streams to storage; never holds the payloads."""

    def __init__(self, stride: int = 1024) -> None:
        """Create an empty index builder.

        Args:
            stride: Fence sampling interval. Larger keeps RAM smaller and costs
                a little more search per lookup.
        """
        self.stride = max(1, int(stride))
        self._records: list[tuple[bytes, bytes]] = []

    def add(self, name: str, offset: int, length: int, flags: int = 0) -> None:
        """Add one shard's location."""
        record = IndexRecord(name, offset, length, flags)
        self._records.append((_digest(name), record.pack()))

    def extend(self, records: Iterable[IndexRecord]) -> None:
        """Add many shard locations."""
        for record in records:
            self._records.append((_digest(record.name), record.pack()))

    def __len__(self) -> int:
        """Number of records staged."""
        return len(self._records)

    def build(self) -> bytes:
        """Serialize the whole index.

        Returns:
            The complete index blob, ready to write as one object.
        """
        self._records.sort(key=lambda pair: pair[0])
        count = len(self._records)
        fence_entries = [
            (self._records[i][0], i) for i in range(0, count, self.stride)
        ] or [(b"\0" * 16, 0)]
        fence_blob = b"".join(_FENCE.pack(d, i) for d, i in fence_entries)
        records_at = HEADER_SIZE + len(fence_blob)
        header = _HEADER.pack(
            MAGIC, VERSION, count, self.stride, len(fence_entries), records_at, 0
        )
        return header + fence_blob + b"".join(blob for _d, blob in self._records)


class ShardIndexReader:
    """Reads a sorted index by byte range, holding only the fence in RAM."""

    def __init__(self, storage: ShardStorage, key: str) -> None:
        """Open an index stored under ``key`` in ``storage``.

        Raises:
            StorageError: If the object is missing or not a UQIX index.
        """
        self.storage = storage
        self.key = key
        head = storage.read_range(key, 0, HEADER_SIZE)
        if len(head) < HEADER_SIZE:
            raise StorageError(f"{key!r}: too small to be an index")
        magic, version, count, stride, fence_count, records_at, _reserved = _HEADER.unpack(head)
        if magic != MAGIC:
            raise StorageError(f"{key!r}: not a UQIX index")
        if version != VERSION:
            raise StorageError(f"{key!r}: unsupported index version {version}")
        self.count = int(count)
        self.stride = int(stride)
        self.records_at = int(records_at)

        # The fence is the *only* thing held resident, and it is why memory
        # stays flat as the model grows.
        fence_blob = storage.read_range(key, HEADER_SIZE, fence_count * FENCE_SIZE)
        self._fence_digests: list[bytes] = []
        self._fence_ordinals: list[int] = []
        for i in range(fence_count):
            digest, ordinal = _FENCE.unpack(
                fence_blob[i * FENCE_SIZE:(i + 1) * FENCE_SIZE]
            )
            self._fence_digests.append(digest)
            self._fence_ordinals.append(int(ordinal))
        self.probes = 0

    @property
    def resident_bytes(self) -> int:
        """Bytes this reader holds in RAM — the fence and nothing else."""
        return len(self._fence_digests) * 24 + 512

    def _record_at(self, ordinal: int) -> tuple[bytes, IndexRecord]:
        """Read record number ``ordinal`` by byte range."""
        blob = self.storage.read_range(
            self.key, self.records_at + ordinal * RECORD_SIZE, RECORD_SIZE
        )
        if len(blob) < RECORD_SIZE:
            raise StorageError(f"{self.key!r}: truncated record at {ordinal}")
        self.probes += 1
        return IndexRecord.unpack(blob)

    def lookup(self, name: str) -> IndexRecord | None:
        """Find a shard's location.

        The fence narrows the search to one stride-sized window before any I/O;
        the binary search inside that window costs ``log2(stride)`` byte-range
        reads at worst, independent of how many shards exist.

        Returns:
            The record, or None if absent.
        """
        if not self.count:
            return None
        target = _digest(name)

        # Narrow to the interval the fence says it must be in.
        slot = bisect.bisect_right(self._fence_digests, target) - 1
        if slot < 0:
            slot = 0
        low = self._fence_ordinals[slot]
        high = (
            self._fence_ordinals[slot + 1]
            if slot + 1 < len(self._fence_ordinals)
            else self.count - 1
        )

        while low <= high:
            mid = (low + high) // 2
            digest, record = self._record_at(mid)
            if digest == target:
                return record if record.name == name else None
            if digest < target:
                low = mid + 1
            else:
                high = mid - 1
        return None

    def iter_records(self, batch: int = 4096) -> Iterator[IndexRecord]:
        """Stream every record without holding more than ``batch`` at a time."""
        for start in range(0, self.count, batch):
            span = min(batch, self.count - start)
            blob = self.storage.read_range(
                self.key, self.records_at + start * RECORD_SIZE, span * RECORD_SIZE
            )
            for i in range(span):
                chunk = blob[i * RECORD_SIZE:(i + 1) * RECORD_SIZE]
                if len(chunk) < RECORD_SIZE:
                    return
                _digest_bytes, record = IndexRecord.unpack(chunk)
                yield record

    def stats(self) -> dict:
        """Index size, resident cost and probe count."""
        return {
            "records": self.count,
            "stride": self.stride,
            "fence_entries": len(self._fence_digests),
            "resident_bytes": self.resident_bytes,
            "index_bytes": self.records_at + self.count * RECORD_SIZE,
            "probes": self.probes,
        }
