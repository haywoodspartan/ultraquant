"""The storage contract: byte-range reads, and the honest capabilities of a medium.

Every backend implements the same small interface. The one operation that matters
is :meth:`ShardStorage.read_range` — the whole paging design rests on being able
to pull one shard's bytes without touching the rest of the library.

:class:`StorageCapabilities` exists so callers can adapt rather than guess: a
backend declares whether its ranged reads are genuinely cheap (a local NVMe seek)
or an emulation that fetches more than asked (an object store without range
support), what alignment its I/O path prefers, and how many reads it can usefully
have in flight at once.
"""

from __future__ import annotations

import abc
import threading
from dataclasses import dataclass, field
from typing import Iterable, Sequence

__all__ = [
    "StorageError",
    "StorageCapabilities",
    "StorageStats",
    "ShardStorage",
]


class StorageError(RuntimeError):
    """Raised when a storage backend cannot satisfy a request."""


@dataclass(frozen=True)
class StorageCapabilities:
    """What a backend can actually do, so callers need not assume."""

    #: True when a ranged read costs about what it asks for, rather than
    #: fetching the whole object and slicing it.
    ranged_reads: bool = True
    #: Preferred I/O alignment in bytes (sector size for a raw block device).
    alignment: int = 1
    #: Useful concurrent reads in flight. 1 means serial.
    queue_depth: int = 1
    #: True when the medium is across a network (SAN, NVMe-oF, RADOS).
    remote: bool = False
    #: True when contents vanish on power loss.
    volatile: bool = False
    #: True when the backend can be written to.
    writable: bool = True
    #: Human label for logs and the GUI.
    medium: str = "unknown"


@dataclass
class StorageStats:
    """Counters describing what a backend has actually done.

    ``bytes_requested`` versus ``bytes_read`` is the interesting pair: on a
    direct-I/O block device the two differ by the read amplification that
    sector alignment forces, which is the cost of bypassing the page cache.
    """

    reads: int = 0
    writes: int = 0
    bytes_requested: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    errors: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_read(self, requested: int, actual: int) -> None:
        """Count one read of ``requested`` bytes that moved ``actual`` bytes."""
        with self._lock:
            self.reads += 1
            self.bytes_requested += requested
            self.bytes_read += actual

    def record_write(self, count: int) -> None:
        """Count one write of ``count`` bytes."""
        with self._lock:
            self.writes += 1
            self.bytes_written += count

    def record_error(self) -> None:
        """Count one failed operation."""
        with self._lock:
            self.errors += 1

    @property
    def amplification(self) -> float:
        """Bytes actually moved per byte asked for (1.0 = no waste)."""
        if not self.bytes_requested:
            return 1.0
        return self.bytes_read / self.bytes_requested

    def as_dict(self) -> dict:
        """JSON-safe snapshot."""
        return {
            "reads": self.reads,
            "writes": self.writes,
            "bytes_requested": self.bytes_requested,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "errors": self.errors,
            "read_amplification": round(self.amplification, 4),
        }


class ShardStorage(abc.ABC):
    """A medium that can store shard bytes and serve ranges of them."""

    def __init__(self) -> None:
        """Initialise the shared counters."""
        self.stats = StorageStats()

    # -- identity ----------------------------------------------------------

    @property
    @abc.abstractmethod
    def uri(self) -> str:
        """The URI this backend was opened from."""

    @property
    @abc.abstractmethod
    def capabilities(self) -> StorageCapabilities:
        """What this backend can do."""

    # -- reads -------------------------------------------------------------

    @abc.abstractmethod
    def read_range(self, key: str, offset: int, length: int) -> bytes:
        """Read ``length`` bytes of ``key`` starting at ``offset``.

        Raises:
            StorageError: If the object is missing or the range is unreadable.
        """

    @abc.abstractmethod
    def read_all(self, key: str) -> bytes:
        """Read an object whole."""

    def read_many(
        self, requests: Sequence[tuple[str, int, int]]
    ) -> list[bytes]:
        """Read several ranges, using the backend's queue depth where useful.

        Args:
            requests: ``(key, offset, length)`` triples.

        Returns:
            One byte string per request, in the order given.
        """
        depth = max(1, self.capabilities.queue_depth)
        if depth == 1 or len(requests) < 2:
            return [self.read_range(k, o, n) for k, o, n in requests]

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(depth, len(requests))) as pool:
            return list(pool.map(lambda r: self.read_range(*r), requests))

    # -- writes ------------------------------------------------------------

    @abc.abstractmethod
    def write(self, key: str, data: bytes) -> None:
        """Store ``data`` under ``key``, replacing anything already there."""

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """Remove ``key``. Missing keys are not an error."""

    # -- inspection --------------------------------------------------------

    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        """Whether ``key`` is present."""

    @abc.abstractmethod
    def size(self, key: str) -> int:
        """Size of ``key`` in bytes.

        Raises:
            StorageError: If the object is missing.
        """

    @abc.abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        """Keys under ``prefix``, sorted."""

    def total_bytes(self, keys: Iterable[str] | None = None) -> int:
        """Total stored size of ``keys`` (default: everything)."""
        return sum(self.size(k) for k in (keys if keys is not None else self.list_keys()))

    def describe(self) -> dict:
        """A JSON-safe summary of this backend, for logs and the GUI."""
        caps = self.capabilities
        return {
            "uri": self.uri,
            "medium": caps.medium,
            "ranged_reads": caps.ranged_reads,
            "alignment": caps.alignment,
            "queue_depth": caps.queue_depth,
            "remote": caps.remote,
            "volatile": caps.volatile,
            "writable": caps.writable,
            "stats": self.stats.as_dict(),
        }

    def close(self) -> None:
        """Release any held handles. Safe to call more than once."""

    def __enter__(self) -> "ShardStorage":
        """Context-manager entry."""
        return self

    def __exit__(self, *exc_info) -> None:
        """Context-manager exit; closes the backend."""
        self.close()

    def __repr__(self) -> str:
        """Readable representation."""
        return f"<{type(self).__name__} {self.uri}>"
