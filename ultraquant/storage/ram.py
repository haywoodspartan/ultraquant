"""RAM as the fast working tier in front of slow storage.

This is *not* where the model lives. The shard library is the model, and it stays
on NVMe, SAN or disk; RAM holds only the shards currently being worked with, so a
store far larger than memory stays usable. See
:class:`ultraquant.storage.tiered.TieredStorage` for the read-through tier that
uses this, and :mod:`ultraquant.shards.working_set` for the pattern-recognition
driven prefetch that decides *what* is worth holding.

:class:`RamStorage` on its own is a complete backend, which makes it useful for
tests and scratch space — but it is volatile, and it says so in its capabilities.
"""

from __future__ import annotations

import ctypes
import os
import threading

from ultraquant.storage.base import ShardStorage, StorageCapabilities, StorageError

__all__ = ["RamStorage", "available_ram", "total_ram", "is_ram_backed"]


def _meminfo() -> tuple[int, int]:
    """``(total, available)`` system memory in bytes; zeros if unknown."""
    if os.name == "nt":
        class _Status(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _Status()
        status.dwLength = ctypes.sizeof(_Status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys), int(status.ullAvailPhys)
        except (AttributeError, OSError):
            return 0, 0
        return 0, 0

    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        total = 0
    available = 0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
    except OSError:
        available = 0
    return total, available


def total_ram() -> int:
    """Total physical memory in bytes (0 if it cannot be determined)."""
    return _meminfo()[0]


def available_ram() -> int:
    """Currently available physical memory in bytes (0 if unknown)."""
    return _meminfo()[1]


def is_ram_backed(path: str | os.PathLike) -> bool:
    """Whether ``path`` sits on a memory-backed filesystem.

    Only meaningful on Unix (tmpfs / ramfs); Windows RAM disks present as
    ordinary volumes and cannot be told apart from an SSD here.
    """
    if os.name == "nt":
        return False
    try:
        with open("/proc/mounts", "r", encoding="utf-8") as handle:
            best, best_type = "", ""
            target = os.path.abspath(str(path))
            for line in handle:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount, fstype = parts[1], parts[2]
                if target.startswith(mount) and len(mount) > len(best):
                    best, best_type = mount, fstype
            return best_type in ("tmpfs", "ramfs")
    except OSError:
        return False


class RamStorage(ShardStorage):
    """An in-process byte store, used as the hot tier over durable storage."""

    def __init__(self, name: str = "working-set", max_bytes: int | None = None) -> None:
        """Create an empty RAM store.

        Args:
            name: Label for the URI.
            max_bytes: Optional hard cap; writes beyond it raise
                :class:`StorageError` so a runaway cache cannot exhaust memory.
        """
        super().__init__()
        self.name = name
        self.max_bytes = max_bytes
        self._objects: dict[str, bytes] = {}
        self._bytes = 0
        self._lock = threading.RLock()

    @property
    def uri(self) -> str:
        """The URI this backend was opened from."""
        return f"ram://{self.name}"

    @property
    def capabilities(self) -> StorageCapabilities:
        """Volatile, unaligned, effectively unlimited parallelism."""
        return StorageCapabilities(
            ranged_reads=True,
            alignment=1,
            queue_depth=1,      # slicing memory gains nothing from threads
            remote=False,
            volatile=True,
            writable=True,
            medium="ram",
        )

    @property
    def used_bytes(self) -> int:
        """Bytes currently held."""
        return self._bytes

    def read_range(self, key: str, offset: int, length: int) -> bytes:
        """Read ``length`` bytes of ``key`` starting at ``offset``."""
        if length <= 0:
            return b""
        with self._lock:
            blob = self._objects.get(key)
        if blob is None:
            self.stats.record_error()
            raise StorageError(f"not in RAM: {key!r}")
        data = blob[offset:offset + length]
        self.stats.record_read(length, len(data))
        return data

    def read_all(self, key: str) -> bytes:
        """Read an object whole."""
        with self._lock:
            blob = self._objects.get(key)
        if blob is None:
            self.stats.record_error()
            raise StorageError(f"not in RAM: {key!r}")
        self.stats.record_read(len(blob), len(blob))
        return blob

    def write(self, key: str, data: bytes) -> None:
        """Store ``data`` under ``key``.

        Raises:
            StorageError: If it would exceed ``max_bytes``.
        """
        blob = bytes(data)
        with self._lock:
            prior = len(self._objects.get(key, b""))
            if self.max_bytes is not None:
                projected = self._bytes - prior + len(blob)
                if projected > self.max_bytes:
                    self.stats.record_error()
                    raise StorageError(
                        f"RAM tier full: {projected:,} > {self.max_bytes:,} bytes"
                    )
            self._objects[key] = blob
            self._bytes += len(blob) - prior
        self.stats.record_write(len(blob))

    def delete(self, key: str) -> None:
        """Drop ``key`` from RAM."""
        with self._lock:
            blob = self._objects.pop(key, None)
            if blob is not None:
                self._bytes -= len(blob)

    def clear(self) -> None:
        """Drop everything."""
        with self._lock:
            self._objects.clear()
            self._bytes = 0

    def exists(self, key: str) -> bool:
        """Whether ``key`` is held."""
        with self._lock:
            return key in self._objects

    def size(self, key: str) -> int:
        """Size of ``key`` in bytes."""
        with self._lock:
            blob = self._objects.get(key)
        if blob is None:
            raise StorageError(f"not in RAM: {key!r}")
        return len(blob)

    def list_keys(self, prefix: str = "") -> list[str]:
        """Keys under ``prefix``, sorted."""
        with self._lock:
            return sorted(k for k in self._objects if k.startswith(prefix))
