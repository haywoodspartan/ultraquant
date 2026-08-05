"""Local filesystem storage — the default durable home for a shard library.

Ordinary buffered I/O, which is the right choice on a normal filesystem: the
operating system's page cache is a genuine asset when the same shards are read
repeatedly, and fighting it buys nothing. For a raw SAN LUN or an NVMe-oF
namespace, where the page cache mostly adds a copy, see
:mod:`ultraquant.storage.blockdev`.
"""

from __future__ import annotations

import os
from pathlib import Path

from ultraquant.storage.base import ShardStorage, StorageCapabilities, StorageError

__all__ = ["LocalStorage"]


class LocalStorage(ShardStorage):
    """Files under a directory, one object per key."""

    def __init__(self, root: str | os.PathLike, medium: str = "local-fs") -> None:
        """Open (or create) a storage root.

        Args:
            root: Directory holding the objects.
            medium: Label reported in :attr:`capabilities`.
        """
        super().__init__()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._medium = medium

    @property
    def uri(self) -> str:
        """The URI this backend was opened from."""
        return f"local://{self.root.as_posix()}"

    @property
    def capabilities(self) -> StorageCapabilities:
        """Buffered local I/O: cheap seeks, no alignment constraints."""
        return StorageCapabilities(
            ranged_reads=True,
            alignment=1,
            queue_depth=4,
            remote=False,
            volatile=False,
            writable=True,
            medium=self._medium,
        )

    def _path(self, key: str) -> Path:
        """Filesystem path for ``key``, kept inside the root."""
        safe = key.replace("\\", "/").lstrip("/")
        if ".." in Path(safe).parts:
            raise StorageError(f"key escapes the storage root: {key!r}")
        return self.root / safe

    def read_range(self, key: str, offset: int, length: int) -> bytes:
        """Read ``length`` bytes of ``key`` starting at ``offset``."""
        if length <= 0:
            return b""
        path = self._path(key)
        try:
            with open(path, "rb") as handle:
                handle.seek(offset)
                data = handle.read(length)
        except OSError as exc:
            self.stats.record_error()
            raise StorageError(f"read failed for {key!r}: {exc}") from exc
        self.stats.record_read(length, len(data))
        return data

    def read_all(self, key: str) -> bytes:
        """Read an object whole."""
        try:
            data = self._path(key).read_bytes()
        except OSError as exc:
            self.stats.record_error()
            raise StorageError(f"read failed for {key!r}: {exc}") from exc
        self.stats.record_read(len(data), len(data))
        return data

    def write(self, key: str, data: bytes) -> None:
        """Store ``data`` under ``key`` atomically."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        try:
            with open(tmp, "wb") as handle:
                handle.write(data)
            os.replace(tmp, path)
        except OSError as exc:
            self.stats.record_error()
            raise StorageError(f"write failed for {key!r}: {exc}") from exc
        self.stats.record_write(len(data))

    def delete(self, key: str) -> None:
        """Remove ``key`` if present."""
        try:
            self._path(key).unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            self.stats.record_error()
            raise StorageError(f"delete failed for {key!r}: {exc}") from exc

    def exists(self, key: str) -> bool:
        """Whether ``key`` is present."""
        return self._path(key).exists()

    def size(self, key: str) -> int:
        """Size of ``key`` in bytes."""
        try:
            return self._path(key).stat().st_size
        except OSError as exc:
            raise StorageError(f"stat failed for {key!r}: {exc}") from exc

    def list_keys(self, prefix: str = "") -> list[str]:
        """Keys under ``prefix``, sorted."""
        out: list[str] = []
        for path in self.root.rglob("*"):
            if not path.is_file() or path.name.endswith(".tmp"):
                continue
            key = path.relative_to(self.root).as_posix()
            if key.startswith(prefix):
                out.append(key)
        return sorted(out)
