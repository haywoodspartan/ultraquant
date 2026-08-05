"""Ceph backends: a CephFS mount, and native RADOS objects.

Two genuinely different things, unlike the SAN family:

* :class:`CephFSStorage` — a mounted CephFS tree. POSIX semantics, so this is
  local-file I/O with remote latency and a deeper queue; the ceph-specific part
  is the tuning, not the interface.
* :class:`RadosStorage` — real RADOS objects through ``librados``. This is an
  object API with native ranged reads (``read(oid, length, offset)``), which is
  exactly the primitive the shard library needs, so no filesystem is involved at
  all. It requires the ``rados`` Python bindings; without them the class reports
  itself unavailable rather than failing at import.
"""

from __future__ import annotations

import os
from pathlib import Path

from ultraquant.storage.base import ShardStorage, StorageCapabilities, StorageError
from ultraquant.storage.local import LocalStorage

__all__ = ["CephFSStorage", "RadosStorage"]


class CephFSStorage(LocalStorage):
    """A shard library on a mounted CephFS tree."""

    def __init__(self, root: str | os.PathLike, queue_depth: int = 16) -> None:
        """Open a CephFS-backed store.

        Args:
            root: Path inside the CephFS mount.
            queue_depth: Concurrent reads; a distributed filesystem rewards more
                parallelism than a local disk because each read costs a round trip.
        """
        super().__init__(root, medium="cephfs")
        self.queue_depth = int(queue_depth)

    @property
    def uri(self) -> str:
        """The URI this backend was opened from."""
        return f"cephfs://{self.root.as_posix()}"

    @property
    def capabilities(self) -> StorageCapabilities:
        """POSIX semantics over a network, so deep queues and no alignment rules."""
        return StorageCapabilities(
            ranged_reads=True,
            alignment=1,
            queue_depth=self.queue_depth,
            remote=True,
            volatile=False,
            writable=True,
            medium="cephfs",
        )


class RadosStorage(ShardStorage):
    """Shards as RADOS objects, read by native byte range."""

    def __init__(
        self,
        pool: str,
        namespace: str | None = None,
        conffile: str | None = None,
        keyring: str | None = None,
        client_name: str = "client.admin",
        queue_depth: int = 32,
        timeout: float = 30.0,
    ) -> None:
        """Connect to a RADOS pool.

        Args:
            pool: Pool holding the shard objects.
            namespace: Optional RADOS namespace.
            conffile: ``ceph.conf`` path (default: librados' own search).
            keyring: Optional keyring path.
            client_name: Cephx client name.
            queue_depth: Concurrent reads.
            timeout: Connection timeout in seconds.

        Raises:
            StorageError: If ``librados`` is unavailable or the connection fails.
        """
        super().__init__()
        self.pool = pool
        self.namespace = namespace
        self.queue_depth = int(queue_depth)
        self._cluster = None
        self._ioctx = None

        try:
            import rados  # noqa: PLC0415 - optional dependency, probed on purpose
        except ImportError as exc:
            raise StorageError(
                "RADOS backend needs the 'rados' Python bindings (ceph-common); "
                "use cephfs:// against a mounted tree if they are not installed"
            ) from exc

        options = {}
        if keyring:
            options["keyring"] = keyring
        try:
            self._cluster = rados.Rados(
                conffile=conffile or rados.Rados.DEFAULT_CONF_FILES,
                rados_id=client_name.replace("client.", ""),
                conf=options or None,
            )
            self._cluster.connect(timeout=int(timeout))
            self._ioctx = self._cluster.open_ioctx(pool)
            if namespace:
                self._ioctx.set_namespace(namespace)
        except Exception as exc:  # noqa: BLE001 - librados raises its own hierarchy
            raise StorageError(f"RADOS connection failed: {exc}") from exc

    @classmethod
    def available(cls) -> bool:
        """Whether the librados bindings can be imported. Never raises."""
        try:
            import rados  # noqa: F401, PLC0415
        except Exception:  # noqa: BLE001
            return False
        return True

    @property
    def uri(self) -> str:
        """The URI this backend was opened from."""
        suffix = f"/{self.namespace}" if self.namespace else ""
        return f"rados://{self.pool}{suffix}"

    @property
    def capabilities(self) -> StorageCapabilities:
        """Native object ranged reads over the network."""
        return StorageCapabilities(
            ranged_reads=True,
            alignment=1,
            queue_depth=self.queue_depth,
            remote=True,
            volatile=False,
            writable=True,
            medium="ceph-rados",
        )

    def _require(self):
        """The open ioctx.

        Raises:
            StorageError: If the connection is closed.
        """
        if self._ioctx is None:
            raise StorageError("RADOS connection is closed")
        return self._ioctx

    def read_range(self, key: str, offset: int, length: int) -> bytes:
        """Read ``length`` bytes of object ``key`` at ``offset``."""
        if length <= 0:
            return b""
        try:
            data = self._require().read(key, length, offset)
        except Exception as exc:  # noqa: BLE001
            self.stats.record_error()
            raise StorageError(f"RADOS read failed for {key!r}: {exc}") from exc
        self.stats.record_read(length, len(data))
        return data

    def read_all(self, key: str) -> bytes:
        """Read an object whole, in chunks."""
        ioctx = self._require()
        chunks: list[bytes] = []
        offset = 0
        step = 8 * 1024 * 1024
        try:
            while True:
                block = ioctx.read(key, step, offset)
                if not block:
                    break
                chunks.append(block)
                offset += len(block)
                if len(block) < step:
                    break
        except Exception as exc:  # noqa: BLE001
            self.stats.record_error()
            raise StorageError(f"RADOS read failed for {key!r}: {exc}") from exc
        data = b"".join(chunks)
        self.stats.record_read(len(data), len(data))
        return data

    def write(self, key: str, data: bytes) -> None:
        """Store an object whole."""
        try:
            self._require().write_full(key, bytes(data))
        except Exception as exc:  # noqa: BLE001
            self.stats.record_error()
            raise StorageError(f"RADOS write failed for {key!r}: {exc}") from exc
        self.stats.record_write(len(data))

    def delete(self, key: str) -> None:
        """Remove an object if present."""
        try:
            self._require().remove_object(key)
        except Exception:  # noqa: BLE001 - absent is not an error
            pass

    def exists(self, key: str) -> bool:
        """Whether the object exists."""
        try:
            self._require().stat(key)
        except Exception:  # noqa: BLE001
            return False
        return True

    def size(self, key: str) -> int:
        """Object size in bytes."""
        try:
            return int(self._require().stat(key)[0])
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"RADOS stat failed for {key!r}: {exc}") from exc

    def list_keys(self, prefix: str = "") -> list[str]:
        """Object names under ``prefix``, sorted."""
        try:
            return sorted(
                obj.key for obj in self._require().list_objects()
                if obj.key.startswith(prefix)
            )
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"RADOS list failed: {exc}") from exc

    def close(self) -> None:
        """Close the ioctx and cluster handle."""
        try:
            if self._ioctx is not None:
                self._ioctx.close()
        finally:
            self._ioctx = None
            if self._cluster is not None:
                try:
                    self._cluster.shutdown()
                finally:
                    self._cluster = None
