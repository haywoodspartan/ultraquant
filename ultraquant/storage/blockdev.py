"""Tuned block-device storage for SAN volumes, NVMe-oF namespaces and the like.

Pure Storage FlashArray, IBM/HPE 3PAR, Lightbits and any NVMe-oF target all
present the same thing to a host: a block device, usually with a filesystem on
it. There is no proprietary wire protocol to speak from Python — the array's
distinctiveness lives in its *control plane* (provisioning, snapshots, replication;
see :mod:`ultraquant.storage.vendors`) and in how the host tunes its I/O.

So this is one backend, parameterised, rather than four near-identical classes
pretending to speak protocols they do not:

* **Unbuffered (direct) I/O** — bypasses the OS page cache. On an array with its
  own large cache and a low-latency fabric, double-caching in host RAM wastes
  memory that the RAM tier uses far better. Requires offset, length and buffer
  address all aligned to the device's sector size, so a request for an arbitrary
  range is widened to sector boundaries and sliced back down; the cost of that
  widening is visible as ``read_amplification`` in the stats.
* **Queue depth** — SAN and NVMe-oF reward many reads in flight. A single
  round trip to an array is far more expensive than to a local SSD, but the
  fabric will happily carry dozens at once, so :meth:`read_many` fans out.
* **Readahead** — a tunable extra span fetched around each read, for arrays
  whose minimum efficient transfer is larger than one shard.
"""

from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path

from ultraquant.storage.base import ShardStorage, StorageCapabilities, StorageError

__all__ = ["BlockDeviceStorage", "PROFILES", "sector_size"]

#: Tuning presets. Every one of these is a block device to the host; they differ
#: in fabric latency and in how much parallelism the target rewards.
PROFILES: dict[str, dict] = {
    "generic": {"queue_depth": 8, "readahead": 0, "direct": False, "medium": "block-device"},
    "nvmeof": {"queue_depth": 32, "readahead": 0, "direct": True, "medium": "nvme-of"},
    "lightbits": {"queue_depth": 32, "readahead": 0, "direct": True, "medium": "lightbits-nvme-tcp"},
    "pure": {"queue_depth": 32, "readahead": 32 * 1024, "direct": True, "medium": "pure-flasharray"},
    "3par": {"queue_depth": 16, "readahead": 64 * 1024, "direct": True, "medium": "ibm-3par"},
}

_WINDOWS = os.name == "nt"
_FILE_FLAG_NO_BUFFERING = 0x20000000
_FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000


def sector_size(path: str | os.PathLike) -> int:
    """Physical sector size of the volume holding ``path``.

    Falls back to 4096 — the safe modern default — when it cannot be determined,
    because over-aligning is merely slightly wasteful while under-aligning makes
    unbuffered reads fail outright.
    """
    try:
        if _WINDOWS:
            drive = os.path.splitdrive(os.path.abspath(str(path)))[0]
            root = f"{drive}\\" if drive else None
            sectors = ctypes.c_ulong(0)
            bytes_per_sector = ctypes.c_ulong(0)
            free_clusters = ctypes.c_ulong(0)
            total_clusters = ctypes.c_ulong(0)
            ok = ctypes.windll.kernel32.GetDiskFreeSpaceW(
                ctypes.c_wchar_p(root),
                ctypes.byref(sectors),
                ctypes.byref(bytes_per_sector),
                ctypes.byref(free_clusters),
                ctypes.byref(total_clusters),
            )
            if ok and bytes_per_sector.value:
                return int(bytes_per_sector.value)
        else:
            return int(os.statvfs(str(path)).f_bsize) or 4096
    except (OSError, AttributeError, ValueError):
        pass
    return 4096


class BlockDeviceStorage(ShardStorage):
    """A shard library on a block volume, with host-side I/O tuning."""

    def __init__(
        self,
        root: str | os.PathLike,
        profile: str = "generic",
        direct: bool | None = None,
        queue_depth: int | None = None,
        readahead: int | None = None,
        align: int | None = None,
    ) -> None:
        """Open a tuned volume-backed store.

        Args:
            root: Directory on the mounted volume.
            profile: One of :data:`PROFILES`.
            direct: Force unbuffered I/O on or off (default: the profile's).
            queue_depth: Concurrent reads for :meth:`read_many`.
            readahead: Extra bytes fetched around each read.
            align: Override the detected sector size.

        Raises:
            StorageError: If ``profile`` is unknown.
        """
        super().__init__()
        if profile not in PROFILES:
            raise StorageError(
                f"unknown profile {profile!r}; expected one of {sorted(PROFILES)}"
            )
        preset = PROFILES[profile]
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.direct = preset["direct"] if direct is None else bool(direct)
        self.queue_depth = int(queue_depth if queue_depth is not None else preset["queue_depth"])
        self.readahead = int(readahead if readahead is not None else preset["readahead"])
        self.align = int(align) if align else sector_size(self.root)
        self._medium = preset["medium"]
        self._direct_failed = False
        self._lock = threading.Lock()

    @property
    def uri(self) -> str:
        """The URI this backend was opened from."""
        return f"{self.profile}://{self.root.as_posix()}"

    @property
    def capabilities(self) -> StorageCapabilities:
        """Aligned, high-queue-depth, network-attached block storage."""
        return StorageCapabilities(
            ranged_reads=True,
            alignment=self.align if self.direct else 1,
            queue_depth=self.queue_depth,
            remote=self.profile != "generic",
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

    # -- unbuffered read ---------------------------------------------------

    def _read_direct(self, path: Path, start: int, span: int) -> bytes:
        """Unbuffered read of an already-aligned ``span`` at ``start``.

        Raises:
            OSError: If the platform refuses the unbuffered open or read.
        """
        if not _WINDOWS:
            flags = os.O_RDONLY | getattr(os, "O_DIRECT", 0)
            fd = os.open(str(path), flags)
            try:
                buffer = _aligned_buffer(span, self.align)
                got = os.preadv(fd, [buffer], start) if hasattr(os, "preadv") else 0
                if not got:
                    os.lseek(fd, start, os.SEEK_SET)
                    return os.read(fd, span)
                return bytes(buffer[:got])
            finally:
                os.close(fd)

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.CreateFileW(
            ctypes.c_wchar_p(str(path)),
            0x80000000,                       # GENERIC_READ
            0x00000001 | 0x00000002,          # FILE_SHARE_READ | FILE_SHARE_WRITE
            None,
            3,                                # OPEN_EXISTING
            _FILE_FLAG_NO_BUFFERING | _FILE_FLAG_SEQUENTIAL_SCAN,
            None,
        )
        if handle == -1 or handle == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_last_error(), "CreateFileW failed")
        try:
            # VirtualAlloc returns page-aligned memory, which satisfies any
            # sector alignment a volume can ask for.
            address = ctypes.windll.kernel32.VirtualAlloc(
                None, ctypes.c_size_t(span), 0x1000 | 0x2000, 0x04
            )
            if not address:
                raise OSError("VirtualAlloc failed")
            try:
                high = ctypes.c_long(start >> 32)
                moved = kernel32.SetFilePointer(
                    ctypes.c_void_p(handle), ctypes.c_long(start & 0xFFFFFFFF),
                    ctypes.byref(high), 0,
                )
                if moved == 0xFFFFFFFF:
                    raise OSError("SetFilePointer failed")
                read = ctypes.c_ulong(0)
                ok = kernel32.ReadFile(
                    ctypes.c_void_p(handle), ctypes.c_void_p(address),
                    ctypes.c_ulong(span), ctypes.byref(read), None,
                )
                if not ok:
                    raise OSError(ctypes.get_last_error(), "ReadFile failed")
                return ctypes.string_at(address, read.value)
            finally:
                ctypes.windll.kernel32.VirtualFree(
                    ctypes.c_void_p(address), ctypes.c_size_t(0), 0x8000
                )
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))

    def read_range(self, key: str, offset: int, length: int) -> bytes:
        """Read ``length`` bytes of ``key`` at ``offset``, aligning if needed."""
        if length <= 0:
            return b""
        path = self._path(key)

        want_end = offset + length + max(0, self.readahead)
        if self.direct and not self._direct_failed:
            start = (offset // self.align) * self.align
            span = ((want_end - start + self.align - 1) // self.align) * self.align
            try:
                raw = self._read_direct(path, start, span)
                self.stats.record_read(length, len(raw))
                head = offset - start
                return raw[head:head + length]
            except OSError:
                # Unbuffered I/O is unavailable here (a filesystem that refuses
                # it, or a non-aligned device). Fall back permanently rather
                # than paying the failure on every read.
                with self._lock:
                    self._direct_failed = True

        try:
            with open(path, "rb") as handle:
                handle.seek(offset)
                data = handle.read(length + max(0, self.readahead))
        except OSError as exc:
            self.stats.record_error()
            raise StorageError(f"read failed for {key!r}: {exc}") from exc
        self.stats.record_read(length, len(data))
        return data[:length]

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
        """Store ``data`` under ``key`` atomically (buffered; writes are rare)."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        try:
            with open(tmp, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
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
        out = []
        for path in self.root.rglob("*"):
            if not path.is_file() or path.name.endswith(".tmp"):
                continue
            key = path.relative_to(self.root).as_posix()
            if key.startswith(prefix):
                out.append(key)
        return sorted(out)

    def describe(self) -> dict:
        """Summary including the tuning actually in force."""
        info = super().describe()
        info.update({
            "profile": self.profile,
            "direct_io": self.direct and not self._direct_failed,
            "direct_io_fell_back": self._direct_failed,
            "readahead": self.readahead,
        })
        return info


def _aligned_buffer(size: int, alignment: int) -> memoryview:
    """A writable buffer whose start address is ``alignment``-aligned."""
    raw = bytearray(size + alignment)
    address = ctypes.addressof((ctypes.c_char * len(raw)).from_buffer(raw))
    shift = (-address) % alignment
    return memoryview(raw)[shift:shift + size]
