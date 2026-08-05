"""Pluggable byte-range storage beneath the shard library.

The vault rests on exactly one primitive: *read `length` bytes at `offset`*.
Everything else — the catalog, the paging budget, the associative routing — is
built on top of that. So the storage layer is defined by that primitive, and any
medium able to serve it can back the model library: a local file, a SAN LUN, an
NVMe-oF namespace, a Ceph object, or plain RAM.

Open one with :func:`open_storage` and a URI::

    local:///models/uq            a directory of files (default)
    ram://scratch                 in-process RAM; volatile, for tests
    blockdev://X:/uq?direct=1     tuned unbuffered I/O on a mounted volume
    nvmeof://X:/uq                NVMe-oF namespace (block tuning preset)
    lightbits://X:/uq             Lightbits NVMe/TCP (block tuning preset)
    pure://X:/uq                  Pure Storage FlashArray volume
    3par://X:/uq                  IBM/HPE 3PAR volume
    cephfs:///mnt/cephfs/uq       a mounted CephFS tree
    rados://pool/namespace        RADOS objects via librados
"""

from ultraquant.storage.base import (
    ShardStorage,
    StorageCapabilities,
    StorageError,
    StorageStats,
)
from ultraquant.storage.blockdev import BlockDeviceStorage
from ultraquant.storage.ceph import CephFSStorage, RadosStorage
from ultraquant.storage.local import LocalStorage
from ultraquant.storage.index import (
    IndexRecord,
    ShardIndexReader,
    ShardIndexWriter,
    index_footprint,
)
from ultraquant.storage.ram import RamStorage, available_ram, total_ram
from ultraquant.storage.tiered import TieredStorage, suggested_ram_budget
from ultraquant.storage.registry import URI_SCHEMES, open_storage

__all__ = [
    "ShardStorage",
    "StorageCapabilities",
    "StorageError",
    "StorageStats",
    "LocalStorage",
    "BlockDeviceStorage",
    "RamStorage",
    "CephFSStorage",
    "RadosStorage",
    "TieredStorage",
    "suggested_ram_budget",
    "ShardIndexWriter",
    "ShardIndexReader",
    "IndexRecord",
    "index_footprint",
    "available_ram",
    "total_ram",
    "open_storage",
    "URI_SCHEMES",
]
