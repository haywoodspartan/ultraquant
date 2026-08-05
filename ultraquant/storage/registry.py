"""Open a storage backend from a URI, and wrap it in a RAM tier.

One entry point so callers never import a specific backend::

    open_storage("local:///models/uq")
    open_storage("nvmeof://X:/uq", cache="2GB")
    open_storage("rados://models/uq", cache="auto")

``cache`` is what makes a library larger than memory usable: it wraps the durable
backend in :class:`~ultraquant.storage.tiered.TieredStorage`, so the model stays
on storage and only the working set occupies RAM.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from ultraquant.storage.base import ShardStorage, StorageError
from ultraquant.storage.blockdev import PROFILES, BlockDeviceStorage
from ultraquant.storage.ceph import CephFSStorage, RadosStorage
from ultraquant.storage.local import LocalStorage
from ultraquant.storage.ram import RamStorage
from ultraquant.storage.tiered import TieredStorage, suggested_ram_budget

__all__ = ["open_storage", "parse_size", "URI_SCHEMES"]

#: Every scheme :func:`open_storage` understands, with a one-line description.
URI_SCHEMES: dict[str, str] = {
    "local": "a directory on an ordinary filesystem (default)",
    "ram": "in-process memory; volatile, for scratch and tests",
    "blockdev": "a mounted block volume with host-side I/O tuning",
    "nvmeof": "NVMe-oF namespace (deep queue, unbuffered)",
    "lightbits": "Lightbits NVMe/TCP volume (deep queue, unbuffered)",
    "pure": "Pure Storage FlashArray volume",
    "3par": "IBM/HPE 3PAR volume",
    "cephfs": "a mounted CephFS tree",
    "rados": "Ceph RADOS objects via librados",
}

_SIZE_RE = re.compile(r"^\s*([\d.]+)\s*([kmgtp]?i?b?)\s*$", re.I)
_UNITS = {
    "": 1, "b": 1,
    "k": 1024, "kb": 1024, "kib": 1024,
    "m": 1024 ** 2, "mb": 1024 ** 2, "mib": 1024 ** 2,
    "g": 1024 ** 3, "gb": 1024 ** 3, "gib": 1024 ** 3,
    "t": 1024 ** 4, "tb": 1024 ** 4, "tib": 1024 ** 4,
    "p": 1024 ** 5, "pb": 1024 ** 5, "pib": 1024 ** 5,
}


def parse_size(text: str | int | None) -> int | None:
    """Parse ``"2GB"``, ``"512MiB"``, ``"auto"`` or an int into bytes.

    Args:
        text: Size string, byte count, ``"auto"``, or None.

    Returns:
        Bytes, or None when no cache is wanted.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    if text is None or text == "":
        return None
    if isinstance(text, int):
        return text
    lowered = str(text).strip().lower()
    if lowered in ("auto", "default"):
        return suggested_ram_budget()
    if lowered in ("none", "off", "0"):
        return None
    match = _SIZE_RE.match(lowered)
    if not match:
        raise ValueError(f"cannot parse size {text!r}")
    value, unit = match.groups()
    if unit not in _UNITS:
        raise ValueError(f"unknown size unit in {text!r}")
    return int(float(value) * _UNITS[unit])


def _windows_path(parsed) -> str:
    """Reassemble a filesystem path from a parsed URI.

    ``blockdev://X:/uq`` puts ``X:`` in netloc and ``/uq`` in path, while
    ``local:///models/uq`` puts everything in path — both must round-trip.
    """
    if parsed.netloc:
        return f"{parsed.netloc}{parsed.path}"
    return parsed.path


def open_storage(uri: str, cache: str | int | None = None, **kwargs) -> ShardStorage:
    """Open a storage backend described by ``uri``.

    Args:
        uri: One of the schemes in :data:`URI_SCHEMES`. A bare path is treated
            as ``local://``.
        cache: RAM tier budget (``"2GB"``, ``"auto"``, bytes, or None for no
            tier). The durable backend always remains the book of record.
        **kwargs: Passed to the backend constructor.

    Returns:
        A ready backend, wrapped in a RAM tier when ``cache`` is given.

    Raises:
        StorageError: If the scheme is unknown or the backend cannot open.
    """
    if "://" not in uri:
        uri = f"local://{uri}"
    # Split the scheme by hand: a URI scheme may not begin with a digit, so
    # urlparse silently refuses to recognise "3par://" and reports no scheme
    # at all. Parsing the remainder under a placeholder keeps netloc, path and
    # query handling intact while letting vendor names start however they like.
    scheme, _, remainder = uri.partition("://")
    scheme = scheme.lower()
    parsed = urlparse(f"uq://{remainder}")
    query = {k: v[0] for k, v in parse_qs(parsed.query).items()}

    def flag(name: str, default: bool | None = None) -> bool | None:
        raw = query.get(name)
        if raw is None:
            return default
        return raw.lower() in ("1", "true", "yes", "on")

    if scheme == "local":
        cold: ShardStorage = LocalStorage(_windows_path(parsed), **kwargs)
    elif scheme == "ram":
        cold = RamStorage(name=parsed.netloc or "scratch", **kwargs)
    elif scheme in ("blockdev", *PROFILES):
        profile = "generic" if scheme == "blockdev" else scheme
        profile = query.get("profile", profile)
        options = dict(kwargs)
        if "direct" not in options:
            direct = flag("direct")
            if direct is not None:
                options["direct"] = direct
        for name, caster in (("queue_depth", int), ("readahead", int), ("align", int)):
            if name in query and name not in options:
                options[name] = caster(query[name])
        cold = BlockDeviceStorage(_windows_path(parsed), profile=profile, **options)
    elif scheme == "cephfs":
        options = dict(kwargs)
        if "queue_depth" in query:
            options["queue_depth"] = int(query["queue_depth"])
        cold = CephFSStorage(_windows_path(parsed), **options)
    elif scheme == "rados":
        pool = parsed.netloc
        namespace = parsed.path.strip("/") or None
        if not pool:
            raise StorageError("rados:// needs a pool, e.g. rados://models/uq")
        options = dict(kwargs)
        for name in ("conffile", "keyring", "client_name"):
            if name in query and name not in options:
                options[name] = query[name]
        if "queue_depth" in query:
            options["queue_depth"] = int(query["queue_depth"])
        cold = RadosStorage(pool, namespace=namespace, **options)
    else:
        raise StorageError(
            f"unknown storage scheme {scheme!r}; expected one of {sorted(URI_SCHEMES)}"
        )

    budget = parse_size(cache if cache is not None else query.get("cache"))
    if budget is None:
        return cold
    return TieredStorage(cold, max_bytes=budget, name=scheme)
