"""Shard vault and the ``.uql`` library container format.

This module realises the project's guiding idea: the model is a *library of
patterns of learned data, catalogued associatively like a human brain*.  Each
slice of learned parameters is a **shard**.  Shards are first written as small
loose ``.uqs`` files, then *consolidated* into large ``.uql`` library files with
an offset index so that any one shard can be read back as a byte-range chunk
**on demand** — the whole store is never loaded at once.

The ``.uql`` container layout is::

    b"UQL1"                         # 4-byte magic
    <index length>                  # 8-byte big-endian unsigned integer
    <index>                         # JSON (utf-8): list of per-shard records
    <payload 0><payload 1> ...      # concatenated stored (compressed) bytes

Offsets recorded in the index (and mirrored into the catalog) are **absolute
file offsets**, so a reader seeks straight to a shard's bytes.

Every shard carries brain-like ``associations`` — keyword -> strength weights
that are reinforced on use and survive re-training, exactly as recall
strengthens a memory trace.

Pure Python standard library only; all persisted state is JSON-safe; every path
is handled with :mod:`pathlib` and encoded to be safe on Windows filesystems.
"""

from __future__ import annotations

import hashlib
import contextlib
import heapq
import json
import os
import zlib
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .sketch import sketch

__all__ = ["ShardIntegrityError", "ShardVault"]

#: Below this many stored prototypes, routing scores everything exactly. The
#: screen only earns its keep once the exact scan stops being free.
_SCREEN_ABOVE = 512

#: The slot a category fills when it never declares one. A library whose
#: categories are alternatives leaves them all here, which is what makes
#: them compete rather than compose.
_DEFAULT_SLOT = "pattern"


def _json_default(value: Any) -> Any:
    """Serialize the compact in-memory forms back to plain JSON."""
    if isinstance(value, array):
        return list(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")

_MAGIC = b"UQL1"
_HEADER_BYTES = 4 + 8  # magic + 8-byte big-endian index length
_ASSOC_CAP = 5.0  # maximum association strength (matches reinforce())

# Characters allowed verbatim in a loose-shard filename.
_SAFE_FILENAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789._-"
)


class ShardIntegrityError(RuntimeError):
    """Raised when a shard's bytes no longer match their recorded sha256."""


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(shard_id: str) -> str:
    """Return a Windows-safe loose-shard filename for ``shard_id``.

    Shard ids in this system may contain characters that are illegal in
    Windows filenames (for example the colon in ``expert:math``).  Each unsafe
    character is reversibly escaped as ``~<hex>~`` and the whole name is given an
    ``s_`` prefix so it can never collide with a reserved device name such as
    ``CON`` or ``NUL``.  The mapping is deterministic and injective.

    Args:
        shard_id: The logical shard identifier.

    Returns:
        A filesystem-safe basename (without extension).
    """
    parts = ["s_"]
    for ch in shard_id:
        if ch in _SAFE_FILENAME_CHARS:
            parts.append(ch)
        else:
            parts.append("~%x~" % ord(ch))
    return "".join(parts)


class ShardVault:
    """A catalogued store of parameter shards with on-demand chunk reads.

    The vault keeps a JSON catalog (``root/catalog.json``) that maps every
    ``shard_id`` to a JSON-safe entry describing where its stored bytes live
    (a loose ``.uqs`` file, or a byte range inside a ``.uql`` library), its
    integrity hash, access statistics, and associative keyword weights.

    Args:
        root: Directory that holds the catalog, the ``loose/`` folder, and
            (by convention) any packed libraries.

    Attributes:
        root: The vault root directory.
        loose_dir: ``root/loose`` — where loose ``.uqs`` shards are written.
        catalog_path: ``root/catalog.json`` — the persisted catalog.
    """

    def __init__(self, root: str | os.PathLike, storage: Any | None = None) -> None:
        """Open (or create) a vault.

        Args:
            root: Directory holding the catalog, loose shards and libraries.
            storage: Optional :class:`~ultraquant.storage.base.ShardStorage` that
                shard *payload* reads are routed through — a tuned block device,
                a Ceph pool, or a RAM tier over any of them. The catalog and the
                on-disk layout are unchanged; only where the bytes come from
                differs, which is what lets the same library sit on NVMe, SAN or
                Ceph without the vault knowing.
        """
        self.root: Path = Path(root)
        self.loose_dir: Path = self.root / "loose"
        self.catalog_path: Path = self.root / "catalog.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.loose_dir.mkdir(parents=True, exist_ok=True)
        self.storage = storage
        # Insertion order is preserved and used as the default pack order.
        self._catalog: dict[str, dict[str, Any]] = {}
        # Derived lookups so callers never have to scan the catalog. Routing a
        # single input would otherwise cost one pass over every shard, which is
        # ruinous once a library holds a million of them.
        self._by_category: dict[str, list[str]] = {}
        self._assoc: dict[str, dict[str, float]] = {}
        self._defer_save = 0
        self._dirty = False
        # Sketch screen over the signatures; built on first use and invalidated
        # by anything that can change what a category looks like.
        self._sketch_dirty = True
        # feature width -> {bits, owner, categories, seen, vectors}
        self._sk_by_dim: dict[int, dict[str, Any]] = {}
        if self.catalog_path.exists():
            self._load_catalog()
        self._rebuild_lookups()

    # ------------------------------------------------------------------ #
    # derived lookups
    # ------------------------------------------------------------------ #

    @contextlib.contextmanager
    def batch(self):
        """Group mutations so the catalog is written once, not per change.

        Every mutation persists the catalog, which is right for a single edit
        and quadratic for a bulk one: reinforcing a category with N shards would
        otherwise rewrite the whole catalog N times. Measured on a 20,000-shard
        library that turned one `learn()` call into 46 seconds.
        """
        self._defer_save += 1
        try:
            yield self
        finally:
            self._defer_save -= 1
            if self._defer_save == 0 and self._dirty:
                self._dirty = False
                self._write_catalog()

    def _rebuild_lookups(self) -> None:
        """Rebuild the category and association indexes from the catalog."""
        self._by_category = {}
        self._assoc = {}
        self._sketch_dirty = True
        for entry in self._catalog.values():
            self._index_entry(entry, +1)

    def _index_entry(self, entry: dict[str, Any], sign: int) -> None:
        """Add (+1) or remove (-1) one entry's contribution to the indexes."""
        shard_id = entry["shard_id"]
        category = entry["category"]
        if entry.get("signature"):
            self._sketch_dirty = True
        members = self._by_category.setdefault(category, [])
        if sign > 0:
            if shard_id not in members:
                members.append(shard_id)
        elif shard_id in members:
            members.remove(shard_id)

        for keyword, weight in (entry.get("associations") or {}).items():
            key = keyword.lower()
            per_category = self._assoc.setdefault(key, {})
            per_category[category] = per_category.get(category, 0.0) + sign * float(weight)
            if per_category[category] <= 0.0:
                per_category.pop(category, None)
            if not per_category:
                self._assoc.pop(key, None)

    def libraries(self) -> list[str]:
        """Library segments currently referenced by the catalog, in use order."""
        seen: list[str] = []
        for entry in self._catalog.values():
            path = entry.get("library_path")
            if path and path not in seen:
                seen.append(path)
        return seen

    def compact(self, target: str | os.PathLike, remove_old: bool = True) -> dict:
        """Merge every library segment into one, dropping anything unreferenced.

        Expansion writes a new segment each time categories are added, which
        keeps existing files immutable — safe, but it accumulates segments, and a
        shard that was retrained leaves its old bytes stranded in an older
        segment where nothing points at them any more. Compaction rewrites the
        live set into a single file and reclaims that space, in the same spirit
        as a log-structured store merging its levels.

        Args:
            target: Path for the merged library.
            remove_old: Delete the superseded segments once every shard has been
                re-read from the new file and verified.

        Returns:
            ``{"segments_before", "shards", "bytes_before", "bytes_after",
            "reclaimed"}``.
        """
        before = self.libraries()
        packed_ids = [
            shard_id for shard_id, entry in self._catalog.items()
            if entry["location"] == "library"
        ]
        bytes_before = 0
        for name in before:
            path = Path(name)
            if not path.is_absolute():
                path = self.root / path
            if path.exists():
                bytes_before += path.stat().st_size

        if not packed_ids:
            return {
                "segments_before": len(before), "shards": 0,
                "bytes_before": bytes_before, "bytes_after": 0, "reclaimed": 0,
            }

        target_path = Path(target)
        old_paths = []
        for name in before:
            path = Path(name)
            if not path.is_absolute():
                path = self.root / path
            if path.resolve() != target_path.resolve():
                old_paths.append(path)

        self.pack(target_path, packed_ids, prune_loose=False)

        # Only now that every shard reads correctly from the merged file may the
        # old segments go.
        for shard_id in packed_ids:
            self.get(shard_id)
        bytes_after = target_path.stat().st_size if target_path.exists() else 0
        if remove_old:
            for path in old_paths:
                try:
                    path.unlink()
                except OSError:
                    pass

        return {
            "segments_before": len(before),
            "shards": len(packed_ids),
            "bytes_before": bytes_before,
            "bytes_after": bytes_after,
            "reclaimed": max(0, bytes_before - bytes_after),
        }

    def set_signature(self, shard_id: str, signature: list[float]) -> None:
        """Attach a compact content signature to a shard's catalog entry.

        Routing a *pattern* cannot use keywords — a glyph has no words in it —
        so it must compare the input against what each category looks like. Doing
        that by reading shards would page the whole library on every input, which
        defeats the point, so a small fingerprint of the category's prototypes
        lives in the catalog instead. Tens of floats per category, and no paging.

        One vector *per prototype* is stored rather than their mean: averaging
        blurs distinct shapes together, and a pattern then matches whichever
        category's average happens to sit nearest it. Measured on the glyph set,
        mean-pooling sent ``plus`` to ``arrows`` because both arrows carry a
        central stem.

        Vectors are held as :class:`array.array` of doubles rather than lists of
        Python floats. This is a memory decision, not a style one: at the 20,000
        category design point the signatures are 1.5 million float objects, and
        CPython charges roughly 32 bytes for each once the object header and the
        list's pointer are counted — **46 MB of the 74 MB resident catalog**.
        The same numbers in a ``'d'`` array cost 12 MB and are bit-identical, so
        nothing downstream has to reason about precision. The on-disk JSON is
        unchanged; only the in-memory representation differs.

        Args:
            shard_id: The shard to fingerprint.
            signature: One vector, or a list of per-prototype vectors.
        """
        entry = self._catalog[shard_id]
        if signature and isinstance(signature[0], (list, tuple, array)):
            rows = signature
        else:
            rows = [signature]
        entry["signature"] = [
            array("d", (round(float(v), 4) for v in row)) for row in rows
        ]
        self._sketch_dirty = True
        self._save_catalog()

    def set_slot(self, shard_id: str, slot: str) -> None:
        """Declare which *aspect* of an input a category describes.

        Two experts with the same slot are alternatives and compete for it; two
        with different slots describe different aspects and compose. Everything
        built before this defaulted to one slot, which is why the library has
        behaved as a set of alternatives: not because composition was impossible
        but because nothing ever said the categories were about different things.

        Args:
            shard_id: The expert shard.
            slot: The aspect it describes, e.g. ``"shape"`` or ``"mark"``.
        """
        self._catalog[shard_id]["slot"] = str(slot)
        self._save_catalog()

    def slot_of(self, category: str) -> str:
        """The slot a category fills; ``"pattern"`` when it never said."""
        for shard_id in self._by_category.get(category, ()):
            slot = self._catalog[shard_id].get("slot")
            if slot:
                return str(slot)
        return _DEFAULT_SLOT

    def slots(self) -> dict[str, str]:
        """``category -> slot`` for every category that has an expert."""
        return {category: self.slot_of(category) for category in self._by_category}

    def signatures(self) -> dict[str, list[float]]:
        """Every category's content signature, for content-based routing."""
        out: dict[str, list[float]] = {}
        for entry in self._catalog.values():
            signature = entry.get("signature")
            if signature:
                out.setdefault(entry["category"], [list(row) for row in signature])
        return out

    # ------------------------------------------------------------------ #
    # sketch screening
    # ------------------------------------------------------------------ #

    def _ensure_sketches(self) -> None:
        """Build the resident sketch index if the catalog has moved on.

        Rebuilt lazily rather than maintained incrementally: signatures are
        written during training and read during routing, so the rebuild lands
        once at the boundary instead of on every ``set_signature`` in a batch.

        The index is deliberately array-backed. The obvious Python shape — a
        dict of lists of per-prototype tuples — costs more memory than the
        signatures it is meant to screen, which would make it worse than
        useless.
        """
        if not self._sketch_dirty:
            return
        # Indexed *per feature width*. Sketches are only comparable within one
        # dimension — the hyperplanes differ — and more importantly a library
        # may hold experts over different modalities at once, where comparing a
        # 30-input text vector against a 25-input glyph prototype is not a
        # near-miss but a category error. Segregating by width makes that
        # impossible rather than merely unlikely.
        by_dim: dict[int, dict[str, Any]] = {}
        for entry in self._catalog.values():
            signature = entry.get("signature")
            if not signature:
                continue
            category = entry["category"]
            for vector in signature:
                width = len(vector)
                bucket = by_dim.get(width)
                if bucket is None:
                    bucket = by_dim[width] = {
                        "bits": array("Q"), "owner": array("I"),
                        "categories": [], "seen": {}, "vectors": [],
                    }
                index = bucket["seen"].get(category)
                if index is None:
                    index = len(bucket["categories"])
                    bucket["seen"][category] = index
                    bucket["categories"].append(category)
                bucket["bits"].append(sketch(vector))
                bucket["owner"].append(index)
                # A reference, not a copy: the exact rerank needs the real
                # numbers and the catalog is already holding them.
                bucket["vectors"].append(vector)
        self._sk_by_dim = by_dim
        self._sketch_dirty = False

    def sketch_index_bytes(self) -> int:
        """Resident size of the sketch index itself, in bytes."""
        self._ensure_sketches()
        total = 0
        for bucket in self._sk_by_dim.values():
            for key in ("bits", "owner"):
                buffer = bucket[key]
                total += buffer.buffer_info()[1] * buffer.itemsize
        return total

    def signature_widths(self) -> dict[int, int]:
        """Feature width -> number of stored prototypes at that width.

        A library holding more than one width is holding more than one
        modality, which is the point of allowing it.
        """
        self._ensure_sketches()
        return {width: len(bucket["bits"])
                for width, bucket in sorted(self._sk_by_dim.items())}

    def screen_signatures(
        self, pixels: list[float], limit: int = 64
    ) -> list[tuple[str, list[list[float]]]]:
        """Shortlist the categories worth scoring exactly against ``pixels``.

        Screens on sketch Hamming distance, which costs one XOR and one
        ``bit_count`` per prototype instead of a 25-element dot product, then
        returns the owning categories of the closest ``limit`` prototypes.

        Below ``_SCREEN_ABOVE`` prototypes the screen is skipped entirely and
        everything is returned: at that size the exact scan is already
        microseconds, and an approximate step could only lose recall for no
        measurable gain.

        Args:
            pixels: The query pattern.
            limit: How many prototypes survive the screen.

        Returns:
            ``(category, prototype vectors)`` pairs to score exactly.
        """
        self._ensure_sketches()
        # Only prototypes of the query's own width are candidates at all.
        bucket = self._sk_by_dim.get(len(pixels))
        if bucket is None:
            return []
        bits = bucket["bits"]
        owner = bucket["owner"]
        names = bucket["categories"]
        stored = bucket["vectors"]

        groups: dict[str, list[list[float]]] = {}
        if len(bits) <= max(limit, _SCREEN_ABOVE):
            for position, index in enumerate(owner):
                groups.setdefault(names[index], []).append(stored[position])
            return list(groups.items())

        query = sketch(pixels)
        # nsmallest over a generator beats sorting the whole array: the pool is
        # tiny next to the library and the scan stays O(prototypes) with a
        # constant of two machine instructions.
        best = heapq.nsmallest(
            limit,
            range(len(bits)),
            key=lambda position: (query ^ bits[position]).bit_count(),
        )
        for position in best:
            groups.setdefault(names[owner[position]], []).append(stored[position])
        return list(groups.items())

    def shards_in(self, category: str) -> list[str]:
        """Shard ids belonging to ``category``, without scanning the catalog."""
        return list(self._by_category.get(category, ()))

    def categories(self) -> list[str]:
        """Every category present, sorted."""
        return sorted(self._by_category)

    def association_scores(self, tokens: set[str]) -> dict[str, float]:
        """Summed association strength per category for the given tokens.

        Costs one dictionary lookup per token rather than a pass over the
        library, which is what keeps routing independent of model size.

        Args:
            tokens: Distinct lowercase query tokens.

        Returns:
            ``{category: summed weight}`` for categories with any match.
        """
        scores: dict[str, float] = {}
        for token in tokens:
            for category, weight in self._assoc.get(token.lower(), {}).items():
                scores[category] = scores.get(category, 0.0) + weight
        return scores

    def storage_key(self, shard_id: str) -> str:
        """The storage key a shard's bytes are read from.

        For a loose shard that is its ``.uqs`` object; for a packed shard it is
        the containing library, since the shard is a byte range inside it.
        """
        entry = self._catalog[shard_id]
        if entry["location"] == "loose":
            return f"loose/{_safe_filename(shard_id)}.uqs"
        library = Path(entry["library_path"])
        try:
            return library.relative_to(self.root).as_posix()
        except ValueError:
            return library.as_posix()

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #

    def _load_catalog(self) -> None:
        """Load the catalog from ``root/catalog.json``."""
        with open(self.catalog_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self._catalog = {}
        # json.load builds a fresh string object per occurrence, so a 20,000
        # shard catalog holds 20,000 copies of "expert-net", of "zlib", and of
        # the same library path. Folding the repeats onto one object each is
        # free and saves megabytes.
        pool: dict[str, str] = {}
        for entry in payload.get("shards", []):
            for field in ("category", "kind", "codec", "location", "library_path"):
                value = entry.get(field)
                if isinstance(value, str):
                    entry[field] = pool.setdefault(value, value)
            signature = entry.get("signature")
            if signature:
                # Straight from json.load these are lists of Python floats, the
                # single largest resident cost in a big catalog. Pack them.
                entry["signature"] = [array("d", row) for row in signature]
            self._catalog[entry["shard_id"]] = entry

    def _save_catalog(self) -> None:
        """Persist the catalog, unless a :meth:`batch` is collecting changes."""
        if self._defer_save:
            self._dirty = True
            return
        self._write_catalog()

    def _write_catalog(self) -> None:
        """Atomically write the catalog to ``root/catalog.json``."""
        payload = {"shards": list(self._catalog.values())}
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          default=_json_default)
        tmp = self.catalog_path.parent / (self.catalog_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, self.catalog_path)

    def _loose_path(self, shard_id: str) -> Path:
        """Return the on-disk path of ``shard_id``'s loose ``.uqs`` file."""
        return self.loose_dir / (_safe_filename(shard_id) + ".uqs")

    # ------------------------------------------------------------------ #
    # writing shards
    # ------------------------------------------------------------------ #

    def add_shard(
        self,
        shard_id: str,
        category: str,
        payload: dict,
        kind: str = "expert-net",
        associations: dict[str, float] | None = None,
    ) -> dict:
        """Serialize, compress, and store ``payload`` as a loose shard.

        The payload is canonically serialized
        (``json.dumps(sort_keys=True, separators=(",", ":"))``, utf-8),
        zlib-compressed, and written to ``root/loose/<safe-name>.uqs``.  A
        catalog entry (location ``"loose"``) is recorded.

        Re-adding an existing ``shard_id`` overwrites the bytes and refreshes
        the entry but **preserves** the accumulated ``access_count``,
        ``associations`` and ``signature`` — learning survives re-training.  Any
        ``associations`` supplied on a re-add are merged in without lowering
        already-learned strengths.

        Carrying the signature over matters as much as the associations: it is
        what pattern routing matches against, so dropping it on a re-train would
        make a category's own glyphs unroutable — trained, stored, and
        unreachable.

        Args:
            shard_id: Unique identifier for the shard.
            category: The category / expert this shard belongs to.
            payload: JSON-serializable dict of shard data (e.g. a net state).
            kind: Free-form shard kind label.
            associations: Optional initial keyword -> strength weights.

        Returns:
            A copy of the recorded catalog entry.
        """
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        stored = zlib.compress(data)
        digest = hashlib.sha256(stored).hexdigest()
        self._loose_path(shard_id).write_bytes(stored)

        now = _utc_now()
        existing = self._catalog.get(shard_id)
        if existing is not None:
            access_count = int(existing["access_count"])
            last_access = existing["last_access"]
            created = existing["created"]
            assoc: dict[str, float] = dict(existing["associations"])
            signature = existing.get("signature")
            slot = existing.get("slot")
        else:
            access_count = 0
            last_access = None
            created = now
            assoc = {}
            signature = None
            slot = None
        if associations:
            for key, weight in associations.items():
                # Seed new keys / raise weak ones, but never lower learning.
                assoc[key] = min(_ASSOC_CAP, max(float(weight), assoc.get(key, 0.0)))

        entry = {
            "shard_id": shard_id,
            "category": category,
            "kind": kind,
            "codec": "zlib",
            "nbytes": len(stored),
            "sha256": digest,
            "created": created,
            "location": "loose",
            "library_path": None,
            "offset": 0,
            "length": len(stored),
            "access_count": access_count,
            "last_access": last_access,
            "associations": assoc,
        }
        if signature:
            entry["signature"] = signature
        if slot:
            entry["slot"] = slot
        prior = self._catalog.get(shard_id)
        if prior is not None:
            self._index_entry(prior, -1)
        self._catalog[shard_id] = entry
        self._index_entry(entry, +1)
        self._save_catalog()
        return self._copy_entry(entry)

    # ------------------------------------------------------------------ #
    # reading shards
    # ------------------------------------------------------------------ #

    def load_bytes(self, shard_id: str) -> bytes:
        """Return the raw *stored* bytes of ``shard_id``.

        For a loose shard the whole ``.uqs`` file is read.  For a library
        shard only the shard's byte range is read: the ``.uql`` is opened, the
        stream is ``seek``-ed to the absolute ``offset``, and exactly
        ``length`` bytes are read — never the rest of the library.

        Args:
            shard_id: The shard to read.

        Returns:
            The stored (compressed) bytes.

        Raises:
            KeyError: If ``shard_id`` is not in the catalog.
        """
        entry = self._catalog[shard_id]

        if self.storage is not None:
            key = self.storage_key(shard_id)
            if entry["location"] == "loose":
                return self.storage.read_all(key)
            return self.storage.read_range(
                key, int(entry["offset"]), int(entry["length"])
            )

        if entry["location"] == "loose":
            return self._loose_path(shard_id).read_bytes()
        library_path = Path(entry["library_path"])
        with open(library_path, "rb") as fh:
            fh.seek(int(entry["offset"]))
            return fh.read(int(entry["length"]))

    def get(self, shard_id: str) -> dict:
        """Load, verify, and decode a shard's payload; count the access.

        Args:
            shard_id: The shard to fetch.

        Returns:
            The decoded payload dict.

        Raises:
            KeyError: If ``shard_id`` is not in the catalog.
            ShardIntegrityError: If the stored bytes fail their sha256 check.
        """
        entry = self._catalog[shard_id]
        raw = self.load_bytes(shard_id)
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ShardIntegrityError(
                f"shard {shard_id!r} failed integrity check "
                f"(expected {entry['sha256']})"
            )
        data = zlib.decompress(raw) if entry["codec"] == "zlib" else raw
        payload = json.loads(data.decode("utf-8"))
        self.touch(shard_id)
        return payload

    # ------------------------------------------------------------------ #
    # reinforcement
    # ------------------------------------------------------------------ #

    def touch(self, shard_id: str) -> None:
        """Record an access: increment ``access_count`` and stamp ``last_access``."""
        entry = self._catalog[shard_id]
        entry["access_count"] = int(entry["access_count"]) + 1
        entry["last_access"] = _utc_now()
        self._save_catalog()

    def prune_associations(self) -> float:
        """Drop uninformative keywords from every shard's associations.

        The router reinforces a shard with the same tokens it learns, so the
        vault accumulated the same stopword weight — and pruning only the
        router's own table left it. Measured on the deployed library: after the
        router table was cleaned, decoy queries still routed on **vault**
        association alone, scoring 0.7 to 1.2 with no base-keyword hit and no
        learned weight at all.

        Returns:
            Total association weight removed.
        """
        from ultraquant.shards.router import _informative

        removed = 0.0
        with self.batch():
            for entry in self._catalog.values():
                assoc = entry.get("associations") or {}
                drop = [k for k in assoc if not _informative(k)]
                if not drop:
                    continue
                self._index_entry(entry, -1)
                for key in drop:
                    removed += float(assoc.pop(key))
                self._index_entry(entry, +1)
        if removed:
            self._save_catalog()
        return removed

    def reinforce(
        self, shard_id: str, keywords: list[str], delta: float = 0.1
    ) -> None:
        """Strengthen this shard's associations for each of ``keywords``.

        ``associations[k] = min(5.0, associations.get(k, 0) + delta)``.

        Args:
            shard_id: The shard whose associations to strengthen.
            keywords: Keywords to reinforce (duplicates are collapsed).
            delta: Strength increment per distinct keyword.
        """
        entry = self._catalog[shard_id]
        self._index_entry(entry, -1)
        assoc: dict[str, float] = entry["associations"]
        for key in dict.fromkeys(keywords):
            assoc[key] = min(_ASSOC_CAP, assoc.get(key, 0.0) + delta)
        self._index_entry(entry, +1)
        self._save_catalog()

    # ------------------------------------------------------------------ #
    # consolidation into .uql libraries
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_index(records: list[dict[str, Any]]) -> tuple[list[dict], bytes]:
        """Compute the ``.uql`` index (with absolute offsets) and its bytes.

        The first payload sits at ``12 + len(index_bytes)``, but the index's
        serialized length depends on the offset values it contains — a fixpoint.
        Because larger offsets never shorten the index, iterating from the
        smallest possible base converges monotonically in a couple of passes.

        Args:
            records: Per-shard records carrying ``length`` and metadata.

        Returns:
            ``(index, index_bytes)`` — the index list and its utf-8 encoding.
        """
        base = _HEADER_BYTES
        while True:
            offset = base
            index: list[dict] = []
            for rec in records:
                item = {
                    "shard_id": rec["shard_id"],
                    "offset": offset,
                    "length": rec["length"],
                    "sha256": rec["sha256"],
                    "codec": rec["codec"],
                    "category": rec["category"],
                    "kind": rec["kind"],
                }
                # The signature travels with the library. Without it a packed
                # library is unroutable by pattern on any machine that did not
                # train it: the expert is present, correct, and unreachable.
                if rec.get("signature"):
                    item["signature"] = rec["signature"]
                if rec.get("slot"):
                    item["slot"] = rec["slot"]
                index.append(item)
                offset += rec["length"]
            index_bytes = json.dumps(
                index, sort_keys=True, separators=(",", ":"), default=_json_default
            ).encode("utf-8")
            new_base = _HEADER_BYTES + len(index_bytes)
            if new_base == base:
                return index, index_bytes
            base = new_base

    def pack(
        self,
        library_path: str | os.PathLike,
        shard_ids: list[str] | None = None,
        prune_loose: bool = False,
    ) -> int:
        """Consolidate shards into a single ``.uql`` library file.

        Writes the magic, the 8-byte big-endian index length, the JSON index,
        and the concatenated stored payloads.  Each packed shard's catalog entry
        is updated to location ``"library"`` with its absolute ``offset`` and
        ``length``.

        If ``prune_loose`` is set, each loose ``.uqs`` file is deleted **only
        after** its chunk has been re-read from the freshly written library and
        its sha256 re-verified.

        Args:
            library_path: Destination ``.uql`` path.
            shard_ids: Shards to pack; ``None`` packs every catalogued shard in
                insertion order.
            prune_loose: Delete loose files after a verified round-trip.

        Returns:
            The number of shards packed.

        Raises:
            KeyError: If a requested ``shard_id`` is unknown.
            ShardIntegrityError: If a post-pack verification read fails.
        """
        lib_path = Path(library_path)
        lib_path.parent.mkdir(parents=True, exist_ok=True)
        ids = list(self._catalog.keys()) if shard_ids is None else list(shard_ids)

        records: list[dict[str, Any]] = []
        for shard_id in ids:
            entry = self._catalog[shard_id]
            stored = self.load_bytes(shard_id)
            records.append(
                {
                    "shard_id": shard_id,
                    "stored": stored,
                    "length": len(stored),
                    "sha256": entry["sha256"],
                    "codec": entry["codec"],
                    "category": entry["category"],
                    "kind": entry["kind"],
                    "signature": entry.get("signature"),
                    "slot": entry.get("slot"),
                }
            )

        index, index_bytes = self._build_index(records)
        with open(lib_path, "wb") as fh:
            fh.write(_MAGIC)
            fh.write(len(index_bytes).to_bytes(8, "big"))
            fh.write(index_bytes)
            for rec in records:
                fh.write(rec["stored"])

        library_str = str(lib_path)
        for idx_entry in index:
            entry = self._catalog[idx_entry["shard_id"]]
            entry["location"] = "library"
            entry["library_path"] = library_str
            entry["offset"] = idx_entry["offset"]
            entry["length"] = idx_entry["length"]
        self._save_catalog()

        if prune_loose:
            for shard_id in ids:
                chunk = self.load_bytes(shard_id)  # now reads from the library
                if hashlib.sha256(chunk).hexdigest() != self._catalog[shard_id][
                    "sha256"
                ]:
                    raise ShardIntegrityError(
                        f"post-pack verification failed for {shard_id!r}"
                    )
                loose = self._loose_path(shard_id)
                if loose.exists():
                    loose.unlink()

        return len(records)

    def pack_sharded(
        self,
        library_dir: str | os.PathLike,
        shard_ids: list[str] | None = None,
        max_part_bytes: int = 64 * 1024 * 1024,
        max_shards_per_part: int | None = None,
        prune_loose: bool = False,
        prefix: str = "part",
    ) -> dict:
        """Pack shards into a directory of bounded, independently-readable parts.

        One enormous ``.uql`` is the wrong shape once a library is real: it
        cannot be synced incrementally, replicated in pieces, spread across
        volumes, or handed to an object store, and touching any shard rewrites
        or shadows the whole file. Large models ship as
        ``part-00001-of-000NN`` for exactly these reasons, and this does the
        same.

        **Every part is a complete ``.uql``** — its own magic, its own index,
        its own payloads — so a single part can be copied, moved, replicated or
        served on its own and still read correctly. Nothing depends on a part's
        neighbours.

        Args:
            library_dir: Destination directory for the parts.
            shard_ids: Shards to pack; defaults to every catalogued shard.
            max_part_bytes: Roll to a new part once this is exceeded. A shard
                larger than the cap gets a part to itself rather than being
                split, because a shard is the unit of a ranged read.
            max_shards_per_part: Optional cap on shards per part; set to 1 for
                one file per shard.
            prune_loose: Delete loose files after a verified round-trip.
            prefix: Base name for the part files.

        Returns:
            ``{"parts", "shards", "bytes", "manifest", "part_files"}``.

        Raises:
            KeyError: If a requested ``shard_id`` is unknown.
            ShardIntegrityError: If a post-pack verification read fails.
        """
        directory = Path(library_dir)
        directory.mkdir(parents=True, exist_ok=True)
        ids = list(self._catalog.keys()) if shard_ids is None else list(shard_ids)
        if not ids:
            return {"parts": 0, "shards": 0, "bytes": 0,
                    "manifest": None, "part_files": []}

        # Group shards into parts before writing anything, so part numbering can
        # carry the usual "k of n" suffix.
        groups: list[list[str]] = []
        current: list[str] = []
        current_bytes = 0
        for shard_id in ids:
            size = int(self._catalog[shard_id]["nbytes"])
            too_big = current and (current_bytes + size > max_part_bytes)
            too_many = (
                max_shards_per_part is not None
                and len(current) >= max_shards_per_part
            )
            if too_big or too_many:
                groups.append(current)
                current, current_bytes = [], 0
            current.append(shard_id)
            current_bytes += size
        if current:
            groups.append(current)

        total = len(groups)
        part_files: list[str] = []
        written = 0
        for number, group in enumerate(groups, start=1):
            part = directory / f"{prefix}-{number:05d}-of-{total:05d}.uql"
            self.pack(part, group, prune_loose=False)
            part_files.append(part.name)
            written += part.stat().st_size

        # Expansion writes further parts into the same directory, so the
        # manifest accumulates rather than being replaced — otherwise each wave
        # would erase every earlier part from the record while leaving the files
        # themselves on disk.
        manifest = directory / "library.json"
        known: list[str] = []
        if manifest.exists():
            try:
                with open(manifest, "r", encoding="utf-8") as handle:
                    known = list(json.load(handle).get("parts", []))
            except (OSError, ValueError):
                known = []
        combined = known + [name for name in part_files if name not in known]
        payload = {
            "format": "uql-sharded/1",
            "created": _utc_now(),
            "parts": combined,
            "shards": len({
                shard_id for shard_id, entry in self._catalog.items()
                if entry["location"] == "library"
            }),
            "max_part_bytes": int(max_part_bytes),
        }
        tmp = manifest.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
        os.replace(tmp, manifest)

        if prune_loose:
            for shard_id in ids:
                chunk = self.load_bytes(shard_id)
                if hashlib.sha256(chunk).hexdigest() != self._catalog[shard_id]["sha256"]:
                    raise ShardIntegrityError(
                        f"post-pack verification failed for {shard_id!r}"
                    )
                loose = self._loose_path(shard_id)
                if loose.exists():
                    loose.unlink()

        return {
            "parts": total,
            "shards": len(ids),
            "bytes": written,
            "manifest": str(manifest),
            "part_files": part_files,
        }

    def attach_directory(self, library_dir: str | os.PathLike) -> dict:
        """Attach every part of a sharded library, reading only their indexes.

        Payload bytes stay on disk; this reads each part's header and index so
        the catalog learns where every shard lives. A missing part is reported
        rather than raised, because a partially replicated library is a normal
        state — the shards that did arrive remain usable.

        Returns:
            ``{"parts", "shards", "missing"}``.
        """
        directory = Path(library_dir)
        manifest = directory / "library.json"
        if manifest.exists():
            with open(manifest, "r", encoding="utf-8") as handle:
                names = json.load(handle).get("parts", [])
        else:
            names = sorted(p.name for p in directory.glob("*.uql"))

        attached = 0
        shards = 0
        missing: list[str] = []
        for name in names:
            path = directory / name
            if not path.exists():
                missing.append(name)
                continue
            shards += self.attach(path)
            attached += 1
        return {"parts": attached, "shards": shards, "missing": missing}

    def attach(self, library_path: str | os.PathLike) -> int:
        """Merge a ``.uql`` library's index into the catalog.

        Reads **only** the magic and the index (never the payload bytes).  New
        ``shard_id``s are added with a fresh ``access_count`` of 0 and empty
        ``associations``; already-known ids keep their statistics while being
        re-pointed at this library.

        Args:
            library_path: The ``.uql`` file to attach.

        Returns:
            The number of shards attached.

        Raises:
            ShardIntegrityError: If the file is not a ``UQL1`` container.
        """
        lib_path = Path(library_path)
        with open(lib_path, "rb") as fh:
            magic = fh.read(4)
            if magic != _MAGIC:
                raise ShardIntegrityError(
                    f"{lib_path} is not a UQL1 library (bad magic {magic!r})"
                )
            index_len = int.from_bytes(fh.read(8), "big")
            index_bytes = fh.read(index_len)
        index = json.loads(index_bytes.decode("utf-8"))

        library_str = str(lib_path)
        now = _utc_now()
        count = 0
        for idx_entry in index:
            shard_id = idx_entry["shard_id"]
            existing = self._catalog.get(shard_id)
            if existing is not None:
                access_count = int(existing["access_count"])
                last_access = existing["last_access"]
                created = existing["created"]
                assoc = dict(existing["associations"])
                # Same rule as a re-add: attaching a library relocates a shard's
                # bytes, it does not un-learn what the category looks like. What
                # is already known locally wins over what the library shipped.
                signature = existing.get("signature") or idx_entry.get("signature")
                slot = existing.get("slot") or idx_entry.get("slot")
            else:
                access_count = 0
                last_access = None
                created = now
                assoc = {}
                signature = idx_entry.get("signature")
                slot = idx_entry.get("slot")
            if signature:
                signature = [array("d", row) for row in signature]
            self._catalog[shard_id] = {
                "shard_id": shard_id,
                "category": idx_entry["category"],
                "kind": idx_entry["kind"],
                "codec": idx_entry["codec"],
                "nbytes": int(idx_entry["length"]),
                "sha256": idx_entry["sha256"],
                "created": created,
                "location": "library",
                "library_path": library_str,
                "offset": int(idx_entry["offset"]),
                "length": int(idx_entry["length"]),
                "access_count": access_count,
                "last_access": last_access,
                "associations": assoc,
            }
            if signature:
                self._catalog[shard_id]["signature"] = signature
            if slot:
                self._catalog[shard_id]["slot"] = slot
            count += 1
        # Newly attached shards are absent from the derived lookups until this
        # runs, which would make them unroutable by both keyword and pattern.
        self._rebuild_lookups()
        self._save_catalog()
        return count

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #

    @staticmethod
    def _copy_entry(entry: dict[str, Any]) -> dict[str, Any]:
        """Return a deep-enough copy of a catalog entry (associations copied)."""
        copy = dict(entry)
        copy["associations"] = dict(entry["associations"])
        signature = entry.get("signature")
        if signature:
            # Callers get plain lists; the compact form is an internal detail.
            copy["signature"] = [list(row) for row in signature]
        return copy

    def has(self, shard_id: str) -> bool:
        """Return whether ``shard_id`` is present in the catalog."""
        return shard_id in self._catalog

    def entry(self, shard_id: str) -> dict:
        """Return a copy of the catalog entry for ``shard_id``."""
        return self._copy_entry(self._catalog[shard_id])

    def catalog(self) -> list[dict]:
        """Return copies of all catalog entries, in insertion order."""
        return [self._copy_entry(entry) for entry in self._catalog.values()]

    def stats(self) -> dict:
        """Return aggregate store statistics (JSON-safe).

        Returns:
            ``{shards, loose_bytes, library_bytes, total_bytes, libraries}``
            where the ``*_bytes`` figures sum each shard's stored ``nbytes`` by
            location and ``libraries`` is the sorted set of library paths.
        """
        loose_bytes = 0
        library_bytes = 0
        libraries: set[str] = set()
        for entry in self._catalog.values():
            if entry["location"] == "loose":
                loose_bytes += int(entry["nbytes"])
            else:
                library_bytes += int(entry["nbytes"])
                if entry["library_path"]:
                    libraries.add(entry["library_path"])
        return {
            "shards": len(self._catalog),
            "loose_bytes": loose_bytes,
            "library_bytes": library_bytes,
            "total_bytes": loose_bytes + library_bytes,
            "libraries": sorted(libraries),
        }
