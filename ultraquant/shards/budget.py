"""Resident-set management under a byte budget: the on-demand paging mechanism.

:class:`ShardCache` is the "load chunks on demand, swap in and out on the fly"
layer.  Callers ask for a shard by id and supply a *loader* that materialises it
only on a miss.  The cache keeps recently used shards resident, evicting the
least-recently-used ones whenever the resident set would exceed the byte budget
— exactly how a store far larger than RAM is paged: the catalog picks the shard,
the budget decides what stays resident.

A single shard larger than the whole budget is permitted to be resident alone
(everything else is evicted to make room); this keeps oversized experts usable
rather than un-loadable.

Pure Python standard library only.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

__all__ = ["ShardCache"]


class ShardCache:
    """Least-recently-used shard cache bounded by a total byte budget.

    The cache stores decoded payloads keyed by ``shard_id`` in
    least-recently-used to most-recently-used order.  Each resident shard costs
    the ``nbytes`` its loader reports; the sum of resident costs is kept at or
    below ``max_bytes`` except for the single-oversize case described above.

    Args:
        max_bytes: The resident-set byte budget.
    """

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes: int = int(max_bytes)
        # Ordered least-recently-used (front) -> most-recently-used (back).
        self._entries: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self.current_bytes: int = 0
        self.peak_bytes: int = 0
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0

    def is_resident(self, shard_id: str) -> bool:
        """Whether this shard is in RAM, WITHOUT making it so.

        Read-only on purpose: it does not touch the recency order, so
        asking cannot change what gets evicted next. §11.104 calls it
        before each consultation to tell a read that cost a page-in
        from one that cost nothing, and a probe that promoted its own
        subject would make that count wrong.
        """
        return shard_id in self._entries

    # ------------------------------------------------------------------ #
    # core operation
    # ------------------------------------------------------------------ #

    def get(self, shard_id: str, loader: Callable[[], tuple[dict, int]]) -> dict:
        """Return ``shard_id``'s payload, loading and caching it on a miss.

        On a hit the shard is promoted to most-recently-used and returned.  On
        a miss ``loader()`` is called for ``(payload, nbytes)``; the shard is
        inserted as most-recently-used, then least-recently-used shards are
        evicted until the resident set fits the budget (leaving at most the new
        shard resident if it alone exceeds the budget).

        Args:
            shard_id: Identifier of the desired shard.
            loader: Zero-argument callable returning ``(payload, nbytes)``; only
                invoked on a miss.

        Returns:
            The (possibly freshly loaded) payload dict.
        """
        entry = self._entries.get(shard_id)
        if entry is not None:
            self.hits += 1
            self._entries.move_to_end(shard_id)
            return entry["payload"]

        self.misses += 1
        payload, nbytes = loader()
        nbytes = int(nbytes)
        self._entries[shard_id] = {"payload": payload, "nbytes": nbytes}
        self._entries.move_to_end(shard_id)
        self.current_bytes += nbytes
        self._evict_to_budget()
        if self.current_bytes > self.peak_bytes:
            self.peak_bytes = self.current_bytes
        return payload

    def _evict_to_budget(self) -> None:
        """Evict least-recently-used shards until within budget.

        Stops once the resident set fits ``max_bytes`` or only one shard
        remains (that lone shard is allowed to exceed the budget).  The
        most-recently-used shard — including one just inserted — is therefore
        never evicted while any other candidate exists.
        """
        while self.current_bytes > self.max_bytes and len(self._entries) > 1:
            shard_id, entry = next(iter(self._entries.items()))  # LRU end
            self.current_bytes -= entry["nbytes"]
            del self._entries[shard_id]
            self.evictions += 1

    # ------------------------------------------------------------------ #
    # management
    # ------------------------------------------------------------------ #

    def invalidate(self, shard_id: str) -> None:
        """Drop ``shard_id`` from the resident set if present (no eviction count)."""
        entry = self._entries.pop(shard_id, None)
        if entry is not None:
            self.current_bytes -= entry["nbytes"]

    def set_budget(self, max_bytes: int) -> None:
        """Set a new byte budget, evicting immediately to fit it.

        Args:
            max_bytes: The new resident-set budget.
        """
        self.max_bytes = int(max_bytes)
        self._evict_to_budget()

    def resident(self) -> list[str]:
        """Return resident shard ids in least-recently-used to most-recent order."""
        return list(self._entries.keys())

    def stats(self) -> dict:
        """Return cache accounting (JSON-safe).

        Returns:
            ``{hits, misses, evictions, current_bytes, peak_bytes,
            budget_bytes, resident}``.
        """
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "current_bytes": self.current_bytes,
            "peak_bytes": self.peak_bytes,
            "budget_bytes": self.max_bytes,
            "resident": list(self._entries.keys()),
        }
