"""Facts in the library, rather than beside it.

Facts were stored in ``memory.json`` and nowhere else. Everything else the model
learns is a catalogued shard — paged on demand, reinforced on use, reachable
through the vault's keyword index, and carried inside a packed ``.uql`` when the
library is copied to another machine. Facts had none of that, which broke four
stated properties at once:

* **Never load what you don't need.** ``memory.json`` is read whole. At a million
  facts the entire semantic store is resident before the first question.
* **The library is the model.** Packing a library and shipping it left every fact
  behind, because the vault did not know they existed.
* **Routing.** The vault keeps an inverted keyword index that makes finding the
  right shard O(1); facts were invisible to it, so anything wanting to know
  "what do I hold about bridges" had to scan the lot.
* **Recall reinforces.** Shards accumulate access counts and association weights
  on use. Facts had their own confidence, but none of the associative structure
  that routing actually reads.

Facts are therefore grouped into **bucket shards**: a fact's key hashes to one
bucket, so a lookup pages exactly one shard rather than one file per fact — a
million single-fact shards would be a catalog disaster, and one shard for all of
them would page the whole store. Each bucket's catalog entry carries the keyword
associations of every key inside it, which is what puts facts back on the same
routing path as everything else.

The behaviour of :class:`~ultraquant.memory.systematic.SystematicMemory` is
unchanged; only where the bytes live differs. A memory with no vault behind it
keeps working exactly as before.

Pure Python standard library.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

__all__ = ["FactShards", "DEFAULT_BUCKETS"]

#: How many bucket shards a fact store is spread across.
#:
#: Sized so a bucket stays worth paging as a unit: at a million facts this is
#: ~4,000 per shard, a few hundred kilobytes compressed. One shard per fact
#: would make the catalog larger than the data; one shard for all of them would
#: page the entire semantic store to answer one question.
DEFAULT_BUCKETS = 256

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class FactShards:
    """Sharded, catalogued storage for semantic facts.

    Args:
        vault: Where shards live.
        cache: Optional :class:`~ultraquant.shards.budget.ShardCache`, so hot
            buckets stay resident under the same byte budget as experts.
        buckets: Number of bucket shards.
    """

    def __init__(self, vault: Any, cache: Any | None = None,
                 buckets: int = DEFAULT_BUCKETS) -> None:
        self.vault = vault
        self.cache = cache
        self.buckets = int(buckets)
        self._dirty: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # addressing
    # ------------------------------------------------------------------ #

    def bucket_of(self, key: str) -> str:
        """The shard id holding ``key``: hashed on the SUBJECT PREFIX.

        The first density measurement found the scale wall here, not in
        the reasoning: full-key hashing scattered one entity's facts
        ("little wall arch material/height/width") across four buckets,
        and at 10,000 facts every bucket associated every common token,
        so ranked bucket selection collapsed into ties and the origin of
        a valid chain was unreachable ~85% of the time.

        Hashing the first two informative tokens co-locates a subject's
        facts in ONE bucket - and makes retrieval ADDRESSABLE: a probe
        that names the subject computes this bucket directly, O(1) at
        any density, no ranking involved. Locality by topic, which is
        also how recall is supposed to feel.
        """
        prefix = " ".join(self.tokens(key)[:2]) or key.lower()
        digest = hashlib.blake2b(prefix.encode("utf-8"),
                                 digest_size=4).digest()
        return f"fact:{int.from_bytes(digest, 'big') % self.buckets:03d}"

    def _legacy_bucket_of(self, key: str) -> str:
        """The pre-prefix-era address, kept so old stores stay readable."""
        digest = hashlib.blake2b(key.lower().encode("utf-8"),
                                 digest_size=4).digest()
        return f"fact:{int.from_bytes(digest, 'big') % self.buckets:03d}"

    @staticmethod
    def tokens(key: str) -> list[str]:
        """Keyword tokens of a fact key, for the vault's inverted index."""
        return _TOKEN_RE.findall(key.lower())

    # ------------------------------------------------------------------ #
    # reading
    # ------------------------------------------------------------------ #

    def _load(self, shard_id: str) -> dict[str, Any]:
        """The facts in one bucket, from the write buffer or from storage."""
        if shard_id in self._dirty:
            return self._dirty[shard_id]
        if not self.vault.has(shard_id):
            return {}
        if self.cache is not None:
            payload = self.cache.get(
                shard_id,
                lambda: (self.vault.get(shard_id),
                         int(self.vault.entry(shard_id)["nbytes"])),
            )
        else:
            payload = self.vault.get(shard_id)
        return dict(payload.get("facts", {}))

    def get(self, key: str) -> dict | None:
        """The record for ``key``, paging its bucket (legacy as fallback)."""
        record = self._load(self.bucket_of(key)).get(key)
        if record is not None:
            return record
        legacy = self._legacy_bucket_of(key)
        if legacy != self.bucket_of(key):
            return self._load(legacy).get(key)
        return None

    def has(self, key: str) -> bool:
        """Whether ``key`` is held."""
        return self.get(key) is not None

    def keys(self) -> list[str]:
        """Every fact key.

        This genuinely pages every bucket, which is the honest cost of asking a
        question about the whole store. Nothing on the recall path uses it.
        """
        out: list[str] = []
        for entry in self.vault.catalog():
            if entry.get("kind") == "fact-bucket":
                out.extend(self._load(entry["shard_id"]))
        for facts in self._dirty.values():
            out.extend(facts)
        return sorted(set(out))

    def count(self) -> int:
        """How many facts are held, from the catalog where possible."""
        total = 0
        counted = set()
        for entry in self.vault.catalog():
            if entry.get("kind") != "fact-bucket":
                continue
            shard_id = entry["shard_id"]
            counted.add(shard_id)
            total += (len(self._dirty[shard_id]) if shard_id in self._dirty
                      else int(entry.get("fact_count", 0)))
        for shard_id, facts in self._dirty.items():
            if shard_id not in counted:
                total += len(facts)
        return total

    def search(self, text: str, top_k: int = 5,
               max_buckets: int = 8) -> list[str]:
        """Fact keys related to ``text``, via the vault's keyword index.

        This is the point of putting facts in the library: finding what is held
        about a topic costs a handful of index lookups and pages only the
        buckets that could contain an answer, instead of scanning every fact.
        """
        wanted = set(self.tokens(text))
        if not wanted:
            return []
        scores = self.vault.association_scores(wanted)
        candidates = [
            (scores[entry["category"]], entry["shard_id"])
            for entry in self.vault.catalog()
            if entry.get("kind") == "fact-bucket" and entry["category"] in scores
        ]
        # A token like "weight" that appears in every key matches every bucket,
        # and paging them all is the whole-store scan this exists to avoid. The
        # buckets are ranked by how strongly they match and only the best are
        # read, so an unselective query costs a bounded number of reads instead
        # of one per bucket. A selective token narrows it to one anyway.
        candidates.sort(key=lambda pair: (-pair[0], pair[1]))
        buckets = [shard_id for _score, shard_id in candidates[:max_buckets]]
        # Addressed retrieval: a probe naming a subject computes that
        # subject's bucket directly. Every adjacent bigram of the probe is
        # tried, because the probe may start mid-phrase; this is what
        # keeps multi-token recall O(1) when the ranked candidates above
        # have collapsed into density ties.
        probe_tokens = self.tokens(text)
        for start in range(max(len(probe_tokens) - 1, 0)):
            prefix = " ".join(probe_tokens[start:start + 2])
            digest = hashlib.blake2b(prefix.encode("utf-8"),
                                     digest_size=4).digest()
            addressed = (f"fact:"
                         f"{int.from_bytes(digest, 'big') % self.buckets:03d}")
            if addressed not in buckets:
                buckets.append(addressed)
        hits: list[tuple[int, float, str]] = []
        for shard_id in buckets or list(self._dirty):
            bucket = self._load(shard_id)
            for key, record in bucket.items():
                overlap = len(wanted & set(self.tokens(key)))
                if overlap:
                    # Recall reinforces (rule 3), finally reaching fact
                    # retrieval: equal-overlap ties break toward the fact
                    # that has been re-attested, not toward the alphabet.
                    # A tie-break only - reinforcement never outranks a
                    # better token match, and the coverage rules upstream
                    # still refuse whatever retrieval surfaces (§11.44).
                    weight = float(record.get("reinforcements", 0))                         + float(record.get("confidence", 0.0))
                    hits.append((overlap, weight, key))
        hits.sort(key=lambda triple: (-triple[0], -triple[1], triple[2]))
        return [key for _o, _w, key in hits[:top_k]]

    # ------------------------------------------------------------------ #
    # writing
    # ------------------------------------------------------------------ #

    def put(self, key: str, record: dict) -> None:
        """Stage a fact. Call :meth:`flush` to persist."""
        shard_id = self.bucket_of(key)
        if shard_id not in self._dirty:
            self._dirty[shard_id] = self._load(shard_id)
        self._dirty[shard_id][key] = dict(record)

    def delete(self, key: str) -> bool:
        """Forget a fact, wherever it lives (legacy bucket included)."""
        removed = False
        for shard_id in {self.bucket_of(key), self._legacy_bucket_of(key)}:
            if shard_id not in self._dirty:
                self._dirty[shard_id] = self._load(shard_id)
            if self._dirty[shard_id].pop(key, None) is not None:
                removed = True
        return removed

    def flush(self) -> int:
        """Write staged buckets into the vault.

        Returns:
            How many bucket shards were written.
        """
        if not self._dirty:
            return 0
        written = 0
        with self.vault.batch():
            for shard_id, facts in self._dirty.items():
                # An EMPTY staged bucket is not "nothing to do" - it means
                # every fact in it was deleted, and skipping it resurrects
                # them: the vault keeps the old bucket, the cache keeps the
                # old payload, and the next recall serves a record that was
                # retracted. Found live when truth maintenance deleted a
                # consolidated fact whose bucket held nothing else - the
                # stale answer came back, and recall-reinforcement then
                # re-persisted the ghost.
                if not facts and not self.vault.has(shard_id):
                    continue
                associations = {}
                for key in facts:
                    for token in self.tokens(key):
                        associations[token] = max(associations.get(token, 0.0), 1.0)
                # Each bucket is its own category so the vault's inverted index
                # resolves to a *bucket* rather than to "facts" as a whole. With
                # one shared category every lookup would page every bucket,
                # which is the whole store -- exactly what sharding them was for.
                self.vault.add_shard(
                    shard_id, shard_id, {"facts": facts},
                    kind="fact-bucket", associations=associations,
                )
                # The count lives in the catalog so `count()` does not have to
                # page every bucket to answer.
                self.vault.entry(shard_id)  # ensure it exists
                self.vault._catalog[shard_id]["fact_count"] = len(facts)
                if self.cache is not None:
                    self.cache.invalidate(shard_id)
                written += 1
        self._dirty.clear()
        return written

    def migrate(self, facts: dict[str, dict]) -> int:
        """Move an existing flat fact store into buckets.

        Returns:
            How many facts were migrated.
        """
        for key, record in facts.items():
            self.put(key, record)
        self.flush()
        return len(facts)

    def stats(self) -> dict:
        """Shape of the fact store."""
        buckets = [
            entry for entry in self.vault.catalog()
            if entry.get("kind") == "fact-bucket"
        ]
        return {
            "facts": self.count(),
            "buckets_used": len(buckets),
            "buckets_total": self.buckets,
            "bytes": sum(int(e["nbytes"]) for e in buckets),
            "pending": sum(len(f) for f in self._dirty.values()),
        }
