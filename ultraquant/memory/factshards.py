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
import unicodedata  # §11.139: language normalization, without domain rules.
from copy import deepcopy
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


# §11.139: shared by addressing, dictionaries and readers in both stores.
def normalize_subject(subject: str) -> str:
    """Fold case, accents, punctuation and one leading article."""
    decomposed = unicodedata.normalize("NFKD", subject.casefold())
    bare = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    words = "".join(ch if ch.isalnum() else " " for ch in bare).split()
    if words and words[0] in ("the", "a", "an"):
        words = words[1:]
    return " ".join(words)


def subject_ngrams(text: str) -> set[str]:
    """Contiguous normalized phrases of one through eight tokens."""
    words = normalize_subject(text).split()
    return {" ".join(words[start:start + size])
            for start in range(len(words))
            for size in range(1, min(8, len(words) - start) + 1)}


# §11.141: token spans, shared by both stores; no inferred grammar.
def _span_start(words: list[str], span: list[str]) -> int | None:
    if span:
        for start in range(len(words) - len(span) + 1):
            if words[start:start + len(span)] == span:
                return start
    return None


def choose_subject(subjects: set[str]) -> str | None:
    """Choose the sole matched subject not contained in a longer match."""
    tokens = {subject: subject.split() for subject in subjects}
    survivors = [subject for subject, span in tokens.items()
                 if not any(len(other) > len(span)
                            and _span_start(other, span) is not None
                            for other in tokens.values())]
    return survivors[0] if len(survivors) == 1 else None


def question_words(text: str, subject: str = "") -> set[str]:
    """Remove one normalized subject span before filtering stopwords."""
    from ultraquant.interpreter.learning import _STOPWORDS

    words = normalize_subject(text).split()
    # §11.141: callers already normalized the subject; do not strip twice.
    span = subject.split()
    start = _span_start(words, span)
    if start is not None:
        del words[start:start + len(span)]
    return set(words) - _STOPWORDS


def _learn_asking(vocabulary: dict, attribute: str, subject: str,
                  question: str) -> None:
    """Index each question word by the distinct subjects that used it."""
    normalized = normalize_subject(attribute)
    subject = normalize_subject(subject)
    if not normalized or not subject or not question.strip():
        return
    item = vocabulary.setdefault(normalized, {"name": attribute, "subjects": 0})
    asked_by = item.setdefault("asked_by", {})
    for word in question_words(question, subject) - set(normalized.split()):
        asked_by[word] = sorted(set(asked_by.get(word, ())) | {subject})


def attribute_words(attribute: str, item: dict) -> set[str]:
    """Attribute tokens and asking words attested by at least two subjects."""
    # §11.141: vocabulary keys are already normalized.
    return set(attribute.split()) | {
        word for word, subjects in item.get("asked_by", {}).items()
        if len(subjects) >= 2}


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
        # §11.139: index pages are cached independently of fact buckets.
        self._indexes: dict[str, dict] = {}
        self._index_before: dict[str, dict | None] = {}
        # review 8: remember submitted pages to detect reversions within a batch.
        self._index_written: dict[str, dict] = {}
        self._indexes_ready = False
        self._duplicates: dict[str, set[str]] = {}

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
        # review 8: preserve the pre-catalogue address without normalization.
        digest = hashlib.blake2b(prefix.encode("utf-8"), digest_size=4).digest()
        return f"fact:{int.from_bytes(digest, 'big') % self.buckets:03d}"

    def subject_bucket(self, subject: str) -> str:
        """The bucket addressed by a normalized subject."""
        digest = hashlib.blake2b(normalize_subject(subject).encode("utf-8"),
                                 digest_size=4).digest()
        return f"fact:{int.from_bytes(digest, 'big') % self.buckets:03d}"

    # §11.139: the only rule assigning a record to its bucket.
    def address(self, key: str, record: dict) -> str:
        """Address explicit structure, falling back to the unstructured key."""
        return (self.subject_bucket(record["subject"]) if record.get("subject")
                else self.bucket_of(key))

    def _legacy_bucket_of(self, key: str) -> str:
        """The pre-prefix-era address, kept so old stores stay readable."""
        digest = hashlib.blake2b(key.lower().encode("utf-8"),
                                 digest_size=4).digest()
        return f"fact:{int.from_bytes(digest, 'big') % self.buckets:03d}"

    @staticmethod
    def tokens(key: str) -> list[str]:
        """Keyword tokens of a fact key, for the vault's inverted index."""
        return _TOKEN_RE.findall(key.lower())

    # §11.139: all four dictionaries use the same paged index machinery.
    @staticmethod
    def _index_id(kind: str, key: str = "") -> str:
        if kind == "attributes":
            return "index:attributes"
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest()
        return f"index:{kind}:{int.from_bytes(digest, 'big') % 64:02d}"

    def _index_data(self, kind: str, key: str = "") -> dict:
        shard_id = self._index_id(kind, key)
        field = "derived" if kind == "derivations" else kind
        if shard_id not in self._indexes:
            payload = self.vault.get(shard_id) if self.vault.has(shard_id) else None
            self._index_before[shard_id] = deepcopy(payload)
            self._indexes[shard_id] = deepcopy(payload) if payload is not None else {field: {}}
        return self._indexes[shard_id][field]

    def _ensure_indexes(self) -> None:
        """Migrate a library lacking derivation indexes in one bucket scan."""
        if self._indexes_ready:
            return
        entries = self.vault.catalog()
        if any(e["shard_id"].startswith("index:derivations:") for e in entries):
            self._ensure_values(entries)
            self._indexes_ready = True
            return
        # §11.139: even an edgeless library persists a migration marker.
        # Reads share one batch so a migration does not fsync per bucket.
        with self.vault.batch():
            for entry in entries:
                sid = entry["shard_id"]
                if entry.get("kind") == "fact-index":
                    payload = self.vault.get(sid)
                    self._index_before[sid] = deepcopy(payload)
                    self._indexes[sid] = {field: {} for field in payload}
            marker = "index:derivations:00"
            self._indexes.setdefault(marker, {"derived": {}})
            self._index_before.setdefault(marker, None)
            self._indexes.setdefault("index:values:00", {"values": {}})
            self._index_before.setdefault("index:values:00", None)
            locations: dict[str, set[str]] = {}
            # §11.139: preserve get's precedence when old stores hold duplicates.
            # Only catalogue fields are retained during this one-time scan.
            selected: dict[str, tuple[str, dict]] = {}
            attribute_subjects: dict[str, set[str]] = {}
            for entry in entries:
                if entry.get("kind") != "fact-bucket":
                    continue
                bucket = entry["shard_id"]
                for key, record in self._load(bucket).items():
                    locations.setdefault(key, set()).add(bucket)
                    # §11.139: an existing directory outranks both fallbacks.
                    prior_index = self._index_before.get(self._index_id("keys", key)) or {}
                    directed = prior_index.get("keys", {}).get(key)
                    preferred = (directed, self.bucket_of(key), self._legacy_bucket_of(key))
                    rank = preferred.index(bucket) if bucket in preferred else 3
                    previous = selected.get(key)
                    prior_rank = (preferred.index(previous[0])
                                  if previous and previous[0] in preferred else 3)
                    if previous is None or rank < prior_rank:
                        selected[key] = (bucket, {field: record[field] for field in
                            ("subject", "attribute", "value", "derived_from") if field in record})
            for key, (bucket, record) in selected.items():
                directory = self._index_data("keys", key)
                if bucket != self.bucket_of(key):
                    directory[key] = bucket
                subject = normalize_subject(record.get("subject") or "")
                if subject:
                    subjects = self._index_data("subjects", subject)
                    item = subjects.setdefault(subject, {
                        "name": record["subject"], "bucket": bucket, "keys": []})
                    item["keys"] = sorted(set(item["keys"]) | {key})
                    attribute = normalize_subject(record.get("attribute") or "")
                    if attribute:
                        attribute_subjects.setdefault(attribute, set()).add(subject)
                        self._index_data("attributes").setdefault(attribute, {
                            "name": record["attribute"], "subjects": 0})
                self._update_derivations(key, None, record)
                self._update_values(key, None, record)
            for attribute, subjects in attribute_subjects.items():
                self._index_data("attributes")[attribute]["subjects"] = len(subjects)
            self._duplicates = {key: buckets for key, buckets in locations.items()
                                if len(buckets) > 1}
        self._indexes_ready = True
        # §11.139: clean historical duplicate locations in the indexed flush,
        # so the directory remains sufficient after a restart.
        for key in list(self._duplicates):
            self.put(key, self.get(key))

    def _ensure_values(self, entries: list[dict]) -> None:
        """Stage the value index once for libraries with older indexes."""
        if any(e["shard_id"].startswith("index:values:") for e in entries):
            return
        with self.vault.batch():
            self._indexes.setdefault("index:values:00", {"values": {}})
            self._index_before.setdefault("index:values:00", None)
            selected: dict[str, tuple[int, dict]] = {}
            for entry in entries:
                if entry.get("kind") != "fact-bucket":
                    continue
                bucket = entry["shard_id"]
                for key, record in self._load(bucket).items():
                    directed = self._index_data("keys", key).get(key)
                    preferred = (directed, self.bucket_of(key), self._legacy_bucket_of(key))
                    rank = preferred.index(bucket) if bucket in preferred else 3
                    if key not in selected or rank < selected[key][0]:
                        selected[key] = (rank, record)
            for key, (_rank, record) in selected.items():
                self._update_values(key, None, record)

    def _update_values(self, key: str, old: dict | None, new: dict | None) -> None:
        """Move a structured fact's membership with its stored value."""
        for record, adding in ((old, False), (new, True)):
            if not record or not record.get("subject") or not record.get("attribute"):
                continue
            value = normalize_subject(str(record["value"]))
            index = self._index_data("values", value)
            keys = set(index.get(value, {}).get("keys", ()))
            if adding:
                keys.add(key)
            else:
                keys.discard(key)
            if keys:
                index[value] = {"keys": sorted(keys)}
            else:
                index.pop(value, None)

    def _locations(self, key: str) -> list[str]:
        """Directory address first, followed by both historical fallbacks."""
        self._ensure_indexes()
        directory = self._index_data("keys", key).get(key)
        return list(dict.fromkeys(b for b in [directory, self.bucket_of(key),
                    self._legacy_bucket_of(key), *sorted(self._duplicates.get(key, ()))]
                    if b is not None))

    def _update_derivations(self, key: str, old: dict | None, new: dict | None) -> None:
        before = {p for p, _v in (old or {}).get("derived_from", [])}
        after = {p for p, _v in (new or {}).get("derived_from", [])}
        for premise in before | after:
            index = self._index_data("derivations", premise)
            children = set(index.get(premise, ()))
            children.discard(key)
            if premise in after:
                children.add(key)
            if children:
                index[premise] = sorted(children)
            else:
                index.pop(premise, None)

    def derived_candidates(self, key: str) -> set[str]:
        """Direct dependants from the persisted reverse premise index."""
        self._ensure_indexes()
        return set(self._index_data("derivations", key).get(key, ()))

    def subjects_in(self, text: str) -> set[str]:
        """Held subjects named by any contiguous one-to-eight-token phrase."""
        self._ensure_indexes()
        return {phrase for phrase in subject_ngrams(text)
                if phrase in self._index_data("subjects", phrase)}

    def subject_names(self, normalized: str) -> set[str]:
        """The recorded name in one addressed subject dictionary entry."""
        self._ensure_indexes()
        item = self._index_data("subjects", normalized).get(normalized)
        return {item["name"]} if item is not None else set()

    # §11.141: the public seams use the same span and learning rules as RAM.
    def _choose_subject(self, subjects: set[str]) -> str | None:
        """Resolve nested matches, declining unrelated surviving subjects."""
        return choose_subject(subjects)

    def learn_asking(self, attribute: str, subject: str, question: str) -> None:
        """Stage learned question words in the normal attribute index flush."""
        self._ensure_indexes()
        _learn_asking(self._index_data("attributes"), attribute, subject, question)

    def _other_attribute(self, subject: str, attribute: str, key: str) -> bool:
        item = self._index_data("subjects", subject).get(subject)
        if item is None:
            return False
        for other in item["keys"]:
            if other != key:
                record = self.get(other) or {}
                if normalize_subject(record.get("attribute") or "") == attribute:
                    return True
        return False

    def _update_structure(self, key: str, old: dict | None, new: dict | None,
                          bucket: str | None) -> None:
        """Maintain subject membership and counts of subjects per attribute."""
        self._update_values(key, old, new)
        old, new = old or {}, new or {}
        before = tuple(normalize_subject(old.get(f) or "") for f in ("subject", "attribute"))
        after = tuple(normalize_subject(new.get(f) or "") for f in ("subject", "attribute"))
        if before != after:
            for (subject, attribute), record, delta in ((before, old, -1), (after, new, 1)):
                if subject and attribute and not self._other_attribute(subject, attribute, key):
                    vocabulary = self._index_data("attributes")
                    item = vocabulary.setdefault(attribute, {"name": record["attribute"], "subjects": 0})
                    item["subjects"] += delta
                    if item["subjects"] <= 0:
                        vocabulary.pop(attribute, None)
        if before[0]:
            subjects = self._index_data("subjects", before[0])
            item = subjects.get(before[0])
            if item is not None:
                item["keys"] = [k for k in item["keys"] if k != key]
                if not item["keys"]:
                    subjects.pop(before[0])
        if after[0]:
            subjects = self._index_data("subjects", after[0])
            item = subjects.setdefault(after[0], {"name": new["subject"], "bucket": bucket, "keys": []})
            item["bucket"] = bucket
            item["keys"] = sorted(set(item["keys"]) | {key})

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
        # §11.139: copies protect old metadata and edges until put sees them.
        for bucket in self._locations(key):
            record = self._load(bucket).get(key)
            if record is not None:
                return deepcopy(record)
        return None

    def has(self, key: str) -> bool:
        """Whether ``key`` is held."""
        return self.get(key) is not None

    def keys(self) -> list[str]:
        """Every fact key.

        This genuinely pages every bucket, which is the honest cost of asking a
        question about the whole store. Nothing on the recall path uses it.
        """
        self._ensure_indexes()  # §11.139: also recognize pre-index libraries.
        out: list[str] = []
        for entry in self.vault.catalog():
            if entry.get("kind") == "fact-bucket":
                out.extend(self._load(entry["shard_id"]))
        for facts in self._dirty.values():
            out.extend(facts)
        return sorted(set(out))

    # §11.139: the catalogue exposes the same key enumeration as memory.
    def fact_keys(self) -> list[str]:
        """Every held key exactly once, including staged moves and drops."""
        return self.keys()

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

    def keys_covering(self, tokens: set[str]) -> list[str]:
        """Exhaust indexed and staged buckets for keys covering every token."""
        if not tokens:
            return []
        categories = set.intersection(*(
            set(self.vault.association_scores({token})) for token in tokens))
        buckets = {entry["shard_id"] for entry in self.vault.catalog()
                   if entry.get("kind") == "fact-bucket"
                   and entry["category"] in categories}
        buckets.update(self._dirty)
        return sorted({key for bucket in buckets for key in self._load(bucket)
                       if tokens <= set(self.tokens(key))})

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
        # §11.139: named subjects select dictionary buckets directly.
        for subject in sorted(self.subjects_in(text)):
            bucket = self._index_data("subjects", subject)[subject]["bucket"]
            if bucket not in buckets:
                buckets.append(bucket)
        # Addressed retrieval: a probe naming a subject computes that
        # subject's bucket directly. Every adjacent bigram of the probe is
        # tried, because the probe may start mid-phrase; this is what
        # keeps multi-token recall O(1) when the ranked candidates above
        # have collapsed into density ties.
        probe_tokens = self.tokens(text)
        for start in range(max(len(probe_tokens) - 1, 0)):
            prefix = " ".join(probe_tokens[start:start + 2])
            addressed = self.bucket_of(prefix)  # §11.139: one addressing rule.
            if addressed not in buckets:
                buckets.append(addressed)
        hits: list[tuple[int, float, str]] = []
        # §11.139: staged facts are searchable before the vault is flushed.
        for shard_id in dict.fromkeys([*buckets, *self._dirty]):
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
        # §11.139: indexes and all locations change in the same flush.
        old = self.get(key)
        locations = self._locations(key)
        shard_id = self.address(key, record)
        self._update_structure(key, old, record, shard_id)
        self._update_derivations(key, old, record)
        for other in locations:
            if other == shard_id:
                continue
            facts = self._load(other)
            if key in facts:
                self._dirty[other] = dict(facts)
                del self._dirty[other][key]
        if shard_id not in self._dirty:
            self._dirty[shard_id] = self._load(shard_id)
        self._dirty[shard_id][key] = deepcopy(record)
        directory = self._index_data("keys", key)
        if shard_id == self.bucket_of(key):
            directory.pop(key, None)
        else:
            directory[key] = shard_id
        self._duplicates.pop(key, None)

    def delete(self, key: str) -> bool:
        """Forget a fact, wherever it lives (legacy bucket included)."""
        # §11.139: remove the record's memberships before its bucket entry.
        old = self.get(key)
        if old is None:
            return False
        self._update_structure(key, old, None, None)
        self._update_derivations(key, old, None)
        for shard_id in self._locations(key):
            facts = self._load(shard_id)
            if key in facts:
                self._dirty[shard_id] = dict(facts)
                del self._dirty[shard_id][key]
        self._index_data("keys", key).pop(key, None)
        self._duplicates.pop(key, None)
        return True

    def flush(self) -> int:
        """Write staged buckets into the vault.

        Returns:
            How many bucket shards were written.
        """
        # §11.139: unchanged index pages never get rewritten, even on reads.
        self._ensure_indexes()
        # review 8: a later nested flush can undo an earlier uncommitted page.
        changed = {sid: payload for sid, payload in self._indexes.items()
                   if (payload != self._index_before[sid]
                       or payload != self._index_written.get(sid, self._index_before[sid]))
                   and (self._index_before[sid] is not None
                        or sid in self._index_written
                        or any(payload.values())
                        or sid in ("index:derivations:00", "index:values:00"))}
        if not self._dirty and not changed:
            return 0
        # review 8: snapshots survive later staging until the outer commit.
        buckets = deepcopy(self._dirty)
        changed = deepcopy(changed)
        written = {sid: {"facts": facts} for sid, facts in buckets.items()}
        written.update(changed)
        count = 0
        with self.vault.batch():
            for shard_id, facts in buckets.items():
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
                count += 1
            # §11.139: a failed index write rolls back the bucket writes too.
            for shard_id, payload in changed.items():
                self.vault.add_shard(shard_id, shard_id, payload, kind="fact-index")
                self._index_written[shard_id] = payload  # review 8: submitted, not committed.
            # review 8: a nested flush has not committed yet.
            self.vault.after_batch(lambda committed: self._finalize(committed, written))
        return count

    # review 8: rollback leaves staged writes and index baselines untouched.
    def _finalize(self, committed, written) -> None:
        """Acknowledge committed snapshots without clearing newer changes."""
        if not committed:
            return
        for shard_id, payload in written.items():
            if shard_id.startswith("index:"):
                self._index_before[shard_id] = deepcopy(payload)
            elif self._dirty.get(shard_id) == payload["facts"]:
                del self._dirty[shard_id]

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
