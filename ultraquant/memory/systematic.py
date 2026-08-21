"""Systematic memory for UltraQuant.

Implements :class:`SystematicMemory`: three cooperating stores plus a
bit-signature index, with JSON persistence.

* **Episodic** store — an append-only log of timestamped events
  (``{id, t, kind, content, tags}``).
* **Working** memory — a bounded FIFO of the most recent episode ids.
* **Semantic** store — key/value facts with confidence that grows on
  reinforcement and resets on revision (conflicting revisions are logged
  as ``"revision"`` episodes).
* **Signature** index — labelled bit vectors queried by Hamming
  similarity via :meth:`SystematicMemory.nearest_signature`.

Pure Python stdlib only.  All persisted state is JSON-safe.
"""

from __future__ import annotations

import json
import re
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class SystematicMemory:
    """Three memory stores + bit-signature index with JSON persistence.

    Parameters
    ----------
    path:
        Optional file path for persistence.  If given and the file
        already exists, the memory auto-loads from it on construction.
    working_capacity:
        Maximum number of episode ids retained in working memory
        (bounded FIFO — oldest ids are evicted first).
    """

    def __init__(
        self,
        path: str | os.PathLike | None = None,
        working_capacity: int = 16,
        shards: Any | None = None,
    ) -> None:
        self.path: Path | None = Path(path) if path is not None else None
        self.working_capacity: int = int(working_capacity)

        self._episodes: list[dict[str, Any]] = []
        self._facts: dict[str, dict[str, Any]] = {}
        self._signatures: list[dict[str, Any]] = []
        self._working: deque[int] = deque(maxlen=self.working_capacity)
        self._next_id: int = 1

        # When a fact-shard backing is supplied, facts become catalogued shards
        # like everything else the model learns: paged on demand, reinforced on
        # use, reachable through the vault's keyword index, and carried inside a
        # packed library. Without one, this class behaves exactly as it always
        # did, which is what keeps it usable on its own.
        self.shards: Any | None = shards

        if self.path is not None and self.path.exists():
            self.load()
        if self.shards is not None and self._facts:
            # A store written before facts were sharded still has them inline.
            self.shards.migrate(self._facts)
            self._facts = {}

    # ------------------------------------------------------------------
    # Episodic store
    # ------------------------------------------------------------------

    def remember_episode(
        self,
        kind: str,
        content: dict,
        tags: list[str] | None = None,
    ) -> int:
        """Append an episode and push its id into working memory.

        Parameters
        ----------
        kind:
            Free-form category of the episode (e.g. ``"recognition"``).
        content:
            JSON-safe payload describing the event.
        tags:
            Optional tags used later for any-overlap filtering.

        Returns
        -------
        int
            The new episode's id (incrementing integer).
        """
        episode_id = self._next_id
        self._next_id += 1
        episode = {
            "id": episode_id,
            "t": _utc_now(),
            "kind": kind,
            "content": content,
            "tags": list(tags) if tags is not None else [],
        }
        self._episodes.append(episode)
        self._working.append(episode_id)
        return episode_id

    def recall_episodes(
        self,
        kind: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Return up to ``limit`` matching episodes, most recent first.

        ``kind`` filters by exact match; ``tags`` filters by any-overlap
        (an episode matches if it shares at least one tag).
        """
        results: list[dict] = []
        for episode in reversed(self._episodes):
            if kind is not None and episode["kind"] != kind:
                continue
            if tags is not None and not set(tags) & set(episode["tags"]):
                continue
            results.append(episode)
            if len(results) >= limit:
                break
        return results

    def working(self) -> list[dict]:
        """Return the episodes currently in working memory, oldest first."""
        by_id = {ep["id"]: ep for ep in self._episodes}
        return [by_id[eid] for eid in self._working if eid in by_id]

    # ------------------------------------------------------------------
    # Semantic store
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Fact backing
    # ------------------------------------------------------------------

    def _fact_record(self, key: str) -> dict | None:
        """The stored record for ``key``, wherever facts happen to live."""
        if self.shards is not None:
            return self.shards.get(key)
        return self._facts.get(key)

    def _put_fact(self, key: str, record: dict) -> None:
        """Write a record back to whichever store is in use."""
        if self.shards is not None:
            self.shards.put(key, record)
        else:
            self._facts[key] = record

    def fact_keys(self) -> list[str]:
        """Every fact key held."""
        if self.shards is not None:
            return self.shards.keys()
        return sorted(self._facts)

    def find_facts(self, text: str, top_k: int = 5) -> list[str]:
        """Fact keys related to ``text``.

        With facts in the library this is an index lookup that pages only the
        buckets which could hold an answer. Without it, there is nothing to do
        but scan, which is exactly the cost sharding removes.
        """
        if self.shards is not None:
            return self.shards.search(text, top_k=top_k)
        wanted = set(re.findall(r"[a-z0-9]+", text.lower()))
        scored = []
        for key, record in self._facts.items():
            overlap = len(wanted & set(re.findall(r"[a-z0-9]+",
                                                  key.lower())))
            if overlap:
                # The same reinforcement tie-break the sharded search
                # applies (§11.44): equal overlap resolves toward the
                # re-attested fact, never past a better token match.
                weight = (float(record.get("reinforcements", 0))
                          + float(record.get("confidence", 0.0)))
                scored.append((overlap, weight, key))
        scored.sort(key=lambda triple: (-triple[0], -triple[1], triple[2]))
        return [key for _o, _w, key in scored[:top_k]]

    def remember_fact(self, key: str, value: Any, confidence: float = 0.5,
                      negated: bool = False) -> None:
        """Store, reinforce, or revise a semantic fact.

        * New key → stored with the given confidence and 0 reinforcements.
        * Same key, equal value AND equal polarity → confidence bumped by
          0.1 (capped at 1.0) and ``reinforcements`` incremented.
        * Same key, different value OR flipped polarity → replaced,
          confidence reset to the given ``confidence``, and a
          ``"revision"`` episode logged. Polarity is part of a fact's
          identity: "the dome material is steel" after "the dome
          material is not steel" is a change of mind, never a
          reinforcement.

        Args:
            key: The fact key.
            value: The believed value — for a negation, the value the
                subject is believed NOT to be (stored bare; the
                ``negated`` flag carries the polarity).
            confidence: Belief strength for a new or revised fact.
            negated: True to store belief-of-absence.
        """
        now = _utc_now()
        existing = self._fact_record(key)
        if existing is None:
            record = {
                "value": value,
                "confidence": float(confidence),
                "reinforcements": 0,
                "first_seen": now,
                "last_seen": now,
            }
            if negated:
                record["negated"] = True
            self._put_fact(key, record)
        elif (existing["value"] == value
              and bool(existing.get("negated")) == bool(negated)):
            existing["confidence"] = min(1.0, existing["confidence"] + 0.1)
            existing["reinforcements"] += 1
            existing["last_seen"] = now
            self._put_fact(key, existing)
        else:
            old_value = existing["value"]
            existing["value"] = value
            if negated:
                existing["negated"] = True
            else:
                existing.pop("negated", None)
            existing["confidence"] = float(confidence)
            existing["last_seen"] = now
            # A revision breaks every conclusion that rested on the old
            # value: truth maintenance retracts derived facts recursively,
            # so the next question re-derives from what is NOW believed
            # instead of recalling a conclusion whose premise is gone.
            existing.pop("derived_from", None)
            self._put_fact(key, existing)
            for gone in self._retract_derivatives(key):
                self.remember_episode(
                    kind="retraction",
                    content={"key": gone,
                             "because": f"premise {key!r} was revised"},
                )
            self.remember_episode(
                "revision",
                {"key": key, "old_value": old_value, "new_value": value},
                tags=["fact", key],
            )

    def confirm_fact(self, key: str, confidence: float = 0.9) -> bool:
        """Set a fact's confidence outright, as direct testimony.

        :meth:`remember_fact` treats a repeat of the same value as *incidental*
        reinforcement and nudges confidence by 0.1, which is right for hearing
        something again in passing. Being told "yes, that is correct" is a
        different and stronger kind of evidence, and this records it as such
        rather than pretending it was another passing mention.

        Args:
            key: The fact to confirm.
            confidence: Confidence to assert.

        Returns:
            True if the fact existed and was updated.
        """
        fact = self._fact_record(key)
        if fact is None:
            return False
        fact["confidence"] = max(0.0, min(1.0, float(confidence)))
        fact["reinforcements"] += 1
        fact["last_seen"] = _utc_now()
        self._put_fact(key, fact)
        return True

    def recall_fact(self, key: str) -> dict | None:
        """Return the stored fact record for ``key``, or None if absent."""
        fact = self._fact_record(key)
        return dict(fact) if fact is not None else None

    # ------------------------------------------------------------------
    # Consolidated (derived) facts and their truth maintenance
    # ------------------------------------------------------------------

    def consolidate_fact(self, key: str, value: Any, confidence: float,
                         premises: list, negated: bool = False) -> None:
        """Store a *derived* fact with the premises it rests on.

        The brain-shaped move (ARCHITECTURE §11.30's registered successor):
        a derivation that earned confirmation stops being re-derived and
        becomes recallable - and usable as a premise for further
        derivation, which is how two honest hops become three.

        The price of materialising a conclusion is staleness, so the
        provenance is load-bearing, not decorative: every premise
        ``(key, value)`` is recorded, and :meth:`remember_fact` retracts
        any derived fact whose premise is later revised. §11.16 measured
        what an entrenched wrong answer costs; a consolidated fact that
        outlives its premises is that error with a memory.
        """
        now = _utc_now()
        record = {
            "value": value,
            "confidence": float(confidence),
            "reinforcements": 0,
            "first_seen": now,
            "last_seen": now,
            "derived_from": [[str(p_key), str(p_value)]
                             for p_key, p_value in premises],
        }
        if negated:
            # A consolidated denial ("believed not temperate", earned
            # through a chain and confirmed) carries its polarity, so it
            # is inert as a bridge exactly like a stated one (§11.48).
            record["negated"] = True
        self._put_fact(key, record)

    def _retract_derivatives(self, revised_key: str) -> list[str]:
        """Drop every derived fact resting on ``revised_key``, recursively.

        Returns the retracted keys, for the caller's episode log.
        """
        retracted: list[str] = []
        stack = [revised_key]
        while stack:
            changed = stack.pop()
            for key in list(self.fact_keys()):
                record = self._fact_record(key)
                if not record or "derived_from" not in record:
                    continue
                if any(p_key == changed
                       for p_key, _v in record["derived_from"]):
                    self._drop_fact(key)
                    retracted.append(key)
                    stack.append(key)
        return retracted

    def _drop_fact(self, key: str) -> None:
        """Remove a fact from whichever store is in use."""
        if self.shards is not None:
            self.shards.delete(key)
        else:
            self._facts.pop(key, None)

    # ------------------------------------------------------------------
    # Signature index
    # ------------------------------------------------------------------

    def store_signature(self, label: str, bits: list[int]) -> None:
        """Store a labelled bit signature in the index."""
        self._signatures.append({"label": label, "bits": [int(b) for b in bits]})

    def nearest_signature(self, bits: list[int]) -> tuple[str, float] | None:
        """Return the (label, similarity) of the closest stored signature.

        Similarity is ``1 - hamming_distance / length``.  Only signatures
        of equal length are compared; returns None if none qualify.
        """
        best: tuple[str, float] | None = None
        query = [int(b) for b in bits]
        for entry in self._signatures:
            stored = entry["bits"]
            if len(stored) != len(query):
                continue
            if not stored:
                continue
            hamming = sum(1 for a, b in zip(stored, query) if a != b)
            similarity = 1.0 - hamming / len(stored)
            if best is None or similarity > best[1]:
                best = (entry["label"], similarity)
        return best

    # ------------------------------------------------------------------
    # Persistence & stats
    # ------------------------------------------------------------------

    def save(self, path: str | os.PathLike | None = None) -> None:
        """Serialize all stores to JSON at ``path`` (or ``self.path``).

        Raises
        ------
        ValueError
            If no path was given here or at construction time.
        """
        if path is not None:
            self.path = Path(path)
        if self.path is None:
            raise ValueError("SystematicMemory.save() requires a path")
        payload = {
            "episodes": self._episodes,
            # Empty when facts live in the library; keeping a second copy
            # here would put the store back in RAM whole, which is the
            # thing sharding them was for.
            "facts": {} if self.shards is not None else self._facts,
            "signatures": self._signatures,
            "working": list(self._working),
            "next_id": self._next_id,
            "working_capacity": self.working_capacity,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True)
        if self.shards is not None:
            self.shards.flush()

    def load(self, path: str | os.PathLike | None = None) -> None:
        """Restore all stores from the JSON file at ``path`` (or ``self.path``).

        The working-memory FIFO keeps this instance's capacity; if the
        saved queue is longer, only the most recent ids are retained.

        Raises
        ------
        ValueError
            If no path was given here or at construction time.
        """
        if path is not None:
            self.path = Path(path)
        if self.path is None:
            raise ValueError("SystematicMemory.load() requires a path")
        with open(self.path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self._episodes = list(payload.get("episodes", []))
        self._facts = dict(payload.get("facts", {}))
        self._signatures = list(payload.get("signatures", []))
        self._working = deque(
            payload.get("working", []), maxlen=self.working_capacity
        )
        self._next_id = int(payload.get("next_id", len(self._episodes) + 1))

    def stats(self) -> dict:
        """Return counts per store (JSON-safe)."""
        return {
            "episodic": len(self._episodes),
            "semantic": (self.shards.count() if self.shards is not None
                         else len(self._facts)),
            "signatures": len(self._signatures),
            "working": len(self._working),
        }
