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
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ultraquant.memory.metering import charge_index, charge_lookup
# §11.139: one normalization and n-gram rule for both fact stores.
from ultraquant.memory.factshards import normalize_subject, subject_ngrams
# §11.141: catalogue selection and question learning share token operations.
from ultraquant.memory.factshards import (
    FactShards, _learn_asking, attribute_words, choose_subject, question_words,
)


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

    #: §11.122: this memory charges the metering hooks itself, so a bill
    #: over it is a measurement. A subclass overriding recall_fact or
    #: find_facts must charge them too (or call super()); a memory that
    #: does not declare this is billed as unknown, never as zero.
    metered = True

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
        # §11.141: unsharded asking evidence stays in RAM.
        self._attributes: dict[str, dict] = {}
        # §11.139: reverse premise edges, maintained with the records.
        self._derived: dict[str, set[str]] = {}
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

    def forget_episodes(self, tag: str) -> int:
        """Remove episodes with ``tag`` from history and the working FIFO."""
        # Review 6: transaction rollback removes its history from both stores.
        removed = {ep["id"] for ep in self._episodes if tag in ep["tags"]}
        self._episodes = [ep for ep in self._episodes if ep["id"] not in removed]
        self._working = deque((eid for eid in self._working if eid not in removed),
                              maxlen=self.working_capacity)
        return len(removed)

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
            # §11.139: remove old edges before replacing the record.
            self._index_derivation(key, self._facts.get(key), record)
            self._facts[key] = record

    # §11.139: the in-RAM reverse index has one write path.
    def _index_derivation(self, key: str, old: dict | None,
                          new: dict | None) -> None:
        """Replace the reverse edges belonging to one fact."""
        for premise, _value in (old or {}).get("derived_from", []):
            children = self._derived.get(premise)
            if children is not None:
                children.discard(key)
                if not children:
                    del self._derived[premise]
        for premise, _value in (new or {}).get("derived_from", []):
            self._derived.setdefault(premise, set()).add(key)

    # §11.139: structure comes from records, never from parsing their keys.
    def subjects_in(self, text: str) -> set[str]:
        """Normalized held subjects named by contiguous tokens in text."""
        if self.shards is not None:
            return self.shards.subjects_in(text)
        subjects = {normalize_subject(record["subject"])
                    for record in self._facts.values() if record.get("subject")}
        return subjects & subject_ngrams(text)

    def subject_names(self, normalized: str) -> set[str]:
        """Recorded spellings of one normalized catalogue subject."""
        if self.shards is not None:
            return self.shards.subject_names(normalized)
        return {record["subject"] for record in self._facts.values()
                if record.get("subject")
                and normalize_subject(record["subject"]) == normalized}

    def value_keys(self, value: str) -> list[str]:
        """Keys of structured facts holding the normalized value."""
        # N-grams are already normalized; do not strip a second article.
        if self.shards is not None:
            self.shards._ensure_indexes()
            return list(self.shards._index_data("values", value).get(
                value, {}).get("keys", ()))
        return sorted(key for key, record in self._facts.items()
                      if record.get("subject") and record.get("attribute")
                      and normalize_subject(str(record["value"])) == value)

    def value_attributes(self, value: str, words: set[str]) -> set[str] | None:
        """Attributes that name a held value, or all for informative values."""
        from ultraquant.shards.router import _informative

        attributes = {normalize_subject(record["attribute"])
                      for key in self.value_keys(value)
                      if (record := self.recall_fact(key)) is not None
                      and record.get("subject") and record.get("attribute")}
        vocabulary = self._attribute_vocabulary()
        named = {attribute for attribute in attributes
                 if set(attribute.split()) <= words
                 or words & (attribute_words(attribute, vocabulary.get(attribute, {}))
                             - set(attribute.split()))}
        if named:
            return named
        if any(_informative(token) for token in value.split()):
            return attributes
        return None

    def catalogue_by_value(self, text: str) -> dict | None:
        """Find every holder of the sole named, non-nested value."""
        if self.subjects_in(text):
            return None
        candidates = {}
        for value in subject_ngrams(text):
            if self.value_keys(value):
                attributes = self.value_attributes(value, question_words(text, value))
                if attributes is not None:
                    candidates[value] = attributes
        value = choose_subject(set(candidates))
        if value is None:
            return None
        attributes = candidates[value]
        records = []
        for key in sorted(self.value_keys(value)):
            record = self.recall_fact(key)
            if (record is not None and record.get("subject") and record.get("attribute")
                    and normalize_subject(record["attribute"]) in attributes):
                records.append({"key": key, "record": record})
        return {"form": "value", "value": value, "attributes": sorted(attributes),
                "records": records}

    # §11.141: use only explicit record slots and indexed asking evidence.
    def _attribute_vocabulary(self) -> dict:
        """The attribute page, or its in-memory counterpart."""
        if self.shards is not None:
            self.shards._ensure_indexes()
            return self.shards._index_data("attributes")
        # §11.141: keep asking evidence learned before a record was filed.
        vocabulary = {attribute: {**item, "subjects": 0}
                      for attribute, item in self._attributes.items()}
        subjects: dict[str, set[str]] = {}
        for record in self._facts.values():
            subject = normalize_subject(record.get("subject") or "")
            attribute = normalize_subject(record.get("attribute") or "")
            if subject and attribute:
                subjects.setdefault(attribute, set()).add(subject)
                vocabulary.setdefault(attribute, {
                    **self._attributes.get(attribute, {}),
                    "name": record["attribute"], "subjects": 0})
        for attribute, members in subjects.items():
            vocabulary[attribute]["subjects"] = len(members)
        self._attributes = vocabulary
        return vocabulary

    def learn_kind(self, attribute: str, kind: str | None) -> None:
        """Store a source's kind label, or its failure to name a shared kind."""
        vocabulary = self._attribute_vocabulary()
        item = vocabulary.setdefault(normalize_subject(attribute), {
            "name": attribute, "subjects": 0})
        item["kind"] = kind

    def learn_property(self, attribute: str, candidate: str, verdict: str) -> None:
        """Store a property verdict and the attribute an adoption extends."""
        if verdict not in {"adopted", "refused"}:
            raise ValueError("Property verdict must be adopted or refused")
        vocabulary = self._attribute_vocabulary()
        item = vocabulary.setdefault(normalize_subject(attribute), {
            "name": attribute, "subjects": 0})
        item.setdefault("properties", {})[candidate] = verdict
        if verdict == "adopted":
            vocabulary.setdefault(normalize_subject(candidate), {
                "name": candidate, "subjects": 0})["extends"] = attribute

    def learn_asking(self, attribute: str, subject: str, question: str) -> None:
        """Learn how an attribute was asked about, once per subject and word."""
        if self.shards is not None:
            self.shards.learn_asking(attribute, subject, question)
        else:
            _learn_asking(self._attributes, attribute, subject, question)

    # §11.142: chat requests accept only exact catalogue answers.
    def catalogue_request(self, text: str) -> dict | None:
        """Offer an exact catalogue answer without reinterpreting chat."""
        answer = self.catalogue_answer(text)
        return answer if answer is not None and answer["form"] == "exact" else None

    def _asked_attributes(self, words: set) -> set:
        """Every catalogued attribute that explains all the requested words."""
        if not words:
            return set()
        return {attribute for attribute, item in self._attribute_vocabulary().items()
                if words <= attribute_words(attribute, item)}

    def _names_other_attribute(self, words: set, explained: set) -> bool:
        """Whether the unexplained words name another indexed attribute."""
        rest = words - explained
        return any(rest & attribute_words(attribute, item)
                   for attribute, item in self._attribute_vocabulary().items())

    def _second_hop(self, value, attribute: str) -> tuple[str, dict] | None:
        """Look up the value as a catalogue subject, then its attribute."""
        subject = normalize_subject(str(value))
        if self.shards is not None:
            self.shards._ensure_indexes()
            keys = self.shards._index_data("subjects", subject).get(
                subject, {}).get("keys", ())
        else:
            keys = sorted(key for key, record in self._facts.items()
                          if record.get("subject")
                          and normalize_subject(record["subject"]) == subject)
        for key in keys:
            record = self.recall_fact(key)
            if (record is not None and record.get("subject")
                    and normalize_subject(record["subject"]) == subject
                    and normalize_subject(record.get("attribute") or "") == attribute):
                return key, record
        return None

    def held_value(self, subject: str, attribute: str) -> str | None:
        """Read a subject's attribute through the catalogue in either store."""
        held = self._second_hop(subject, normalize_subject(attribute))
        value = held[1].get("value") if held is not None else None
        return str(value) if value is not None else None

    def key_form(self, attribute: str) -> str | None:
        """The most common held key form, with lexical ties resolved first."""
        attribute = normalize_subject(attribute)
        forms = Counter()
        for key in self.fact_keys():
            record = self.recall_fact(key)
            if (record is None or not record.get("subject")
                    or normalize_subject(record.get("attribute") or "") != attribute):
                continue
            subject = record["subject"].lower()
            if key.count(subject) == 1:
                forms[key.replace(subject, "{subject}", 1)] += 1
        return min(forms, key=lambda form: (-forms[form], form)) if forms else None

    # §11.142: only an indexed unstructured key can explain an unheld subject.
    def _unheld_subject(self, words: set, known: set) -> bool:
        """Whether remaining words name no held unstructured fact."""
        from ultraquant.shards.router import _informative

        others = {word for word in words - known if _informative(word)}
        if not others:
            return False
        keys = (self.shards.keys_covering(others) if self.shards is not None
                else [key for key in self._facts
                      if others <= set(FactShards.tokens(key))])
        for key in keys:
            record = self.recall_fact(key)
            if record is not None and not record.get("subject"):
                return False
        return True

    def catalogue_answers(self, text: str) -> list[dict] | None:
        """One indexed attribute for every named, non-nested subject."""
        subjects = self.subjects_in(text)
        subjects = {subject for subject in subjects
                    if not any(other != subject
                               and choose_subject({subject, other}) == other
                               for other in subjects)}
        if len(subjects) < 2:
            return None
        from ultraquant.interpreter.learning import _STOPWORDS

        tokens = normalize_subject(text).split()
        removed = set()
        for subject in subjects:
            span = subject.split()
            for start in range(len(tokens) - len(span) + 1):
                if tokens[start:start + len(span)] == span:
                    removed.update(range(start, start + len(span)))
        words = {word for index, word in enumerate(tokens)
                 if index not in removed} - _STOPWORDS
        asked = self._asked_attributes(words)
        if len(asked) != 1:
            return None
        attribute = next(iter(asked))
        answers = []
        for subject in sorted(subjects):
            held = self._second_hop(subject, attribute)
            key, record = held if held is not None else (None, None)
            answers.append({"subject": subject, "attribute": attribute,
                            "key": key, "record": record})
        return answers

    def catalogue_answer(self, text: str) -> dict | None:
        """Answer through catalogued subjects without registering curiosity."""
        subjects = self.subjects_in(text)
        chosen = (self.shards._choose_subject(subjects) if self.shards is not None
                  else choose_subject(subjects))
        vocabulary = self._attribute_vocabulary()
        words = question_words(text, chosen or "")
        if chosen is not None:
            if self.shards is not None:
                keys = self.shards._index_data("subjects", chosen)[chosen]["keys"]
            else:
                keys = [key for key, record in self._facts.items()
                        if normalize_subject(record.get("subject") or "") == chosen]
            # §11.145: ambiguity is judged against the whole vocabulary, before
            # the subject's held facts can hide a competing attribute.
            asked = self._asked_attributes(words)
            best, winners = 0, []
            for key in keys:
                record = self.recall_fact(key)
                if not record or not record.get("attribute"):
                    continue
                attribute = normalize_subject(record["attribute"])
                explained = attribute_words(attribute, vocabulary.get(attribute, {}))
                score = len(words & explained)
                # §11.150: the first value can itself name a held subject.
                if score:
                    rest = words - explained
                    following = self._asked_attributes(rest)
                    if len(following) == 1:
                        second = self._second_hop(record["value"], next(iter(following)))
                        if second is not None:
                            key2, record2 = second
                            return {"form": "chain", "key": key2, "record": record2,
                                    "via": {"key": key, "record": record}}
                if score > best:
                    best, winners = score, []
                if score == best and score >= 1:
                    exact = words <= explained and asked == {attribute}
                    if not exact and self._names_other_attribute(words, explained):
                        continue
                    winners.append({"form": "exact" if exact else "reading",
                                    "key": key, "record": record})
            return winners[0] if len(winners) == 1 else None
        if subjects:
            return None
        known = set().union(*(attribute_words(attribute, item)
                              for attribute, item in vocabulary.items()))
        # §11.142: shared words in structured subjects do not establish identity.
        from ultraquant.interpreter.learning import _STOPWORDS

        if (words & known and self._unheld_subject(
                set(FactShards.tokens(text)) - _STOPWORDS, known)
                and self.catalogue_by_value(text) is None):
            return {"form": "unknown-subject"}
        return None

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
        charge_index()                # §11.122: the bill, counted here
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
                      negated: bool = False, *, subject=None, attribute=None) -> dict:  # §11.139
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

        Returns:
            ``{"outcome": "new" | "reinforced" | "revised"}`` — a
            revision additionally carries ``"was"`` (the old belief in
            spoken form, polarity included) and ``"retracted"`` (the
            derived keys truth maintenance took down), so the surface
            can SAY what changed instead of noting a change silently.
        """
        now = _utc_now()
        existing = self._fact_record(key)
        # §11.139: copy before mutation so the index can see the old edges.
        structure = {name: value for name, value in
                     (("subject", subject), ("attribute", attribute))
                     if value is not None}
        if existing is not None:
            existing = dict(existing)
            existing.update(structure)
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
            record.update(structure)  # §11.139: retain the writer's slots.
            self._put_fact(key, record)
            return {"outcome": "new"}
        if (existing["value"] == value
                and bool(existing.get("negated")) == bool(negated)):
            existing["confidence"] = min(1.0, existing["confidence"] + 0.1)
            existing["reinforcements"] += 1
            existing["last_seen"] = now
            self._put_fact(key, existing)
            return {"outcome": "reinforced"}
        old_value = existing["value"]
        was = (f"not {old_value}" if existing.get("negated")
               else str(old_value))
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
        retracted = list(self._retract_derivatives(key))
        for gone in retracted:
            # §11.65: tagged by the retracted key, so "what was X?"
            # can find the takedown after the fact is gone - the same
            # gap the polarity fix closed for revisions.
            self.remember_episode(
                kind="retraction",
                content={"key": gone,
                         "because": f"premise {key!r} was revised"},
                tags=["fact", gone],
            )
        # §11.64: the episode logs SPOKEN forms, polarity included -
        # a §11.48 flip's history must read "was not steel", not
        # "was steel". Old episodes keep their old shape; history
        # answers read whichever form an episode carries.
        now_shown = f"not {value}" if negated else str(value)
        self.remember_episode(
            "revision",
            {"key": key, "old_value": was, "new_value": now_shown},
            tags=["fact", key],
        )
        return {"outcome": "revised", "was": was, "retracted": retracted}

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
        fact = dict(fact)  # §11.139: preserve the indexed pre-write record.
        fact["confidence"] = max(0.0, min(1.0, float(confidence)))
        fact["reinforcements"] += 1
        fact["last_seen"] = _utc_now()
        self._put_fact(key, fact)
        return True

    def recall_fact(self, key: str) -> dict | None:
        """Return the stored fact record for ``key``, or None if absent."""
        charge_lookup()               # §11.122: the bill, counted here
        fact = self._fact_record(key)
        return dict(fact) if fact is not None else None

    # ------------------------------------------------------------------
    # Consolidated (derived) facts and their truth maintenance
    # ------------------------------------------------------------------

    def consolidate_fact(self, key: str, value: Any, confidence: float,
                         premises: list, negated: bool = False, *,
                         subject=None, attribute=None) -> None:  # §11.139
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
        # §11.139: explicit slots replace metadata; omitted slots survive.
        previous = self._fact_record(key) or {}
        for name, supplied in (("subject", subject), ("attribute", attribute)):
            if supplied is not None:
                record[name] = supplied
            elif name in previous:
                record[name] = previous[name]
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
        retracted = self.derivatives_of(revised_key)
        for key in retracted:
            self._drop_fact(key)
        return retracted

    # §11.139: the sole candidate source for recursive truth maintenance.
    def _derived_candidates(self, key) -> set:
        """Keys indexed as depending directly on this premise."""
        if self.shards is not None:
            return self.shards.derived_candidates(key)
        return set(self._derived.get(key, ()))

    def derivatives_of(self, key: str) -> list[str]:
        """Read the recursive retraction set without changing any records."""
        retracted: list[str] = []
        seen: set[str] = set()
        stack = [key]
        while stack:
            changed = stack.pop()
            for key in sorted(self._derived_candidates(changed)):  # §11.139
                if key in seen:
                    continue
                record = self._fact_record(key)
                if not record or "derived_from" not in record:
                    continue
                if any(p_key == changed
                       for p_key, _v in record["derived_from"]):
                    seen.add(key)
                    retracted.append(key)
                    stack.append(key)
        return retracted

    def restore_fact(self, key: str, record_or_None: dict | None) -> None:
        """Restore a full record without reinforcement or truth maintenance."""
        from copy import deepcopy

        if record_or_None is None:
            self._drop_fact(key)
        else:
            self._put_fact(key, deepcopy(record_or_None))

    def _drop_fact(self, key: str) -> None:
        """Remove a fact from whichever store is in use."""
        if self.shards is not None:
            self.shards.delete(key)
        else:
            # §11.139: drops remove only this record's outgoing premise edges.
            self._index_derivation(key, self._facts.get(key), None)
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
            "attributes": {} if self.shards is not None else self._attributes,
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
        self._attributes = dict(payload.get("attributes", {}))
        # §11.139: one rebuild at load; writes maintain it thereafter.
        self._derived = {}
        for key, record in self._facts.items():
            self._index_derivation(key, None, record)
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
