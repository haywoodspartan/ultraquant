"""The contemporary stash: quarantine and analysis for anything found on the web.

Nothing fetched from the network is allowed to become knowledge on arrival.  Web
text lands here first, is split into individual claims, and is classified before
any of it can reach :class:`~ultraquant.memory.systematic.SystematicMemory`:

* ``factual-claim`` — a declarative, checkable assertion.
* ``opinion``       — judgement or preference; never promoted automatically.
* ``hedged``        — reported, alleged, rumoured, possible; not asserted.
* ``unclassified``  — shape not recognised; stays put.

A claim is only marked ``corroborated`` once **independent sources** (distinct
network locations) assert it.  A claim that contradicts something already held as
fact is marked ``disputed`` rather than silently overwriting what is known.  Only
``factual-claim`` entries that are staged or corroborated may be promoted into
memory as fact; opinion, hedged and disputed entries require an explicit forced
promotion by the user, and rejection is always available.

The stash is therefore the boundary between *found* and *believed*.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["StashError", "ContemporaryStash"]


class StashError(RuntimeError):
    """Raised when a stash entry is not eligible for the requested transition."""


#: Markers of judgement/preference rather than assertion of fact.
_OPINION_MARKERS = (
    "i think", "i believe", "in my view", "in my opinion", "imo", "we feel",
    "best", "worst", "greatest", "amazing", "terrible", "beautiful", "awful",
    "favorite", "favourite", "should", "ought to", "arguably", "prefer",
    "wonderful", "horrible", "stunning", "disappointing", "overrated",
    "underrated",
)

#: Markers of reported or uncertain speech.
_HEDGE_MARKERS = (
    "reportedly", "allegedly", "rumored", "rumoured", "may ", "might ",
    "possibly", "some say", "claims that", "is said to", "supposedly",
    "apparently", "could be", "unconfirmed",
)

#: Declarative copulas that make "<subject> <cop> <value>" a checkable claim.
_COPULAS = (" is ", " are ", " was ", " were ", " has ", " have ")

_NUMBER_UNIT = re.compile(r"\b\d+(?:\.\d+)?\s*[a-zA-Z%°]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

VALID_STATUSES = ("staged", "corroborated", "disputed", "promoted", "rejected")


def _utc_now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _normalize(claim: str) -> str:
    """Normalized form used to recognise the same claim from another source."""
    return " ".join(claim.lower().split()).strip(" .!?")


#: Words carrying no content for similarity purposes.
_SIMILARITY_STOPWORDS = frozenset(
    "the a an and or of to in on is are was were for with as by at from into "
    "it its this that called can be has have".split()
)


def _content_tokens(text: str) -> set[str]:
    """The content words of a claim, for paraphrase comparison."""
    return {
        word for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in _SIMILARITY_STOPWORDS and len(word) > 2
    }


_NUMBER_IN_VALUE = re.compile(r"\d+(?:\.\d+)?")


def claim_relation(claim_a: str, claim_b: str) -> str | None:
    """How two declarative claims stand to each other, if it can be told.

    Returns ``"agrees"``, ``"contradicts"``, or ``None`` when no verdict is
    honest. The old dispute test was string inequality on exact-matching keys,
    and it produced a live false dispute: "arithmetic is the study of numbers
    and their operations" against Wikipedia's "an elementary branch of
    mathematics that deals with numerical operations" — two *agreeing*
    definitions flagged as conflict.

    Measured on labelled pairs, the signal is in the **type structure of the
    values**, not in their similarity:

    * **numeric vs numeric** — the numbers are the claim. Same number set
      agrees; different number sets contradict (324 vs 330, 4 vs 6, 486 vs
      512 all separated cleanly). Known limit: units are not normalised, so
      324 m vs 0.324 km would wrongly contradict.
    * **short vs short** — a short definite value competes for an identity
      slot. Every true contradiction measured (Paris/Lyon, blue/red,
      square/diamond) had *zero* token overlap; overlap means agreement.
    * **descriptive values** — agreement pairs measured 0.11-0.12 similarity
      and different-aspect pairs 0.00: not separable. Descriptive values
      therefore **never contradict** (one subject can support many true
      descriptions), and agree only when the dual judges are emphatic
      (J >= 0.5 and H >= 0.45).

    The measurement also surfaced a trap now guarded here: two values with no
    content tokens at all ("4" vs "6" after stopword and length filtering)
    produce two *empty* hypervector bundles, and empty bundles score
    similarity 1.0.
    """
    split_a = _split_claim(claim_a)
    split_b = _split_claim(claim_b)
    if split_a is None or split_b is None:
        return None
    if not (_content_tokens(split_a[0]) & _content_tokens(split_b[0])):
        return None

    value_a, value_b = split_a[1], split_b[1]
    numbers_a = _NUMBER_IN_VALUE.findall(value_a)
    numbers_b = _NUMBER_IN_VALUE.findall(value_b)
    if numbers_a and numbers_b:
        if sorted(numbers_a) == sorted(numbers_b):
            return "agrees"
        return "contradicts"
    if numbers_a or numbers_b:
        return None  # a measurement and a description are different aspects

    tokens_a = _content_tokens(value_a)
    tokens_b = _content_tokens(value_b)
    if not tokens_a or not tokens_b:
        return None  # the empty-bundle trap: no content, no verdict
    short_a = len(tokens_a) <= 3
    short_b = len(tokens_b) <= 3
    overlap = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    if short_a and short_b:
        return "agrees" if tokens_a & tokens_b else "contradicts"
    if overlap >= 0.5:
        from ultraquant.reason.hypervector import similarity

        if similarity(_claim_vector(value_a), _claim_vector(value_b)) >= 0.45:
            return "agrees"
    return None


def _claim_vector(claim: str) -> int:
    """A claim as a hypervector bag of its content words.

    Deterministic (seed vectors are name-derived), fixed-width, and compared
    by XOR + popcount — the structural machinery of §11.7 doing paraphrase
    duty. Word order is deliberately discarded; corroboration asks whether two
    sentences talk about the same things, not whether they share syntax.
    """
    from ultraquant.reason.hypervector import bundle, seed_vector

    words = sorted(_content_tokens(claim))
    if not words:
        return 0
    return bundle([seed_vector(word) for word in words])


def _split_claim(claim: str) -> tuple[str, str] | None:
    """Split a declarative claim into ``(subject, value)`` if it has that shape.

    The subject is normalized the same way user-typed statements are (lowercased,
    leading article dropped) so that a claim found on the web and a fact told
    directly to the model land on the same key — which is what lets the stash
    notice a contradiction instead of silently storing a rival value.
    """
    lowered = claim.lower()
    for cop in _COPULAS:
        idx = lowered.find(cop)
        if idx > 0:
            subject = claim[:idx].strip().strip(",").lower()
            for article in ("the ", "a ", "an "):
                if subject.startswith(article):
                    subject = subject[len(article):]
                    break
            value = claim[idx + len(cop):].strip().strip(" .!?")
            if subject and value and len(subject.split()) <= 6:
                return subject, value
    return None


class ContemporaryStash:
    """Holding area where web claims are analysed before they can become fact."""

    def __init__(self, path: str | os.PathLike) -> None:
        """Open (or create) a stash file.

        Args:
            path: JSON file the stash persists to; loaded if it already exists.
        """
        self.path = Path(path)
        self._entries: dict[int, dict] = {}
        self._next_id = 1
        if self.path.exists():
            self.load()

    # ------------------------------------------------------------ persistence

    def load(self) -> None:
        """Read the stash from disk."""
        with open(self.path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._entries = {int(e["id"]): e for e in data.get("entries", [])}
        self._next_id = int(data.get("next_id", max(self._entries, default=0) + 1))

    def save(self) -> None:
        """Write the stash to disk (called after every mutation)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": [self._entries[k] for k in sorted(self._entries)],
            "next_id": self._next_id,
        }
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------ intake

    def add_page(self, url: str, title: str, text: str, max_claims: int = 8) -> list[int]:
        """Stage the claims of a fetched page.

        Sentences already staged from a *different* network location are not
        duplicated: the new source is appended to the existing entry instead,
        which is what later lets the claim corroborate.

        Args:
            url: Source URL.
            title: Page title (recorded on new entries).
            text: Extracted page text.
            max_claims: Maximum number of sentences taken from this page.

        Returns:
            Ids of the entries created or updated.
        """
        from urllib.parse import urlparse

        netloc = urlparse(url).netloc or url
        touched: list[int] = []
        seen_here: set[str] = set()

        # Real pages front-load navigation - hatnotes, cross-references,
        # "X redirects here" - and taking the first ``max_claims`` sentences in
        # document order filled the whole budget with it. Measured live: the
        # lead sentence actually defining 'pixel' never made the cut because
        # eight lines of chrome came first. Candidates are therefore gathered
        # first, cross-reference boilerplate is dropped outright (an imperative
        # pointing at another page is not a claim about the world, on any
        # site), and declarative sentences spend the budget before the rest.
        declarative: list[str] = []
        other: list[str] = []
        for sentence in _SENTENCE_SPLIT.split(text.replace("\n", " ")):
            claim = " ".join(sentence.split()).strip()
            # The upper bound was 300 and it silently cost the best claim on a
            # real page: encyclopedia lead sentences routinely embed an
            # em-dashed list of examples ("...rules to convert information -
            # such as a letter, word, sound, image, or gesture - into another
            # form...") and the one sentence that actually defined 'code' was
            # dropped for length while hatnote-adjacent filler survived.
            if not (20 <= len(claim) <= 420):
                continue
            key = _normalize(claim)
            if key in seen_here:
                continue
            seen_here.add(key)
            lowered = claim.lower()
            if (lowered.startswith(("for ", "see ", "not to be confused"))
                    or "redirects here" in lowered
                    or "disambiguation" in lowered):
                continue
            if " is " in lowered or " are " in lowered:
                declarative.append(claim)
            else:
                other.append(claim)

        for claim in (declarative + other)[:max_claims]:
            key = _normalize(claim)

            existing = next((e for e in self._entries.values() if _normalize(e["claim"]) == key), None)
            if existing is not None:
                if netloc not in existing["sources"]:
                    existing["sources"].append(netloc)
                    if existing["classification"] == "factual-claim":
                        existing["classification"] = "unclassified"  # re-analyse
                touched.append(existing["id"])
                continue

            entry = {
                "id": self._next_id,
                "url": url,
                "netloc": netloc,
                "title": title,
                "fetched": _utc_now(),
                "claim": claim,
                "classification": "unclassified",
                "status": "staged",
                "sources": [netloc],
                "notes": "",
            }
            self._entries[entry["id"]] = entry
            self._next_id += 1
            touched.append(entry["id"])

        self.save()
        return touched

    # ---------------------------------------------------------------- analysis

    @staticmethod
    def classify(claim: str) -> str:
        """Classify a single claim as opinion, hedged, factual or unclassified."""
        lowered = " " + claim.lower() + " "
        if any(marker in lowered for marker in _OPINION_MARKERS):
            return "opinion"
        if any(marker in lowered for marker in _HEDGE_MARKERS):
            return "hedged"
        if _split_claim(claim) is not None or _NUMBER_UNIT.search(claim):
            return "factual-claim"
        return "unclassified"

    def analyze(self, memory: Any | None = None) -> dict:
        """Classify staged entries and update their corroboration status.

        Args:
            memory: Optional memory consulted to detect contradictions with
                what is already held as fact.

        Returns:
            Counts by classification and by status.
        """
        for entry in self._entries.values():
            if entry["status"] in ("promoted", "rejected"):
                continue
            if entry["classification"] == "unclassified":
                entry["classification"] = self.classify(entry["claim"])

            if entry["classification"] != "factual-claim":
                continue

            disputed = False
            if memory is not None:
                split = _split_claim(entry["claim"])
                if split is not None:
                    known = memory.recall_fact(split[0])
                    if known is not None:
                        stored = f"{split[0]} is {known.get('value')}"
                        relation = claim_relation(entry["claim"], stored)
                        if relation == "contradicts":
                            disputed = True
                            entry["notes"] = (
                                f"conflicts with stored fact: "
                                f"{known.get('value')!r}"
                            )
                        elif relation == "agrees":
                            # The web restating what is already held is
                            # evidence for it, not a rival to it. The old
                            # string-inequality test called this a dispute.
                            memory.confirm_fact(
                                split[0],
                                min(0.95, float(known.get("confidence", 0.5))
                                    + 0.05),
                            )
                            entry["notes"] = "agrees with stored fact"
            # A claim that memory backs does not become disputed because some
            # other source clashes with it - the *rival* carries the dispute
            # (it will fail its own memory check above). Only claims with no
            # memory backing are marked when sources disagree among
            # themselves, and then both sides become questions for the human.
            if not disputed and entry.get("notes") != "agrees with stored fact":
                rival = self._contradicting_source(entry)
                if rival is not None:
                    disputed = True
                    note = f"sources disagree: entry {rival}"
                    if note not in (entry["notes"] or ""):
                        entry["notes"] = ((entry["notes"] or "") + " "
                                          + note).strip()
            if disputed:
                entry["status"] = "disputed"
            elif len(entry["sources"]) >= 2:
                entry["status"] = "corroborated"
            elif (supporter := self._paraphrase_support(entry)) is not None:
                entry["status"] = "corroborated"
                note = f"paraphrased by entry {supporter}"
                if note not in (entry["notes"] or ""):
                    entry["notes"] = ((entry["notes"] or "") + " " + note).strip()
            else:
                entry["status"] = "staged"

        self.save()
        return self.stats()

    def _contradicting_source(self, entry: dict) -> int | None:
        """Another *source* asserting a rival value for the same subject.

        The old dispute path only compared web claims against memory, so two
        fetched pages flatly disagreeing with each other went unnoticed — the
        model would happily hold both at 0.55. Cross-source contradiction uses
        the same typed relation as everything else: a numeric or identity
        clash from a different network location marks both claims disputed,
        and disputed claims become questions for the human
        (``LearningSession._stash_gaps``). Descriptive claims can never reach
        here, by the measured design of :func:`claim_relation`.
        """
        for other in self._entries.values():
            if other["id"] == entry["id"]:
                continue
            if other["status"] in ("rejected", "promoted"):
                continue
            if other["classification"] not in ("factual-claim", "unclassified"):
                continue
            if set(other["sources"]) <= set(entry["sources"]):
                continue  # one source contradicting itself is just noise
            if claim_relation(entry["claim"], other["claim"]) == "contradicts":
                return other["id"]
        return None

    def _paraphrase_support(self, entry: dict) -> int | None:
        """Another source saying the same thing in different words, if any.

        Source-count corroboration matches normalized sentences verbatim, and
        measured against the live web it fired exactly never: two sites
        describing the same thing do not phrase it identically, so verbatim
        matching is mirror detection only. This is the paraphrase test.

        A claim corroborates this one when it comes from a different network
        location, its declarative *subject* shares a content word, and the two
        sentences agree under **both** of two independent similarity judges —
        content-token Jaccard and hypervector bag similarity (content words
        bundled into one 10,000-bit vector, compared by Hamming agreement; the
        structural half of §11.7 earning a production role).

        The thresholds come from measurement on live en/simple Wikipedia
        pairs, and the boundary is genuinely tight: the true paraphrase pair
        ("the main arithmetic operations are addition..." / "the four basic
        arithmetic operations are addition...") scored J=0.250, H=0.311, while
        the best same-subject *different-assertion* pair scored J=0.207,
        H=0.231, and different-subject pairs never exceeded J=0.038, H=0.052.
        Requiring both judges (J >= 0.24 AND H >= 0.28) is deliberate
        conservatism: a false corroboration inflates belief at 0.8 confidence,
        while a missed one merely leaves a claim staged at 0.55 — the errors
        are not symmetric, so the gate leans toward refusing.
        """
        my_tokens = _content_tokens(entry["claim"])
        my_split = _split_claim(entry["claim"])
        if not my_tokens or my_split is None:
            return None
        my_subject = _content_tokens(my_split[0])
        my_vector = _claim_vector(entry["claim"])
        for other in self._entries.values():
            if other["id"] == entry["id"]:
                continue
            if other["status"] in ("rejected",):
                continue
            if set(other["sources"]) <= set(entry["sources"]):
                continue  # same place saying it twice is not support
            other_split = _split_claim(other["claim"])
            if other_split is None:
                continue
            if not (my_subject & _content_tokens(other_split[0])):
                continue
            # "The tower height is 330 metres" shares almost every content
            # word with "...324 metres" and sails past both similarity
            # judges - a contradiction reading as a paraphrase of its own
            # rival. The typed relation is the authority on conflict, and
            # conflict trumps resemblance.
            if claim_relation(entry["claim"], other["claim"]) == "contradicts":
                continue
            other_tokens = _content_tokens(other["claim"])
            union = my_tokens | other_tokens
            if not union:
                continue
            jaccard = len(my_tokens & other_tokens) / len(union)
            if jaccard < 0.24:
                continue
            from ultraquant.reason.hypervector import similarity

            if similarity(my_vector, _claim_vector(other["claim"])) >= 0.28:
                return other["id"]
        return None

    def subject_support(self, subject: str) -> dict:
        """Graded cross-source agreement about a *subject*, not a sentence.

        Sentence-level corroboration — even paraphrase-level — fired zero
        times against the live web, because two sites describing the same
        thing choose **different aspects** to state: one defines a pixel, the
        other describes its RGB representation. No sentence pair aligns, yet
        the sites plainly agree about what a pixel is.

        So agreement is accumulated at the subject level instead: everything
        each source says about the subject is bundled into one hypervector
        (order-free superposition of its claims' content words), and support
        is the strongest cross-source agreement between those bundles. Partial
        overlaps that no sentence pair would clear add up — which is the
        point, and is why this is a new mechanism rather than a lower
        threshold on the old one.

        Measured live (en vs simple Wikipedia): same subject scored 0.111 to
        0.327 across four subjects, different subjects never exceeded 0.048.
        Every subject fired; sentence-level had fired for none.

        Args:
            subject: The subject term.

        Returns:
            ``{"support": strongest cross-source similarity, "sources": n,
            "claims": n}`` — support 0.0 when fewer than two sources have
            anything to say.
        """
        from ultraquant.reason.hypervector import bundle, similarity

        stems = {subject.lower().strip(), subject.lower().strip().rstrip("s")}
        by_netloc: dict[str, list[int]] = {}
        total = 0
        for entry in self._entries.values():
            if entry["classification"] not in ("factual-claim", "unclassified"):
                continue
            if entry["status"] == "rejected":
                continue
            split = _split_claim(entry["claim"])
            if split is None:
                continue
            if not (stems & _content_tokens(split[0])):
                continue
            vector = _claim_vector(entry["claim"])
            if not vector:
                continue
            total += 1
            for netloc in entry["sources"]:
                by_netloc.setdefault(netloc, []).append(vector)

        if len(by_netloc) < 2:
            return {"support": 0.0, "sources": len(by_netloc), "claims": total}
        bundles = {netloc: bundle(vectors)
                   for netloc, vectors in by_netloc.items()}
        names = sorted(bundles)
        support = max(
            similarity(bundles[a], bundles[b])
            for i, a in enumerate(names) for b in names[i + 1:]
        )
        return {"support": round(max(0.0, support), 4),
                "sources": len(by_netloc), "claims": total}

    # --------------------------------------------------------------- decisions

    def promote(self, entry_id: int, memory: Any, force: bool = False) -> str:
        """Promote a stashed claim into memory as fact.

        Args:
            entry_id: Entry to promote.
            memory: Memory that will hold the resulting fact.
            force: Required to promote opinion, hedged or disputed entries.

        Returns:
            The memory fact key that was written.

        Raises:
            KeyError: If the entry does not exist.
            StashError: If the entry is not eligible and ``force`` is False.
        """
        entry = self._entries[int(entry_id)]
        if entry["status"] in ("promoted", "rejected") and not force:
            raise StashError(f"entry {entry_id} is already {entry['status']}")
        eligible = (
            entry["classification"] == "factual-claim"
            and entry["status"] in ("staged", "corroborated")
        )
        if not eligible and not force:
            raise StashError(
                f"entry {entry_id} is {entry['classification']}/{entry['status']} — "
                "not eligible for promotion; use force to override"
            )

        split = _split_claim(entry["claim"])
        if split is not None:
            key, value = split
        else:
            key, value = f"web:{entry['netloc']}:{entry['id']}", entry["claim"]

        confidence = 0.8 if entry["status"] == "corroborated" else 0.55
        if not eligible:
            confidence = 0.3
        memory.remember_fact(key, value, confidence=confidence)
        memory.remember_episode(
            "promotion",
            {
                "key": key,
                "value": value,
                "url": entry["url"],
                "sources": list(entry["sources"]),
                "classification": entry["classification"],
                "prior_status": entry["status"],
                "forced": bool(force and not eligible),
            },
            tags=["web", "stash"],
        )
        entry["status"] = "promoted"
        self.save()
        return key

    def reject(self, entry_id: int, reason: str = "") -> None:
        """Mark an entry as rejected so it is never promoted."""
        entry = self._entries[int(entry_id)]
        entry["status"] = "rejected"
        if reason:
            entry["notes"] = reason
        self.save()

    # ------------------------------------------------------------------ access

    def get(self, entry_id: int) -> dict:
        """Return one entry (a copy)."""
        return dict(self._entries[int(entry_id)])

    def entries(self, status: str | None = None, classification: str | None = None) -> list[dict]:
        """Return entries, optionally filtered by status and/or classification."""
        out = [dict(self._entries[k]) for k in sorted(self._entries)]
        if status is not None:
            out = [e for e in out if e["status"] == status]
        if classification is not None:
            out = [e for e in out if e["classification"] == classification]
        return out

    def stats(self) -> dict:
        """Counts of entries by classification and by status."""
        by_class: dict[str, int] = {}
        by_status: dict[str, int] = {}
        for entry in self._entries.values():
            by_class[entry["classification"]] = by_class.get(entry["classification"], 0) + 1
            by_status[entry["status"]] = by_status.get(entry["status"], 0) + 1
        return {
            "entries": len(self._entries),
            "by_classification": by_class,
            "by_status": by_status,
        }
