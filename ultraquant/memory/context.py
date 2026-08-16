"""A byte-bounded context window: content in RAM, references to everything else.

The system already had a working memory, and it was the wrong shape for a
machine where memory is the binding constraint. ``SystematicMemory`` keeps
*every* episode in a Python list and bounds "working memory" to the last 16 by
**count**; :meth:`~SystematicMemory.working` then rebuilds a dict over the whole
log on each call. So the resident cost grows without limit, the bound is in the
wrong unit, and there is no reference to anything that fell out — recall means
scanning.

This module is the other arrangement. Two structures, sized very differently:

* **Resident content** — recent turns held whole, bounded in **bytes**, because
  bytes are what a card has. This is the context window proper.
* **A reference index** — for *every* turn ever written, a fixed 24-byte record:
  its id, its byte offset in the log, and a 64-bit sign-random-projection
  sketch of its text. Nothing else. This stays resident permanently.

Recall is then a screen over the index rather than a scan of the content: sketch
the query, XOR it against the resident sketches, ``bit_count`` the results, and
seek to the byte offsets of the best few. The same mechanism
``shards/sketch.py`` already uses for category routing, pointed at conversation
history — and the same reason it is fast, which is that Python's big integers
are bit-parallel.

**The ratio is the point.** A turn averages ~200 bytes of text; its reference is
24. So a **million** turns of history costs ~24 MB resident while the window
itself holds whatever the byte budget allows — a few dozen turns. Disk holds the
information; RAM holds only enough to know *where* it is and *roughly what it is
about*. Both numbers are measured in
``ultraquant/experiments/context_gate.py`` rather than asserted here.

**What this is not.** It is not a transformer's KV cache and does not pretend to
be. A KV cache holds computed attention state and is discarded when it overflows
— information genuinely lost. This holds text and an index, so an overflowed
turn is *recoverable* rather than forgotten. That is a different trade, better
in one direction (nothing is lost) and worse in another (a recalled turn must be
re-read, and the sketch may fetch the wrong one).

Append-only, pure stdlib, and safe to lose: the index is rebuilt from the log on
open, so a torn index file is not data loss.
"""

from __future__ import annotations

import hashlib
import json
import os
from array import array
from pathlib import Path
from typing import Any, Iterable

__all__ = ["ContextWindow", "REFERENCE_BYTES", "DEFAULT_BUDGET_BYTES"]

#: Bytes of resident index per turn: id (8) + offset (8) + sketch (8), held in
#: three ``array('Q')`` buffers rather than Python objects. A dict-of-dicts of
#: the same three fields costs roughly 400 bytes per turn, which is the
#: difference between indexing a million turns in 24 MB and in 400 MB.
REFERENCE_BYTES = 24

#: Default resident content budget. Deliberately small: the whole design is that
#: the window is the scarce thing and the log is not.
DEFAULT_BUDGET_BYTES = 64 * 1024

#: Bits in a turn signature. 64 so one Python int holds it and the whole index
#: screens with AND + bit_count, without unpacking anything.
SKETCH_BITS = 64

#: Bits each content token sets. Two keeps a five-token turn at ~10 bits of 64,
#: which leaves the filter sparse enough that overlap means something; more bits
#: per token saturates the word and every turn starts matching every query.
_BITS_PER_TOKEN = 2


def _tokens(text: str) -> list[str]:
    """Content-bearing tokens of a turn, lowercased and stopword-free.

    Stopwords are dropped because they appear in every turn and would set the
    same bits everywhere, which is pure noise in a 64-bit signature.
    """
    from ultraquant.interpreter.learning import _STOPWORDS

    words = "".join(c.lower() if c.isalnum() else " " for c in text).split()
    return [w for w in words if len(w) > 2 and w not in _STOPWORDS]


def _sketch(text: str, bits: int = SKETCH_BITS) -> int:
    """A 64-bit membership signature: which token-slots this turn occupies.

    Each content token sets :data:`_BITS_PER_TOKEN` bits, chosen by BLAKE2b of
    the token. The signature is the OR of them — a Bloom filter over the turn's
    vocabulary — and relevance is measured by how many of a *query's* bits the
    turn already has set (:func:`_overlap`), not by Hamming distance.

    **This is the second design; the first two were both wrong, in instructive
    ways.**

    The original mixed a per-token CRC32 seed with the bit index, also via
    CRC32. CRC32 is *linear over GF(2)*, so every bit index reduced to the same
    linear functional of the seed differing only by a constant flip: the whole
    signature carried about one bit. Three unrelated sentences all produced
    ``0xaaaa5555aaaa5555``, and recall returned the same three turns for four
    different questions — while the gate reported a comfortable pass.

    Replacing it with a proper BLAKE2b **SimHash** fixed the degeneracy and was
    still wrong for this data. SimHash is a random projection: it needs many
    terms for the per-bit vote sums to be stable. A conversational turn has four
    to six content tokens, so each bit is the sign of a sum of five ±1 votes and
    flips on noise. Measured: against "how tall is the tower", unrelated
    chatter scored Hamming 23 while the actual answer scored 25 — the wrong
    ranking, from a mechanism that was working exactly as designed.

    A membership bitmap is the right structure for short texts, because the
    question really is set overlap rather than vector angle. It is also why
    scoring is asymmetric: a long turn setting many bits should not be
    penalised for covering a short query, so the measure is *coverage of the
    query*, not distance between signatures.

    Deterministic across processes — Python's ``hash()`` for str is randomised
    per interpreter, which would make a rebuilt index disagree with its writer.
    """
    value = 0
    for token in _tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        packed = int.from_bytes(digest, "little")
        for slot in range(_BITS_PER_TOKEN):
            value |= 1 << ((packed >> (slot * 8)) % bits)
    return value


def _overlap(query_bits: int, turn_bits: int) -> int:
    """How many of the query's bits this turn covers. Higher is better."""
    return (query_bits & turn_bits).bit_count()


class ContextWindow:
    """Recent turns in memory, every turn on disk, references to all of them.

    Args:
        path: Directory for ``context.jsonl``. Created if absent.
        budget_bytes: Resident content budget. Turns beyond it are evicted from
            memory but keep their reference, so they remain recallable.
        bits: Sketch width.

    Attributes:
        budget_bytes: The resident ceiling, in bytes.
    """

    def __init__(self, path: str | os.PathLike,
                 budget_bytes: int = DEFAULT_BUDGET_BYTES,
                 bits: int = SKETCH_BITS) -> None:
        self.root = Path(path)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log = self.root / "context.jsonl"
        self.budget_bytes = int(budget_bytes)
        self.bits = int(bits)

        #: Parallel arrays, one entry per turn ever written. C-contiguous, so
        #: the whole index is three allocations rather than N objects.
        self._ids: array = array("Q")
        self._offsets: array = array("Q")
        self._sketches: array = array("Q")

        #: Resident content: turn id -> the turn dict. Insertion-ordered, so the
        #: oldest resident turn is the first key.
        self._resident: dict[int, dict] = {}
        self._resident_bytes = 0
        #: Counted so a caller can tell a cheap turn from an expensive one.
        self.recalls = 0
        self.recall_reads = 0

        self._rebuild_index()

    # ---- writing --------------------------------------------------------

    def add(self, text: str, **meta: Any) -> int:
        """Append a turn, keep it resident, evict until the budget holds.

        Args:
            text: The turn's text.
            **meta: Anything else to store with it.

        Returns:
            The turn id.
        """
        turn_id = len(self._ids) + 1
        record = {"id": turn_id, "text": text, **meta}
        line = json.dumps(record, separators=(",", ":")) + "\n"
        encoded = line.encode("utf-8")
        with open(self.log, "ab") as handle:
            offset = handle.tell()
            handle.write(encoded)

        self._ids.append(turn_id)
        self._offsets.append(offset)
        self._sketches.append(_sketch(text, self.bits))

        self._resident[turn_id] = record
        self._resident_bytes += len(encoded)
        self._evict()
        return turn_id

    def _evict(self) -> None:
        """Drop the oldest resident turns until the byte budget is met.

        Only the *content* goes; the reference stays, which is the whole point.
        A turn evicted here is still findable by :meth:`recall`.
        """
        while self._resident_bytes > self.budget_bytes and len(self._resident) > 1:
            oldest = next(iter(self._resident))
            record = self._resident.pop(oldest)
            self._resident_bytes -= len(
                json.dumps(record, separators=(",", ":")).encode("utf-8")) + 1

    # ---- reading --------------------------------------------------------

    def resident(self) -> list[dict]:
        """The turns currently in memory, oldest first."""
        return list(self._resident.values())

    def read(self, turn_id: int) -> dict | None:
        """Fetch one turn, from memory if resident and from disk if not."""
        if turn_id in self._resident:
            return self._resident[turn_id]
        try:
            position = self._ids.index(turn_id)
        except ValueError:
            return None
        return self._read_at(self._offsets[position])

    def _read_at(self, offset: int) -> dict | None:
        """One seek and one line. This is the disk reference being followed."""
        try:
            with open(self.log, "rb") as handle:
                handle.seek(offset)
                line = handle.readline()
        except OSError:
            return None
        if not line:
            return None
        try:
            return json.loads(line)
        except ValueError:
            return None

    def recall(self, query: str, top_k: int = 3,
               include_resident: bool = False) -> list[dict]:
        """Find the turns most like ``query`` and page them back in.

        The screen is XOR plus ``bit_count`` over the resident sketch array —
        no content is touched until the winners are known, and only those are
        read from disk. That is the entire reason the index can cover a history
        the machine could not hold.

        Args:
            query: What to match against.
            top_k: How many turns to return.
            include_resident: Whether turns already in memory may be returned.
                Off by default, since a caller already has those and the
                interesting question is what recall adds beyond the window.

        Returns:
            Turn dicts, closest first.
        """
        if not self._ids:
            return []
        target = _sketch(query, self.bits)
        scored: list[tuple[int, int]] = []
        for position in range(len(self._ids)):
            turn_id = self._ids[position]
            if not include_resident and turn_id in self._resident:
                continue
            # Negated so the existing ascending sort puts best coverage first.
            scored.append((-_overlap(target, self._sketches[position]), position))
        if not scored:
            return []
        scored.sort()
        self.recalls += 1
        found = []
        for negated, position in scored[:top_k]:
            if negated == 0:
                break        # no shared vocabulary at all is not a candidate
            record = self._read_at(self._offsets[position])
            self.recall_reads += 1
            if record is not None:
                record["_overlap"] = -negated
                found.append(record)
        return found

    # ---- accounting -----------------------------------------------------

    def _rebuild_index(self) -> None:
        """Rebuild the reference index by streaming the log.

        The index is never persisted separately, so it cannot go stale or be
        torn: a half-written index file would be a corruption risk, and the log
        is the only thing that has to survive.
        """
        if not self.log.exists():
            return
        offset = 0
        with open(self.log, "rb") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    offset += len(line)
                    continue
                self._ids.append(int(record.get("id", len(self._ids) + 1)))
                self._offsets.append(offset)
                self._sketches.append(_sketch(str(record.get("text", "")),
                                              self.bits))
                offset += len(line)

    def stats(self) -> dict:
        """Sizes, in the units that decide whether this design is worth it."""
        turns = len(self._ids)
        index_bytes = (self._ids.buffer_info()[1] * self._ids.itemsize
                       + self._offsets.buffer_info()[1] * self._offsets.itemsize
                       + self._sketches.buffer_info()[1]
                       * self._sketches.itemsize)
        stored = self.log.stat().st_size if self.log.exists() else 0
        return {
            "turns": turns,
            "resident_turns": len(self._resident),
            "resident_bytes": self._resident_bytes,
            "budget_bytes": self.budget_bytes,
            "index_bytes": index_bytes,
            "stored_bytes": stored,
            "resident_to_stored": (
                (self._resident_bytes + index_bytes) / stored if stored else 0.0),
            "recalls": self.recalls,
            "recall_reads": self.recall_reads,
        }
