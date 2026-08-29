"""The embedding suggester: surface forms translated at the boundary.

§11.37 measured the language wall and its remedy in the same run: the
synonym family fails the lexical core outright (0.000 asserted), and an
embedding model buys exactly the direct-attribute half — at cosine>0.75
with zero decoy falls, and at 0.65 with a 0.833 fabrication rate. This
module is that measurement turned into wiring, and it keeps the
division of labor the measurement dictated:

* **The embedding SUGGESTS, the lexical core VERIFIES.** A suggestion
  must clear two independent checks: the measured cosine threshold, and
  an **anchor rule** — the matched key must share at least one
  informative token with the question. "How tall is the tower" anchors
  on "tower" and may read as ``tower height``; "how tall is the
  obelisk" anchors on nothing and refuses, whatever the cosine says.
  The anchor is subject consistency (§11.38) applied at the boundary.
* **Candidates come from the library's own index**, not a scan: the
  same phrase-and-token probes the spread uses, a bounded few dozen
  keys embedded in one batched call. The store stays on disk; the
  embedding model never sees more than the working set.
* **Named, never silent.** A suggested reading is labelled in the
  reply — the key it read the question as, and the match strength — so
  the reader can veto it, exactly like the modifier label (§11.36).
* **Optional by construction.** No LM Studio, no suggester, no change:
  sessions default to None and every lexical behavior stays identical.
  The live surfaces enable it from settings; the gate measures it
  against the OFF arm before any of this counts as a capability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ultraquant.shards.router import _informative, normalize_token

__all__ = ["SemanticSuggester", "Suggestion", "COSINE_FLOOR"]

#: The measured operating point: §11.37 found zero decoy falls at 0.75
#: and a 0.833 fabrication rate at 0.65. This is not tunable politely.
COSINE_FLOOR = 0.75

#: Seconds between availability re-checks when LM Studio was down.
_RETRY_SECONDS = 60.0


@dataclass
class Suggestion:
    """One verified reading of a question.

    Attributes:
        key: The fact key the question was read as.
        value: That fact's stored value.
        confidence: The fact's stored confidence.
        similarity: The cosine match that proposed the reading.
    """

    key: str
    value: str
    confidence: float
    similarity: float


class SemanticSuggester:
    """Question-to-key reading via the local embedding model.

    Args:
        client: An :class:`~ultraquant.interpreter.lmstudio.LMStudioClient`,
            or None to construct one lazily on first use.
        embedder: Anything exposing ``embed(texts) -> list[vector]`` and
            ``available()``. Supplying one bypasses LM Studio entirely -
            see :class:`~ultraquant.model.distilled.DistilledEmbedder`,
            which is that interface over weights distilled offline.
            The suggester cannot tell the two apart and does not try.
        floor: Cosine threshold. Defaults to :data:`COSINE_FLOOR`, which
            was measured for the LIVE embedding. A different embedder
            lives on a different scale and must bring its own number
            rather than inherit this one.
    """

    def __init__(self, client=None, embedder=None,
                 floor: float | None = None) -> None:
        self._client = client
        self._embedder = embedder
        self._model: str | None = None
        self._down_until = 0.0
        self.floor = COSINE_FLOOR if floor is None else float(floor)

    # ------------------------------------------------------------- plumbing

    def _ready(self):
        """The (embedder, model) pair, or None while it is unusable.

        A supplied embedder short-circuits every part of this: there
        is no server to be down, no model list to consult and no
        retry window to wait out.
        """
        if self._embedder is not None:
            return (self._embedder, None) if self._embedder.available()                 else None
        if time.monotonic() < self._down_until:
            return None
        try:
            if self._client is None:
                from ultraquant.interpreter.lmstudio import LMStudioClient

                self._client = LMStudioClient()
            if not self._client.available():
                raise RuntimeError("unreachable")
            if self._model is None:
                models = self._client.embedding_models()
                if not models:
                    raise RuntimeError("no embedding model")
                self._model = models[0]
            return self._client, self._model
        except Exception:  # noqa: BLE001 - absence must cost nothing
            self._down_until = time.monotonic() + _RETRY_SECONDS
            return None

    @staticmethod
    def _fold(text: str) -> set[str]:
        import re

        return {normalize_token(tok)
                for tok in re.findall(r"[a-z0-9]+", text.lower())
                if _informative(tok)}

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    # ------------------------------------------------------------ the work

    def suggest(self, question: str, memory) -> Suggestion | None:
        """A verified reading of ``question`` over held fact keys, or None.

        Args:
            question: The question the lexical core could not assert on.
            memory: The session's :class:`SystematicMemory`.

        Returns:
            The best :class:`Suggestion` clearing BOTH the cosine floor
            and the anchor rule, or None.
        """
        ready = self._ready()
        if ready is None:
            return None
        embedder, model = ready

        question_tokens = self._fold(question)
        if not question_tokens:
            return None
        from ultraquant.reason.inference import (_ordered_fold,
                                                 _probe_phrases,
                                                 _reachable_facts)

        ordered = _ordered_fold(question)
        probes = [" ".join(sorted(question_tokens))]
        probes += _probe_phrases(ordered)
        probes += sorted(question_tokens)
        candidates = _reachable_facts(memory, probes)
        anchored = {}
        for key, record in candidates.items():
            if self._fold(key) & question_tokens:
                anchored[key] = record
        if not anchored:
            return None

        keys = sorted(anchored)
        try:
            vectors = embedder.embed([question] + keys, model=model)
        except Exception:  # noqa: BLE001 - a failed call is a down server
            self._down_until = time.monotonic() + _RETRY_SECONDS
            return None
        question_vector, key_vectors = vectors[0], vectors[1:]

        ordered = [normalize_token(tok) for tok in
                   __import__("re").findall(r"[a-z0-9]+", question.lower())
                   if _informative(tok)]

        def _uncovered_ok(key_tokens: set) -> bool:
            # Positional verification (§11.36's rule, transitive): every
            # question token the key does not cover must precede - walking
            # rightward - a covered token. "how TALL is the tower" leaves
            # 'tall' uncovered before covered 'tower': a synonym slot.
            # "the height of the OBELISK" leaves 'obelisk' uncovered at
            # the end: a SUBJECT the key does not hold, and reading it
            # away is the wrong-metal fabrication with a cosine excuse -
            # the first gate run measured exactly that fall.
            allowed = False
            for token in reversed(ordered):
                if token in key_tokens:
                    allowed = True
                elif not allowed:
                    return False
            return True

        best: Suggestion | None = None
        for key, vector in zip(keys, key_vectors):
            similarity = self._cosine(question_vector, vector)
            if similarity < self.floor:
                continue
            if not _uncovered_ok(self._fold(key)):
                continue
            record = anchored[key]
            if best is None or similarity > best.similarity:
                best = Suggestion(
                    key=key,
                    value=str(record.get("value", "")),
                    confidence=float(record.get("confidence", 0.0)),
                    similarity=similarity,
                )
        return best
