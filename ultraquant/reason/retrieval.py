"""On demand: what this question needs, and nothing else.

The claim this whole library rests on is that a model does not have
to spend itself entirely on every question - it can read the shards
the question points at and leave the rest on storage. The pieces for
that have existed for a long time and have never been one thing:

* `memory.find_facts` ranks keys through the inverted index,
* `inference._reachable_facts` widens that with phrase probes,
* `SemanticSuggester.suggest` reaches what shares no token at all,
* `PatternWorkingSet` predicts and prefetches shards,
* `ShardCache` decides what stays resident.

Five mechanisms, five callers, and no single answer to *"what did
this question actually cost?"* This module is that answer: one call
in, the facts and the bill out.

**It invents no retrieval.** Every route here was measured
separately; what is new is the composition and the accounting.

**The width is a constant, and that is a finding rather than a
default.** §11.104 built the obvious alternative - let the answer's
own entropy decide how many shards to read - and it **lost to a plain
constant three separate ways**, at 2.56 reads against a fixed 3 that
scored what reading all twelve scored. So the engine reads a fixed
few and the record says why, instead of a loop that costs more to
arrive at the same place.

**Routes are tried in trust order and stop early.** Lexical first,
because it is exact, cheapest, and cannot invent; then phrase reach,
which finds chains; then the semantic route, which is the only one
that can read *"how tall"* onto ``tower height`` and the only one
that costs a representation. A route runs only if the ones before it
did not already cover the question, so the expensive path is not paid
for the easy questions.

**Coverage decides, not similarity.** §11.29's rule: a key that
matches some of a question is not an answer to it. The engine reports
`covered` rather than deciding for the caller, because the pipeline
above it has refusal machinery that is measured and this layer has
none.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ultraquant.shards.router import _informative, normalize_token

__all__ = ["Retrieved", "Retrieval", "RetrievalEngine", "DEFAULT_WIDTH"]

#: Keys pulled per route. Three, because §11.104 measured a fixed
#: three of twelve experts scoring exactly what all twelve scored -
#: and measured an adaptive loop losing to that constant. A width
#: this small is the whole economic claim; widening it should require
#: a measurement, not a hunch.
DEFAULT_WIDTH = 3


def _fold(text: str) -> set:
    return {normalize_token(token)
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if _informative(token)}


@dataclass
class Retrieved:
    """One fact, and how it was reached.

    Attributes:
        key: The fact key.
        record: The stored record.
        route: "lexical", "reach" or "semantic" - which mechanism
            surfaced it, so a caller can weigh them differently and a
            reader can see which one is doing the work.
        strength: Route-specific score; cosine for the semantic route,
            rank-derived otherwise. Not comparable across routes, and
            deliberately not normalised into looking as though it is.
    """

    key: str
    record: dict
    route: str
    strength: float = 0.0


@dataclass
class Retrieval:
    """What the question needed and what reading it cost.

    Attributes:
        question: The question asked.
        facts: What was retrieved, best first within each route.
        examined: Keys the engine looked at, whether kept or not.
        routes: Route -> how many facts it contributed.
        covered: Whether some retrieved key covers every informative
            token of the question (§11.29). False is common and is
            not an error - it is what the refusal machinery above
            reads.
        stopped_after: The last route that ran. Routes after it were
            never paid for.
    """

    question: str
    facts: list = field(default_factory=list)
    examined: int = 0
    routes: dict = field(default_factory=dict)
    covered: bool = False
    stopped_after: str = ""

    @property
    def keys(self) -> list:
        return [item.key for item in self.facts]

    def by_route(self, route: str) -> list:
        return [item for item in self.facts if item.route == route]


class RetrievalEngine:
    """One call: the memory this question needs, and the bill.

    Args:
        memory: The session's :class:`SystematicMemory`.
        suggester: An optional
            :class:`~ultraquant.reason.semantic.SemanticSuggester`.
            Absent, the semantic route simply never runs and every
            other route behaves identically - the same
            optional-by-construction rule the suggester itself keeps.
        width: Keys per route. See :data:`DEFAULT_WIDTH`.
    """

    def __init__(self, memory, suggester=None,
                 width: int = DEFAULT_WIDTH) -> None:
        self.memory = memory
        self.suggester = suggester
        self.width = int(width)

    def _covers(self, key: str, asked: set) -> bool:
        return asked <= _fold(key)

    def _lexical(self, question: str, asked: set, seen: set) -> list:
        """Exact index ranking. Cheapest, and cannot invent."""
        out = []
        for key in self.memory.find_facts(question, top_k=self.width):
            if key in seen:
                continue
            record = self.memory.recall_fact(key)
            if record is None:
                continue
            seen.add(key)
            out.append(Retrieved(key, record, "lexical",
                                 len(asked & _fold(key)) / max(1, len(asked))))
        return out

    def _reach(self, question: str, asked: set, seen: set) -> list:
        """Phrase probes - what the spread uses to find chains."""
        from ultraquant.reason.inference import _ordered_fold, _probe_phrases

        ordered = _ordered_fold(question)
        probes = [" ".join(sorted(asked))] + _probe_phrases(ordered) \
            + sorted(asked)
        out = []
        for probe in probes:
            for key in self.memory.find_facts(probe, top_k=self.width):
                if key in seen:
                    continue
                record = self.memory.recall_fact(key)
                if record is None:
                    continue
                seen.add(key)
                out.append(Retrieved(key, record, "reach",
                                     len(asked & _fold(key))
                                     / max(1, len(asked))))
        return out

    def _semantic(self, question: str, seen: set) -> list:
        """The only route that reads 'how tall' onto 'tower height'.

        It runs last and only when nothing else covered the question,
        because it is the one route that costs a representation - a
        network call with the live embedder, an encode with the
        distilled one.
        """
        if self.suggester is None:
            return []
        try:
            suggestion = self.suggester.suggest(question, self.memory)
        except Exception:                          # noqa: BLE001
            return []
        if suggestion is None or suggestion.key in seen:
            return []
        record = self.memory.recall_fact(suggestion.key)
        if record is None:
            return []
        seen.add(suggestion.key)
        return [Retrieved(suggestion.key, record, "semantic",
                          float(getattr(suggestion, "similarity", 0.0)))]

    def retrieve(self, question: str) -> Retrieval:
        """The facts this question points at, and what they cost."""
        asked = _fold(question)
        result = Retrieval(question=question)
        if not asked:
            return result

        seen: set = set()
        for name, route in (("lexical", self._lexical),
                            ("reach", self._reach)):
            found = route(question, asked, seen)
            result.facts.extend(found)
            result.routes[name] = len(found)
            result.stopped_after = name
            result.examined = len(seen)
            if any(self._covers(item.key, asked) for item in result.facts):
                result.covered = True
                return result

        found = self._semantic(question, seen)
        result.facts.extend(found)
        result.routes["semantic"] = len(found)
        result.stopped_after = "semantic"
        result.examined = len(seen)
        result.covered = any(self._covers(item.key, asked)
                             for item in result.facts)
        return result
