"""Associative category routing: text in, ranked categories out.

:class:`CategoryRouter` is the recall step that decides *which* expert shards a
piece of text is about.  It scores every registered category by three additive
signals:

* **base-keyword overlap** — the hand-registered keyword set for the category;
* **learned keyword weights** — token -> strength weights grown by :meth:`learn`;
* **vault associations** — the summed association strengths of the category's
  own shards for the tokens that appear (the brain-like traces reinforced on
  use, read straight from :class:`~ultraquant.shards.vault.ShardVault`).

Ties break alphabetically so routing is fully deterministic.  Learning both
grows the internal token weights and reinforces the category's vault shards, so
association strength accumulates in the durable store as well as in the router.

Pure Python standard library only; persisted state is JSON-safe.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .vault import ShardVault

__all__ = ["CategoryRouter"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: How many prototypes survive the sketch screen before exact scoring.
#: Measured against a full exact scan over 60,000 prototypes: 32 and 64 agreed
#: on 194/200 and 196/200 queries, 128 agreed on 200/200, and going beyond it
#: only cost time. The test case is deliberately hostile -- 25-dimensional
#: all-positive random vectors sit at ~0.75 mean pairwise cosine, so almost
#: everything looks like everything -- and real signatures separate far more.
_SCREEN_WIDTH = 128


def _tokenize(text: str) -> list[str]:
    """Split ``text`` into lowercase alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


class CategoryRouter:
    """Rank categories for a piece of text using associative recall.

    Args:
        vault: The shard vault whose per-category associations feed routing.
        memory: Optional :class:`SystematicMemory`, retained for callers that
            want to consult episodic/semantic recall alongside routing.
        path: Optional ``router.json`` persistence path; if it already exists on
            construction the router state is auto-loaded from it.
    """

    def __init__(
        self,
        vault: ShardVault,
        memory: Any | None = None,
        path: str | os.PathLike | None = None,
    ) -> None:
        self.vault: ShardVault = vault
        self.memory: Any | None = memory
        self.path: Path | None = Path(path) if path is not None else None
        # category -> set of base keywords
        self._base: dict[str, set[str]] = {}
        # category -> {token: learned weight}
        self._learned: dict[str, dict[str, float]] = {}
        if self.path is not None and self.path.exists():
            self.load()

    # ------------------------------------------------------------------ #
    # registration & learning
    # ------------------------------------------------------------------ #

    def register(self, category: str, keywords: list[str]) -> None:
        """Register ``category`` with its base keyword set (lowercased).

        Calling again for the same category unions in the extra keywords.

        Args:
            category: The category name.
            keywords: Base keywords that name this category.
        """
        base = self._base.setdefault(category, set())
        for keyword in keywords:
            base.add(keyword.lower())
        self._learned.setdefault(category, {})

    def learn(self, text: str, category: str, delta: float = 0.1) -> None:
        """Strengthen the association from ``text``'s tokens to ``category``.

        Each distinct token's learned weight for the category is raised by
        ``delta``, and every shard belonging to the category has those tokens
        reinforced in the vault (so the durable store learns too).  An unknown
        category is registered on the fly so its learned weights take effect.

        Args:
            text: The text whose tokens should point at the category.
            category: The category to strengthen towards.
            delta: Strength increment per distinct token.
        """
        tokens = _tokenize(text)
        if not tokens:
            return
        distinct = list(dict.fromkeys(tokens))
        self._base.setdefault(category, set())
        learned = self._learned.setdefault(category, {})
        for token in distinct:
            learned[token] = learned.get(token, 0.0) + delta
        # One catalog write for the whole category, not one per shard.
        with self.vault.batch():
            for shard_id in self.vault.shards_in(category):
                self.vault.reinforce(shard_id, distinct, delta)

    # ------------------------------------------------------------------ #
    # routing
    # ------------------------------------------------------------------ #

    def _vault_association_scores(self, token_set: set[str]) -> dict[str, float]:
        """Sum matching vault-shard association weights per category.

        Args:
            token_set: The distinct tokens of the query.

        Returns:
            Mapping of category -> summed association strength over that
            category's shards for keywords present in ``token_set``.
        """
        # The vault keeps an inverted keyword index, so this costs one lookup
        # per query token instead of a pass over every shard in the library.
        return self.vault.association_scores(token_set)

    def route_pattern(
        self, pixels: list[float], top_k: int = 3,
        min_similarity: float = 0.0, screen: int = _SCREEN_WIDTH,
    ) -> list[tuple[str, float]]:
        """Rank categories by how much a *pattern* looks like each of them.

        Text routing is useless for an image: a glyph contains no words, so
        keyword overlap finds nothing and every pattern falls through to the same
        default category — which means the library's pattern experts are never
        reached. This routes on content instead, comparing the input against each
        category's stored signature by cosine similarity.

        The signatures live in the catalog, so this costs no shard reads at all.

        Large libraries are screened by sketch first (see
        :mod:`ultraquant.shards.sketch`) so this does not become the O(library)
        scan that the keyword path was rewritten to avoid. Similarities returned
        are always the true cosine — the sketch only decides what gets scored.

        Args:
            pixels: The flattened pattern.
            top_k: How many categories to return.
            min_similarity: Drop categories below this similarity, so a pattern
                unlike anything stored routes nowhere instead of to whichever
                category happens to sit nearest.
            screen: How many prototypes survive the sketch screen. Raising it
                trades speed for recall; below the vault's screening threshold
                it has no effect because everything is scored.

        Returns:
            ``(category, similarity)`` pairs, best first; ``[]`` when no
            category has a signature yet.
        """
        if not pixels:
            return []
        norm = sum(v * v for v in pixels) ** 0.5
        if norm == 0.0:
            return []

        # Screening first keeps this off an O(library) path. Small libraries are
        # returned whole, so the common case is exact and unchanged.
        candidates = self.vault.screen_signatures(pixels, limit=screen)
        if not candidates:
            return []

        scored: list[tuple[str, float]] = []
        for category, vectors in candidates:
            # A category is as close as its nearest prototype, not as close as
            # the average of them.
            best = 0.0
            for signature in vectors:
                # The screen only returns matching widths, so this is a straight
                # comparison. Truncating to the shorter vector, as this used to
                # do, silently made a text pattern comparable to a glyph one.
                other = sum(v * v for v in signature) ** 0.5
                if other == 0.0:
                    continue
                dot = sum(a * b for a, b in zip(pixels, signature))
                best = max(best, dot / (norm * other))
            if best > 0.0:
                scored.append((category, best))

        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        floor = max(0.0, float(min_similarity))
        return [pair for pair in scored[:top_k] if pair[1] > floor]

    def route(self, text: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Rank registered categories for ``text``.

        The score for each registered category is the sum of base-keyword
        overlap count, learned token weights, and vault-association strength.
        Only categories with a positive score are returned, ordered by
        descending score with alphabetical tie-breaking.

        Args:
            text: The input text to route.
            top_k: Maximum number of ranked categories to return.

        Returns:
            Up to ``top_k`` ``(category, score)`` pairs; ``[]`` when the text is
            empty, nothing is registered, or nothing matches.
        """
        tokens = _tokenize(text)
        if not tokens or not self._base:
            return []
        token_set = set(tokens)
        assoc_scores = self._vault_association_scores(token_set)

        ranked: dict[str, float] = {}
        for category, base_keywords in self._base.items():
            base_overlap = float(len(token_set & base_keywords))
            learned = self._learned.get(category, {})
            learned_score = sum(learned.get(token, 0.0) for token in token_set)
            total = base_overlap + learned_score + assoc_scores.get(category, 0.0)
            if total > 0.0:
                ranked[category] = total

        if not ranked:
            return []
        ordered = sorted(ranked.items(), key=lambda item: (-item[1], item[0]))
        return ordered[:top_k]

    # ------------------------------------------------------------------ #
    # persistence & snapshots
    # ------------------------------------------------------------------ #

    def state(self) -> dict:
        """Return a JSON-safe snapshot of the router's learned state."""
        return {
            "base": {cat: sorted(kws) for cat, kws in self._base.items()},
            "learned": {
                cat: dict(weights) for cat, weights in self._learned.items()
            },
        }

    def save(self, path: str | os.PathLike | None = None) -> None:
        """Persist :meth:`state` to ``router.json``.

        Args:
            path: Optional override for the persistence path.

        Raises:
            ValueError: If no path was supplied here or at construction time.
        """
        if path is not None:
            self.path = Path(path)
        if self.path is None:
            raise ValueError("CategoryRouter.save() requires a path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.state(), sort_keys=True, separators=(",", ":"))
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(text)

    def load(self, path: str | os.PathLike | None = None) -> None:
        """Restore router state from ``router.json``.

        Args:
            path: Optional override for the persistence path.

        Raises:
            ValueError: If no path was supplied here or at construction time.
        """
        if path is not None:
            self.path = Path(path)
        if self.path is None:
            raise ValueError("CategoryRouter.load() requires a path")
        with open(self.path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._base = {
            cat: set(kws) for cat, kws in data.get("base", {}).items()
        }
        self._learned = {
            cat: {token: float(weight) for token, weight in weights.items()}
            for cat, weights in data.get("learned", {}).items()
        }
