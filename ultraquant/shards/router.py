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
    """Split ``text`` into lowercase alphanumeric tokens, plurals folded."""
    return [normalize_token(t) for t in _TOKEN_RE.findall(text.lower())]


#: Endings that look plural but are not, so stripping them would corrupt the
#: word. ``-ss`` (class, cross), ``-us`` (status, focus), ``-is`` (analysis),
#: ``-os`` (chaos). Without these, "class" becomes "clas" and stops matching
#: itself.
_NOT_PLURAL = ("ss", "us", "is", "os")


def normalize_token(token: str) -> str:
    """Fold a regular English plural onto its singular.

    Registered vocabulary and query text are written by different people at
    different times, and they disagree about number. Measured on the deployed
    library: ``arithmetic`` registers ``number`` while "add these two numbers"
    says ``numbers``, so the category scored **zero** on a query that is
    obviously its own, and the query went to ``crosses`` instead.

    Deliberately conservative — a *plural folder*, not a stemmer. Porter-style
    stemming also strips ``-ing``, ``-ed`` and ``-ation``, which merges words
    that mean different things and, in a router that had just been taught not
    to over-claim, would widen matching in exactly the direction that
    regressed. The rules here are:

    * ``-ies`` -> ``-y``      (bodies -> body)
    * ``-ches``/``-shes``/``-xes``/``-zes``/``-sses`` -> drop ``-es``
    * ``-s`` -> drop, unless the word ends ``-ss``, ``-us``, ``-is`` or ``-os``

    and nothing is folded below three characters, so ``is``, ``as`` and ``us``
    survive untouched.

    Applied consistently at registration, learning and routing, so the fold is
    invisible to callers: whichever spelling goes in, both match.
    """
    if len(token) < 4:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    for ending in ("ches", "shes", "xes", "zes", "sses"):
        if token.endswith(ending) and len(token) - 2 >= 3:
            return token[:-2]
    if token.endswith("s") and not token.endswith(_NOT_PLURAL):
        stripped = token[:-1]
        if len(stripped) >= 3:
            return stripped
    return token


def _informative(token: str) -> bool:
    """Whether a token says anything about which category text belongs to.

    Stopwords do not: they appear in every query, so weight on them makes a
    category match everything. See :meth:`CategoryRouter.learn` for the
    measurement that forced this.
    """
    from ultraquant.interpreter.learning import _STOPWORDS

    return len(token) > 2 and token not in _STOPWORDS


def prune_uninformative(learned: dict) -> tuple[dict, float]:
    """Strip stopword weight from an already-trained router's tables.

    Filtering :meth:`CategoryRouter.learn` fixes routers trained from now on
    and does nothing for one that has already accumulated the weight — which
    every deployed library has. Pruning on load repairs those in place.

    Returns:
        ``(pruned, removed_weight)``.
    """
    pruned: dict = {}
    removed = 0.0
    for category, weights in learned.items():
        kept = {}
        for token, weight in weights.items():
            if _informative(token):
                kept[token] = weight
            else:
                removed += float(weight)
        pruned[category] = kept
    return pruned, removed


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
        #: Learned weight discarded by :func:`prune_uninformative` on load, so
        #: a caller can report that a library was repaired rather than guess.
        self.pruned_weight: float = 0.0
        #: category -> base keywords refused as uninformative. Recorded rather
        #: than dropped silently: a deployer who registered 'what' deserves to
        #: see that it was not accepted.
        self.rejected_keywords: dict[str, set] = {}
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
            lowered = normalize_token(keyword.lower())
            if not _informative(lowered):
                # Registering a question word is how a category comes to claim
                # every question. Measured: 'world' shipped with 'what', 'who'
                # and 'where' as base keywords, and after both learned-weight
                # and vault-association stopwords were cleaned, those three
                # were still enough to claim "what time does the train leave"
                # and "who won the football match" outright.
                self.rejected_keywords.setdefault(category, set()).add(lowered)
                continue
            base.add(lowered)
        self._learned.setdefault(category, {})

    def learn(self, text: str, category: str, delta: float = 0.1,
              filter_tokens: bool = True) -> None:
        """Strengthen the association from ``text``'s content tokens to ``category``.

        Each distinct token's learned weight for the category is raised by
        ``delta``, and every shard belonging to the category has those tokens
        reinforced in the vault (so the durable store learns too).  An unknown
        category is registered on the fly so its learned weights take effect.

        **Stopwords are not learned, and they were.** Learning every token meant
        ``the``, ``a``, ``is`` and ``what`` accumulated weight for whichever
        category an input happened to route to. Since :meth:`route` admits any
        category scoring above zero, one such token was enough to claim a query
        outright. Measured on the deployed library: **21% of all learned weight
        sat on stopwords** — ``is`` at 1.05, ``a`` at 0.60, ``the`` at 0.50 —
        and of five queries belonging to no category, four were claimed anyway,
        none of them matching a single base keyword.

        The same filter already guarded
        :func:`~ultraquant.forge.distill.distil_into_router`, on the reasoning
        that bulk-generated phrasings were the volume case and one user typing
        one query at a time was not. That reasoning was wrong, and the
        measurement is what says so: ten ordinary learning turns are enough.

        Args:
            text: The text whose tokens should point at the category.
            category: The category to strengthen towards.
            delta: Strength increment per distinct token.
            filter_tokens: Drop stopwords and very short tokens. On by default;
                off only so the abstention gate can measure the old behaviour
                against the new one.
        """
        tokens = _tokenize(text)
        if filter_tokens:
            tokens = [t for t in tokens if _informative(t)]
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

    def correct(self, text: str, category: str, delta: float = 0.4) -> dict:
        """Record that ``text`` belongs to ``category``, against a wrong winner.

        The only ground truth for a routing error is a person saying so, and
        until now there was no way to say it. ``:learn`` cannot: it reinforces
        from the *reply* text rather than the misrouted query, and none of its
        eight question kinds surfaces a misroute at all. Measured on the
        deployed library — answering a fact about arithmetic moved it 1.2 -> 1.4
        while ``crosses`` stayed at 2.15, so the 0.95 gap would close 0.2 at a
        time and only when the answer's words happened to overlap the query.

        So this does two things, and the second is the one that matters against
        an entrenched error (§11.16): it strengthens the query's own content
        tokens for the right category, **and weakens whichever category is
        currently winning them**. Adding weight alone has to out-climb
        accumulated weight; removing the accumulation is what actually reverses
        the decision.

        Weakening is bounded to what was learned. Base keywords are never
        touched, because those were registered deliberately and a single
        correction is not grounds for unregistering vocabulary — and learned
        weight never goes below zero, so a correction cannot invent negative
        evidence.

        Args:
            text: The query that routed wrongly.
            category: Where it should have gone.
            delta: Strength of the correction. Larger than ordinary learning,
                because it carries a person's explicit judgement rather than a
                turn's incidental success.

        Returns:
            ``{"tokens": [...], "strengthened": category, "weakened": {cat: amount}}``
            so a caller can report exactly what changed.
        """
        tokens = [tok for tok in dict.fromkeys(_tokenize(text))
                  if _informative(tok)]
        if not tokens:
            return {"tokens": [], "strengthened": category, "weakened": {}}

        ranked = self.route(text, top_k=3)
        wrong = [name for name, _score in ranked if name != category]

        self.learn(" ".join(tokens), category, delta=delta)

        weakened: dict[str, float] = {}
        for other in wrong:
            learned = self._learned.get(other)
            if not learned:
                continue
            removed = 0.0
            for token in tokens:
                had = learned.get(token, 0.0)
                if had > 0.0:
                    # Down to zero, never below: a correction may remove
                    # accumulated evidence but must not manufacture the
                    # opposite claim.
                    learned[token] = max(0.0, had - delta)
                    removed += had - learned[token]
            if removed:
                weakened[other] = round(removed, 4)
        return {"tokens": tokens, "strengthened": category,
                "weakened": weakened}

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
            cat: {normalize_token(kw) for kw in kws if _informative(kw)}
            for cat, kws in data.get("base", {}).items()
        }
        loaded = {
            cat: {token: float(weight) for token, weight in weights.items()}
            for cat, weights in data.get("learned", {}).items()
        }
        # Repair on load. Filtering learn() fixes routers trained from here on
        # and does nothing for one that already accumulated stopword weight -
        # which every deployed library has, at 21% of its total on the one
        # measured. Without this the fix would only reach new installations.
        self._learned, self.pruned_weight = prune_uninformative(loaded)
        # The vault carries the same stopwords, because learn() reinforced it
        # with the same tokens. Cleaning only this table left decoys routing on
        # vault association alone.
        try:
            self.pruned_weight += self.vault.prune_associations()
        except Exception:  # noqa: BLE001 - a repair must never block loading
            pass
