"""Training corpora for the forge, built from pattern recognition.

A model is grown from *patterns*: each category gets a set of 5x5 glyph
prototypes, and each prototype is augmented into many noisy variants so the
expert learns the shape rather than the exact pixels.  Two taxonomies ship:

* :func:`builtin_taxonomy` — the eight reference glyphs grouped into four
  semantic families (lines, arrows, frames, crosses).
* :func:`synthetic_taxonomy` — procedurally generated prototypes, well separated
  in Hamming distance, for building a library of arbitrary size from nothing.

Features are the 30 numbers the rest of the system uses: 25 pixels plus the five
row means (SPEC.md §10).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ultraquant.pattern.recognition import (PATTERNS, center_glyph,
                                            render, row_means,
                                            scale_normalize)

__all__ = [
    "CategoryCorpus",
    "builtin_taxonomy",
    "synthetic_taxonomy",
    "build_corpora",
    "features_of",
]

#: The eight reference glyphs, grouped into families of two.
BUILTIN_FAMILIES: dict[str, list[str]] = {
    "lines": ["stripes_h", "stripes_v"],
    "arrows": ["arrow_up", "arrow_down"],
    "frames": ["square", "diamond"],
    "crosses": ["plus", "cross"],
}

#: Keywords registered with the router for each built-in family.
BUILTIN_KEYWORDS: dict[str, list[str]] = {
    "lines": ["line", "lines", "stripe", "stripes", "bars", "horizontal", "vertical"],
    "arrows": ["arrow", "arrows", "up", "down", "direction", "pointer"],
    "frames": ["frame", "frames", "square", "diamond", "box", "outline"],
    "crosses": ["cross", "crosses", "plus", "x", "star", "intersection"],
}


@dataclass
class CategoryCorpus:
    """One category's labelled training data."""

    name: str
    labels: list[str]
    xs: list[list[float]] = field(default_factory=list)
    ys: list[int] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    prototypes: dict[str, list[str]] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of training samples."""
        return len(self.xs)


def features_of(rows: list[str]) -> list[float]:
    """Feature vector for a glyph: 25 pixels followed by the five row means."""
    pixels = render(rows)
    return pixels + row_means(pixels)


def centered_features(rows: list[str]) -> list[float]:
    """The same 30 features, position normalised away first.

    :func:`features_of` over :func:`~ultraquant.pattern.recognition.center_glyph`:
    same dimensionality, same downstream machinery, but a glyph and its
    translation produce (near-)identical vectors. Kept as a separate function
    rather than changed in place because every deployed expert was trained on
    positional features — swapping the definition under them would silently
    break existing libraries, where offering the centred variant lets new
    training choose it and the gate measure it.
    """
    return features_of(center_glyph(rows))


def scaled_features(rows: list[str]) -> list[float]:
    """The 30 features over a scale- and position-normalised glyph.

    :func:`~ultraquant.pattern.recognition.scale_normalize` first, then the
    unchanged feature extraction — same dimensionality, same downstream
    machinery, but a shrunk plus and the canonical plus produce (near-)
    identical vectors. Separate from :func:`features_of` for the same reason
    :func:`centered_features` is: deployed experts trained on positional
    features must not have the definition changed under them.
    """
    return features_of(scale_normalize(rows))


def builtin_taxonomy() -> dict[str, dict[str, list[str]]]:
    """The reference glyphs grouped into families.

    Returns:
        ``{category: {label: rows}}``.
    """
    return {
        family: {label: list(PATTERNS[label]) for label in labels}
        for family, labels in BUILTIN_FAMILIES.items()
    }


def _random_glyph(rng: random.Random, density: float = 0.4) -> list[str]:
    """A random 5x5 glyph with roughly ``density`` lit pixels."""
    bits = [1 if rng.random() < density else 0 for _ in range(25)]
    if not any(bits):
        bits[rng.randrange(25)] = 1
    return ["".join("#" if bits[r * 5 + c] else "." for c in range(5)) for r in range(5)]


def _hamming(a: list[str], b: list[str]) -> int:
    """Pixel distance between two glyphs."""
    fa = "".join(a)
    fb = "".join(b)
    return sum(1 for x, y in zip(fa, fb) if x != y)


def synthetic_taxonomy(
    n_categories: int,
    n_labels: int = 4,
    seed: int = 0,
    min_distance: int = 8,
    attempts: int = 400,
) -> dict[str, dict[str, list[str]]]:
    """Generate fresh glyph prototypes — a model built from nothing at all.

    Prototypes within a category are rejected unless they differ by at least
    ``min_distance`` pixels, so each expert faces a genuinely separable problem.

    Args:
        n_categories: How many categories to invent.
        n_labels: Prototypes per category.
        seed: RNG seed.
        min_distance: Minimum Hamming separation inside a category.
        attempts: Tries per prototype before accepting the best candidate.

    Returns:
        ``{category: {label: rows}}``.
    """
    rng = random.Random(seed)
    taxonomy: dict[str, dict[str, list[str]]] = {}
    for c in range(n_categories):
        chosen: list[list[str]] = []
        for _ in range(n_labels):
            best: list[str] | None = None
            best_gap = -1
            for _ in range(attempts):
                candidate = _random_glyph(rng, density=rng.uniform(0.3, 0.55))
                gap = min((_hamming(candidate, prior) for prior in chosen), default=25)
                if gap >= min_distance:
                    best = candidate
                    break
                if gap > best_gap:
                    best, best_gap = candidate, gap
            chosen.append(best or _random_glyph(rng))
        name = f"family_{c:03d}"
        taxonomy[name] = {f"sym_{i}": rows for i, rows in enumerate(chosen)}
    return taxonomy


def _default_keywords(category: str, labels: list[str]) -> list[str]:
    """Router keywords derived from a category and its label names.

    Names are split on underscores and a naive singular is added, because the
    router matches whole tokens: without this, ``"arrow_up"`` would only ever be
    found by the literal string ``arrow_up``, and a category called ``arrows``
    would not match the word "arrow".
    """
    words: list[str] = []
    for name in [category, *labels]:
        parts = [p for p in name.replace("-", "_").split("_") if p]
        words.append(name)
        words.extend(parts)
        for part in parts:
            if len(part) > 3 and part.endswith("s"):
                words.append(part[:-1])
    seen: set[str] = set()
    out: list[str] = []
    for word in words:
        word = word.lower()
        if word and word not in seen:
            seen.add(word)
            out.append(word)
    return out


def build_corpora(
    taxonomy: dict[str, dict[str, list[str]]],
    n_per_class: int = 40,
    flips: int = 2,
    seed: int = 0,
    keywords: dict[str, list[str]] | None = None,
) -> list[CategoryCorpus]:
    """Augment each prototype into a labelled training set.

    Args:
        taxonomy: ``{category: {label: rows}}``.
        n_per_class: Noisy variants generated per prototype.
        flips: Pixels flipped per variant.
        seed: RNG seed (one shared stream, so the whole build is reproducible).
        keywords: Optional per-category router keywords.

    Returns:
        One :class:`CategoryCorpus` per category, in taxonomy order.
    """
    rng = random.Random(seed)
    keywords = keywords or {}
    corpora: list[CategoryCorpus] = []

    for category, prototypes in taxonomy.items():
        labels = list(prototypes)
        corpus = CategoryCorpus(
            name=category,
            labels=labels,
            keywords=list(keywords.get(category) or _default_keywords(category, labels)),
            prototypes={k: list(v) for k, v in prototypes.items()},
        )
        for index, label in enumerate(labels):
            base = render(prototypes[label])
            for _ in range(n_per_class):
                variant = list(base)
                for pixel in rng.sample(range(len(variant)), min(flips, len(variant))):
                    variant[pixel] = 1.0 - variant[pixel]
                corpus.xs.append(variant + row_means(variant))
                corpus.ys.append(index)
        corpora.append(corpus)
    return corpora
