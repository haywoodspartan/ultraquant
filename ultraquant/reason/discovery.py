"""Concepts the system proposes for itself.

Everything the model has learned so far, it learned because a human named a
category and demonstrated it. Learning mode already decides *which* gap is worth
asking about, but the vocabulary of possible answers is still supplied from
outside. This module is the other half: it looks at what the model has actually
seen, finds groups in it, and offers those groups as candidate categories — so
the question becomes "what is this cluster?" rather than "here is a category,
learn it".

The material is what recognition already produces. Patterns the router could not
place are recorded as ``unknown-pattern`` episodes
([§4.1.1](#411-routing-a-pattern-that-contains-no-words)); those are exactly the
observations a new concept would have to be made of.

**k is never told to the algorithm.** The number of groups is chosen by
silhouette score, which uses only the geometry of the observations and never a
label. Passing in the true number of classes would leak the answer into the
result and make the measurement worthless, so :func:`discover` searches a range
and picks for itself.

Whether the groups it finds correspond to anything real is an empirical question
with a gate attached, scored by adjusted Rand index — chance-corrected, so a
random clustering scores 0 rather than something that looks encouraging. See
``ultraquant/experiments/discovery_gate.py``.

Pure Python standard library.
"""

from __future__ import annotations

import math
import random
from typing import Any

__all__ = [
    "SILHOUETTE_FLOOR",
    "distance",
    "kmeans",
    "silhouette",
    "discover",
    "adjusted_rand_index",
    "ConceptDiscovery",
]


#: Below this silhouette, a grouping is not offered as a concept at all.
#:
#: k-means always returns clusters, so a system that proposes its own categories
#: will confidently invent them out of noise unless something stops it. Measured
#: over 20 datasets each: genuine structure never scored below **0.314**, and
#: data with no structure at all never scored above **0.104**. The floor sits in
#: that gap, and it tracks usefulness rather than merely separating the two
#: extremes — at the noise levels where it starts refusing, the clustering it
#: would have proposed had already fallen to ARI 0.46 and below.
SILHOUETTE_FLOOR = 0.12


def distance(a: list[float], b: list[float]) -> float:
    """Euclidean distance."""
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def _kmeans_plus_plus(
    vectors: list[list[float]], k: int, rng: random.Random
) -> list[list[float]]:
    """Seed centres spread out rather than clumped.

    Plain random seeding regularly puts two centres inside one true group and
    none in another, which no amount of iterating recovers. Choosing each centre
    with probability proportional to its squared distance from the nearest
    existing centre costs one pass per centre and removes most of that failure.
    """
    centres = [list(vectors[rng.randrange(len(vectors))])]
    while len(centres) < k:
        weights = []
        for vector in vectors:
            nearest = min(distance(vector, centre) for centre in centres)
            weights.append(nearest * nearest)
        total = sum(weights)
        if total <= 0.0:
            centres.append(list(vectors[rng.randrange(len(vectors))]))
            continue
        threshold = rng.random() * total
        running = 0.0
        for vector, weight in zip(vectors, weights):
            running += weight
            if running >= threshold:
                centres.append(list(vector))
                break
    return centres


def kmeans(
    vectors: list[list[float]], k: int, seed: int = 0, iterations: int = 40,
    restarts: int = 4,
) -> tuple[list[int], float]:
    """Cluster ``vectors`` into ``k`` groups.

    Args:
        vectors: Observations.
        k: Number of clusters.
        seed: Determines initialisation; results are reproducible.
        iterations: Maximum Lloyd iterations per restart.
        restarts: Independent initialisations; the tightest is kept.

    Returns:
        ``(assignment, inertia)`` — cluster index per observation, and the
        summed squared distance to assigned centres.
    """
    if k <= 1 or len(vectors) <= k:
        return [0] * len(vectors), 0.0

    best_assignment: list[int] = [0] * len(vectors)
    best_inertia = math.inf
    for restart in range(max(1, restarts)):
        rng = random.Random(f"{seed}|{k}|{restart}")
        centres = _kmeans_plus_plus(vectors, k, rng)
        assignment = [0] * len(vectors)
        for _ in range(iterations):
            moved = False
            for index, vector in enumerate(vectors):
                nearest = min(
                    range(k), key=lambda c: distance(vector, centres[c])
                )
                if nearest != assignment[index]:
                    assignment[index] = nearest
                    moved = True
            for cluster in range(k):
                members = [v for v, a in zip(vectors, assignment) if a == cluster]
                if not members:
                    # An emptied cluster is re-seeded on the worst-fit point
                    # rather than dropped, so k stays what was asked for.
                    worst = max(
                        range(len(vectors)),
                        key=lambda i: distance(vectors[i], centres[assignment[i]]),
                    )
                    centres[cluster] = list(vectors[worst])
                    continue
                centres[cluster] = [
                    sum(values) / len(members) for values in zip(*members)
                ]
            if not moved:
                break
        inertia = sum(
            distance(vector, centres[cluster]) ** 2
            for vector, cluster in zip(vectors, assignment)
        )
        if inertia < best_inertia:
            best_inertia, best_assignment = inertia, list(assignment)
    return best_assignment, best_inertia


def silhouette(vectors: list[list[float]], assignment: list[int]) -> float:
    """How well-separated a clustering is, using no labels at all.

    For each point: ``a`` is its mean distance to its own cluster, ``b`` the
    smallest mean distance to any other cluster, and the score is
    ``(b - a) / max(a, b)``. Averaged, it runs from -1 to 1.

    This is what chooses ``k``. It is a purely internal criterion — it can be
    computed on data whose labels are unknown, which is the entire situation
    this module exists for.
    """
    clusters = sorted(set(assignment))
    if len(clusters) < 2:
        return -1.0
    members = {c: [i for i, a in enumerate(assignment) if a == c] for c in clusters}
    scores = []
    for index, vector in enumerate(vectors):
        own = members[assignment[index]]
        if len(own) <= 1:
            scores.append(0.0)  # a singleton has no within-cluster distance
            continue
        a = sum(distance(vector, vectors[j]) for j in own if j != index) / (
            len(own) - 1
        )
        b = math.inf
        for cluster, group in members.items():
            if cluster == assignment[index] or not group:
                continue
            mean = sum(distance(vector, vectors[j]) for j in group) / len(group)
            b = min(b, mean)
        if b is math.inf:
            scores.append(0.0)
            continue
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return sum(scores) / len(scores) if scores else -1.0


def discover(
    vectors: list[list[float]], k_range: tuple[int, int] = (2, 8), seed: int = 0,
) -> dict:
    """Find groups, choosing how many for itself.

    Args:
        vectors: Unlabelled observations.
        k_range: Inclusive range of cluster counts to consider.
        seed: Reproducibility.

    Returns:
        ``{"k", "assignment", "silhouette", "scores"}`` where ``scores`` maps
        each candidate ``k`` to its silhouette.
    """
    if len(vectors) < 4:
        return {"k": 1, "assignment": [0] * len(vectors),
                "silhouette": -1.0, "scores": {}}
    low, high = k_range
    high = min(high, len(vectors) - 1)
    scores: dict[int, float] = {}
    best = {"k": 1, "assignment": [0] * len(vectors), "silhouette": -1.0}
    for k in range(max(2, low), max(2, high) + 1):
        assignment, _inertia = kmeans(vectors, k, seed=seed)
        score = silhouette(vectors, assignment)
        scores[k] = score
        if score > best["silhouette"]:
            best = {"k": k, "assignment": assignment, "silhouette": score}
    best["scores"] = scores
    return best


def adjusted_rand_index(truth: list[Any], found: list[Any]) -> float:
    """Agreement between two groupings, invariant to how clusters are named.

    Cluster 0 in one grouping and cluster 3 in another may be the same set, so
    accuracy is meaningless here. The Rand index counts agreeing *pairs*, and
    the adjustment subtracts the agreement expected by chance — which is what
    makes "above chance" a statement about zero rather than about a baseline
    that has to be argued for.

    Returns:
        1.0 for identical groupings, ~0.0 for unrelated ones, negative for
        agreement worse than chance.
    """
    if len(truth) != len(found):
        raise ValueError("groupings must describe the same observations")
    n = len(truth)
    if n < 2:
        return 1.0

    table: dict[tuple[Any, Any], int] = {}
    rows: dict[Any, int] = {}
    columns: dict[Any, int] = {}
    for a, b in zip(truth, found):
        table[(a, b)] = table.get((a, b), 0) + 1
        rows[a] = rows.get(a, 0) + 1
        columns[b] = columns.get(b, 0) + 1

    def choose2(value: int) -> float:
        return value * (value - 1) / 2.0

    index = sum(choose2(count) for count in table.values())
    row_sum = sum(choose2(count) for count in rows.values())
    column_sum = sum(choose2(count) for count in columns.values())
    expected = row_sum * column_sum / choose2(n)
    maximum = 0.5 * (row_sum + column_sum)
    if maximum == expected:
        return 0.0
    return (index - expected) / (maximum - expected)


class ConceptDiscovery:
    """Proposes categories from what a session has actually seen.

    Args:
        session: The interpreter session to read observations from.
        min_cluster: Smallest group worth proposing as a concept. One or two
            stray patterns are noise, not a category.
        min_silhouette: Below this the grouping is not offered at all. See
            :data:`SILHOUETTE_FLOOR`.
    """

    def __init__(self, session: Any, min_cluster: int = 4,
                 min_silhouette: float = SILHOUETTE_FLOOR) -> None:
        self.session = session
        self.min_cluster = int(min_cluster)
        self.min_silhouette = float(min_silhouette)

    def observations(self, limit: int = 200) -> list[tuple[list[float], list[str]]]:
        """Unplaced patterns the model has met, as ``(features, rows)``.

        These are the ones recognition already flagged as belonging to no known
        category, which makes them the only honest raw material for a *new*
        one.
        """
        from ultraquant.pattern.recognition import render, row_means

        out = []
        seen: set[str] = set()
        for episode in self.session.memory.recall_episodes(
            kind="unknown-pattern", limit=limit
        ):
            rows = (episode.get("content") or {}).get("rows") or []
            if len(rows) != 5:
                continue
            key = "".join(rows)
            if key in seen:
                continue
            seen.add(key)
            pixels = render(rows)
            out.append((pixels + row_means(pixels), list(rows)))
        return out

    def propose(self, seed: int = 0) -> list[dict]:
        """Group the unplaced patterns and offer each group as a concept.

        Returns:
            One dict per proposed concept: ``{"members", "rows", "size",
            "silhouette", "k"}``, largest group first.
        """
        observations = self.observations()
        if len(observations) < self.min_cluster * 2:
            return []
        vectors = [features for features, _rows in observations]
        result = discover(vectors, seed=seed)
        if result["silhouette"] < self.min_silhouette:
            # Saying nothing is the right answer here. Proposing a category the
            # model would then be taught is expensive to undo, and clustering
            # noise produces exactly that with no outward sign anything is
            # wrong.
            return []
        groups: dict[int, list[int]] = {}
        for index, cluster in enumerate(result["assignment"]):
            groups.setdefault(cluster, []).append(index)

        proposals = []
        for cluster, members in groups.items():
            if len(members) < self.min_cluster:
                continue
            proposals.append({
                "cluster": cluster,
                "size": len(members),
                "members": members,
                "rows": [observations[i][1] for i in members],
                "silhouette": round(result["silhouette"], 4),
                "k": result["k"],
            })
        proposals.sort(key=lambda p: (-p["size"], p["cluster"]))
        return proposals
