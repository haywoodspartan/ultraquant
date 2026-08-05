"""Prototype consolidation: episodic traces become semantic prototypes.

Every lesson appends what it saw to the category's signature — the right
instinct (each demonstration is real evidence) with an unbounded consequence.
Measured after 200 lessons into one category: **202 prototypes**, 25.6 KB of
catalog-resident signature for a single category, 377 us per routing call —
and, far worse, **8-glyph routing degraded to 6/8**, because a cloud of noisy
near-duplicates reaches into neighbouring categories' territory. Unbounded
episodic memory does not merely cost space; it makes recognition *worse*.

Consolidation is the semantic step: when a category holds more traces than its
budget, they are clustered and each cluster keeps its **medoid** — the real
stored prototype nearest the cluster's centre. Two prior measurements shape
that choice:

* The cluster count is chosen by silhouette, never supplied — Stage 3's
  machinery (`reason.discovery`) doing maintenance duty.
* Medoids, not means: mean-pooling prototypes was measured harmful in §4.1.1
  (7/8 vs 8/8 — the average of ``plus`` and ``cross`` sat nearer ``arrows``
  than either member). Every surviving prototype is something actually seen.

This runs inside :meth:`SelfLearner.consolidate` — the same housekeeping pass
that packs hot shards and snapshots the archive, which is where a brain-like
system does its consolidation — and automatically from teaching when a
category's traces exceed twice the budget, so a long-lived session cannot
degrade before anyone thinks to run maintenance.

Pure Python standard library.
"""

from __future__ import annotations

from typing import Any

from ultraquant.reason.discovery import discover, distance

__all__ = ["PROTOTYPE_BUDGET", "consolidate_signatures"]

#: Prototypes a category may hold before consolidation reduces it. Enough to
#: cover a category's genuine variety (the built-in families need 2-6), small
#: enough that routing stays microseconds and the catalog stays lean.
PROTOTYPE_BUDGET = 12


def _medoid(vectors: list[list[float]], members: list[int]) -> int:
    """The member closest to the cluster's centre — a real trace, not a mean."""
    centre = [sum(vectors[i][d] for i in members) / len(members)
              for d in range(len(vectors[members[0]]))]
    return min(members, key=lambda i: distance(vectors[i], centre))


def consolidate_signatures(
    vault: Any, budget: int = PROTOTYPE_BUDGET, seed: int = 0,
) -> dict:
    """Reduce every oversized signature to representative medoids.

    Args:
        vault: The shard vault whose catalog holds the signatures.
        budget: Maximum prototypes per category afterwards.
        seed: Clustering seed, for reproducibility.

    Returns:
        ``{"categories": n consolidated, "before": total prototypes in them,
        "after": total kept}`` — zeros when nothing was oversized.
    """
    consolidated = 0
    before_total = 0
    after_total = 0
    for entry in list(vault.catalog()):
        signature = entry.get("signature") or []
        if len(signature) <= budget:
            continue
        vectors = [list(vector) for vector in signature]

        result = discover(vectors, k_range=(2, budget), seed=seed)
        groups: dict[int, list[int]] = {}
        for index, cluster in enumerate(result["assignment"]):
            groups.setdefault(cluster, []).append(index)

        keep: list[int] = [_medoid(vectors, members)
                           for members in groups.values()]

        # Medoids cover the dense middles; boundary traces are what stop a
        # neighbouring category's pattern being claimed, so the budget's
        # remainder goes to farthest-point coverage - each addition is the
        # trace worst-served by everything kept so far.
        chosen = set(keep)
        while len(keep) < min(budget, len(vectors)):
            farthest = max(
                (i for i in range(len(vectors)) if i not in chosen),
                key=lambda i: min(distance(vectors[i], vectors[j])
                                  for j in keep),
            )
            keep.append(farthest)
            chosen.add(farthest)

        keep.sort()
        vault.set_signature(entry["shard_id"], [vectors[i] for i in keep])
        consolidated += 1
        before_total += len(vectors)
        after_total += len(keep)
    return {"categories": consolidated, "before": before_total,
            "after": after_total}
