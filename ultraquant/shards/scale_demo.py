"""Demonstration: serving a store far larger than RAM, one chunk at a time.

Builds a synthetic shard library whose total size dwarfs the configured budget,
then runs a category-skewed access pattern through the cache.  The point is the
invariant printed at the end: resident bytes never exceed the budget, no matter
how large the store grows, because the catalog decides *what* is needed and the
budget decides *what stays*.  That is the mechanism by which a 300-billion
parameter store is usable on a machine that could never hold it whole.
"""

from __future__ import annotations

import argparse
import random

from ultraquant.shards.budget import ShardCache
from ultraquant.shards.vault import ShardVault

CATEGORIES = [
    "geometry", "arithmetic", "language", "world",
    "music", "chemistry", "code", "biology",
]


def build_store(vault: ShardVault, n_shards: int, seed: int) -> int:
    """Create ``n_shards`` synthetic shards spread over the categories.

    Returns:
        Total stored bytes.
    """
    rng = random.Random(seed)
    for i in range(n_shards):
        category = CATEGORIES[i % len(CATEGORIES)]
        payload = {
            "slice": i,
            "category": category,
            # Weight-like payload: incompressible enough to have a real size.
            "weights": [round(rng.uniform(-1, 1), 6) for _ in range(600)],
        }
        vault.add_shard(f"slice:{category}:{i:04d}", category, payload, kind="weight-slice")
    return vault.stats()["total_bytes"]


def main(argv: list[str] | None = None) -> int:
    """Run the demonstration.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="On-demand shard paging at scale")
    parser.add_argument("--shards", type=int, default=512)
    parser.add_argument("--budget-kb", type=int, default=64)
    parser.add_argument("--accesses", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", default=None, help="vault directory (default: temp)")
    args = parser.parse_args(argv)

    import tempfile
    root = args.root or tempfile.mkdtemp(prefix="uq_scale_")
    vault = ShardVault(root)

    print(f"Building {args.shards} shards under {root} ...")
    total = build_store(vault, args.shards, args.seed)
    budget = args.budget_kb * 1024

    library = f"{root}/uq_scale.uql"
    packed = vault.pack(library, prune_loose=True)
    print(f"Packed {packed} shards into one library container.")

    stats = vault.stats()
    print()
    print(f"  store size      : {total:,} bytes across {stats['shards']} shards")
    print(f"  RAM budget      : {budget:,} bytes "
          f"({100.0 * budget / max(total, 1):.1f}% of the store)")
    print(f"  categories      : {', '.join(sorted({e['category'] for e in vault.catalog()}))}")

    cache = ShardCache(max_bytes=budget)
    catalog = vault.catalog()
    by_category: dict[str, list[str]] = {}
    for entry in catalog:
        by_category.setdefault(entry["category"], []).append(entry["shard_id"])

    # Skewed access: reasoning stays inside a category for a while, then moves on.
    rng = random.Random(args.seed + 1)
    category = rng.choice(CATEGORIES)
    for step in range(args.accesses):
        if step % 12 == 0:
            category = rng.choice(CATEGORIES)
        shard_id = rng.choice(by_category[category])

        def loader(sid: str = shard_id) -> tuple[dict, int]:
            return vault.get(sid), int(vault.entry(sid)["nbytes"])

        cache.get(shard_id, loader)

    cstats = cache.stats()
    print()
    print(f"  accesses        : {args.accesses}")
    print(f"  cache hits      : {cstats['hits']}  "
          f"({100.0 * cstats['hits'] / max(args.accesses, 1):.1f}%)")
    print(f"  cache misses    : {cstats['misses']} (chunk reads from the library)")
    print(f"  evictions       : {cstats['evictions']} (swapped back out)")
    print(f"  peak resident   : {cstats['peak_bytes']:,} bytes")
    print(f"  resident now    : {len(cstats['resident'])} shards, "
          f"{cstats['current_bytes']:,} bytes")

    assert cstats["peak_bytes"] <= budget, "budget invariant violated"
    print()
    print(f"  Budget invariant held: peak {cstats['peak_bytes']:,} <= budget {budget:,}.")
    print(f"  Only {100.0 * cstats['peak_bytes'] / max(total, 1):.1f}% of the store was ever "
          f"resident at once.")
    print("  This is how a 300B-parameter store is served: the catalog picks the")
    print("  shard, a byte-range read pulls just that chunk, and the budget decides")
    print("  what stays. Store size is bounded by disk, not RAM.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
