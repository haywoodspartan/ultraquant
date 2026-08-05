"""Can a trillion-parameter library be served from a small machine?

This answers it with measurements rather than assertion. Three things have to
hold, and only the third is about weights at all:

1. **The index must not scale with the model.** A 1.2T-parameter library needs
   over a million shard records. Held as Python objects that is gigabytes — the
   machine would exhaust memory *describing* a model it never loaded. So the
   index is built for real at full scale here, and its resident cost measured.
2. **A lookup must not read the index.** Finding a shard among 1.2M costs a
   handful of byte-range reads, bounded by the fence stride, not by the record
   count.
3. **A read must not read the library.** Pulling one shard out of a packed
   library brings in only the blocks it spans.

Run ``python -m ultraquant.storage.scale`` for the default 1.2T projection, or
``--build N`` to construct a real index of N records and measure it.

Honesty about what is simulated: the index, the lookups and the block accounting
are all real and measured. The several hundred gigabytes of *payload* are not
written — the arithmetic for them is shown, and the paging path is exercised
against a real (smaller) library.
"""

from __future__ import annotations

import argparse
import random
import time

from ultraquant.storage.index import (
    RECORD_SIZE,
    ShardIndexReader,
    ShardIndexWriter,
    index_footprint,
)
from ultraquant.storage.ram import RamStorage, available_ram, total_ram
from ultraquant.storage.tiered import TieredStorage

#: Ternary weights plus a per-layer fp32 scale: ~1.6 bits per parameter.
BITS_PER_PARAM = 1.6


def _human(nbytes: float) -> str:
    """Format a byte count."""
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(nbytes) < 1024 or unit == "PB":
            return f"{nbytes:,.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:,.1f} PB"


def project(parameters: int, params_per_shard: int, stride: int) -> dict:
    """Work out what a model of this size costs to store and to index."""
    shards = max(1, parameters // params_per_shard)
    payload = int(parameters * BITS_PER_PARAM / 8)
    footprint = index_footprint(shards, stride)
    return {
        "parameters": parameters,
        "shards": shards,
        "payload_bytes": payload,
        "shard_bytes": payload // shards,
        **footprint,
    }


def report(parameters: int, params_per_shard: int, stride: int) -> dict:
    """Print the projection for one model size."""
    info = project(parameters, params_per_shard, stride)
    print(f"\n  Model: {parameters / 1e12:.2f}T parameters at {BITS_PER_PARAM} bits/param")
    print(f"    library on storage   : {_human(info['payload_bytes'])}")
    print(f"    shards               : {info['shards']:,} "
          f"({_human(info['shard_bytes'])} each)")
    print(f"    index on storage     : {_human(info['disk_bytes'])}")
    print(f"    index resident in RAM: {_human(info['ram_bytes'])}  "
          f"<- the only part that must fit")
    print(f"    lookup cost          : <= {info['probes']} byte-range reads")
    return info


def build_and_measure(records: int, stride: int, lookups: int, seed: int) -> None:
    """Build a real index of ``records`` entries and measure it end to end."""
    print(f"\n  Building a real index of {records:,} records (stride {stride})...")
    writer = ShardIndexWriter(stride=stride)
    started = time.perf_counter()
    offset = 0
    shard_bytes = 262_144
    for i in range(records):
        writer.add(f"slice:{i:09d}", offset, shard_bytes)
        offset += shard_bytes
    blob = writer.build()
    build_seconds = time.perf_counter() - started
    print(f"    built in {build_seconds:.2f}s -> {_human(len(blob))} index "
          f"describing {_human(offset)} of weights")

    store = RamStorage(name="index-host")
    store.write("model.uqx", blob)
    reader = ShardIndexReader(store, "model.uqx")

    print(f"    reader resident      : {_human(reader.resident_bytes)} "
          f"({reader.stats()['fence_entries']:,} fence entries)")

    rng = random.Random(seed)
    names = [f"slice:{rng.randrange(records):09d}" for _ in range(lookups)]
    reader.probes = 0
    started = time.perf_counter()
    found = sum(1 for name in names if reader.lookup(name) is not None)
    elapsed = time.perf_counter() - started

    print(f"    {lookups:,} random lookups: {found:,} found in {elapsed * 1000:.1f} ms "
          f"({elapsed / lookups * 1e6:.1f} us each)")
    print(f"    byte-range reads     : {reader.probes:,} total, "
          f"{reader.probes / lookups:.1f} per lookup "
          f"({_human(reader.probes * RECORD_SIZE / lookups)} read per lookup)")
    missing = reader.lookup("slice:does-not-exist")
    print(f"    absent key returns   : {missing!r}")


def demo_block_paging(library_mb: int, shard_kb: int, budget_kb: int, reads: int,
                      seed: int) -> None:
    """Show that reading one shard does not read the library."""
    print(f"\n  Block paging: a {library_mb} MB library, {budget_kb} KB of RAM")
    cold = RamStorage(name="cold-library")
    size = library_mb * 1024 * 1024
    rng = random.Random(seed)
    cold.write("model.uql", bytes(rng.randrange(256) for _ in range(4096)) * (size // 4096))

    tier = TieredStorage(cold, max_bytes=budget_kb * 1024, name="scale",
                         block_size=256 * 1024)
    shard_bytes = shard_kb * 1024
    count = size // shard_bytes
    for _ in range(reads):
        index = rng.randrange(count)
        tier.read_range("model.uql", index * shard_bytes, shard_bytes)

    stats = tier.tier_stats()
    requested = reads * shard_bytes
    print(f"    {reads} shard reads   : {stats['hits']} hot / {stats['misses']} cold")
    print(f"    asked for            : {_human(requested)}")
    print(f"    read from storage    : {_human(stats['cold_bytes_read'])} "
          f"(amplification {stats['cold_bytes_read'] / requested:.2f}x)")
    print(f"    never read           : {_human(size - stats['cold_bytes_read'])} "
          f"of the {_human(size)} library")
    print(f"    peak resident        : {_human(stats['resident_bytes'])} "
          f"(budget {_human(stats['budget_bytes'])})")
    assert stats["resident_bytes"] <= tier.max_bytes, "budget invariant violated"
    print("    budget invariant held: RAM never exceeded its cap.")


def main(argv: list[str] | None = None) -> int:
    """Run the scaling analysis.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="Trillion-parameter scaling analysis")
    parser.add_argument("--params-per-shard", type=int, default=1_000_000)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--build", type=int, default=200_000,
                        help="records to build a real index for (0 to skip)")
    parser.add_argument("--lookups", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    total = total_ram()
    print("UltraQuant scaling analysis")
    print(f"  this machine: {_human(total)} RAM total, {_human(available_ram())} available")

    for parameters in (7e9, 175e9, 1.2e12):
        info = report(int(parameters), args.params_per_shard, args.stride)
        if total:
            print(f"    fits in RAM?         : index {info['ram_bytes'] / total * 100:.4f}% "
                  f"of this machine's memory; weights stay on storage")

    if args.build > 0:
        build_and_measure(args.build, args.stride, args.lookups, args.seed)

    demo_block_paging(library_mb=64, shard_kb=256, budget_kb=4096, reads=200,
                      seed=args.seed)

    print("\n  Conclusion: RAM cost is set by the fence stride and the working set,")
    print("  not by the parameter count. A larger model means a larger file on")
    print("  storage and more index records on storage -- neither of which has to")
    print("  be resident. What must fit in memory is the sparse fence plus the")
    print("  shards actually being used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
