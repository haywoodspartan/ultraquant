"""Tensors that go INTO the library. The packing gate.

§11.95 measured what ternary conversion costs. This measures whether
the result can live in the shard library as a first-class citizen -
catalogued, addressable, paged - rather than as a pile of files in
the same directory.

Three things are at stake and only one of them is obvious.

**The bytes.** The vault serialises a payload as JSON and
zlib-compresses it, which is an awkward home for millions of trits.
Four spellings were measured on real tensors and the dense packings
beat the obvious one by a third; base-243 and two-bit then tied,
because density and compressibility pull against each other and very
nearly cancel. base-243 won on decode speed instead - 3.4x - which
is the axis that matters for a store that decodes on every page-in.

**The exactness.** The conversion is already lossy. If the PACKING
were lossy too the two losses would be indistinguishable, and no
future measurement of the quantiser could be trusted. So the
round-trip through JSON, base64, zlib and sha256 must return the
same trits and the same scales, bit for bit.

**The address.** A shard nobody can find is not in a library. Every
packed tensor must be reachable by the route the vault actually
offers: its category holds its shard id, and its own name tokens
score that category.

**The criteria, written before the run.**

1. **Round-trip is exact** - every trit and every scale recovered
   bit-for-bit from the vault. One difference is a fail, and there
   is no tolerance to widen: this is the layer that must not add
   loss to a lossy conversion.
2. **Gain** - stored bits per weight for the packed spelling must
   beat the naive JSON-of-integers by more than the tensor-to-tensor
   sd of that contrast.
3. **Everything is addressable** - for every shard written, its id
   appears in its category and its name tokens give that category a
   non-zero association score. One unreachable shard is a fail.
4. **The decode price is named** - weights per second, reported and
   not gated, because a pageable library pays it on every page-in.

**It took three runs. The mechanism was wrong once and the harness
once, and both are worth keeping.**

Run one failed criterion 2: +0.187 bits/weight saved against a
tensor-to-tensor sd of 0.507. The per-tensor numbers appeared to say
why - packing seemed to LOSE 0.38 to 0.59 bits/weight on four 1-D
tensors - and that reading produced a tidy story about skewed
distributions and a per-tensor spelling choice to fix it.

**The story was an artifact and is retracted here.** The baseline it
rested on was a bare {q, s} dict carrying no name and no shape,
while the packed payload carried both, so the comparison was
measuring the spelling AND the metadata at once; on small tensors,
where a 25-character name is a real share of the bytes, that gap is
most of the difference. Compared like for like, packing wins on
**twelve of fourteen** real tensors, and the two exceptions favour
text by 0.014 and 0.062 bits/weight - not 0.4 to 0.6.

The per-tensor spelling choice survives the correction, but for a
much smaller reason than the one it was built for: it buys about
0.005 bits/weight on this checkpoint. Both spellings are written,
the smaller is kept, and the payload records which so the decoder
can dispatch - kept because it can never be worse, not because it
pays.

Run two barely moved (1.824 -> 1.820), which is what sent me back to
the baseline. Run three, comparing like for like - the same shard
with the same metadata, and only the trits spelled differently:

| | |
|---|---:|
| tensors packed | 20 (1,018,016 weights) |
| packed spelling | **1.820 bits/weight** |
| naive spelling | 2.298 bits/weight |
| saved | **+0.477 at 0.300 tensor sd** |
| trit differences | **0** |
| scale differences | **0** |
| unreachable shards | **0** |
| decode rate | 11.6M weights/s |

Criterion 1 is the one with no tolerance, and it held: every trit
and every scale came back bit-for-bit through JSON, base64, zlib and
the vault's own sha256 check. The conversion above this layer is
lossy by 47%; if this layer were lossy too the two would be
indistinguishable, and no future measurement of the quantiser could
be trusted.

The decode rate is end-to-end - zlib, sha256, JSON parse and unpack
together - not the codec in isolation, which reads 118M weights/s.
The gap is the price of being IN the library rather than beside it,
and it is reported rather than gated because that price buys the
catalog.

Run it::

    python -m ultraquant.experiments.packlibrary_gate
"""

from __future__ import annotations

import json
import shutil
import statistics
import tempfile
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from ultraquant.convert import gguf, library, pack, ternary
from ultraquant.experiments.convert_gate import checkpoint_path
from ultraquant.shards.vault import ShardVault

__all__ = ["PackParityReport", "run_gate"]


@dataclass
class PackParityReport:
    """Exactness, size, reachability and the decode clock.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        shards: Tensors packed.
        weights: Ternary weights stored.
        trit_differences: Trits that came back wrong.
        scale_differences: Scales that came back wrong.
        packed_bits: Stored bits per weight, packed.
        naive_bits: Stored bits per weight, JSON list of ints.
        delta: Naive minus packed, in bits per weight.
        tensor_sd: Tensor-to-tensor sd of that contrast.
        unreachable: Shards not findable by category or keyword.
        decode_rate: Weights per second decoded from stored bytes.
        categories: Category -> shard count.
        reason: Plain-language verdict.
    """

    passes: bool
    shards: int = 0
    weights: int = 0
    trit_differences: int = 0
    scale_differences: int = 0
    packed_bits: float = 0.0
    naive_bits: float = 0.0
    delta: float = 0.0
    tensor_sd: float = 0.0
    unreachable: int = 0
    decode_rate: float = 0.0
    categories: dict = field(default_factory=dict)
    reason: str = ""


def _stored_bytes(payload: dict) -> int:
    """Exactly what the vault will write: canonical JSON, zlibbed."""
    raw = json.dumps(payload, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return len(zlib.compress(raw))


def _pack_all(opened, vault, tensors: int, rows_each: int):
    """Write tensors as shards; return what was written and its cost."""
    expected: dict = {}
    packed_sizes: list = []
    naive_sizes: list = []
    contrasts: list = []
    weights = 0
    written = 0
    for info in opened.tensors:
        if written >= tensors:
            break
        if not info.readable:
            continue
        rows = opened.rows_of(info, 0, min(rows_each, info.rows))
        if not rows or not rows[0]:
            continue
        quantised, scales, _rule = ternary.choose_rules(rows)
        payload = pack.to_payload(quantised, scales, info.name,
                                  info.type_name)
        vault.add_shard(shard_id=f"tensor:{info.name}",
                        category=library.category_of(info.name),
                        payload=payload, kind="ternary-tensor",
                        associations=library.associations_of(info.name))
        expected[f"tensor:{info.name}"] = (quantised, scales, info.name)
        count = len(quantised) * len(quantised[0])
        weights += count
        written += 1
        # Like for like: the same shard, the same metadata, the only
        # difference being how the trits are spelled. The first
        # version of this compared the packed payload against a bare
        # {q, s} dict, which carries no name and no shape - so it was
        # measuring the spelling AND the metadata at once, and small
        # tensors, where the metadata is a real share of the bytes,
        # made the contrast look like noise.
        stored = _stored_bytes(payload)
        naive = _stored_bytes(pack.to_payload(quantised, scales,
                                              info.name, info.type_name,
                                              spelling="ints"))
        packed_sizes.append(8.0 * stored / count)
        naive_sizes.append(8.0 * naive / count)
        contrasts.append(8.0 * (naive - stored) / count)
    return expected, packed_sizes, naive_sizes, contrasts, weights, written


def _read_back(vault, expected: dict):
    """(trit differences, scale differences, unreachable, rate)."""
    trit_differences = 0
    scale_differences = 0
    unreachable = 0
    decoded = 0
    started = time.perf_counter()
    for shard_id, (quantised, scales, name) in expected.items():
        # Through the vault's own path, which verifies sha256 on the
        # way - so this measures the store, not just the codec.
        back, back_scales = pack.from_payload(vault.get(shard_id))
        if len(back) != len(quantised):
            trit_differences += abs(len(back) - len(quantised))
        for index, row in enumerate(quantised):
            if index >= len(back):
                trit_differences += len(row)
                continue
            for column, value in enumerate(row):
                if back[index][column] != value:
                    trit_differences += 1
            decoded += len(row)
        for index, scale in enumerate(scales):
            if index >= len(back_scales) or back_scales[index] != scale:
                scale_differences += 1
        category = library.category_of(name)
        if shard_id not in vault.shards_in(category):
            unreachable += 1
            continue
        scores = vault.association_scores(set(library.associations_of(name)))
        if scores.get(category, 0.0) <= 0.0:
            unreachable += 1
    elapsed = time.perf_counter() - started
    rate = (decoded / elapsed) if elapsed > 0 else 0.0
    return trit_differences, scale_differences, unreachable, rate


def run_gate(tensors: int = 20, rows_each: int = 64,
             path: Path | None = None) -> PackParityReport:
    """Pack real tensors into a real vault and read them back."""
    checkpoint = Path(path) if path is not None else checkpoint_path()
    if not checkpoint.exists():
        return PackParityReport(
            passes=False,
            reason="FAIL - no checkpoint to pack; set UQ_CONVERT_MODEL")
    opened = gguf.read(checkpoint)
    root = Path(tempfile.mkdtemp(prefix="uq_packgate_"))
    try:
        vault = ShardVault(root)
        (expected, packed_sizes, naive_sizes, contrasts,
         weights, written) = _pack_all(opened, vault, tensors, rows_each)
        trits, scales_wrong, unreachable, rate = _read_back(vault, expected)

        packed_bits = statistics.mean(packed_sizes) if packed_sizes else 0.0
        naive_bits = statistics.mean(naive_sizes) if naive_sizes else 0.0
        delta = statistics.mean(contrasts) if contrasts else 0.0
        tensor_sd = statistics.stdev(contrasts) if len(contrasts) > 1 else 0.0
        categories = {category: len(vault.shards_in(category))
                      for category in vault.categories()}

        passes = (written > 0 and trits == 0 and scales_wrong == 0
                  and unreachable == 0 and tensor_sd > 0.0
                  and delta > tensor_sd)
        reason = (
            f"{'PASS' if passes else 'FAIL'} - {written} tensors packed "
            f"({weights:,} weights); {trits} trit and {scales_wrong} "
            f"scale differences on the way back; {packed_bits:.3f} "
            f"bits/weight stored against the naive spelling's "
            f"{naive_bits:.3f} ({delta:+.3f} at {tensor_sd:.3f} tensor "
            f"sd); {unreachable} unreachable shards; "
            f"{rate/1e6:.1f}M weights/s decoded")
        return PackParityReport(
            passes=passes, shards=written, weights=weights,
            trit_differences=trits, scale_differences=scales_wrong,
            packed_bits=packed_bits, naive_bits=naive_bits, delta=delta,
            tensor_sd=tensor_sd, unreachable=unreachable,
            decode_rate=rate, categories=categories, reason=reason)
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":                        # pragma: no cover
    report = run_gate()
    print(f"tensors packed : {report.shards}")
    print(f"weights stored : {report.weights:,}")
    print(f"categories     : {report.categories}")
    print()
    print(f"packed spelling: {report.packed_bits:.3f} bits/weight")
    print(f"naive spelling : {report.naive_bits:.3f} bits/weight")
    print(f"saved          : {report.delta:+.3f} at "
          f"{report.tensor_sd:.3f} tensor sd")
    print()
    print(f"trit differences : {report.trit_differences}")
    print(f"scale differences: {report.scale_differences}")
    print(f"unreachable      : {report.unreachable}")
    print(f"decode rate      : {report.decode_rate/1e6:.1f}M weights/s")
    print()
    print(report.reason)
