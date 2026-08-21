"""A glyph for every imported tensor. The signature gate.

A packed tensor is addressed three ways: its shard id names it
exactly, its category names its layer, its name tokens name its
kind. All three are things somebody already knew how to type. This
unit adds a fourth that is derived from the WEIGHTS - a 5x5 glyph
rendered from the trits - so the recognizer the system already has
can name a tensor by what is in it.

**The obvious objection, taken first.** The library already has a
signature for content: `shards/sketch.py`, a 64-bit sign-random
projection built precisely for cheap similarity routing. A 25-bit
thresholded downsample is not going to beat it at estimating cosine,
and pretending otherwise would be the kind of comparison that makes
a new mechanism look good by not running the old one. So the sketch
is measured on the same tensors, side by side, and the glyph is
required to be USEFUL, not to win.

What the glyph has that the sketch does not is a reader. Eight named
classes, a trained classifier and a memory that logs what it saw
already exist for this exact shape. A sketch is 64 bits nobody can
look at; a glyph is a picture with a name, and the name is an
association that retrieves the shard.

**The criteria, written before the run.**

1. **Deterministic** - the same tensor renders the same glyph on
   every call and after a round-trip through the vault's JSON, zlib
   and sha256. One difference is a fail; a signature that drifts is
   worse than none, because it is stored.
2. **The glyph carries tensor identity** - mean within-kind Hamming
   distance must be lower than mean between-kind by more than the
   within-kind sd. Tensors of the same kind across different layers
   should look alike, and tensors of different kinds should not.
   Failing this means the glyph is a picture of nothing.
3. **The standard option is measured beside it** - the same
   separation, computed from the 64-bit sketch of the same rows,
   reported in the same units (share of bits differing) so the two
   are comparable. The glyph does not have to win. The comparison
   has to be made, and stated, because a signature that adds nothing
   the sketch already gives is decoration.
4. **The name retrieves the shard** - every glyph is classifiable,
   the label is written as an association, and asking the catalog
   for that label scores the tensor's category above zero. One
   unreachable-by-shape tensor is a fail.
5. **The label is not noise** - two tensors of the same kind must
   share a label more often than two tensors picked at random. If
   the shape label is at the chance rate, it is decoration with a
   trained classifier behind it, and this criterion says so.

**FAILED criterion 2, and it took three passes to find out how
badly.** The headline number moved twice, both times because the
harness was measuring something other than what the criterion asked.

**Pass one rigged the comparison.** The sketch was fed 64 truncated
row means while the glyph saw the whole sampled block, and the glyph
duly separated 1.5x better (+0.179 against +0.117). That was an
artifact of handing two mechanisms different data. Given the same 25
block densities - with the sketch holding strictly more of them,
real values against a threshold, 64 bits against 25 - the sketch
came out ahead instead (1.32 sds against 0.85). A comparison against
a standard option is only worth running if the standard option is
allowed to win.

**Pass two was confounded by dimensionality, and this is the real
finding.** 26 of the 40 tensors are 1-D, and 26 1-D tensors render
only **6 distinct glyphs** - a bias has one row, so four of the
glyph's five row-blocks are empty and it renders a bar. 1-D tensors
therefore sit ~0.03 apart while 2-D tensors sit ~0.48 apart, and
kinds are almost perfectly dimensionality-homogeneous. "Within kind"
was mostly measuring "within dimensionality". Both criteria that ask
about KIND are now measured on the 2-D pool:

| 2-D tensors only | within | between | margin | sd | margin/sd |
|---|---:|---:|---:|---:|---:|
| glyph (25 bits) | 0.478 | 0.498 | +0.020 | 0.103 | **0.19** |
| sketch (64 bits) | 0.038 | 0.037 | **-0.001** | 0.021 | **-0.04** |

**Neither signature separates tensor kind.** The glyph gets 0.19 of
the one-sd bar criterion 2 asked for, and the sketch separates
nothing at all. This is not a glyph problem, it is a summary
problem: block density over a ternary matrix does not distinguish an
attention projection from a feed-forward one. The mixed-pool number
this gate would have reported - margin +0.179 at sd 0.211 - is kept
in the report so the confound stays visible rather than being
quietly corrected away.

**Criterion 5 survives the same correction, thinly.** On the 2-D
pool, same-kind pairs share a label **0.222** against **0.143** for
any pair - a real lift of +0.079 over a baseline that already
absorbs the label skew, but across 18 same-kind pairs. On the 1-D
pool the lift is **-0.019**: 26 tensors, two labels, nothing. The
uncontrolled +0.169 was dimensionality again.

**What passed, and what the glyph is therefore for.** Criterion 1:
zero drift, on a second render and through JSON, zlib and sha256.
Criterion 4: zero tensors unreachable by shape - every label is a
working address, and the end-to-end run retrieves real Qwen tensors
by asking the catalog for "stripes_v". So the glyph keeps the job it
demonstrably does - being rendered once, looked at, named, and
retrieved by that name - and does not get the job it was measured
for and failed. **Nothing routes by glyph distance, and nothing
should route by sketch distance over these summaries either**, which
is the more useful half of the result and the half only the standard
option could have supplied.

This is the third time 1-D tensors have contaminated a measurement
in this module: §11.95 read 2.41 relative error by pushing a vector
through a 1xN dot product, §11.96 compared small tensors against a
baseline carrying no metadata, and §11.99 measured dimensionality
and called it kind. In a real checkpoint most tensors are 1-D and
they behave nothing like the matrices. Any aggregate that mixes them
is measuring the mix.

Run it::

    python -m ultraquant.experiments.tensorglyph_gate
"""

from __future__ import annotations

import re
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ultraquant.convert import gguf, glyph, library, ternary
from ultraquant.experiments.convert_gate import checkpoint_path
from ultraquant.pattern.recognition import PatternRecognizer
from ultraquant.shards import sketch
from ultraquant.shards.vault import ShardVault

__all__ = ["GlyphReport", "run_gate", "kind_of"]

_DIGITS = re.compile(r"\d+")


def kind_of(name: str) -> str:
    """A tensor's kind: its name with the layer numbers taken out.

    "blk.0.attn_q.weight" and "blk.7.attn_q.weight" are the same kind
    of thing in different places, which is exactly the grouping
    criterion 2 asks the glyph to respect.
    """
    return _DIGITS.sub("#", name.lower())


@dataclass
class GlyphReport:
    """What the glyph separated, and what the sketch separated.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        tensors: Tensors rendered.
        kinds: Distinct kinds among them.
        drift: Glyphs that changed on a second render or a round-trip.
        glyph_within: Mean within-kind distance, share of cells.
        glyph_between: Mean between-kind distance, share of cells.
        glyph_sd: Within-kind sd, same units.
        sketch_within: The same, from the 64-bit sketch.
        sketch_between: The same, from the 64-bit sketch.
        sketch_sd: The same, from the 64-bit sketch.
        unreachable: Tensors not retrievable by their shape label.
        label_same_kind: Share of same-kind pairs sharing a label.
        label_any_pair: Share of all pairs sharing a label.
        labels: Label -> how many tensors carry it.
        one_d: Tensors with a single row, excluded from the
            kind measurements and counted here instead.
        mixed_margin: The margin a pool mixing dimensionalities
            would have reported - kept to show the confound.
        mixed_sd: That pool's within-kind sd.
        reason: Plain-language verdict.
    """

    passes: bool
    tensors: int = 0
    kinds: int = 0
    drift: int = 0
    glyph_within: float = 0.0
    glyph_between: float = 0.0
    glyph_sd: float = 0.0
    sketch_within: float = 0.0
    sketch_between: float = 0.0
    sketch_sd: float = 0.0
    unreachable: int = 0
    label_same_kind: float = 0.0
    label_any_pair: float = 0.0
    labels: dict = field(default_factory=dict)
    one_d: int = 0
    mixed_margin: float = 0.0
    mixed_sd: float = 0.0
    reason: str = ""

    @property
    def glyph_margin(self) -> float:
        return self.glyph_between - self.glyph_within

    @property
    def sketch_margin(self) -> float:
        return self.sketch_between - self.sketch_within


def _separation(items, distance, scale):
    """Mean within-kind and between-kind distance, and the within sd."""
    within, between = [], []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            gap = distance(items[i][1], items[j][1]) / scale
            (within if items[i][0] == items[j][0] else between).append(gap)
    return (
        statistics.fmean(within) if within else 0.0,
        statistics.fmean(between) if between else 0.0,
        statistics.pstdev(within) if len(within) > 1 else 0.0,
    )


def run_gate(tensors: int = 40, rows_each: int = 96,
             path: Path | None = None, seed: int = 0) -> GlyphReport:
    """Render, store, page back, and compare against the sketch."""
    checkpoint = Path(path) if path is not None else checkpoint_path()
    if not checkpoint.exists():
        return GlyphReport(passes=False,
                           reason="FAIL - no checkpoint; set "
                                  "UQ_CONVERT_MODEL")
    opened = gguf.read(checkpoint)
    recognizer = PatternRecognizer(mode="classical", seed=seed)
    recognizer.train(epochs=30, n_per_class=30, flips=2, seed=1)

    root = Path(tempfile.mkdtemp(prefix="uq_glyphgate_"))
    try:
        vault = ShardVault(root)
        drawn: list[tuple[str, int]] = []
        sketched: list[tuple[str, int]] = []
        dims: list[int] = []
        names: list[str] = []
        labels: dict = {}
        drift = 0
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
            picture = glyph.glyph_of(quantised)
            # Criterion 1, the cheap half: rendering twice.
            if glyph.glyph_of(quantised) != picture:
                drift += 1
            kind = kind_of(info.name)
            # Dimensionality is carried alongside because a pool that
            # mixes 1-D and 2-D tensors does not measure KIND. See
            # the confound note in the gate docstring.
            dims.append(len(quantised))
            drawn.append((kind, glyph.glyph_bits(picture)))
            # The standard option, on IDENTICAL input. The first
            # version of this line fed the sketch 64 row means while
            # the glyph saw the whole sampled block - which would
            # have made the new mechanism win a race the old one was
            # not allowed to run. Both now summarise the same 25
            # block densities, and the sketch has strictly more to
            # work with: real values against a threshold, 64 bits
            # against 25.
            grid = glyph.densities_of(quantised)
            flat = [value for line in grid for value in line]
            sketched.append((kind, sketch.sketch(flat)))
            names.append(info.name)
            written += 1

        report_pack = library.pack_checkpoint(checkpoint, vault,
                                              limit=tensors,
                                              rows_each=rows_each,
                                              recognizer=recognizer)
        labels = dict(report_pack.labels)

        # Criterion 1, the half that matters: through JSON, zlib and
        # sha256 and back.
        rendered = {name: bits for name, (_kind, bits)
                    in zip(names, drawn)}
        for name in names:
            here = vault.get(f"tensor:{name}").get("glyph")
            if here is None or glyph.glyph_bits(list(here)) != rendered[name]:
                drift += 1

        # Criterion 4: the label is an address that works.
        unreachable = 0
        for name in names:
            stored = vault.get(f"tensor:{name}")
            shape = stored.get("shape")
            if not shape:
                unreachable += 1
                continue
            category = library.category_of(name)
            if vault.association_scores({shape}).get(category, 0.0) <= 0.0:
                unreachable += 1

        # Criterion 5: does the label say anything a coin would not?
        # On the 2-D pool, for the same reason criterion 2 is: with
        # 1-D tensors in, "same kind" and "same dimensionality" are
        # nearly the same predicate and the lift measures the wrong
        # one.
        pool = [(kind_of(name),
                 str(vault.get(f"tensor:{name}").get("shape", "")))
                for name, rows in zip(names, dims) if rows > 1]
        same_kind_hits = same_kind_pairs = 0
        any_hits = any_pairs = 0
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                agree = pool[i][1] == pool[j][1] and pool[i][1] != ""
                any_pairs += 1
                any_hits += int(agree)
                if pool[i][0] == pool[j][0]:
                    same_kind_pairs += 1
                    same_kind_hits += int(agree)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # Criterion 2 and criterion 5 both ask about KIND, so both are
    # measured on the 2-D pool. Mixing dimensionalities answers a
    # different question - see the confound note above.
    matrices = [item for item, rows in zip(drawn, dims) if rows > 1]
    matrix_sketches = [item for item, rows in zip(sketched, dims) if rows > 1]
    g_within, g_between, g_sd = _separation(matrices, glyph.hamming, 25.0)
    s_within, s_between, s_sd = _separation(matrix_sketches, sketch.hamming,
                                            float(sketch.SKETCH_BITS))
    mixed = _separation(drawn, glyph.hamming, 25.0)
    label_same = (same_kind_hits / same_kind_pairs) if same_kind_pairs else 0.0
    label_any = (any_hits / any_pairs) if any_pairs else 0.0
    kinds = len({kind for kind, _bits in matrices})
    passes = (drift == 0
              and g_sd > 0 and g_between - g_within > g_sd
              and unreachable == 0
              and label_same > label_any)
    report = GlyphReport(
        passes=passes, tensors=len(drawn), kinds=kinds, drift=drift,
        glyph_within=g_within, glyph_between=g_between, glyph_sd=g_sd,
        sketch_within=s_within, sketch_between=s_between, sketch_sd=s_sd,
        unreachable=unreachable, label_same_kind=label_same,
        label_any_pair=label_any, labels=labels,
        one_d=sum(1 for rows in dims if rows == 1),
        mixed_margin=mixed[1] - mixed[0], mixed_sd=mixed[2])
    report.reason = (
        f"{'PASS' if passes else 'FAIL'} - {len(drawn)} tensors in "
        f"{kinds} kinds; {drift} glyphs drifted; glyph separates "
        f"{g_within:.3f} within vs {g_between:.3f} between "
        f"(margin {report.glyph_margin:+.3f} at sd {g_sd:.3f}); sketch "
        f"separates {s_within:.3f} vs {s_between:.3f} "
        f"(margin {report.sketch_margin:+.3f}); {unreachable} tensors "
        f"unreachable by shape; same-kind pairs share a label "
        f"{label_same:.3f} against {label_any:.3f} for any pair; "
        f"{report.one_d} 1-D tensors excluded (mixing them in would "
        f"have reported margin {report.mixed_margin:+.3f} at sd "
        f"{report.mixed_sd:.3f})")
    return report


if __name__ == "__main__":                        # pragma: no cover
    result = run_gate()
    print(f"tensors:              {result.tensors} in {result.kinds} kinds")
    print(f"glyphs drifted:       {result.drift}")
    print(f"glyph  within/between {result.glyph_within:.3f} / "
          f"{result.glyph_between:.3f}  margin "
          f"{result.glyph_margin:+.3f} at sd {result.glyph_sd:.3f}")
    print(f"sketch within/between {result.sketch_within:.3f} / "
          f"{result.sketch_between:.3f}  margin "
          f"{result.sketch_margin:+.3f} at sd {result.sketch_sd:.3f}")
    print(f"unreachable by shape: {result.unreachable}")
    print(f"same-kind label rate: {result.label_same_kind:.3f} "
          f"(any pair {result.label_any_pair:.3f})")
    for name, count in sorted(result.labels.items(),
                              key=lambda kv: -kv[1]):
        print(f"  {name:<14} {count:3d} tensors")
    print()
    print(result.reason)
