"""Weights that come back out and still recognise. The import gate.

§11.96 proved tensors go INTO the library: packed, catalogued,
addressable, recovered bit-for-bit through JSON, base64, zlib and
sha256. That is a claim about BYTES. It says nothing about whether a
network built from those bytes computes what the original computed,
and "the weights are imported" is not true until something runs on
them.

The glyph recognizer is the instrument. It is the system's one
existing consumer of ternary weights, it has a labelled dataset and
a real accuracy number, and it is small enough that every logit can
be compared rather than sampled. So: train a recognizer, push its
layers into the vault through the SAME codec a converted transformer
uses, page them back, and ask whether the network that comes out is
the network that went in.

**Why the criterion is exactness rather than a tolerance.** The
conversion above this layer is already lossy - §11.95 measured 0.467
relative output error per layer. If the IMPORT were lossy too, the
two losses would be indistinguishable, and no future measurement of
the quantiser could be trusted. The same argument §11.96 made about
bytes applies to arithmetic, so the bar is the same: bit-for-bit, no
tolerance to widen.

That bar is only reachable because of something worth checking
rather than assuming: **ternary quantisation is idempotent.** A
matrix whose entries are all in {-alpha, 0, +alpha} has
``0.75 * mean|w| < alpha`` for any non-zero fraction, so every
survivor survives and the recovered scale is alpha again. An
imported layer can therefore be handed back to the network still
flagged quantized, and its forward pass reproduces the original's
exactly. The gate measures this rather than trusting the algebra.

**Biases are stored as exact floats, not trits.** §11.95 measured a
1-D tensor at 0.527 relative error, and a bias is added to every
output of its layer - the one place a per-element error cannot
cancel. A handful of numbers against a matrix's millions is free, so
ternarising them would be a measurable loss taken for nothing.
Criterion 3 exists so that decision cannot be quietly reversed.

**The criteria, written before the run.**

1. **Inference is unchanged, exactly** - for every glyph in the
   dataset, the imported network's logits equal the original's
   bit-for-bit. One difference is a fail.
2. **Recognition is unchanged** - same predicted label on every
   glyph, and the same accuracy to the last digit. A network can
   drift in its logits and still agree on the argmax, so this is
   the weaker check and criterion 1 is the real one; it is here
   because it is what a user would notice.
3. **Biases survive exactly** - every bias element bit-for-bit. A
   bias that came back through the trit codec is a fail even if
   the accuracy held.
4. **Every layer is addressable** - each layer's shard is reachable
   by its category AND scores non-zero on its own name tokens. A
   weight nobody can find is not in a library.
5. **The cost is named** - bits per weight of the stored recognizer,
   reported and not gated, because a library that cannot say what
   it charges is not offering a trade.

**It took two runs, and the first one found a real defect.**

Run one FAILED criteria 1 and 2: all 128 glyphs differed in their
logits, 26 in their label, and accuracy fell **0.900 -> 0.792**. The
cause was one layer. `UltraQuantNet`'s head is full precision by
default, and the packer pushed it through the trit codec like
everything else - a 0.151 error in the weights of the layer that
produces the logits. **Trits are how you store a ternary matrix, not
how you store a matrix.** A layer that is not ternary now travels as
exact floats and says so in the payload; a converted transformer is
ternary by construction and never takes that path.

That is the bias decision one level up, and finding it twice in one
module is the useful part: the rule is not "biases are special", it
is that a lossy codec may only carry what is already in its
alphabet.

**A mis-aimed measurement was corrected on the way.** The idempotence
check first ran on the ORIGINAL layers, whose master weights are
full precision and were never expected to survive re-quantisation.
It reported 0/2, which reads as a refuted claim and was a question
asked of the wrong object. Idempotence is a property of what
ARRIVES; measured there, it holds.

Run two:

| | |
|---|---:|
| glyphs through both networks | 128 |
| differing logits | **0** |
| differing labels | **0** |
| accuracy | **0.9000 -> 0.9000** |
| bias elements changed | **0** |
| unreachable layers | **0** |
| ternary layers idempotent | **1/1** |

**Criterion 5, the cost, and it is not a flattering number.** 1,216
weights in 3,606 bytes is 23.7 bits/weight - but the blend hides the
trade, so it is broken out:

| | weights | bytes | bits/weight |
|---|---:|---:|---:|
| ternary | 960 | 745 | **6.208** |
| exact floats | 256 | 2,768 | **86.500** |

The exactly-stored head is 21% of the weights and 77% of the bytes.
That is what criterion 1 costs, paid deliberately and named here
rather than averaged away. And the ternary rate is itself 3.4x
§11.96's 1.820, for a reason that is not the codec: at 960 weights
the per-shard metadata - name, shape, scales, base64 padding, zlib
header, sha256 - is most of the shard. §11.96's number was measured
over 1,018,016 weights, and the two are not comparable at face
value.

Neither number is optimised here, and one of them plainly could be:
86.5 bits/weight is JSON writing decimal digits, which is a spelling
problem rather than an information one. It is recorded as the next
question rather than quietly left as an achievement.

Run it::

    python -m ultraquant.experiments.glyphimport_gate
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ultraquant.convert import load
from ultraquant.pattern.recognition import (LABELS, PATTERNS, make_dataset,
                                            render, PatternRecognizer)
from ultraquant.shards.vault import ShardVault

__all__ = ["ImportReport", "run_gate"]


@dataclass
class ImportReport:
    """What survived the trip through the library, and what it cost.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        samples: Glyphs pushed through both networks.
        logit_differences: Samples whose logits differ at any bit.
        label_differences: Samples classified differently.
        accuracy_before: Accuracy of the trained network.
        accuracy_after: Accuracy of the imported network.
        bias_differences: Bias elements that changed.
        unreachable: Layer shards not reachable by category or index.
        idempotent: Layers whose weights survive re-quantisation.
        layers: Layers stored.
        weights: Weights stored.
        stored_bytes: Bytes the vault holds for them.
        max_logit_gap: Largest absolute logit difference seen.
        reason: Plain-language verdict.
    """

    passes: bool
    samples: int = 0
    logit_differences: int = 0
    label_differences: int = 0
    accuracy_before: float = 0.0
    accuracy_after: float = 0.0
    bias_differences: int = 0
    unreachable: int = 0
    idempotent: int = 0
    layers: int = 0
    weights: int = 0
    stored_bytes: int = 0
    max_logit_gap: float = 0.0
    cost_split: dict = field(default_factory=dict)
    reason: str = ""

    @property
    def bits_per_weight(self) -> float:
        if not self.weights:
            return 0.0
        return 8.0 * self.stored_bytes / self.weights


def run_gate(epochs: int = 30, seed: int = 0) -> ImportReport:
    """Train, store, page back, and compare - logit by logit."""
    root = Path(tempfile.mkdtemp(prefix="uq_glyphimport_"))
    try:
        recognizer = PatternRecognizer(mode="classical", seed=seed,
                                       memory_path=root / "memory")
        recognizer.train(epochs=epochs, n_per_class=30, flips=2, seed=1)
        before = recognizer.evaluate(n_per_class=15, flips=3, seed=99)

        vault = ShardVault(root / "vault")
        written = load.store_net(vault, recognizer.net, prefix="glyphnet",
                                 category="pattern")
        imported = load.load_net(vault, prefix="glyphnet")

        # Idempotence is a property of what ARRIVES, not of what was
        # sent: the master weights of a trained layer are full
        # precision and were never expected to survive requantisation
        # unchanged. The first version of this line asked the wrong
        # question and got 0/2, which read as a refuted claim and was
        # a mis-aimed measurement.
        ternary_layers = [layer for layer in [*imported.hidden, imported.head]
                          if layer.quantized]
        idempotent = sum(1 for layer in ternary_layers
                         if load.is_idempotent(layer))

        # Criterion 3, before anything else touches the layers: a
        # bias that came back through the trit codec would be a
        # measurable loss taken for nothing.
        bias_differences = 0
        originals = [*recognizer.net.hidden, recognizer.net.head]
        arrived = [*imported.hidden, imported.head]
        for was, now in zip(originals, arrived):
            for old, new in zip(was.b, now.b):
                if old != new:
                    bias_differences += 1

        # Criterion 1 and 2, over the whole labelled dataset plus the
        # eight clean glyphs - every logit compared, none sampled.
        samples, logit_differences, label_differences = 0, 0, 0
        max_gap = 0.0
        xs, _ys = make_dataset(15, 3, seed=99)
        clean = [render(PATTERNS[name]) for name in LABELS]
        for flat in [*xs, *clean]:
            features = recognizer.features(flat)
            want = recognizer.net.forward(features)
            have = imported.forward(features)
            samples += 1
            if want != have:
                logit_differences += 1
                for a, b in zip(want, have):
                    max_gap = max(max_gap, abs(a - b))
            if (want.index(max(want)) != have.index(max(have))):
                label_differences += 1

        after_net = recognizer.net
        recognizer.net = imported
        after = recognizer.evaluate(n_per_class=15, flips=3, seed=99)
        recognizer.net = after_net

        # Criterion 4: reachable by category AND by its own name.
        unreachable = 0
        for shard_id in written:
            if shard_id.endswith(":shape"):
                continue
            in_category = shard_id in vault.shards_in("pattern")
            tokens = [t for t in shard_id.replace(":", " ").replace(".", " ")
                      .split() if not t.isdigit()]
            scores = vault.association_scores(tokens)
            if not in_category or scores.get("pattern", 0.0) <= 0.0:
                unreachable += 1

        weights = sum(len(layer.w) * len(layer.w[0])
                      for layer in originals if layer.w)
        loose = Path(vault.root) / "loose"
        stored = sum(path.stat().st_size for path in loose.glob("*.uqs"))
        # Criterion 5 done properly: one blended bits/weight hides the
        # trade this unit made. A ternary layer and an exactly-stored
        # float layer are charged at wildly different rates, and a
        # library that cannot say which is which is not offering a
        # trade, it is quoting an average.
        split: dict = {}
        for layer, shard_id in zip(originals, [i for i in written
                                               if not i.endswith(":shape")]):
            count = len(layer.w) * len(layer.w[0]) if layer.w else 0
            safe = shard_id.replace(":", "~3a~")
            path = loose / f"s_{safe}.uqs"
            size = path.stat().st_size if path.exists() else 0
            tag = "ternary" if layer.quantized else "exact floats"
            was = split.get(tag, (0, 0))
            split[tag] = (was[0] + count, was[1] + size)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    passes = (logit_differences == 0 and label_differences == 0
              and before == after and bias_differences == 0
              and unreachable == 0)
    report = ImportReport(
        passes=passes, samples=samples,
        logit_differences=logit_differences,
        label_differences=label_differences,
        accuracy_before=before, accuracy_after=after,
        bias_differences=bias_differences, unreachable=unreachable,
        idempotent=idempotent, layers=len(ternary_layers), weights=weights,
        stored_bytes=stored, max_logit_gap=max_gap, cost_split=split)
    report.reason = (
        f"{'PASS' if passes else 'FAIL'} - {samples} glyphs through both "
        f"networks; {logit_differences} with differing logits, "
        f"{label_differences} classified differently, accuracy "
        f"{before:.4f} -> {after:.4f}, {bias_differences} bias elements "
        f"changed, {unreachable} unreachable layers, "
        f"{idempotent}/{len(ternary_layers)} ternary layers idempotent "
        f"under "
        f"re-quantisation, {report.bits_per_weight:.3f} bits/weight stored"
        + (f", largest logit gap {max_gap:g}" if max_gap else ""))
    return report


if __name__ == "__main__":                        # pragma: no cover
    result = run_gate()
    print(f"glyphs compared:      {result.samples}")
    print(f"differing logits:     {result.logit_differences}")
    print(f"differing labels:     {result.label_differences}")
    print(f"accuracy before:      {result.accuracy_before:.4f}")
    print(f"accuracy after:       {result.accuracy_after:.4f}")
    print(f"bias differences:     {result.bias_differences}")
    print(f"unreachable layers:   {result.unreachable}")
    print(f"idempotent layers:    {result.idempotent}/{result.layers}")
    print(f"stored:               {result.weights} weights, "
          f"{result.stored_bytes} bytes, "
          f"{result.bits_per_weight:.3f} bits/weight")
    for tag, (count, size) in sorted(result.cost_split.items()):
        rate = 8.0 * size / count if count else 0.0
        print(f"  {tag:<18} {count:5d} weights, {size:6d} bytes, "
              f"{rate:7.3f} bits/weight")
    print()
    print(result.reason)
