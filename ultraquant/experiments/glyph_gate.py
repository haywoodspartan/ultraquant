"""Do model-drawn variants make the pattern experts better? A pre-registered gate.

§11.13 distilled *phrasings* into the router. This distils *shapes* into the
experts: a local model draws validated variants of each canonical glyph — a
plus in a corner, offset stripes, a thick square — and the question is whether
training on them improves recognition of variants the expert has never seen,
without costing recognition of the canonicals it already knew.

**The gate, written before the run.**

1. **Gain** — accuracy on *held-out* variants (drawn by the model, validated,
   and never trained on) must rise over the noise-only baseline by more than
   the seed-to-seed standard deviation.
2. **Control** — accuracy on noise-perturbed canonicals must not fall by more
   than one seed sd. Widening a family must not cost its centre; an expert
   that recognises exotic pluses but fumbles the ordinary one has been made
   worse in the case that dominates real use.

The two arms are identical in every respect — same network shape, same seed,
same epochs, same noise — except that the variant arm's training set also
contains the taught half of the validated variants. Anything the gate reports
is therefore attributable to the variants and not to training-run luck.

**It was run, and it FAILED — and the diagnosis is the §11.5 lesson again.**
53 validated variants, 24 taught per seed: held-variant recognition moved
0.435 → 0.448, a gain of +0.013 at **0.23x** seed sd, with the canonical
control held. Training on half the model's variants barely helps with the
other half, because the features are **position-bound**: 25 raw pixels plus
five row means, so a plus drawn in a corner shares almost no feature mass with
the centered plus that anchors its family. The validator needed shift-invariant
matching to judge these variants fairly (see
:func:`~ultraquant.forge.glyph_distill._agreement`), and the experts have the
identical blindness in their representation — a fix in the metric cannot reach
them. The variants are good data (validated, kept in
``uq_home/vault/glyph_variants.json``) waiting on a representation that can
carry translation; shift-augmentation or invariant features would be a new
mechanism deserving its own gate, not a retry of this one.

Requires ``variants.json`` (produced by the generation runs); reports SKIPPED
without it, never a pass.

Run it::

    python -m ultraquant.experiments.glyph_gate
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from ultraquant.forge.corpus import features_of
from ultraquant.model.network import UltraQuantNet
from ultraquant.pattern.recognition import LABELS, PATTERNS

__all__ = ["GlyphGateReport", "run_gate"]

#: Noisy copies of each canonical per arm - the baseline's whole diet, and the
#: control's test distribution.
_NOISE_COPIES = 12

#: Cells flipped per noisy copy. Two of twenty-five keeps every noisy canonical
#: recognisable to a reader.
_NOISE_FLIPS = 2

_EPOCHS = 60
_LR = 0.05
_HIDDEN = (24,)


@dataclass
class GlyphGateReport:
    """What the gate concluded.

    Attributes:
        passes: The verdict under the pre-registered criterion.
        skipped: True when no variants file exists. Never a pass.
        held_baseline: Held-variant accuracy, noise-only training.
        held_variant: The same with variant training. The gain.
        delta: ``held_variant - held_baseline``.
        seed_sd: Seed-to-seed standard deviation of the delta.
        margin_ratio: ``delta / seed_sd``.
        canon_baseline: Noisy-canonical accuracy, baseline arm. The control's
            reference.
        canon_variant: The same, variant arm. Must not fall.
        variants_taught: Mean variants in the teaching split.
        variants_held: Mean variants in the held split.
        reason: Plain-language verdict.
    """

    passes: bool = False
    skipped: bool = False
    held_baseline: float = 0.0
    held_variant: float = 0.0
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    canon_baseline: float = 0.0
    canon_variant: float = 0.0
    variants_taught: float = 0.0
    variants_held: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """A printable report, control beside the gain."""
        if self.skipped:
            return f"SKIPPED - {self.reason}"
        return "\n".join([
            f"held-out variants     baseline {self.held_baseline:.3f}   "
            f"with variants {self.held_variant:.3f}   delta {self.delta:+.3f}",
            f"noisy canonicals (CTRL) {self.canon_baseline:.3f} -> "
            f"{self.canon_variant:.3f}   <- widening a family must not cost "
            "its centre",
            f"seed sd {self.seed_sd:.3f}   margin ratio "
            f"{self.margin_ratio:.2f}x  (gate needs > 1.0)",
            f"variants: {self.variants_taught:.0f} taught, "
            f"{self.variants_held:.0f} held per seed",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ])


def _noisy(rows: list[str], rng: random.Random, flips: int = _NOISE_FLIPS) -> list[str]:
    """A copy of ``rows`` with ``flips`` random cells inverted."""
    grid = [list(row) for row in rows]
    for _ in range(flips):
        y = rng.randrange(5)
        x = rng.randrange(5)
        grid[y][x] = "#" if grid[y][x] == "." else "."
    return ["".join(row) for row in grid]


def _train(xs: list, ys: list, seed: int) -> UltraQuantNet:
    net = UltraQuantNet(input_dim=30, hidden_dims=list(_HIDDEN),
                        num_classes=len(LABELS), seed=seed)
    order = list(range(len(xs)))
    rng = random.Random(seed)
    for _ in range(_EPOCHS):
        rng.shuffle(order)
        net.train_batch([xs[i] for i in order], [ys[i] for i in order], _LR)
    return net


def _accuracy(net: UltraQuantNet, cases: list) -> float:
    if not cases:
        return 0.0
    hit = 0
    for rows, label_index in cases:
        predicted, _confidence = net.predict(features_of(rows))
        if predicted == label_index:
            hit += 1
    return hit / len(cases)


def run_gate(variants_path: str | Path | None = None,
             seeds: int = 8) -> GlyphGateReport:
    """Run the gate over the validated variants file.

    Args:
        variants_path: The generation runs' output. Defaults to
            ``uq_home/vault/glyph_variants.json`` then the scratch file.
        seeds: Train/split repetitions. Must exceed 1 or the sd is meaningless.

    Returns:
        The :class:`GlyphGateReport`.
    """
    candidates = ([Path(variants_path)] if variants_path else
                  [Path("uq_home/vault/glyph_variants.json")])
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return GlyphGateReport(skipped=True,
                               reason=f"no variants file at {candidates[0]}")
    variants: dict = json.loads(path.read_text(encoding="utf-8"))
    variants = {label: rows for label, rows in variants.items()
                if label in PATTERNS and rows}
    if not variants:
        return GlyphGateReport(skipped=True, reason="variants file is empty")

    label_index = {label: i for i, label in enumerate(LABELS)}
    deltas, canon_b, canon_v, held_b, held_v = [], [], [], [], []
    taught_n, held_n = [], []

    for seed in range(seeds):
        rng = random.Random(seed)
        teach: list = []
        held: list = []
        for label, rows_list in variants.items():
            shuffled = list(rows_list)
            rng.shuffle(shuffled)
            cut = max(1, len(shuffled) // 2) if len(shuffled) > 1 else 1
            teach += [(rows, label) for rows in shuffled[:cut]]
            held += [(rows, label_index[label]) for rows in shuffled[cut:]]
        taught_n.append(len(teach))
        held_n.append(len(held))

        base_xs, base_ys = [], []
        for label in variants:
            index = label_index[label]
            base_xs.append(features_of(PATTERNS[label]))
            base_ys.append(index)
            for _ in range(_NOISE_COPIES):
                base_xs.append(features_of(_noisy(PATTERNS[label], rng)))
                base_ys.append(index)

        var_xs = list(base_xs)
        var_ys = list(base_ys)
        for rows, label in teach:
            var_xs.append(features_of(rows))
            var_ys.append(label_index[label])

        control = [(_noisy(PATTERNS[label], rng), label_index[label])
                   for label in variants for _ in range(4)]

        baseline = _train(base_xs, base_ys, seed)
        with_variants = _train(var_xs, var_ys, seed)

        held_b.append(_accuracy(baseline, held))
        held_v.append(_accuracy(with_variants, held))
        deltas.append(held_v[-1] - held_b[-1])
        canon_b.append(_accuracy(baseline, control))
        canon_v.append(_accuracy(with_variants, control))

    delta = statistics.fmean(deltas)
    # stdev, not pstdev: the seeds are a sample of splits and noise draws.
    seed_sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0

    report = GlyphGateReport(
        held_baseline=statistics.fmean(held_b),
        held_variant=statistics.fmean(held_v),
        delta=delta, seed_sd=seed_sd,
        margin_ratio=(delta / seed_sd) if seed_sd > 0 else 0.0,
        canon_baseline=statistics.fmean(canon_b),
        canon_variant=statistics.fmean(canon_v),
        variants_taught=statistics.fmean(taught_n),
        variants_held=statistics.fmean(held_n),
    )

    tolerance = (statistics.stdev(canon_v) if len(canon_v) > 1 else 0.0)
    if report.canon_variant < report.canon_baseline - tolerance:
        report.reason = (
            f"noisy-canonical recognition fell {report.canon_baseline:.3f} -> "
            f"{report.canon_variant:.3f}; the variants cost the family centre")
        return report
    if delta <= 0:
        report.reason = f"no gain on held-out variants ({delta:+.3f})"
        return report
    if seed_sd <= 0.0:
        report.reason = (f"every split gave the same delta ({delta:+.3f}); "
                         "nothing varied")
        return report
    if report.margin_ratio <= 1.0:
        report.reason = (f"gain {delta:+.3f} is {report.margin_ratio:.2f}x "
                         f"the seed sd ({seed_sd:.3f})")
        return report
    report.passes = True
    report.reason = (
        f"held-variant recognition {report.held_baseline:.3f} -> "
        f"{report.held_variant:.3f} ({delta:+.3f}, {report.margin_ratio:.2f}x "
        f"seed sd) with the canonical control held "
        f"({report.canon_baseline:.3f} -> {report.canon_variant:.3f})")
    return report


if __name__ == "__main__":                        # pragma: no cover
    import sys

    override = sys.argv[1] if len(sys.argv) > 1 else None
    print(run_gate(override).as_text())
