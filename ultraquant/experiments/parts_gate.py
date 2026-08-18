"""Part-based features against the variant data. The last §11.21 candidate.

The diagnosis chain so far: §11.20 blamed position and was measured wrong
(§11.21); §11.21 blamed structure-and-scale and the scale half was measured
wrong (§11.22); capacity was binding, hidden 64 became the monolith champion —
and still fails 46% of held variants. Of the three mechanisms §11.21 named,
**part-based features** is the one whose claim was never measured.

The mechanism is :func:`~ultraquant.forge.corpus.part_features`: the sixteen
overlapping 2x2 patches read as 4-bit patterns, histogrammed — local shapes
counted with their positions thrown away — plus the ten row/column means so
arrangement is not entirely lost. The bet it embodies: a corner plus and the
centred plus share crossing-patch *counts* even though they share almost no
pixels. :func:`~ultraquant.forge.corpus.hybrid_features` concatenates the 30
positional numbers with the 16 bins, because §11.21 measured what happens when
a candidate is only tested in one combination.

Width is fixed at hidden 64 — the champion §11.22 measured — so this gate
varies representation only. Chance is 0.125 (8-way monolith), directly
comparable to §11.20–§11.22.

**The criteria, written before the run.**

1. **Gain** — the best parts arm must beat positional/64 on held-out variants
   by more than the seed-to-seed sd of that contrast.
2. **Control** — that arm's noisy-canonical accuracy must not fall below
   positional/64's by more than one sd.
3. **Attribution is reported, not gated** — parts-only vs hybrid says whether
   the histogram replaces position or supplements it.

**It was run, and it FAILED — the last candidate measures nothing.**

| arm | held variants | noisy canon (control) |
|---|---:|---:|
| positional/64 (§11.22 champion) | 0.539 | 0.867 |
| parts/64 | 0.397 | 0.453 |
| hybrid/64 | 0.539 | 0.867 |

Parts alone are not merely weaker — they collapse the *control* to 0.453,
because local texture cannot even hold the noisy canonicals apart: two flipped
pixels perturb patch counts more than they perturb pixel positions. And the
hybrid buys nothing: **-0.000 at seed sd 0.106** against positional/64, per-seed
swings of ±0.1 averaging to exactly zero. The invariance also has a limit that
lands exactly on the hard cases: patch counts are translation-invariant only
while a shape's one-cell halo stays inside the grid, and a border-hugging
glyph loses its boundary patches to clipping — the difficult variants hug
edges and corners, which is precisely where the histogram stops being the
same histogram.

That exact zero is recorded deliberately, because it cost an investigation: the
two arms' 8-seed hit totals tie at 125/232 *and* their control totals tie,
while every per-seed value differs — a coincidence that looked so much like a
plumbing defect that it was debugged as one. The chase found no defect in the
arms and one real defect in the report: the contrast was computed
"best-overall vs reference", which degenerates to a row of exact zeros
whenever the reference wins and reports nothing at all. The registered text
said "the best parts arm", the code now matches it, and the seed sd this
exposes (0.106) is the honest scale of the null result.

With this, §11.21's candidate list is fully measured: position (no, §11.21),
scale (no, §11.22), capacity (the one real lift, §11.22), parts (no, here).
Every representational mechanism named for the variant problem is now spent;
hidden 64 still fails 46% of held variants, and the remaining lever is
**data** — 26 taught variants per split is what the mechanisms starved on, and
§11.20 already measured which models can draw more.

Run it::

    python -m ultraquant.experiments.parts_gate
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from ultraquant.forge.corpus import (features_of, hybrid_features,
                                     part_features)
from ultraquant.model.network import UltraQuantNet
from ultraquant.pattern.recognition import LABELS, PATTERNS

__all__ = ["PartsReport", "run_gate"]

_NOISE_COPIES = 12
_NOISE_FLIPS = 2
_EPOCHS = 60
_LR = 0.05
_HIDDEN = (64,)

#: The arms: (name, featurizer). Width fixed at the §11.22 champion.
_ARMS = (
    ("positional/64", features_of),
    ("parts/64", part_features),
    ("hybrid/64", hybrid_features),
)

_REFERENCE = "positional/64"


@dataclass
class PartsReport:
    """The three-way table and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criterion.
        skipped: True when no variants file exists. Never a pass.
        held: Arm name -> held-variant accuracy.
        canon: Arm name -> noisy-canonical accuracy.
        best: The best arm by held-variant accuracy.
        delta: Best arm minus the reference arm, held variants.
        seed_sd: Seed-to-seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    skipped: bool = False
    held: dict = None
    canon: dict = None
    best: str = ""
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        if self.skipped:
            return f"SKIPPED - {self.reason}"
        lines = [f"{'arm':<16} {'held variants':>14} {'noisy canon (CTRL)':>19}"]
        for name, _f in _ARMS:
            marker = "  <- best" if name == self.best else ""
            lines.append(f"{name:<16} {self.held[name]:>14.3f} "
                         f"{self.canon[name]:>19.3f}{marker}")
        lines += [
            f"delta (best parts arm vs {_REFERENCE}) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _noisy(rows, rng, flips=_NOISE_FLIPS):
    grid = [list(row) for row in rows]
    for _ in range(flips):
        y = rng.randrange(5)
        x = rng.randrange(5)
        grid[y][x] = "#" if grid[y][x] == "." else "."
    return ["".join(row) for row in grid]


def _train(xs, ys, seed):
    net = UltraQuantNet(input_dim=len(xs[0]), hidden_dims=list(_HIDDEN),
                        num_classes=len(LABELS), seed=seed)
    order = list(range(len(xs)))
    rng = random.Random(seed)
    for _ in range(_EPOCHS):
        rng.shuffle(order)
        net.train_batch([xs[i] for i in order], [ys[i] for i in order], _LR)
    return net


def _accuracy(net, cases, featurize):
    if not cases:
        return 0.0
    hit = 0
    for rows, label_index in cases:
        predicted, _conf = net.predict(featurize(rows))
        if predicted == label_index:
            hit += 1
    return hit / len(cases)


def run_gate(variants_path: str | Path | None = None,
             seeds: int = 8) -> PartsReport:
    """Run the three-arm gate over the kept variants.

    Args:
        variants_path: Defaults to ``uq_home/vault/glyph_variants.json``.
        seeds: Split/noise repetitions.

    Returns:
        The :class:`PartsReport`.
    """
    path = Path(variants_path or "uq_home/vault/glyph_variants.json")
    if not path.exists():
        return PartsReport(skipped=True,
                           reason=f"no variants file at {path}")
    variants = {label: rows for label, rows in
                json.loads(path.read_text(encoding="utf-8")).items()
                if label in PATTERNS and rows}
    if not variants:
        return PartsReport(skipped=True, reason="variants file empty")

    label_index = {label: i for i, label in enumerate(LABELS)}
    held_acc = {name: [] for name, _f in _ARMS}
    canon_acc = {name: [] for name, _f in _ARMS}

    for seed in range(seeds):
        rng = random.Random(seed)
        teach, held = [], []
        for label, rows_list in variants.items():
            shuffled = list(rows_list)
            rng.shuffle(shuffled)
            cut = max(1, len(shuffled) // 2) if len(shuffled) > 1 else 1
            teach += [(rows, label) for rows in shuffled[:cut]]
            held += [(rows, label_index[label]) for rows in shuffled[cut:]]

        rows_set = []
        for label in variants:
            rows_set.append((PATTERNS[label], label))
            for _ in range(_NOISE_COPIES):
                rows_set.append((_noisy(PATTERNS[label], rng), label))
        rows_set += teach
        control = [(_noisy(PATTERNS[label], rng), label_index[label])
                   for label in variants for _ in range(4)]

        for name, featurize in _ARMS:
            xs = [featurize(rows) for rows, _label in rows_set]
            ys = [label_index[label] for _rows, label in rows_set]
            net = _train(xs, ys, seed)
            held_acc[name].append(_accuracy(net, held, featurize))
            canon_acc[name].append(_accuracy(net, control, featurize))

    held_mean = {name: statistics.fmean(vals) for name, vals in held_acc.items()}
    canon_mean = {name: statistics.fmean(vals) for name, vals in canon_acc.items()}
    best = max(held_mean, key=held_mean.get)
    # The registered contrast is the best PARTS arm against the reference —
    # not "best overall vs reference", which degenerates to a row of exact
    # zeros whenever the reference wins and reports nothing at all.
    candidate = max((name for name, _f in _ARMS if name != _REFERENCE),
                    key=held_mean.get)
    contrasts = [held_acc[candidate][i] - held_acc[_REFERENCE][i]
                 for i in range(len(held_acc[candidate]))]
    delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of splits and noise draws.
    seed_sd = statistics.stdev(contrasts) if len(contrasts) > 1 else 0.0

    report = PartsReport(
        held=held_mean, canon=canon_mean, best=best, delta=delta,
        seed_sd=seed_sd,
        margin_ratio=(delta / seed_sd) if seed_sd > 0 else 0.0,
    )

    canon_sd = (statistics.stdev(canon_acc[candidate])
                if len(canon_acc[candidate]) > 1 else 0.0)
    if canon_mean[candidate] < canon_mean[_REFERENCE] - canon_sd:
        report.reason = (
            f"the control failed: {candidate} scores "
            f"{canon_mean[candidate]:.3f} on noisy canonicals against the "
            f"reference's {canon_mean[_REFERENCE]:.3f}")
        return report
    if delta <= 0:
        report.reason = (
            f"the best parts arm ({candidate}) does not beat positional/64 "
            f"on held-out variants ({delta:+.3f} at sd {seed_sd:.3f})")
        return report
    if seed_sd <= 0.0:
        report.reason = (f"every split gave the same contrast ({delta:+.3f}); "
                         "nothing varied")
        return report
    if report.margin_ratio <= 1.0:
        report.reason = (f"gain {delta:+.3f} is {report.margin_ratio:.2f}x "
                         f"the seed sd ({seed_sd:.3f})")
        return report
    report.passes = True
    report.reason = (
        f"{candidate} lifts held-variant recognition "
        f"{held_mean[_REFERENCE]:.3f} -> {held_mean[candidate]:.3f} "
        f"({delta:+.3f}, {report.margin_ratio:.2f}x seed sd) with the "
        f"canonical control held ({canon_mean[_REFERENCE]:.3f} -> "
        f"{canon_mean[candidate]:.3f})")
    return report


if __name__ == "__main__":                        # pragma: no cover
    import sys

    print(run_gate(sys.argv[1] if len(sys.argv) > 1 else None).as_text())
