"""Scale normalisation and capacity against the variant data. The third gate.

§11.20 measured that model-drawn variants do not transfer under the positional
features and the 30→24→8 net (+0.013 at 0.23x sd). §11.21 built translation
capability and measured the diagnosis wrong: 62% of the variants sit exactly
where their canonical sits, so the difficulty is **structure and scale**. It
named three candidate mechanisms — scale normalisation, part-based features,
capacity — and this gate takes the two cheap ones, factorially, because §11.21
also demonstrated what happens when arms are not crossed: a mechanism can help,
hurt, or do nothing depending on what it is combined with.

The mechanism under test is
:func:`~ultraquant.pattern.recognition.scale_normalize` — stretch the set-pixel
bounding box to fill the grid, one uniform factor so a bar cannot become a
block, centred so scale and position normalise together. Capacity is the probe
result folded in: hidden (64,) lifted *both* held-variants and the control in a
pre-gate sweep, so the small net was under-fitting generally, and any
representation claim has to survive being crossed with it.

**The criteria, written before the run.**

1. **Gain** — the best arm must beat §11.20's arm (positional features, hidden
   24) on held-out variants by more than the seed-to-seed sd of that contrast.
2. **Control** — that arm's noisy-canonical accuracy must not fall below the
   §11.20 arm's control by more than one sd.
3. **Attribution is reported, not gated** — the single-factor arms say whether
   scale, capacity, or only their combination does the work.

**It was run, and it PASSED — on the arm neither prior diagnosis predicted.**

| arm | held variants | noisy canon (control) |
|---|---:|---:|
| positional/24 (§11.20 reference) | 0.448 | 0.758 |
| **positional/64** | **0.539** | **0.867** |
| scaled/24 | 0.422 | 0.770 |
| scaled/64 | 0.457 | 0.867 |

Capacity alone: **+0.091 at 1.86x seed sd**, with the control not merely held
but improved. Scale normalisation — the mechanism §11.21's diagnosis pointed
at — does nothing: worse than the reference at hidden 24, and at hidden 64 it
*loses* to plain positional features. The diagnosis chain closes on the least
glamorous member of §11.21's candidate list: §11.20 said position, §11.21
measured that wrong and said structure-and-scale, and this gate measures the
scale half of *that* wrong too. The 30→24→8 ternary network was simply
under-fitting everything — variants and canonicals alike — and widening it
lifts both.

Two honest limits. Hidden 64 still fails 46% of held variants, so capacity is
the *binding* constraint on this data, not a solution. And this does not
change the deployed forge default (hidden 32) by itself: a wider expert is a
larger shard in a library whose size ratios are load-bearing (§6), so adopting
it is a sizing decision with a measured disk cost, not a silent constant bump.

Run it::

    python -m ultraquant.experiments.capacity_gate
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from ultraquant.forge.corpus import features_of, scaled_features
from ultraquant.model.network import UltraQuantNet
from ultraquant.pattern.recognition import LABELS, PATTERNS

__all__ = ["StructureReport", "run_gate"]

_NOISE_COPIES = 12
_NOISE_FLIPS = 2
_EPOCHS = 60
_LR = 0.05

#: The four arms: (name, featurizer, hidden). Variants are taught in every arm;
#: §11.20 already measured the no-variant baselines.
_ARMS = (
    ("positional/24", features_of, (24,)),
    ("positional/64", features_of, (64,)),
    ("scaled/24", scaled_features, (24,)),
    ("scaled/64", scaled_features, (64,)),
)

_REFERENCE = "positional/24"


@dataclass
class StructureReport:
    """The four-way table and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criterion.
        skipped: True when no variants file exists. Never a pass.
        held: Arm name -> held-variant accuracy.
        canon: Arm name -> noisy-canonical accuracy.
        best: The best arm by held-variant accuracy.
        delta: Best arm minus the §11.20 reference arm, held variants.
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
        for name, _f, _h in _ARMS:
            marker = "  <- best" if name == self.best else ""
            lines.append(f"{name:<16} {self.held[name]:>14.3f} "
                         f"{self.canon[name]:>19.3f}{marker}")
        lines += [
            f"delta (best vs {_REFERENCE}) {self.delta:+.3f}   "
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


def _train(xs, ys, seed, hidden):
    net = UltraQuantNet(input_dim=30, hidden_dims=list(hidden),
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
             seeds: int = 8) -> StructureReport:
    """Run the four-arm gate over the kept variants.

    Args:
        variants_path: Defaults to ``uq_home/vault/glyph_variants.json``.
        seeds: Split/noise repetitions.

    Returns:
        The :class:`StructureReport`.
    """
    path = Path(variants_path or "uq_home/vault/glyph_variants.json")
    if not path.exists():
        return StructureReport(skipped=True,
                               reason=f"no variants file at {path}")
    variants = {label: rows for label, rows in
                json.loads(path.read_text(encoding="utf-8")).items()
                if label in PATTERNS and rows}
    if not variants:
        return StructureReport(skipped=True, reason="variants file empty")

    label_index = {label: i for i, label in enumerate(LABELS)}
    held_acc = {name: [] for name, _f, _h in _ARMS}
    canon_acc = {name: [] for name, _f, _h in _ARMS}

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

        for name, featurize, hidden in _ARMS:
            xs = [featurize(rows) for rows, _label in rows_set]
            ys = [label_index[label] for _rows, label in rows_set]
            net = _train(xs, ys, seed, hidden)
            held_acc[name].append(_accuracy(net, held, featurize))
            canon_acc[name].append(_accuracy(net, control, featurize))

    held_mean = {name: statistics.fmean(vals) for name, vals in held_acc.items()}
    canon_mean = {name: statistics.fmean(vals) for name, vals in canon_acc.items()}
    best = max(held_mean, key=held_mean.get)
    contrasts = [held_acc[best][i] - held_acc[_REFERENCE][i]
                 for i in range(len(held_acc[best]))]
    delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of splits and noise draws.
    seed_sd = statistics.stdev(contrasts) if len(contrasts) > 1 else 0.0

    report = StructureReport(
        held=held_mean, canon=canon_mean, best=best, delta=delta,
        seed_sd=seed_sd,
        margin_ratio=(delta / seed_sd) if seed_sd > 0 else 0.0,
    )

    canon_sd = (statistics.stdev(canon_acc[best])
                if len(canon_acc[best]) > 1 else 0.0)
    if canon_mean[best] < canon_mean[_REFERENCE] - canon_sd:
        report.reason = (
            f"the control failed: {best} scores "
            f"{canon_mean[best]:.3f} on noisy canonicals against the "
            f"reference's {canon_mean[_REFERENCE]:.3f}")
        return report
    if best == _REFERENCE or delta <= 0:
        report.reason = ("no arm beats the §11.20 reference on held-out "
                         f"variants ({delta:+.3f})")
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
        f"{best} lifts held-variant recognition "
        f"{held_mean[_REFERENCE]:.3f} -> {held_mean[best]:.3f} ({delta:+.3f}, "
        f"{report.margin_ratio:.2f}x seed sd) with the canonical control held "
        f"({canon_mean[_REFERENCE]:.3f} -> {canon_mean[best]:.3f})")
    return report


if __name__ == "__main__":                        # pragma: no cover
    import sys

    print(run_gate(sys.argv[1] if len(sys.argv) > 1 else None).as_text())
