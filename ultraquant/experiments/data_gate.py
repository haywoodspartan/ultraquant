"""More variants against the fixed held set. The data gate.

§11.24 spent §11.21's candidate list: position, scale, and parts measured
nothing; capacity was the one real lift and still fails 46% of held variants.
Every mechanism starved on the same diet — 26 taught variants per split. This
gate measures the lever that remains: **data**. New model-drawn variants,
validated by the same Jaccard-shift contradiction check as the original bank
(§11.20), are added to the *training* side only.

The design constraint that matters: the evaluation universe is **frozen**.
Held sets are split from the original bank (``glyph_variants.json``, 53
variants) with the same seeded protocol as §11.20–§11.24, and new variants
(``glyph_variants_new.json``) never enter a held set — otherwise the target
would move with the data and no number would be comparable to the chain this
gate extends. One arm trains on the original teach half; the other adds every
new variant. Width is hidden 64, the measured champion; chance is 0.125.

**The criteria, written before any new variant was drawn.**

1. **Gain** — the enlarged arm must beat the baseline arm on the original
   held variants by more than the seed-to-seed sd of that contrast.
2. **Control** — its noisy-canonical accuracy must not fall below the
   baseline's by more than one sd.
3. **Provenance is reported, not gated** — which model drew how many kept
   variants is recorded beside the verdict, one model loaded at a time,
   smallest first among the voices not yet measured for drawing.

**It was run, and it FAILED — the data lever, pulled naively, is negative.**

Six voices drew, one model at a time, through the same validation as the
original bank (fresh kept / rejected): katarau-9b 12/32, cydonia-22b 15/15,
qwen3.8-27b 11/31, gemma-4-31b 6/32 (mostly duplicating its own earlier
draws), qwen3.6-35b-a3b 18/22, command-r 5/14 — and qwen3-48b **0/0**: a
runaway reasoner that burns any budget in hidden tokens under this prompt and
is not tamed by ``reasoning_effort="none"``, unlike the models §11.13
measured. 67 fresh validated variants staged. Then:

| arm | held variants | noisy canon (control) |
|---|---:|---:|
| baseline (original teach half) | 0.539 | 0.867 |
| enlarged (+67 new variants) | 0.466 | 0.766 |

**-0.073 at -1.05x seed sd, and the control fell with it.** More validated
data made everything worse. Three post-hoc probes (labelled as such — none
was pre-registered) aim the diagnosis:

* Class balance is not the cause: capping additions at 4 per label recovers
  the control (0.852) but not the held set (0.496 < 0.539).
* Capacity is not the cause: all 67 at hidden 96 scores 0.504; at hidden
  **128**, 0.526 — twice the champion width still below the no-new-data
  baseline.
* What remains is **off-distribution interference**: the held set is
  original-era drawing style, the additions are five other models' styles,
  and training on the mixture pulls the features away from the very
  distribution the frozen evaluation measures.

The design did its job — freezing the held set is what made this measurable —
but it also bounds the claim: this gate says new-style data does not help
*old-style recognition*, not that the new data is worthless. Whether "glyph
variant" is even one distribution across drawing voices is the successor
question, and it needs its own gate (train on one voice's bank, test on
another's, both directions) before any more drawing is worth the tokens.

Run it::

    python -m ultraquant.experiments.data_gate
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

__all__ = ["DataReport", "run_gate"]

_NOISE_COPIES = 12
_NOISE_FLIPS = 2
_EPOCHS = 60
_LR = 0.05
_HIDDEN = (64,)

_ARMS = ("baseline", "enlarged")


@dataclass
class DataReport:
    """The two-way table and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criterion.
        skipped: True when either variants file is missing. Never a pass.
        held: Arm name -> held-variant accuracy on the original bank.
        canon: Arm name -> noisy-canonical accuracy.
        new_count: How many new variants the enlarged arm trained on.
        delta: Enlarged minus baseline, held variants.
        seed_sd: Seed-to-seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    skipped: bool = False
    held: dict = None
    canon: dict = None
    new_count: int = 0
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        if self.skipped:
            return f"SKIPPED - {self.reason}"
        lines = [f"{'arm':<12} {'held variants':>14} {'noisy canon (CTRL)':>19}"]
        for name in _ARMS:
            lines.append(f"{name:<12} {self.held[name]:>14.3f} "
                         f"{self.canon[name]:>19.3f}")
        lines += [
            f"new variants trained on: {self.new_count}",
            f"delta (enlarged vs baseline) {self.delta:+.3f}   "
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
    net = UltraQuantNet(input_dim=30, hidden_dims=list(_HIDDEN),
                        num_classes=len(LABELS), seed=seed)
    order = list(range(len(xs)))
    rng = random.Random(seed)
    for _ in range(_EPOCHS):
        rng.shuffle(order)
        net.train_batch([xs[i] for i in order], [ys[i] for i in order], _LR)
    return net


def _accuracy(net, cases):
    if not cases:
        return 0.0
    hit = 0
    for rows, label_index in cases:
        predicted, _conf = net.predict(features_of(rows))
        if predicted == label_index:
            hit += 1
    return hit / len(cases)


def _load(path: Path) -> dict:
    return {label: rows for label, rows in
            json.loads(path.read_text(encoding="utf-8")).items()
            if label in PATTERNS and rows}


def run_gate(variants_path: str | Path | None = None,
             new_path: str | Path | None = None,
             seeds: int = 8) -> DataReport:
    """Run the two-arm gate: the original bank against the enlarged one.

    Args:
        variants_path: The frozen original bank; defaults to
            ``uq_home/vault/glyph_variants.json``. Held sets come only from
            here.
        new_path: The staged new variants; defaults to
            ``uq_home/vault/glyph_variants_new.json``. Training-side only.
        seeds: Split/noise repetitions.

    Returns:
        The :class:`DataReport`.
    """
    path = Path(variants_path or "uq_home/vault/glyph_variants.json")
    npath = Path(new_path or "uq_home/vault/glyph_variants_new.json")
    if not path.exists():
        return DataReport(skipped=True, reason=f"no variants file at {path}")
    if not npath.exists():
        return DataReport(skipped=True,
                          reason=f"no new-variants file at {npath}")
    variants = _load(path)
    fresh = _load(npath)
    if not variants:
        return DataReport(skipped=True, reason="original bank empty")
    new_flat = [(rows, label) for label, rows_list in fresh.items()
                for rows in rows_list]
    if not new_flat:
        return DataReport(skipped=True, reason="new-variants file empty")

    label_index = {label: i for i, label in enumerate(LABELS)}
    held_acc = {name: [] for name in _ARMS}
    canon_acc = {name: [] for name in _ARMS}

    for seed in range(seeds):
        rng = random.Random(seed)
        teach, held = [], []
        for label, rows_list in variants.items():
            shuffled = list(rows_list)
            rng.shuffle(shuffled)
            cut = max(1, len(shuffled) // 2) if len(shuffled) > 1 else 1
            teach += [(rows, label) for rows in shuffled[:cut]]
            held += [(rows, label_index[label]) for rows in shuffled[cut:]]

        base_set = []
        for label in variants:
            base_set.append((PATTERNS[label], label))
            for _ in range(_NOISE_COPIES):
                base_set.append((_noisy(PATTERNS[label], rng), label))
        base_set += teach
        control = [(_noisy(PATTERNS[label], rng), label_index[label])
                   for label in variants for _ in range(4)]

        arm_sets = {"baseline": base_set,
                    "enlarged": base_set + new_flat}
        for name in _ARMS:
            xs = [features_of(rows) for rows, _label in arm_sets[name]]
            ys = [label_index[label] for _rows, label in arm_sets[name]]
            net = _train(xs, ys, seed)
            held_acc[name].append(_accuracy(net, held))
            canon_acc[name].append(_accuracy(net, control))

    held_mean = {name: statistics.fmean(vals) for name, vals in held_acc.items()}
    canon_mean = {name: statistics.fmean(vals) for name, vals in canon_acc.items()}
    contrasts = [held_acc["enlarged"][i] - held_acc["baseline"][i]
                 for i in range(seeds)]
    delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of splits and noise draws.
    seed_sd = statistics.stdev(contrasts) if len(contrasts) > 1 else 0.0

    report = DataReport(
        held=held_mean, canon=canon_mean, new_count=len(new_flat),
        delta=delta, seed_sd=seed_sd,
        margin_ratio=(delta / seed_sd) if seed_sd > 0 else 0.0,
    )

    canon_sd = (statistics.stdev(canon_acc["enlarged"])
                if seeds > 1 else 0.0)
    if canon_mean["enlarged"] < canon_mean["baseline"] - canon_sd:
        report.reason = (
            f"the control failed: the enlarged arm scores "
            f"{canon_mean['enlarged']:.3f} on noisy canonicals against the "
            f"baseline's {canon_mean['baseline']:.3f}")
        return report
    if delta <= 0:
        report.reason = (f"{len(new_flat)} new variants do not lift the "
                         f"original held set ({delta:+.3f} at sd {seed_sd:.3f})")
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
        f"{len(new_flat)} new variants lift the original held set "
        f"{held_mean['baseline']:.3f} -> {held_mean['enlarged']:.3f} "
        f"({delta:+.3f}, {report.margin_ratio:.2f}x seed sd) with the "
        f"canonical control held ({canon_mean['baseline']:.3f} -> "
        f"{canon_mean['enlarged']:.3f})")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
