"""Is "glyph variant" one distribution across drawing voices? The test.

§11.25 measured 67 new validated variants making old-style recognition
*worse*, and the post-hoc probes left one suspect standing:
off-distribution interference between drawing styles. This gate turns that
suspicion into a falsifiable hypothesis.

**The one-distribution hypothesis**: variants of the same eight glyphs are
one learnable distribution regardless of which model drew them, so a net
trained on one bank should recognise the other bank about as well as a bank
recognises its own held-out half.

Four arms at hidden 64, every training set anchored by the canonicals and
their noisy copies, a noisy-canonical control beside each:

* ``within-original`` — teach the original bank's half, test its other half.
* ``within-new`` — the same, inside the staged bank.
* ``cross original->new`` — teach the **whole** original bank, test the whole
  staged bank.
* ``cross new->original`` — the reverse.

The cross arms train on a *full* bank while the within arms train on half —
twice the data, biased **in favour** of transfer. A rejection under that bias
is conservative.

**The criterion, written before the run.** The hypothesis is REJECTED if each
cross arm falls below its matching within arm (the one sharing its test
distribution) by more than one seed-to-seed sd of that contrast, in **both**
directions. Controls must hold in every arm for the run to count at all.
This is a hypothesis test, not a capability gate: the honest outcomes are
REJECTED / NOT REJECTED, and either one aims the roadmap — rejected means
per-voice structure (and no more drawing tokens until it is respected);
not-rejected means §11.25's diagnosis was wrong and the interference lives
elsewhere.

**It was run, and the hypothesis was NOT rejected — §11.25's diagnosis dies.**

| arm | variant acc | noisy canon (control) |
|---|---:|---:|
| within-original | 0.504 | 0.879 |
| within-new | 0.615 | 0.859 |
| cross original->new | 0.556 | 0.824 |
| cross new->original | 0.460 | 0.809 |

Gaps: original->new **-0.059 at sd 0.115**, new->original **-0.044 at sd
0.092** — both inside one sd. Each bank recognises the other nearly as well
as its own held half, so the drawing voices are not meaningfully different
distributions, and the off-distribution-interference diagnosis §11.25 left
standing is measured wrong. The chain corrects itself again: this is the
second time a section's parting diagnosis died in the next section's gate
(§11.20's "position" died in §11.21; §11.25's "style" dies here).

What the numbers point at instead — stated as the next hypothesis, not a
finding: **load**. §11.25's enlarged arm (27 original + 67 new taught)
scored 0.466 on the original held set; this gate's cross arm trained on the
67 *alone* scores 0.460 on the whole original bank — adding the 27
same-style variants bought almost nothing, while every heavily-loaded arm
lands in the 0.46–0.56 band and the lightly-loaded within arms sit at their
band's top. The suspect is the fixed canonical anchor thinning as variants
take a larger share of training — §11.21's augmentation collapse was this at
25x. Its gate would vary taught-set size at fixed composition from the
pooled banks against a fixed pooled held set. Until then, the banks stay
separate and no more drawing tokens are spent: transfer works, so volume is
not the bottleneck either — *how much variant mass one net can absorb* is.

Run it::

    python -m ultraquant.experiments.distribution_gate
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from ultraquant.forge.corpus import features_of
from ultraquant.model.network import UltraQuantNet
from ultraquant.pattern.recognition import LABELS, PATTERNS

__all__ = ["DistributionReport", "run_gate"]

_NOISE_COPIES = 12
_NOISE_FLIPS = 2
_EPOCHS = 60
_LR = 0.05
_HIDDEN = (64,)

_ARMS = ("within-original", "within-new",
         "cross original->new", "cross new->original")


@dataclass
class DistributionReport:
    """The four-way table and the hypothesis verdict.

    Attributes:
        rejected: True when the one-distribution hypothesis is rejected.
        valid: False when a control failed and the run does not count.
        skipped: True when a bank is missing. Never a verdict.
        accuracy: Arm name -> variant accuracy on its test set.
        canon: Arm name -> noisy-canonical control.
        gaps: Direction -> (delta, seed_sd) for cross minus within.
        reason: Plain-language verdict.
    """

    rejected: bool = False
    valid: bool = True
    skipped: bool = False
    accuracy: dict = field(default_factory=dict)
    canon: dict = field(default_factory=dict)
    gaps: dict = field(default_factory=dict)
    reason: str = ""

    def as_text(self) -> str:
        """The table, the gaps, then the verdict."""
        if self.skipped:
            return f"SKIPPED - {self.reason}"
        lines = [f"{'arm':<22} {'variant acc':>12} {'noisy canon (CTRL)':>19}"]
        for name in _ARMS:
            lines.append(f"{name:<22} {self.accuracy[name]:>12.3f} "
                         f"{self.canon[name]:>19.3f}")
        for direction, (delta, sd) in self.gaps.items():
            lines.append(f"gap {direction}: {delta:+.3f} at sd {sd:.3f}")
        if not self.valid:
            lines.append("INVALID - " + self.reason)
        else:
            lines.append(("REJECTED - " if self.rejected
                          else "NOT REJECTED - ") + self.reason)
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


def _half_split(bank: dict, rng, label_index):
    teach, held = [], []
    for label, rows_list in bank.items():
        shuffled = list(rows_list)
        rng.shuffle(shuffled)
        cut = max(1, len(shuffled) // 2) if len(shuffled) > 1 else 1
        teach += [(rows, label) for rows in shuffled[:cut]]
        held += [(rows, label_index[label]) for rows in shuffled[cut:]]
    return teach, held


def _flat(bank: dict, label_index, as_labels: bool):
    if as_labels:
        return [(rows, label) for label, rows_list in bank.items()
                for rows in rows_list]
    return [(rows, label_index[label]) for label, rows_list in bank.items()
            for rows in rows_list]


def run_gate(variants_path: str | Path | None = None,
             new_path: str | Path | None = None,
             seeds: int = 8) -> DistributionReport:
    """Run the four-arm hypothesis test over the two banks.

    Args:
        variants_path: The original bank; defaults to
            ``uq_home/vault/glyph_variants.json``.
        new_path: The staged bank; defaults to
            ``uq_home/vault/glyph_variants_new.json``.
        seeds: Split/noise repetitions.

    Returns:
        The :class:`DistributionReport`.
    """
    opath = Path(variants_path or "uq_home/vault/glyph_variants.json")
    npath = Path(new_path or "uq_home/vault/glyph_variants_new.json")
    if not opath.exists() or not npath.exists():
        return DistributionReport(
            skipped=True,
            reason=f"needs both banks ({opath.name}, {npath.name})")
    original = _load(opath)
    fresh = _load(npath)
    if not original or not fresh:
        return DistributionReport(skipped=True, reason="a bank is empty")

    label_index = {label: i for i, label in enumerate(LABELS)}
    acc = {name: [] for name in _ARMS}
    canon = {name: [] for name in _ARMS}

    for seed in range(seeds):
        rng = random.Random(seed)
        base = []
        for label in PATTERNS:
            base.append((PATTERNS[label], label))
            for _ in range(_NOISE_COPIES):
                base.append((_noisy(PATTERNS[label], rng), label))
        control = [(_noisy(PATTERNS[label], rng), label_index[label])
                   for label in PATTERNS for _ in range(4)]

        o_teach, o_held = _half_split(original, rng, label_index)
        n_teach, n_held = _half_split(fresh, rng, label_index)

        arm_data = {
            "within-original": (base + o_teach, o_held),
            "within-new": (base + n_teach, n_held),
            "cross original->new": (base + _flat(original, label_index, True),
                                    _flat(fresh, label_index, False)),
            "cross new->original": (base + _flat(fresh, label_index, True),
                                    _flat(original, label_index, False)),
        }
        for name in _ARMS:
            teach_set, test_set = arm_data[name]
            xs = [features_of(rows) for rows, _label in teach_set]
            ys = [label_index[label] for _rows, label in teach_set]
            net = _train(xs, ys, seed)
            acc[name].append(_accuracy(net, test_set))
            canon[name].append(_accuracy(net, control))

    report = DistributionReport(
        accuracy={n: statistics.fmean(v) for n, v in acc.items()},
        canon={n: statistics.fmean(v) for n, v in canon.items()},
    )

    # Controls first: a run with a failed control decides nothing.
    canon_floor = min(report.canon.values())
    within_ref = max(report.canon["within-original"],
                     report.canon["within-new"])
    canon_sds = {n: (statistics.stdev(v) if seeds > 1 else 0.0)
                 for n, v in canon.items()}
    for name in _ARMS:
        if report.canon[name] < within_ref - 2 * max(canon_sds.values()):
            report.valid = False
            report.reason = (f"the control failed in arm '{name}' "
                             f"({report.canon[name]:.3f}); no verdict")
            return report

    pairs = {
        "original->new": ("cross original->new", "within-new"),
        "new->original": ("cross new->original", "within-original"),
    }
    below = {}
    for direction, (cross_arm, within_arm) in pairs.items():
        contrasts = [acc[cross_arm][i] - acc[within_arm][i]
                     for i in range(seeds)]
        delta = statistics.fmean(contrasts)
        # stdev, not pstdev: the seeds are a sample of splits and noise draws.
        sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
        report.gaps[direction] = (delta, sd)
        below[direction] = sd > 0 and delta < -sd

    if all(below.values()):
        report.rejected = True
        report.reason = (
            "one distribution is rejected: each bank recognises its own "
            "held-out half better than the other bank recognises it, beyond "
            "seed sd in both directions, despite the cross arms training on "
            "twice the data")
    else:
        holding = [d for d, b in below.items() if not b]
        report.reason = (
            f"not rejected: transfer holds within noise in "
            f"{' and '.join(holding)}; §11.25's interference diagnosis "
            "needs another explanation")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
