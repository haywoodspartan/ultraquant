"""Variant load against a fixed or scaled anchor. The load gate.

§11.26 could not reject one-distribution, which killed §11.25's style
diagnosis and left one suspect: **load** — every heavily-loaded arm across
two gates landed in the same accuracy band regardless of style match, as if
the fixed canonical anchor (8 canonicals + 96 noisy copies) thins as variant
mass grows and the ternary net spreads over more modes than it can hold.

This gate makes load falsifiable, and crosses it with its own remedy in the
same run (§11.22's lesson: uncrossed arms mislead). Both banks pool into one
120-variant universe. Per seed, a fixed held set of 30 is drawn first; taught
sets of size N in {15, 30, 60, 90} come from the remaining 90. Two anchor
policies per N:

* **fixed** — the historical 96 noisy copies, so variant share grows with N
  (13% at N=15, 46% at N=90 — §11.25's failing arm sat here).
* **scaled** — noisy copies grown proportionally so the anchor:variant ratio
  stays the light-load baseline's (~3.85:1), capped at 48 copies per label.

**The claims, written before the run.**

1. **Load is SUPPORTED** if, under the fixed anchor, the best N beats the
   largest N on the held set by more than the seed-to-seed sd of that
   contrast — more data measurably worse than less, with style controlled by
   pooling.
2. **The anchor remedy is SUPPORTED** if, at the largest N, the scaled
   anchor beats the fixed anchor beyond seed sd on the held set.
3. Controls are reported per arm; a run where every control collapses
   decides nothing.

If neither claim holds, load dies like style died, and the honest state is
that the variant ceiling is none of: position, scale, parts, capacity,
style, or load — pointing at the data itself (validation letting through
ambiguous drawings) or irreducible task ambiguity.

**It was run, and neither claim survived.**

| arm | held (pooled) | noisy canon (control) |
|---|---:|---:|
| fixed/15 | 0.575 | 0.879 |
| fixed/30 | 0.571 | 0.855 |
| fixed/60 | 0.588 | 0.852 |
| fixed/90 | 0.588 | 0.820 |
| scaled/15 | 0.550 | 0.848 |
| scaled/30 | 0.575 | 0.848 |
| scaled/60 | 0.625 | 0.867 |
| scaled/90 | 0.604 | 0.871 |

Load: the fixed-anchor curve is **flat** — N=60 "beats" N=90 by +0.000 at sd
0.040. Six times the variant mass, no damage on a pooled target. Remedy:
+0.017 at sd 0.047 — the scaled anchor keeps the control healthier (0.871 vs
0.820 at N=90, the one visible anchor-thinning trace) but buys no held-set
gain beyond noise. NOT SUPPORTED, twice.

That closes the triangulation the last three gates form. §11.25 measured
adding 67 variants as -0.073 (at -1.05x sd) against the original-only held
set; §11.26 showed the styles transfer; this gate shows volume is harmless
on a pooled target. The §11.25 regression was a borderline-noise event on
the hardest sub-target (the original bank reads hardest: within-original
0.504 vs within-new 0.615), not a mechanism. The suspects are now all dead:
position (§11.21), scale (§11.22), parts (§11.24), style (§11.26), load
(here) — with capacity (§11.22) the one real but bounded lift. What remains
is the data itself: shift-Jaccard validation admits drawings a reader might
not call their label, and no training arrangement can learn past mislabeled
or ambiguous targets. Measuring *that* needs an oracle the system does not
have — a human-labeled sample of the banks — which is the learn-queue's
"ask the human second" territory, not another training gate.

Run it::

    python -m ultraquant.experiments.load_gate
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

__all__ = ["LoadReport", "run_gate"]

_NOISE_FLIPS = 2
_EPOCHS = 60
_LR = 0.05
_HIDDEN = (64,)
_HELD_SIZE = 30
_SIZES = (15, 30, 60, 90)
_BASE_COPIES = 12          # the historical anchor: 12 noisy copies per label
_ANCHOR_RATIO = 3.85       # anchor:variant ratio of the light-load baseline
_MAX_COPIES = 48


@dataclass
class LoadReport:
    """The size-by-anchor table and the two claim verdicts.

    Attributes:
        load_supported: Claim 1 — a smaller N beats the largest, fixed anchor.
        remedy_supported: Claim 2 — scaling the anchor rescues the largest N.
        valid: False when every control collapsed; no verdict.
        skipped: True when a bank is missing. Never a verdict.
        held: ``(policy, N)`` -> held accuracy on the pooled held set.
        canon: ``(policy, N)`` -> noisy-canonical control.
        best_fixed: The best N under the fixed anchor.
        load_delta: Best-N minus largest-N under the fixed anchor.
        load_sd: Seed sd of that contrast.
        remedy_delta: Scaled minus fixed at the largest N.
        remedy_sd: Seed sd of that contrast.
        reason: Plain-language verdict.
    """

    load_supported: bool = False
    remedy_supported: bool = False
    valid: bool = True
    skipped: bool = False
    held: dict = field(default_factory=dict)
    canon: dict = field(default_factory=dict)
    best_fixed: int = 0
    load_delta: float = 0.0
    load_sd: float = 0.0
    remedy_delta: float = 0.0
    remedy_sd: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, the contrasts, then the verdicts."""
        if self.skipped:
            return f"SKIPPED - {self.reason}"
        lines = [f"{'arm':<14} {'held (pooled)':>13} {'noisy canon (CTRL)':>19}"]
        for policy in ("fixed", "scaled"):
            for size in _SIZES:
                key = (policy, size)
                marker = ("  <- best fixed"
                          if policy == "fixed" and size == self.best_fixed
                          else "")
                lines.append(f"{policy}/{size:<8} {self.held[key]:>13.3f} "
                             f"{self.canon[key]:>19.3f}{marker}")
        lines += [
            f"load contrast (fixed: best N={self.best_fixed} vs N={_SIZES[-1]}) "
            f"{self.load_delta:+.3f} at sd {self.load_sd:.3f}",
            f"remedy contrast (N={_SIZES[-1]}: scaled vs fixed) "
            f"{self.remedy_delta:+.3f} at sd {self.remedy_sd:.3f}",
        ]
        if not self.valid:
            lines.append("INVALID - " + self.reason)
        else:
            lines.append(
                f"load {'SUPPORTED' if self.load_supported else 'NOT SUPPORTED'}"
                f" / remedy "
                f"{'SUPPORTED' if self.remedy_supported else 'NOT SUPPORTED'}"
                f" - {self.reason}")
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


def _load_bank(path: Path) -> dict:
    return {label: rows for label, rows in
            json.loads(path.read_text(encoding="utf-8")).items()
            if label in PATTERNS and rows}


def _anchor(rng, copies_per_label):
    anchor = []
    for label in PATTERNS:
        anchor.append((PATTERNS[label], label))
        for _ in range(copies_per_label):
            anchor.append((_noisy(PATTERNS[label], rng), label))
    return anchor


def run_gate(variants_path: str | Path | None = None,
             new_path: str | Path | None = None,
             seeds: int = 8) -> LoadReport:
    """Run the size-by-anchor factorial over the pooled banks.

    Args:
        variants_path: The original bank; defaults to
            ``uq_home/vault/glyph_variants.json``.
        new_path: The staged bank; defaults to
            ``uq_home/vault/glyph_variants_new.json``.
        seeds: Split/noise repetitions.

    Returns:
        The :class:`LoadReport`.
    """
    opath = Path(variants_path or "uq_home/vault/glyph_variants.json")
    npath = Path(new_path or "uq_home/vault/glyph_variants_new.json")
    if not opath.exists() or not npath.exists():
        return LoadReport(skipped=True,
                          reason=f"needs both banks ({opath.name}, "
                                 f"{npath.name})")
    pool_by_label: dict[str, list] = {}
    for bank in (_load_bank(opath), _load_bank(npath)):
        for label, rows_list in bank.items():
            pool_by_label.setdefault(label, []).extend(rows_list)
    pool = [(rows, label) for label, rows_list in pool_by_label.items()
            for rows in rows_list]
    if len(pool) < _HELD_SIZE + _SIZES[-1]:
        return LoadReport(skipped=True,
                          reason=f"pool too small ({len(pool)}) for held "
                                 f"{_HELD_SIZE} + taught {_SIZES[-1]}")

    label_index = {label: i for i, label in enumerate(LABELS)}
    held_acc = {(p, s): [] for p in ("fixed", "scaled") for s in _SIZES}
    canon_acc = {(p, s): [] for p in ("fixed", "scaled") for s in _SIZES}

    for seed in range(seeds):
        rng = random.Random(seed)
        shuffled = list(pool)
        rng.shuffle(shuffled)
        held = [(rows, label_index[label])
                for rows, label in shuffled[:_HELD_SIZE]]
        remainder = shuffled[_HELD_SIZE:]
        control = [(_noisy(PATTERNS[label], rng), label_index[label])
                   for label in PATTERNS for _ in range(4)]

        for size in _SIZES:
            taught = remainder[:size]
            for policy in ("fixed", "scaled"):
                if policy == "fixed":
                    copies = _BASE_COPIES
                else:
                    copies = min(_MAX_COPIES,
                                 max(_BASE_COPIES,
                                     round((_ANCHOR_RATIO * size - 8) / 8)))
                train_set = _anchor(rng, copies) + taught
                xs = [features_of(rows) for rows, _label in train_set]
                ys = [label_index[label] for _rows, label in train_set]
                net = _train(xs, ys, seed)
                held_acc[(policy, size)].append(_accuracy(net, held))
                canon_acc[(policy, size)].append(_accuracy(net, control))

    report = LoadReport(
        held={k: statistics.fmean(v) for k, v in held_acc.items()},
        canon={k: statistics.fmean(v) for k, v in canon_acc.items()},
    )

    if max(report.canon.values()) < 0.5:
        report.valid = False
        report.reason = "every control collapsed below 0.5; no verdict"
        return report

    largest = _SIZES[-1]
    report.best_fixed = max(
        _SIZES, key=lambda s: report.held[("fixed", s)])
    load_contrasts = [held_acc[("fixed", report.best_fixed)][i]
                      - held_acc[("fixed", largest)][i]
                      for i in range(seeds)]
    report.load_delta = statistics.fmean(load_contrasts)
    # stdev, not pstdev: the seeds are a sample of splits and noise draws.
    report.load_sd = (statistics.stdev(load_contrasts)
                      if seeds > 1 else 0.0)
    report.load_supported = (report.best_fixed != largest
                             and report.load_sd > 0
                             and report.load_delta > report.load_sd)

    remedy_contrasts = [held_acc[("scaled", largest)][i]
                        - held_acc[("fixed", largest)][i]
                        for i in range(seeds)]
    report.remedy_delta = statistics.fmean(remedy_contrasts)
    report.remedy_sd = (statistics.stdev(remedy_contrasts)
                        if seeds > 1 else 0.0)
    report.remedy_supported = (report.remedy_sd > 0
                               and report.remedy_delta > report.remedy_sd)

    bits = []
    if report.load_supported:
        bits.append(
            f"N={report.best_fixed} beats N={largest} by "
            f"{report.load_delta:+.3f} ({report.load_delta / report.load_sd:.2f}x sd) "
            "under the fixed anchor")
    else:
        bits.append(
            f"the largest load is not beaten beyond sd "
            f"({report.load_delta:+.3f} at sd {report.load_sd:.3f})")
    if report.remedy_supported:
        bits.append(
            f"scaling the anchor rescues N={largest} by "
            f"{report.remedy_delta:+.3f} "
            f"({report.remedy_delta / report.remedy_sd:.2f}x sd)")
    else:
        bits.append(
            f"anchor scaling does not rescue N={largest} "
            f"({report.remedy_delta:+.3f} at sd {report.remedy_sd:.3f})")
    report.reason = "; ".join(bits)
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
