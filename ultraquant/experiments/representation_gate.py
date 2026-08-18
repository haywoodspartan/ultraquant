"""Does a translation-capable representation unlock the variant data? A gate.

§11.20 distilled 53 validated glyph variants from a local model and found that
training on them did not transfer: **+0.013 at 0.23x seed sd**. The diagnosis
was the representation — 25 raw pixels plus row means, so a corner plus shares
almost no feature mass with the centred plus anchoring its family. The data was
kept for "the day a translation-capable representation gets its own gate".

This is that gate, and the representation is **centroid centering**
(:func:`~ultraquant.pattern.recognition.center_glyph`): translate every glyph so
its mass sits at the grid centre before extracting the same 30 features. It is
the O(1) form of the shift-search the variant validator already needed, applied
at feature time.

**The design is factorial** — two representations crossed with two training
sets, four arms trained identically otherwise — because the claim under test is
about an *interaction*, not a main effect. §11.20 showed variants alone do
nothing; centering alone might solve translation so completely that the
variants add nothing on top; only the four-way table attributes the gain.

**The criteria, written before the run.**

1. **Gain** — centred features with variants must beat §11.20's failed arm
   (positional features with variants) on held-out variants by more than the
   seed-to-seed sd.
2. **Control** — noisy-canonical accuracy under centred features must not fall
   below the positional baseline by more than one sd. Normalising position must
   not cost the family centres.
3. **Attribution is reported, not gated** — the centred-no-variants arm shows
   how much the representation does alone. If it captures most of the gain,
   the honest conclusion is that the representation was the missing piece and
   the variant data was mostly a probe that found it.

**It was run, and it FAILED — twice, with the two failures pointing different
ways.**

| arm | held variants | noisy canon (control) |
|---|---:|---:|
| positional | 0.435 | 0.773 |
| positional+variants | 0.448 | 0.758 |
| centered | 0.397 | 0.785 |
| centered+variants | 0.405 | 0.746 |

Centering does not merely fail to help; on the **off-centre subset it was built
for** it is worse than doing nothing: 0.319 positional → **0.200** centred.
Two measurements explain it. First, 33 of the 53 variants are already centred —
models draw centred — so the mechanism is the identity on 62% of the data.
Second, for the 20 off-centre variants, centring translates them into the grid
centre where all eight canonical families' pixels crowd; position is
normalised, and the residual scale/structure mismatch (a shrunk 3x3 plus is
*small*, not just displaced) now competes in the most contested region of the
grid.

The standard alternative was probed in the same session and failed harder:
**shift augmentation** (every training row also at ±2-cell offsets) collapsed
the canonical control from 0.711 to **0.348**. A 30→24→8 ternary network does
not have the capacity to represent eight families crossed with twenty-five
positions; asked to, it stops representing the families.

So at this scale, translation capability is not available by either standard
route: normalising position destroys structure discrimination, and learning
position multiplies the problem past the network's capacity. The variants
remain banked. What their difficulty actually is — measured, not assumed — is
**structure and scale, not translation**: 62% of them sit exactly where the
canonical sits and still only 44% are recognised. A representation for *that*
is a different mechanism again (scale normalisation, part-based features, or
simply capacity), each deserving its own gate.

Run it::

    python -m ultraquant.experiments.representation_gate
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from ultraquant.forge.corpus import centered_features, features_of
from ultraquant.model.network import UltraQuantNet
from ultraquant.pattern.recognition import LABELS, PATTERNS

__all__ = ["RepresentationReport", "run_gate"]

_NOISE_COPIES = 12
_NOISE_FLIPS = 2
_EPOCHS = 60
_LR = 0.05
_HIDDEN = (24,)

#: The four arms: (name, featurizer, include_variants).
_ARMS = (
    ("positional", features_of, False),
    ("positional+variants", features_of, True),
    ("centered", centered_features, False),
    ("centered+variants", centered_features, True),
)


@dataclass
class RepresentationReport:
    """The four-way table and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criterion.
        skipped: True when no variants file exists. Never a pass.
        held: Arm name -> held-variant accuracy.
        canon: Arm name -> noisy-canonical accuracy (the control).
        delta: Centred+variants minus positional+variants, on held variants.
        seed_sd: Seed-to-seed sd of that delta.
        margin_ratio: ``delta / seed_sd``.
        representation_share: How much of the gain the representation alone
            captures: (centered - positional) / (centered+variants -
            positional+variants), on held variants. Reported for attribution.
        reason: Plain-language verdict.
    """

    passes: bool = False
    skipped: bool = False
    held: dict = None
    canon: dict = None
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    representation_share: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        if self.skipped:
            return f"SKIPPED - {self.reason}"
        lines = [f"{'arm':<22} {'held variants':>14} {'noisy canon (CTRL)':>19}"]
        for name, _f, _v in _ARMS:
            lines.append(f"{name:<22} {self.held[name]:>14.3f} "
                         f"{self.canon[name]:>19.3f}")
        lines += [
            f"delta (centered+v vs positional+v) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            f"representation alone captures {self.representation_share:.0%} "
            "of the gain (reported, not gated)",
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
             seeds: int = 8) -> RepresentationReport:
    """Run the four-arm gate over the kept variants.

    Args:
        variants_path: Defaults to ``uq_home/vault/glyph_variants.json``.
        seeds: Split/noise repetitions.

    Returns:
        The :class:`RepresentationReport`.
    """
    path = Path(variants_path or "uq_home/vault/glyph_variants.json")
    if not path.exists():
        return RepresentationReport(skipped=True,
                                    reason=f"no variants file at {path}")
    variants = {label: rows for label, rows in
                json.loads(path.read_text(encoding="utf-8")).items()
                if label in PATTERNS and rows}
    if not variants:
        return RepresentationReport(skipped=True, reason="variants file empty")

    label_index = {label: i for i, label in enumerate(LABELS)}
    held_acc = {name: [] for name, _f, _v in _ARMS}
    canon_acc = {name: [] for name, _f, _v in _ARMS}
    deltas = []

    for seed in range(seeds):
        rng = random.Random(seed)
        teach, held = [], []
        for label, rows_list in variants.items():
            shuffled = list(rows_list)
            rng.shuffle(shuffled)
            cut = max(1, len(shuffled) // 2) if len(shuffled) > 1 else 1
            teach += [(rows, label) for rows in shuffled[:cut]]
            held += [(rows, label_index[label]) for rows in shuffled[cut:]]

        base_rows = []
        for label in variants:
            base_rows.append((PATTERNS[label], label))
            for _ in range(_NOISE_COPIES):
                base_rows.append((_noisy(PATTERNS[label], rng), label))
        control = [(_noisy(PATTERNS[label], rng), label_index[label])
                   for label in variants for _ in range(4)]

        per_seed = {}
        for name, featurize, include_variants in _ARMS:
            rows_set = list(base_rows) + (teach if include_variants else [])
            xs = [featurize(rows) for rows, _label in rows_set]
            ys = [label_index[label] for _rows, label in rows_set]
            net = _train(xs, ys, seed)
            per_seed[name] = _accuracy(net, held, featurize)
            held_acc[name].append(per_seed[name])
            canon_acc[name].append(_accuracy(net, control, featurize))
        deltas.append(per_seed["centered+variants"]
                      - per_seed["positional+variants"])

    delta = statistics.fmean(deltas)
    # stdev, not pstdev: seeds are a sample of splits and noise draws.
    seed_sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    held_mean = {name: statistics.fmean(vals) for name, vals in held_acc.items()}
    canon_mean = {name: statistics.fmean(vals) for name, vals in canon_acc.items()}

    gain_total = held_mean["centered+variants"] - held_mean["positional+variants"]
    gain_repr = held_mean["centered"] - held_mean["positional"]
    report = RepresentationReport(
        held=held_mean, canon=canon_mean, delta=delta, seed_sd=seed_sd,
        margin_ratio=(delta / seed_sd) if seed_sd > 0 else 0.0,
        representation_share=(gain_repr / gain_total) if gain_total else 0.0,
    )

    canon_sd = (statistics.stdev(canon_acc["centered+variants"])
                if len(canon_acc["centered+variants"]) > 1 else 0.0)
    if canon_mean["centered+variants"] < canon_mean["positional"] - canon_sd:
        report.reason = (
            f"the control failed: noisy canonicals fell "
            f"{canon_mean['positional']:.3f} -> "
            f"{canon_mean['centered+variants']:.3f} under centering")
        return report
    if delta <= 0:
        report.reason = f"no gain over the positional arm ({delta:+.3f})"
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
        f"held-variant recognition {held_mean['positional+variants']:.3f} -> "
        f"{held_mean['centered+variants']:.3f} ({delta:+.3f}, "
        f"{report.margin_ratio:.2f}x seed sd) with the canonical control held")
    return report


if __name__ == "__main__":                        # pragma: no cover
    import sys

    print(run_gate(sys.argv[1] if len(sys.argv) > 1 else None).as_text())
