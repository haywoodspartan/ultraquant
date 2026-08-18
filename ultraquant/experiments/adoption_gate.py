"""Does the capacity result survive the deployed shape? The adoption gate.

§11.22 measured that hidden 64 lifts variant recognition — **on an 8-way
monolith over all the glyph families**, the protocol the variant gates
inherited from §11.20. The deployed forge builds nothing of that shape. It
builds four family experts of **two classes each** (lines, arrows, frames,
crosses) at hidden 32 — a far easier problem per expert — and a width that
under-fits an 8-way net may be perfectly adequate for a 2-way one. Adopting 64
in the forge on §11.22's evidence alone would be the silent constant bump that
gate explicitly refused to make.

So this gate re-asks the question at the deployed shape, 2x2: width {32
deployed, 64 candidate} x training corpus {deployed recipe, deployed recipe +
banked variants}, using the deployed corpus recipe (40 augmented samples per
prototype, 2-pixel noise, 40 epochs). Held-out variants are scored **within
their family expert** — a 2-way choice, chance 0.5, so numbers here are not
comparable to the monolith gates' (chance 0.125). The real disk cost is
measured beside it: the builtin library forged end-to-end at both widths, and
the store bytes reported.

**The criteria, written before the run.**

1. **Gain** — hidden 64 with variants must beat hidden 32 with variants on
   held-out variants by more than the seed-to-seed sd of that contrast.
2. **Control** — its noisy-canonical accuracy must not fall below the 32-arm's
   by more than one sd.
3. **Cost is stated, not gated** — four experts are a fixed set, not the
   scaling family; the bytes are reported for the record.

A FAIL here is a finding, not a disappointment: it would mean capacity binds
the monolith and not the family experts, and the deployed default survives
because it was measured, not because it was never questioned.

**It was run, and it FAILED — hidden 64 is worse at the deployed shape.**

| arm | held variants | noisy canon (control) |
|---|---:|---:|
| hidden 32 (deployed) | 0.784 | 1.000 |
| **hidden 32 + variants** | **0.797** | 0.996 |
| hidden 64 | 0.754 | 1.000 |
| hidden 64 + variants | 0.772 | 1.000 |

The contrast the gate was registered on — 64+variants vs 32+variants — is
**-0.026 at -1.06x seed sd**: the candidate width is not merely unhelpful at
the family shape, it is slightly worse, and the library it would forge costs
**82,820 bytes against 42,595** (+94%) for that regression. Capacity binds the
8-way monolith and not the 2-way family experts; §11.22's pass stops at the
shape it was measured on, and the deployed default survives measured.

Two notes for honesty. The control sits at or near ceiling (1.000/0.996)
because the deployed problem — two well-separated prototypes, forty augmented
samples each — genuinely is that easy, which is itself the explanation of the
verdict; it moved off 1.000 in one arm, so it is not structurally unable to
fail (§11.11's defect). And variant-teaching at this shape buys +0.013 at
hidden 32, roughly what it bought the monolith in §11.20 — the banked variants
remain data waiting on a better mechanism, now with part-based features as the
one §11.21 candidate still unmeasured.

Run it::

    python -m ultraquant.experiments.adoption_gate
"""

from __future__ import annotations

import json
import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ultraquant.forge.corpus import (BUILTIN_FAMILIES, BUILTIN_KEYWORDS,
                                     build_corpora, builtin_taxonomy,
                                     features_of)
from ultraquant.model.network import UltraQuantNet
from ultraquant.pattern.recognition import PATTERNS

__all__ = ["AdoptionReport", "run_gate"]

_WIDTHS = (32, 64)
_EPOCHS = 40
_LR = 0.05
_PER_CLASS = 40
_FLIPS = 2

#: label -> family, inverted from the deployed grouping.
_FAMILY_OF = {label: family
              for family, labels in BUILTIN_FAMILIES.items()
              for label in labels}


@dataclass
class AdoptionReport:
    """The 2x2 table, the byte cost, and the verdict.

    Attributes:
        passes: True only if hidden 64 earns adoption at the deployed shape.
        skipped: True when no variants file exists. Never a pass.
        held: ``(width, with_variants)`` -> held-variant accuracy (2-way).
        canon: ``(width, with_variants)`` -> noisy-canonical accuracy.
        store_bytes: width -> end-to-end forged library bytes.
        delta: 64-with-variants minus 32-with-variants, held variants.
        seed_sd: Seed-to-seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    skipped: bool = False
    held: dict = field(default_factory=dict)
    canon: dict = field(default_factory=dict)
    store_bytes: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, the cost, then the verdict."""
        if self.skipped:
            return f"SKIPPED - {self.reason}"
        lines = [f"{'arm':<22} {'held variants':>14} {'noisy canon (CTRL)':>19}"]
        for width in _WIDTHS:
            for taught in (False, True):
                name = f"hidden {width}" + ("+variants" if taught else "")
                key = (width, taught)
                lines.append(f"{name:<22} {self.held[key]:>14.3f} "
                             f"{self.canon[key]:>19.3f}")
        if self.store_bytes:
            b32 = self.store_bytes.get(32, 0)
            b64 = self.store_bytes.get(64, 0)
            lines.append(f"forged library bytes: hidden 32 = {b32:,}, "
                         f"hidden 64 = {b64:,} ({b64 - b32:+,})")
        lines += [
            f"delta (64+variants vs 32+variants) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _noisy(rows, rng, flips=_FLIPS):
    grid = [list(row) for row in rows]
    for _ in range(flips):
        y = rng.randrange(5)
        x = rng.randrange(5)
        grid[y][x] = "#" if grid[y][x] == "." else "."
    return ["".join(row) for row in grid]


def _train_family(corpus, extra, seed, width):
    """One family expert at the deployed recipe, optionally with variants."""
    xs = list(corpus.xs) + [features_of(rows) for rows, _idx in extra]
    ys = list(corpus.ys) + [idx for _rows, idx in extra]
    net = UltraQuantNet(input_dim=30, hidden_dims=[width],
                        num_classes=len(corpus.labels), seed=seed)
    order = list(range(len(xs)))
    rng = random.Random(seed)
    for _ in range(_EPOCHS):
        rng.shuffle(order)
        net.train_batch([xs[i] for i in order], [ys[i] for i in order], _LR)
    return net


def _measure_store_bytes() -> dict:
    """Forge the builtin library end-to-end at each width; report the bytes.

    The pure-Python tier, because it is the reference and the byte count does
    not depend on the tier.
    """
    from ultraquant.forge.forge import ModelForge

    corpora = build_corpora(builtin_taxonomy(), n_per_class=_PER_CLASS,
                            flips=_FLIPS, seed=0, keywords=BUILTIN_KEYWORDS)
    sizes = {}
    for width in _WIDTHS:
        root = Path(tempfile.mkdtemp(prefix=f"uq_adopt_{width}_"))
        try:
            forge = ModelForge(root, seed=0, hidden=width, tier="python")
            report = forge.build(corpora, epochs=_EPOCHS, lr=_LR,
                                 batch_size=16)
            sizes[width] = report.store_bytes
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return sizes


def run_gate(variants_path: str | Path | None = None,
             seeds: int = 8, measure_cost: bool = True) -> AdoptionReport:
    """Run the 2x2 gate at the deployed family-expert shape.

    Args:
        variants_path: Defaults to ``uq_home/vault/glyph_variants.json``.
        seeds: Split/noise repetitions.
        measure_cost: Also forge the library at both widths for the byte cost.

    Returns:
        The :class:`AdoptionReport`.
    """
    path = Path(variants_path or "uq_home/vault/glyph_variants.json")
    if not path.exists():
        return AdoptionReport(skipped=True,
                              reason=f"no variants file at {path}")
    variants = {label: rows for label, rows in
                json.loads(path.read_text(encoding="utf-8")).items()
                if label in PATTERNS and rows}
    if not variants:
        return AdoptionReport(skipped=True, reason="variants file empty")

    held_acc = {(w, t): [] for w in _WIDTHS for t in (False, True)}
    canon_acc = {(w, t): [] for w in _WIDTHS for t in (False, True)}

    for seed in range(seeds):
        rng = random.Random(seed)
        corpora = {c.name: c for c in build_corpora(
            builtin_taxonomy(), n_per_class=_PER_CLASS, flips=_FLIPS,
            seed=seed, keywords=BUILTIN_KEYWORDS)}
        label_pos = {family: {label: i for i, label in
                              enumerate(corpora[family].labels)}
                     for family in corpora}

        teach = {family: [] for family in corpora}
        held = {family: [] for family in corpora}
        for label, rows_list in variants.items():
            family = _FAMILY_OF[label]
            idx = label_pos[family][label]
            shuffled = list(rows_list)
            rng.shuffle(shuffled)
            cut = max(1, len(shuffled) // 2) if len(shuffled) > 1 else 1
            teach[family] += [(rows, idx) for rows in shuffled[:cut]]
            held[family] += [(rows, idx) for rows in shuffled[cut:]]

        control = {family: [(_noisy(PATTERNS[label], rng), pos)
                            for label, pos in label_pos[family].items()
                            for _ in range(4)]
                   for family in corpora}

        for width in _WIDTHS:
            for taught in (False, True):
                hits = total = c_hits = c_total = 0
                for family, corpus in corpora.items():
                    extra = teach[family] if taught else []
                    net = _train_family(corpus, extra, seed, width)
                    for rows, idx in held[family]:
                        predicted, _conf = net.predict(features_of(rows))
                        hits += int(predicted == idx)
                        total += 1
                    for rows, idx in control[family]:
                        predicted, _conf = net.predict(features_of(rows))
                        c_hits += int(predicted == idx)
                        c_total += 1
                held_acc[(width, taught)].append(hits / max(total, 1))
                canon_acc[(width, taught)].append(c_hits / max(c_total, 1))

    report = AdoptionReport(
        held={k: statistics.fmean(v) for k, v in held_acc.items()},
        canon={k: statistics.fmean(v) for k, v in canon_acc.items()},
        store_bytes=_measure_store_bytes() if measure_cost else {},
    )

    contrasts = [held_acc[(64, True)][i] - held_acc[(32, True)][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of splits and noise draws.
    report.seed_sd = statistics.stdev(contrasts) if len(contrasts) > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    canon_sd = (statistics.stdev(canon_acc[(64, True)])
                if seeds > 1 else 0.0)
    if report.canon[(64, True)] < report.canon[(32, True)] - canon_sd:
        report.reason = (
            f"the control failed: hidden 64 scores "
            f"{report.canon[(64, True)]:.3f} on noisy canonicals against "
            f"hidden 32's {report.canon[(32, True)]:.3f}")
        return report
    if report.delta <= 0:
        report.reason = (
            "hidden 64 does not beat the deployed width on held-out variants "
            f"at the family shape ({report.delta:+.3f}); capacity binds the "
            "monolith, not the family experts, and the deployed default "
            "survives measured")
        return report
    if report.seed_sd <= 0.0:
        report.reason = (f"every split gave the same contrast "
                         f"({report.delta:+.3f}); nothing varied")
        return report
    if report.margin_ratio <= 1.0:
        report.reason = (
            f"gain {report.delta:+.3f} is {report.margin_ratio:.2f}x the "
            f"seed sd ({report.seed_sd:.3f}); within noise, the deployed "
            "default survives measured")
        return report
    report.passes = True
    report.reason = (
        f"hidden 64 lifts held-variant recognition at the deployed shape "
        f"{report.held[(32, True)]:.3f} -> {report.held[(64, True)]:.3f} "
        f"({report.delta:+.3f}, {report.margin_ratio:.2f}x seed sd) with the "
        f"canonical control held ({report.canon[(32, True)]:.3f} -> "
        f"{report.canon[(64, True)]:.3f})")
    return report


if __name__ == "__main__":                        # pragma: no cover
    import sys

    print(run_gate(sys.argv[1] if len(sys.argv) > 1 else None).as_text())
