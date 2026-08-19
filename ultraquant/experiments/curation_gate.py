"""Reader-curated variants against the label-noise hypothesis. The gate.

§11.27 closed the variant chain with every training-side suspect dead and
one question standing: is the ~0.54–0.62 ceiling the *data's* fault — does
the shift-Jaccard validator admit drawings a reader would not call their
label? Answering needs an oracle independent of the validator and of the
drawing models. This gate uses one, and names it honestly: **the curator is
Claude Fable 5, not a human** — an independent reader, blind-protocolled,
its verdicts stored beside the banks for any human to spot-check.

**The blind protocol.** All banked variants (both banks pooled) are exported
shuffled and *unlabeled*. The reader sees each 5x5 grid cold, with only the
eight canonical glyphs as a reference card — exactly what a human curator
would have — and names what it reads as: one of the eight labels, or
"none". The readings are joined against the claimed labels afterwards, by
script. KEEP means the blind reading matched the claim; everything else is
REJECT, with the misreading kept for the record. The reader never sees a
claim before judging, so expectation cannot anchor the verdicts.

**The criteria, written before a single glyph was read.**

1. **Label-noise claim** — on the held half of the KEPT variants (the only
   items where "correct" has a reader's endorsement), training on kept-only
   variants must beat training on all variants by more than the seed-to-seed
   sd of that contrast. Removing reader-rejected items from training is the
   treatment; if they were mislabeled noise, removing them helps.
2. **Concentration diagnostic, reported not gated** — under the historical
   protocol (train on a half-split of everything), accuracy on
   reader-REJECTED held items vs reader-KEPT held items. If the ceiling was
   the data, the errors should concentrate in the rejected pile.
3. **Control** — noisy canonicals must hold within one sd across arms.
4. The curation itself is reported: kept/rejected counts per label, and the
   rejection reasons (what the rejected items read as instead).

If claim 1 fails AND the diagnostic shows no concentration, then the reader
agrees with the validator, the data was never the problem, and the ceiling
is the honest capability of these features and this net on this task —
recorded as such, with no further suspect to chase.

**It was run, and it PASSED — the last suspect is confirmed, not buried.**

The blind reader rejected **31 of 120** banked variants (26%). The
disagreements concentrate where the geometry predicts: the arrow families
lose the most (claimed arrows reading as plus, diamond, or nothing — arrows
and plus share the full middle bar, so a sloppy taper collapses the
distinction), then plus -> none and stripes_v -> none. Then the gate:

| arm | held (kept-only) | noisy canon (control) |
|---|---:|---:|
| trained on kept only | **0.633** | 0.840 |
| trained on everything | 0.561 | 0.797 |

**+0.072 at exactly 2.00x seed sd, control improved.** And the
concentration diagnostic corroborates from the other side: under the
historical protocol, reader-kept held items score 0.580 while
reader-rejected items score **0.406** — the ceiling's error mass sits in
precisely the pile the reader refused, measured before any retraining.

Three honesty notes. First, the oracle: these verdicts are Claude Fable 5's
readings, not a human's — blind to the claims, independent of the validator
and of every drawing model, stored in full (``glyph_curation.json``,
including what each rejected item read as) for any human to overturn.
Second, ambiguity was *part* of the ceiling, not all of it: endorsed data
still only reaches 0.633, so the features-and-capacity limits measured in
§11.20–§11.24 remain real. Third, the validator itself is unchanged — the
shift-Jaccard check admitted these 31, and tightening it is a new mechanism
that would need its own gate against the curation file as ground truth;
until then, future variant training should train on the curated bank, and
the curation file is the standing record of which variants earned belief.

Run it::

    python -m ultraquant.experiments.curation_gate
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

__all__ = ["CurationReport", "run_gate"]

_NOISE_COPIES = 12
_NOISE_FLIPS = 2
_EPOCHS = 60
_LR = 0.05
_HIDDEN = (64,)


@dataclass
class CurationReport:
    """The curation summary, the two arms, and the verdict.

    Attributes:
        passes: Claim 1 under the pre-registered criterion.
        skipped: True when banks or curation are missing. Never a pass.
        kept: Variants the reader endorsed, per label.
        rejected: Variants the reader rejected, per label.
        held_kept_arm: Accuracy on kept-held, trained kept-only.
        held_all_arm: Accuracy on kept-held, trained on everything.
        rejected_acc: Historical-protocol accuracy on rejected held items.
        kept_acc: Historical-protocol accuracy on kept held items.
        canon: Arm name -> noisy-canonical control.
        delta: kept-arm minus all-arm on the kept held set.
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    skipped: bool = False
    kept: dict = field(default_factory=dict)
    rejected: dict = field(default_factory=dict)
    held_kept_arm: float = 0.0
    held_all_arm: float = 0.0
    rejected_acc: float = 0.0
    kept_acc: float = 0.0
    canon: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The counts, the arms, the diagnostic, then the verdict."""
        if self.skipped:
            return f"SKIPPED - {self.reason}"
        n_kept = sum(len(v) for v in self.kept.values())
        n_rej = sum(len(v) for v in self.rejected.values())
        lines = [
            f"curation: {n_kept} kept, {n_rej} rejected "
            f"({n_rej / max(n_kept + n_rej, 1):.0%} of the banks)",
            f"{'arm':<26} {'held (kept-only)':>16} {'noisy canon':>12}",
            f"{'trained on kept only':<26} {self.held_kept_arm:>16.3f} "
            f"{self.canon.get('kept', 0.0):>12.3f}",
            f"{'trained on everything':<26} {self.held_all_arm:>16.3f} "
            f"{self.canon.get('all', 0.0):>12.3f}",
            f"concentration diagnostic (historical protocol): "
            f"kept items {self.kept_acc:.3f} vs rejected items "
            f"{self.rejected_acc:.3f}",
            f"delta (kept-trained vs all-trained) {self.delta:+.3f}   "
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


def _load_bank(path: Path) -> dict:
    return {label: rows for label, rows in
            json.loads(path.read_text(encoding="utf-8")).items()
            if label in PATTERNS and rows}


def _anchor(rng):
    anchor = []
    for label in PATTERNS:
        anchor.append((PATTERNS[label], label))
        for _ in range(_NOISE_COPIES):
            anchor.append((_noisy(PATTERNS[label], rng), label))
    return anchor


def run_gate(variants_path: str | Path | None = None,
             new_path: str | Path | None = None,
             curation_path: str | Path | None = None,
             seeds: int = 8) -> CurationReport:
    """Run the curation gate over the pooled banks and the reader verdicts.

    Args:
        variants_path: The original bank; defaults to
            ``uq_home/vault/glyph_variants.json``.
        new_path: The staged bank; defaults to
            ``uq_home/vault/glyph_variants_new.json``.
        curation_path: The reader's verdicts; defaults to
            ``uq_home/vault/glyph_curation.json``. Format:
            ``{"oracle": ..., "protocol": ..., "verdicts": [
            {"label": claim, "rows": [...], "keep": bool,
            "read_as": label-or-none}, ...]}``.
        seeds: Split/noise repetitions.

    Returns:
        The :class:`CurationReport`.
    """
    opath = Path(variants_path or "uq_home/vault/glyph_variants.json")
    npath = Path(new_path or "uq_home/vault/glyph_variants_new.json")
    cpath = Path(curation_path or "uq_home/vault/glyph_curation.json")
    for p in (opath, npath, cpath):
        if not p.exists():
            return CurationReport(skipped=True, reason=f"missing {p.name}")

    curation = json.loads(cpath.read_text(encoding="utf-8"))
    verdicts = curation.get("verdicts", [])
    if not verdicts:
        return CurationReport(skipped=True, reason="curation file empty")

    verdict_by_glyph = {(v["label"], tuple(v["rows"])): v for v in verdicts}
    kept: dict[str, list] = {}
    rejected: dict[str, list] = {}
    missing = 0
    for bank in (_load_bank(opath), _load_bank(npath)):
        for label, rows_list in bank.items():
            for rows in rows_list:
                v = verdict_by_glyph.get((label, tuple(rows)))
                if v is None:
                    missing += 1
                    continue
                (kept if v["keep"] else rejected).setdefault(
                    label, []).append(rows)
    if missing:
        return CurationReport(
            skipped=True,
            reason=f"{missing} banked variants lack a verdict; curation "
                   "must cover the banks completely before the gate runs")
    if not kept or not rejected:
        return CurationReport(
            skipped=True,
            reason="curation kept or rejected nothing; the contrast "
                   "does not exist")

    label_index = {label: i for i, label in enumerate(LABELS)}
    kept_arm, all_arm = [], []
    canon_kept, canon_all = [], []
    rej_hits, rej_total, keep_hits, keep_total = 0, 0, 0, 0

    all_pool = {label: (kept.get(label, []) + rejected.get(label, []))
                for label in set(kept) | set(rejected)}

    for seed in range(seeds):
        rng = random.Random(seed)

        k_teach, k_held = [], []
        for label, rows_list in kept.items():
            shuffled = list(rows_list)
            rng.shuffle(shuffled)
            cut = max(1, len(shuffled) // 2) if len(shuffled) > 1 else 1
            k_teach += [(rows, label) for rows in shuffled[:cut]]
            k_held += [(rows, label_index[label]) for rows in shuffled[cut:]]

        rej_flat = [(rows, label) for label, rows_list in rejected.items()
                    for rows in rows_list]
        anchor = _anchor(rng)
        control = [(_noisy(PATTERNS[label], rng), label_index[label])
                   for label in PATTERNS for _ in range(4)]

        arms = {"kept": anchor + k_teach,
                "all": anchor + k_teach + rej_flat}
        nets = {}
        for name, train_set in arms.items():
            xs = [features_of(rows) for rows, _label in train_set]
            ys = [label_index[label] for _rows, label in train_set]
            nets[name] = _train(xs, ys, seed)
        kept_arm.append(_accuracy(nets["kept"], k_held))
        all_arm.append(_accuracy(nets["all"], k_held))
        canon_kept.append(_accuracy(nets["kept"], control))
        canon_all.append(_accuracy(nets["all"], control))

        # Diagnostic under the historical protocol: half-split everything,
        # score kept vs rejected held items separately.
        h_teach, h_held = [], []
        for label, rows_list in all_pool.items():
            shuffled = list(rows_list)
            rng.shuffle(shuffled)
            cut = max(1, len(shuffled) // 2) if len(shuffled) > 1 else 1
            h_teach += [(rows, label) for rows in shuffled[:cut]]
            h_held += [(rows, label) for rows in shuffled[cut:]]
        xs = [features_of(rows) for rows, _label in _anchor(rng) + h_teach]
        ys = [label_index[label] for _rows, label in _anchor(rng) + h_teach]
        hist = _train(xs, ys, seed)
        rejected_keys = {(label, tuple(rows))
                         for label, rows_list in rejected.items()
                         for rows in rows_list}
        for rows, label in h_held:
            predicted, _conf = hist.predict(features_of(rows))
            hit = int(predicted == label_index[label])
            if (label, tuple(rows)) in rejected_keys:
                rej_hits += hit
                rej_total += 1
            else:
                keep_hits += hit
                keep_total += 1

    contrasts = [kept_arm[i] - all_arm[i] for i in range(seeds)]
    delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of splits and noise draws.
    seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0

    report = CurationReport(
        kept=kept, rejected=rejected,
        held_kept_arm=statistics.fmean(kept_arm),
        held_all_arm=statistics.fmean(all_arm),
        rejected_acc=rej_hits / max(rej_total, 1),
        kept_acc=keep_hits / max(keep_total, 1),
        canon={"kept": statistics.fmean(canon_kept),
               "all": statistics.fmean(canon_all)},
        delta=delta, seed_sd=seed_sd,
        margin_ratio=(delta / seed_sd) if seed_sd > 0 else 0.0,
    )

    canon_sd = statistics.stdev(canon_kept) if seeds > 1 else 0.0
    if report.canon["kept"] < report.canon["all"] - canon_sd:
        report.reason = (
            f"the control failed: kept-trained scores "
            f"{report.canon['kept']:.3f} against all-trained "
            f"{report.canon['all']:.3f}")
        return report
    if delta <= 0:
        report.reason = (
            f"removing reader-rejected variants does not help "
            f"({delta:+.3f} at sd {seed_sd:.3f}); the reader's rejections "
            "were not the training poison")
        return report
    if seed_sd <= 0.0:
        report.reason = (f"every split gave the same contrast ({delta:+.3f});"
                         " nothing varied")
        return report
    if report.margin_ratio <= 1.0:
        report.reason = (f"gain {delta:+.3f} is {report.margin_ratio:.2f}x "
                         f"the seed sd ({seed_sd:.3f})")
        return report
    report.passes = True
    report.reason = (
        f"label noise confirmed: dropping reader-rejected variants lifts "
        f"kept-held recognition {report.held_all_arm:.3f} -> "
        f"{report.held_kept_arm:.3f} ({delta:+.3f}, "
        f"{report.margin_ratio:.2f}x seed sd) with the control held")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
