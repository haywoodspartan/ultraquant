"""The tightened validator against the reader's verdicts. The gate.

§11.28 measured the shift-Jaccard validator admitting 31 of 120 banked
variants that a blind independent reader refuses — and said tightening it
"is a new mechanism that would need its own gate against the curation
file as ground truth". This is that gate, and the mechanism is
:func:`~ultraquant.forge.glyph_distill.strict_contradiction`: the old
nearest-anchor rule plus the two checks whose absence the rejections
decompose onto — a **floor** (18 of the 31 read as *nothing*: nearest,
but weakly, and "nearest" was the whole old test) and a **margin** (13
read as a *different* label, most having won their claim by a whisker).

**Calibration and evaluation are split**, because tuning thresholds on
the data that then grades them manufactures a pass: per split seed, the
120 verdicts are stratified-halved; the (margin, floor) grid is searched
on the calibration half (maximise reject-recall subject to keeping >=95%
of the reader-kept items); the chosen point is scored on the held half
it never saw. Five split seeds, means and sds reported.

The baseline states itself: the deployed validator accepted all 120 by
construction, so it keeps 100% and catches 0% — any recall at all beats
it, which is why the bar is set against the reader instead.

**The criteria, written before the run.**

1. **Recall** — the calibrated validator must reject at least half of the
   reader-rejected items on the held half (mean recall >= 0.50).
2. **Keep-rate** — it must keep at least 95% of the reader-kept items on
   the held half (mean >= 0.95); a validator that buys recall by shedding
   good variants has just moved the curation cost.
3. **Stability is reported** — the chosen (margin, floor) per seed; a
   point that jumps across splits is a warning even when the means pass.

**It was run, and it FAILED — the family cannot see what the reader sees.**

Held-half rejection recall: **0.050** against the 0.50 bar. Keep rate
0.947, under its own 0.95 bar — even five percent recall was bought by
shedding good variants. And the chosen operating points never stabilised:
two of five splits found NO grid point keeping 95% on calibration and
fell back to the plain rule, the rest chose floors of 0.50–0.55 with zero
margin. The rejected items do not sit lower in shift-Jaccard agreement
than the kept ones, and no threshold on that axis separates them: the
reader rejects on **shape gestalt** — whether a taper points, whether a
silhouette reads as its label — which Jaccard-under-shift does not
encode at all.

So §11.28's registered question closes negative, and the consequence is
the protocol, made permanent rather than stopgap: mechanical validation
catches the gross errors at drawing time, and **admission to the
training bank goes through a blind reading**. `strict_contradiction`
stays in the codebase as the third resident of the built-but-not-adopted
shelf, beside `center_glyph` (§11.21) and `scale_normalize` (§11.22):
correct at what it does, tested, and measured out of a job.

Run it::

    python -m ultraquant.experiments.validator_gate
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from ultraquant.forge.glyph_distill import strict_contradiction

__all__ = ["ValidatorReport", "run_gate"]

_MARGINS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
_FLOORS = (0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
_KEEP_BAR = 0.95
_RECALL_BAR = 0.50


@dataclass
class ValidatorReport:
    """The held-half scores, the chosen operating points, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        skipped: True when the curation file is missing. Never a pass.
        keep_rate: Mean held-half keep rate over reader-kept items.
        recall: Mean held-half rejection recall over reader-rejected items.
        keep_sd: Seed sd of the keep rate.
        recall_sd: Seed sd of the recall.
        chosen: Per-seed ``(margin, floor)`` the calibration selected.
        reason: Plain-language verdict.
    """

    passes: bool = False
    skipped: bool = False
    keep_rate: float = 0.0
    recall: float = 0.0
    keep_sd: float = 0.0
    recall_sd: float = 0.0
    chosen: list = field(default_factory=list)
    reason: str = ""

    def as_text(self) -> str:
        """The scores, the operating points, then the verdict."""
        if self.skipped:
            return f"SKIPPED - {self.reason}"
        lines = [
            f"held-half keep rate (reader-kept):     "
            f"{self.keep_rate:.3f} (sd {self.keep_sd:.3f}, bar "
            f">={_KEEP_BAR})",
            f"held-half rejection recall (rejected): "
            f"{self.recall:.3f} (sd {self.recall_sd:.3f}, bar "
            f">={_RECALL_BAR})",
            "chosen (margin, floor) per seed: "
            + ", ".join(f"({m:.2f}, {f:.2f})" for m, f in self.chosen),
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _score(items: list, margin: float, floor: float) -> tuple[float, float]:
    """(keep rate over kept items, rejection recall over rejected items)."""
    kept_ok = kept_total = caught = rejected_total = 0
    for verdict in items:
        rejected = strict_contradiction(verdict["rows"], verdict["label"],
                                        margin=margin, floor=floor)
        if verdict["keep"]:
            kept_total += 1
            kept_ok += int(rejected is None)
        else:
            rejected_total += 1
            caught += int(rejected is not None)
    return (kept_ok / max(kept_total, 1),
            caught / max(rejected_total, 1))


def run_gate(curation_path: str | Path | None = None,
             seeds: int = 5) -> ValidatorReport:
    """Calibrate on one half of the reader's verdicts, score on the other.

    Args:
        curation_path: Defaults to ``uq_home/vault/glyph_curation.json``.
        seeds: Stratified split repetitions.

    Returns:
        The :class:`ValidatorReport`.
    """
    path = Path(curation_path or "uq_home/vault/glyph_curation.json")
    if not path.exists():
        return ValidatorReport(skipped=True,
                               reason=f"no curation file at {path}")
    verdicts = json.loads(path.read_text(encoding="utf-8")).get(
        "verdicts", [])
    kept_pool = [v for v in verdicts if v["keep"]]
    rejected_pool = [v for v in verdicts if not v["keep"]]
    if not kept_pool or not rejected_pool:
        return ValidatorReport(skipped=True,
                               reason="curation holds no contrast")

    keep_scores, recall_scores, chosen = [], [], []
    for seed in range(seeds):
        rng = random.Random(seed)
        kept = list(kept_pool)
        rejected = list(rejected_pool)
        rng.shuffle(kept)
        rng.shuffle(rejected)
        cal = kept[:len(kept) // 2] + rejected[:len(rejected) // 2]
        held = kept[len(kept) // 2:] + rejected[len(rejected) // 2:]

        best = None
        for margin in _MARGINS:
            for floor in _FLOORS:
                keep_rate, recall = _score(cal, margin, floor)
                if keep_rate < _KEEP_BAR:
                    continue
                candidate = (recall, keep_rate, -margin, -floor)
                if best is None or candidate > best[0]:
                    best = (candidate, margin, floor)
        if best is None:
            # No grid point keeps enough on this calibration half; score
            # the plain rule so the seed is a 0-recall entry, not a hole.
            margin, floor = 0.0, 0.0
        else:
            _c, margin, floor = best
        chosen.append((margin, floor))
        keep_rate, recall = _score(held, margin, floor)
        keep_scores.append(keep_rate)
        recall_scores.append(recall)

    report = ValidatorReport(
        keep_rate=statistics.fmean(keep_scores),
        recall=statistics.fmean(recall_scores),
        # stdev, not pstdev: the seeds are a sample of stratified splits.
        keep_sd=(statistics.stdev(keep_scores) if seeds > 1 else 0.0),
        recall_sd=(statistics.stdev(recall_scores) if seeds > 1 else 0.0),
        chosen=chosen,
    )

    if report.keep_rate < _KEEP_BAR:
        report.reason = (
            f"the keep bar failed: {report.keep_rate:.3f} of reader-kept "
            f"items survive (bar {_KEEP_BAR}); recall was bought by "
            "shedding good variants")
        return report
    if report.recall < _RECALL_BAR:
        report.reason = (
            f"recall {report.recall:.3f} is below the {_RECALL_BAR} bar: "
            "the margin-and-floor family cannot see what the reader sees, "
            "and the validator stays as it is")
        return report
    report.passes = True
    report.reason = (
        f"the calibrated validator catches {report.recall:.3f} of the "
        f"reader-rejected items while keeping {report.keep_rate:.3f} of "
        f"the kept (baseline: 0.000 caught by construction)")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
