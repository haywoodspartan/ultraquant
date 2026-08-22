"""An encoder that refuses its own output. The refusal gate.

§11.101 measured what ternary conversion does to a 27-block tower:
the projected output ends at cosine **0.0185** to the f16 original -
not degraded, uncorrelated. **And the tower did not notice.** Every
activation stayed finite, every shape stayed right, and it emitted
plausible numbers all the way through. The only thing that knew was
the f16 reference, which in a real deployment does not exist.

§11.102 solved the same problem for the prediction layer, where the
output IS a distribution and its entropy is available for the
asking. An encoder's output is not: it is a list of real vectors,
and there is no softmax to take the entropy of.

**The attention rows are distributions, and they are the only
genuine ones inside a transformer.** Each is a softmax over
positions; its entropy, normalised by `log2(seq)`, says how much
structure the tower found. That is computable from the forward pass
alone, which is the whole requirement.

**Measured against what production code actually does**, which is
check for non-finite values. §11.101 already showed that fires zero
times on a completely destroyed tower, and criterion 3 confirms it
here rather than citing it.

**One-class by construction.** The band is fitted on healthy runs
only, because that is the deployment situation - you have the model
you shipped and no oracle. A detector tuned on examples of the
failure it is meant to catch has been told the answer.

**The criteria, written before the run.**

1. **The refusal never consults the reference** - `fit_band` sees
   only f16 traces and `judge` takes only a trace and a band. A
   detector that needs the answer is not a detector, and this is
   checked structurally rather than promised.
2. **The band accepts healthy and refuses destroyed** - fitted on
   f16 traces from two images, it must ACCEPT a third f16 trace it
   has never seen and REFUSE the ternary tower on that same third
   image. Either direction failing is a fail; accepting everything
   and refusing everything are both easy and both useless.
3. **It beats the standard option** - the non-finite check must fire
   zero times on the destroyed tower while this fires. If watching
   for NaN already catches it, this unit is unnecessary and should
   say so.
4. **Each measure is judged alone** - attention entropy by itself
   and peak residual RMS by itself, reported separately, so it is
   visible which one does the work. §11.102 learned that a
   conjunction can hide a measure that contributes nothing, and
   peak RMS is close enough to ordinary norm-watching that entropy
   must be shown to add something rather than assumed to.
5. **The profile is reported** - per-block attention entropy for
   both towers, so the shape is visible and not just its summary.

**Run one FAILED, on the detector rather than on the measure.** The
held-out healthy tower was REFUSED. The band had been fitted on two
runs whose mean attention entropies happened to sit 0.017 apart, and
the width was a fraction of that observed spread - so the band was
0.017 wide and the third healthy tower, at 0.6233, fell outside it.
**A band fitted on two runs is a band the width of the gap between
them**, which is a detector that has learned the two runs it saw
rather than what healthy looks like. `fit_band` now refuses fewer
than three traces outright, uses mean +/- 4 standard deviations, and
floors the assumed spread at 2% of the mean so runs that agree
closely do not produce a band of nearly zero width.

The measure itself was never in doubt on that run: 0.6233 healthy
against 0.9790 broken is a gap of 0.36 against a healthy spread of
0.017.

**Run two, five images, four fitted and one held out: PASS.**

| | healthy | ternary |
|---|---:|---:|
| mean attention entropy | 0.6170 | **0.9787** |
| peak residual RMS | 68.663 | 53.969 |
| verdict | **accepted** | **refused** |

**Criterion 4 is where this unit earned its keep.** Each measure was
judged alone:

| measure alone | accepts healthy | refuses broken |
|---|:--|:--|
| attention entropy | yes | **yes** |
| peak residual RMS | yes | **NO** |

**Attention entropy does all the work; peak residual RMS
contributes nothing.** In run one RMS appeared to catch the broken
tower, but that was the too-tight band: given a properly fitted one,
53.969 sits comfortably inside the healthy range and the ternary
tower is ACCEPTED on that measure. Peak RMS is the closest thing
here to what production monitoring actually watches, and on a tower
whose output has been reduced to cosine 0.0185 it notices nothing.
Criterion 3's non-finite check notices nothing either. The
conjunction still ships, because RMS may catch failures this one did
not exercise, but the record says plainly which measure fired.

**The per-block profile is the reason it works, and it is
interpretable.** The healthy tower's attention entropy rises to 0.91
in the first blocks, falls steadily to about 0.30 by block 21 as the
tower learns what to look at, then broadens again at the end. The
ternary tower starts at 0.83 and climbs monotonically to **0.997**:

| block | 0 | 5 | 10 | 15 | 21 | 26 |
|---|---:|---:|---:|---:|---:|---:|
| healthy | 0.65 | 0.76 | 0.64 | ~0.52 | **0.30** | 0.66 |
| ternary | 0.83 | 0.98 | 0.98 | 0.99 | **0.99** | 1.00 |

**The converted tower never becomes selective.** At 0.997 of maximum
entropy it is attending to every position equally, which is to say
attending to nothing - and that is a far more legible statement of
what ternary conversion did than "cosine 0.0185" was.

Run it::

    python -m ultraquant.experiments.refusal_gate
"""

from __future__ import annotations

import math
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ultraquant.convert import gguf
from ultraquant.experiments.convert_gate import checkpoint_path
from ultraquant.experiments.encoder_gate import _ternarise_block, synthetic_frames
from ultraquant.infer import confidence, vit

__all__ = ["RefusalReport", "run_gate"]


def _rms(stream: list) -> float:
    count = sum(len(token) for token in stream)
    return math.sqrt(sum(v * v for token in stream for v in token) / count)


@dataclass
class RefusalReport:
    """Whether a tower could tell it had been destroyed.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        blocks: Blocks traced.
        healthy_attention: Mean normalised attention entropy, f16.
        broken_attention: The same for the ternary tower.
        healthy_rms: Peak residual RMS, f16.
        broken_rms: The same for the ternary tower.
        accepted_healthy: Whether the held-out f16 trace was accepted.
        refused_broken: Whether the ternary trace was refused.
        reasons: Why the ternary trace was refused.
        attention_alone: (accepted healthy, refused broken) using
            attention entropy and nothing else.
        rms_alone: The same for peak residual RMS alone.
        nonfinite_fired: Whether the standard non-finite check fired.
        profile_healthy: Per-block attention entropy, f16.
        profile_broken: Per-block attention entropy, ternary.
        reason: Plain-language verdict.
    """

    passes: bool
    blocks: int = 0
    healthy_attention: float = 0.0
    broken_attention: float = 0.0
    healthy_rms: float = 0.0
    broken_rms: float = 0.0
    accepted_healthy: bool = False
    refused_broken: bool = False
    reasons: list = field(default_factory=list)
    attention_alone: tuple = (False, False)
    rms_alone: tuple = (False, False)
    nonfinite_fired: bool = False
    profile_healthy: list = field(default_factory=list)
    profile_broken: list = field(default_factory=list)
    reason: str = ""


def run_gate(patches: int = 8, blocks: int | None = None,
             images: int = 5, width: float = 4.0) -> RefusalReport:
    """Trace several towers in one pass and see which notices itself."""
    checkpoint = checkpoint_path()
    if not checkpoint.exists():
        return RefusalReport(passes=False,
                             reason="FAIL - no checkpoint; set "
                                    "UQ_CONVERT_MODEL")
    opened = gguf.read(checkpoint)
    by = {tensor.name: tensor for tensor in opened.tensors}
    config = vit.load_config(opened.metadata)
    count = config.blocks if blocks is None else min(blocks, config.blocks)

    embedding = vit.load_patch_embedding(opened, by, config)
    starts = [vit.patch_embed(synthetic_frames(config, seed=seed),
                              embedding, config, patches)
              for seed in range(images)]
    del embedding

    # Every tower walks the same pass over the file. The LAST image is
    # held out: its f16 trace must be accepted by a band that never
    # saw it, and its ternary trace must be refused.
    streams = [[list(token) for token in start] for start in starts]
    broken = [list(token) for token in starts[-1]]
    traces = [confidence.TowerTrace() for _ in starts]
    broken_trace = confidence.TowerTrace()

    for index in range(count):
        weights = vit.load_block(opened, by, index)
        quantised = _ternarise_block(weights)
        for position, stream in enumerate(streams):
            seen: dict = {}
            streams[position] = vit.apply_block(stream, weights, config,
                                                seen)
            traces[position].attention.append(
                confidence.attention_entropy(seen["attention_rows"][0]))
            traces[position].rms.append(_rms(streams[position]))
        seen = {}
        broken = vit.apply_block(broken, quantised, config, seen)
        broken_trace.attention.append(
            confidence.attention_entropy(seen["attention_rows"][0]))
        broken_trace.rms.append(_rms(broken))
        del weights, quantised

    # Criterion 1: the band is fitted on healthy traces ONLY, and
    # never on the held-out one.
    band = confidence.fit_band(traces[:-1], width=width)
    accepted, _why = confidence.judge(traces[-1], band)
    intact, reasons = confidence.judge(broken_trace, band)
    refused = not intact

    # Criterion 4: each measure alone, so a conjunction cannot hide a
    # measure that contributes nothing.
    wide = (-math.inf, math.inf)

    def alone(which: str) -> tuple:
        only = confidence.Band(
            attention=band.attention if which == "attention" else wide,
            rms=band.rms if which == "rms" else wide, width=width)
        ok_healthy, _r = confidence.judge(traces[-1], only)
        ok_broken, _r = confidence.judge(broken_trace, only)
        return (ok_healthy, not ok_broken)

    attention_alone = alone("attention")
    rms_alone = alone("rms")

    # Criterion 3: the standard option, on the same destroyed tower.
    nonfinite = any(not math.isfinite(v) for v in
                    broken_trace.attention + broken_trace.rms)

    passes = accepted and refused and not nonfinite
    report = RefusalReport(
        passes=passes, blocks=count,
        healthy_attention=traces[-1].mean_attention,
        broken_attention=broken_trace.mean_attention,
        healthy_rms=traces[-1].peak_rms,
        broken_rms=broken_trace.peak_rms,
        accepted_healthy=accepted, refused_broken=refused,
        reasons=reasons, attention_alone=attention_alone,
        rms_alone=rms_alone, nonfinite_fired=nonfinite,
        profile_healthy=list(traces[-1].attention),
        profile_broken=list(broken_trace.attention))
    report.reason = (
        f"{'PASS' if passes else 'FAIL'} - {count} blocks; held-out healthy "
        f"tower {'accepted' if accepted else 'REFUSED'}, ternary tower "
        f"{'refused' if refused else 'ACCEPTED'}; attention entropy "
        f"{report.healthy_attention:.4f} healthy against "
        f"{report.broken_attention:.4f} broken; peak RMS "
        f"{report.healthy_rms:.3f} against {report.broken_rms:.3f}; "
        f"attention alone (accept, refuse) = {attention_alone}, RMS alone "
        f"= {rms_alone}; the non-finite check "
        f"{'FIRED' if nonfinite else 'never fired'}"
        + (f"; refused because: {'; '.join(reasons)}" if reasons else ""))
    return report


if __name__ == "__main__":                        # pragma: no cover
    result = run_gate()
    print(f"blocks {result.blocks}")
    print()
    print(f"{'blk':>4}{'healthy H':>12}{'broken H':>12}")
    for i, (a, b) in enumerate(zip(result.profile_healthy,
                                   result.profile_broken)):
        print(f"{i:>4}{a:>12.4f}{b:>12.4f}")
    print()
    print(f"mean attention entropy  {result.healthy_attention:.4f} healthy, "
          f"{result.broken_attention:.4f} broken")
    print(f"peak residual RMS       {result.healthy_rms:.3f} healthy, "
          f"{result.broken_rms:.3f} broken")
    print("held-out healthy        "
          + ("accepted" if result.accepted_healthy else "REFUSED"))
    print("ternary tower           "
          + ("refused" if result.refused_broken else "ACCEPTED"))
    print(f"attention alone         accept={result.attention_alone[0]}, "
          f"refuse={result.attention_alone[1]}")
    print(f"RMS alone               accept={result.rms_alone[0]}, "
          f"refuse={result.rms_alone[1]}")
    print("non-finite check        "
          + ("FIRED" if result.nonfinite_fired else "never fired"))
    print()
    print(result.reason)
