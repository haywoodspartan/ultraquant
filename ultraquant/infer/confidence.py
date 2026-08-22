"""An encoder that can refuse its own output.

§11.101 measured what ternary conversion does to a 27-block tower:
the projected output ends at cosine **0.0185** to the f16 original -
not degraded, uncorrelated. And the tower did not notice. Every
activation stayed finite, every shape stayed right, and it emitted
plausible numbers all the way through. The only thing that knew
anything was wrong was the f16 reference, which in any real
deployment does not exist.

§11.102 gave the prediction layer an answer to the same problem: an
untrained classifier reports 0.184 of 3.00 bits and says so. This is
that idea carried to the encoder, where it is harder, because **an
encoder's output is not a distribution.** It is a list of real
vectors. There is no softmax to take the entropy of.

**But the attention rows are distributions**, and they are the only
genuine ones inside a transformer. Each is a softmax over positions,
and its entropy - normalised by `log2(seq)` so 0 is perfect focus
and 1 is attending to everything equally - says how much structure
the tower found. That is computable from the forward pass alone,
with nothing to compare against, which is the whole requirement.

**The standard option this is measured against is "check for
non-finite values"**, which is what production code actually does.
On the fully destroyed tower of §11.101 it fires zero times.

**One-class by construction.** The band is fitted on healthy runs
only, because that is the deployment situation: you have the model
you shipped and no oracle. A detector tuned on examples of the
failure it is meant to catch has been told the answer.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from ultraquant.model.uncertainty import entropy, max_entropy

__all__ = ["attention_entropy", "TowerTrace", "Band", "fit_band", "judge"]


def attention_entropy(rows: list) -> float:
    """Mean attention entropy over a block, as a share of the maximum.

    Normalised by ``log2(positions)`` so the number means the same
    thing at any sequence length: **0 is a head that has decided, 1
    is a head attending to everything equally.** A fully masked row
    contributes nothing and is skipped rather than counted as
    certain, since it carries no attention to be certain about.
    """
    ceiling = max_entropy(len(rows[0])) if rows and rows[0] else 0.0
    if ceiling <= 0.0:
        return 0.0
    scores = [entropy(row) / ceiling for row in rows if any(row)]
    return statistics.fmean(scores) if scores else 0.0


@dataclass
class TowerTrace:
    """What one forward pass said about itself.

    Attributes:
        attention: Per-block mean normalised attention entropy.
        rms: Per-block residual RMS.
    """

    attention: list = field(default_factory=list)
    rms: list = field(default_factory=list)

    @property
    def mean_attention(self) -> float:
        return statistics.fmean(self.attention) if self.attention else 0.0

    @property
    def peak_rms(self) -> float:
        return max(self.rms) if self.rms else 0.0


@dataclass
class Band:
    """The range healthy runs occupied, and the margin allowed.

    Attributes:
        attention: (low, high) for mean normalised attention entropy.
        rms: (low, high) for peak residual RMS.
        width: How many standard deviations wide each side is.
    """

    attention: tuple = (0.0, 1.0)
    rms: tuple = (0.0, math.inf)
    width: float = 4.0


#: Fewer than this many healthy runs cannot estimate a spread. Run
#: one of the refusal gate fitted on two, whose entropies happened to
#: sit 0.017 apart, and produced a band 0.017 wide that refused the
#: next healthy tower - a detector that had learned the two runs it
#: saw rather than what healthy looks like.
_MIN_TRACES = 3

#: The smallest spread a band will assume, as a share of the mean.
#: Without it, runs that agree closely give a band of nearly zero
#: width and every later run is an anomaly.
_MIN_RELATIVE_SD = 0.02


def _band(values: list, width: float) -> tuple:
    centre = statistics.fmean(values)
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    spread = max(spread, abs(centre) * _MIN_RELATIVE_SD, 1e-9)
    return (centre - spread * width, centre + spread * width)


def fit_band(traces: list, width: float = 4.0) -> Band:
    """The band healthy runs occupy. Healthy runs ONLY.

    Passing a broken tower in here would be fitting the detector on
    the failure it exists to catch, which is how a detector comes to
    report exactly what it was shown and nothing else.

    Args:
        traces: At least :data:`_MIN_TRACES` healthy traces.
        width: Standard deviations either side of the healthy mean.

    Raises:
        ValueError: With too few traces to estimate a spread at all.
            Refusing is better than returning a band that looks
            fitted and is not.
    """
    if len(traces) < _MIN_TRACES:
        raise ValueError(
            f"{len(traces)} healthy runs is not enough to estimate a "
            f"spread; {_MIN_TRACES} is the minimum. A band fitted on "
            "two runs is a band the width of the gap between them.")
    return Band(
        attention=_band([t.mean_attention for t in traces], width),
        rms=_band([t.peak_rms for t in traces], width),
        width=width)


def judge(trace: TowerTrace, band: Band) -> tuple:
    """``(accepted, reasons)`` - whether this pass looks like itself.

    Returns the reasons rather than a bare boolean because "refused"
    without a cause is the thing this whole module exists to replace.
    """
    reasons: list = []
    low, high = band.attention
    if not low <= trace.mean_attention <= high:
        reasons.append(
            f"attention entropy {trace.mean_attention:.4f} outside "
            f"[{low:.4f}, {high:.4f}]")
    low, high = band.rms
    if not low <= trace.peak_rms <= high:
        reasons.append(
            f"peak residual RMS {trace.peak_rms:.3f} outside "
            f"[{low:.3f}, {high:.3f}]")
    if any(not math.isfinite(v) for v in trace.attention + trace.rms):
        reasons.append("a non-finite activation")
    return (not reasons), reasons
