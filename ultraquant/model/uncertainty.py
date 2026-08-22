"""What a prediction knows, in bits.

`UltraQuantNet.predict` returns the argmax and its softmax
probability. That number is the height of one bar, and it is the
standard thing to report and the standard thing to be misled by: a
network that has learned nothing still produces an argmax, and it
still produces a probability for it, and nothing in the pair
distinguishes "this is a plus" from "one of these eight had to
win".

The whole distribution knows more than its tallest bar. Its
**entropy** is how much is still unknown; ``log2(n) - H`` is how
much has been learned. Both in bits, because bits are the unit the
question is actually in: naming one of eight classes is a 3-bit
question, so a prediction carrying 0.2 bits has answered a
fifteenth of it, and saying so is more honest than reporting 0.31
and letting the reader assume that means something.

**This is the same line the rest of the system keeps.** §11.48 says
absence of belief is not belief in absence, and refuses to answer
"is X Y?" about a subject nobody holds. A prediction layer without
an uncertainty measure breaks that line at the one place the system
cannot see it: the argmax always names a class, so the network
always sounds certain. `decide` is where it stops - below a measured
floor the layer declines rather than naming a winner.

**Entropy against the standard option, and it LOST.** Maximum
softmax probability is not a strawman; it is what almost everything
reports. The gate beside this module ranked the same predictions
three ways and discarded the least certain fifth:

===============  ==================
ranker            error at 80% cover
===============  ==================
margin            0.0078
max-softmax       0.0130
entropy           0.0260
===============  ==================

**Entropy is the worst of the three at deciding which predictions to
throw away**, and the reason is structural: entropy is moved by the
whole tail, so a confident, correct prediction with probability
spread thinly over the other seven classes scores as uncertain.
Max-softmax and margin look only at the top, which is what decides
whether the argmax is right.

**So the three measures are not interchangeable, and the gate
measured which does what.** At matched coverage:

- **`margin` catches WRONG predictions**: accepted error 0.0077
  against evidence's 0.0229.
- **`evidence` catches inputs never seen**: 98.5% of non-glyph
  noise declined against margin's 93.8%.

Those are different failures. A hard in-distribution case is a
two-way confusion - small margin, respectable evidence. An
out-of-distribution input spreads everywhere - low evidence, and a
margin that can still be accidentally wide. Use `margin` to decide
whether to trust an answer and `evidence` to decide whether the
question was answerable.

**Requiring both was tried and does not compose**: taking the weaker
of the two lands at evidence's error rate (0.0222) with no gain,
because whichever measure ranks a sample lower dominates and that is
usually evidence. Measured, not assumed, and recorded here so it is
not retried.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["entropy", "max_entropy", "evidence", "margin", "Prediction",
           "describe", "decide"]


def entropy(probabilities: list[float]) -> float:
    """Shannon entropy in BITS, with ``0 log 0`` taken as 0.

    Bits rather than nats because the question is "which of n", and
    in bits the answer is directly comparable to ``log2(n)``: an
    entropy of 3 over eight classes means nothing has been learned,
    and the reader does not have to divide by ``ln 2`` to see it.
    """
    total = 0.0
    for value in probabilities:
        if value > 0.0:
            total -= value * math.log2(value)
    return total


def max_entropy(count: int) -> float:
    """``log2(n)`` - the entropy of knowing nothing about n classes."""
    return math.log2(count) if count > 0 else 0.0


def evidence(probabilities: list[float]) -> float:
    """Bits the distribution has actually learned: ``log2(n) - H``.

    Zero for a uniform distribution and ``log2(n)`` for a certain
    one. This is the number worth putting in front of a person,
    because it grows with knowledge rather than shrinking with it.
    """
    return max_entropy(len(probabilities)) - entropy(probabilities)


def margin(probabilities: list[float]) -> float:
    """Gap between the top two probabilities.

    Kept because it answers a question entropy does not: a
    distribution split between exactly two classes has moderate
    entropy and a tiny margin, and "it is one of these two and I
    cannot say which" is a different state from "it could be any of
    them".
    """
    if len(probabilities) < 2:
        return 1.0 if probabilities else 0.0
    ordered = sorted(probabilities, reverse=True)
    return ordered[0] - ordered[1]


@dataclass
class Prediction:
    """One prediction, and everything it knows about itself.

    Attributes:
        label: Index of the argmax class.
        probability: Its softmax probability - the standard measure,
            kept so the two can be compared rather than swapped.
        entropy: Bits still unknown.
        evidence: Bits learned, ``log2(n) - entropy``.
        margin: Gap to the runner-up.
        distribution: The whole thing, because a summary that
            discards it cannot be re-examined later.
        declined: True when the layer refused to name a class.
        floor: The evidence threshold it was judged against.
    """

    label: int
    probability: float
    entropy: float
    evidence: float
    margin: float
    distribution: list = field(default_factory=list)
    declined: bool = False
    floor: float = 0.0

    @property
    def bits_of(self) -> str:
        """The question this prediction is an answer to, in bits."""
        total = max_entropy(len(self.distribution))
        return f"{self.evidence:.2f} of {total:.2f} bits"


def describe(probabilities: list[float], floor: float = 0.0,
             margin_floor: float = 0.0) -> Prediction:
    """Everything a distribution says about itself.

    Args:
        probabilities: The whole softmax.
        floor: Bits of evidence required - the "was this question
            answerable" test, which is what catches inputs the
            network has never seen.
        margin_floor: Gap to the runner-up required - the "is this
            answer trustworthy" test, which is what catches wrong
            predictions. Separate because the gate measured them
            catching different failures.
    """
    if not probabilities:
        return Prediction(-1, 0.0, 0.0, 0.0, 0.0, [], True, floor)
    best = max(range(len(probabilities)), key=lambda k: probabilities[k])
    learned = evidence(probabilities)
    gap = margin(probabilities)
    return Prediction(
        label=best, probability=probabilities[best],
        entropy=entropy(probabilities), evidence=learned,
        margin=gap, distribution=list(probabilities),
        declined=learned < floor or gap < margin_floor, floor=floor)


def decide(probabilities: list[float], floor: float,
           margin_floor: float = 0.0) -> int | None:
    """The argmax, or None when the layer knows too little to say.

    The floor is in BITS of evidence, and it is meant to be measured
    rather than chosen - a threshold picked because it looks round is
    a confidence interval invented on the spot. The gate beside this
    module fits it on one split and reports the cost on another.
    """
    result = describe(probabilities, floor, margin_floor)
    return None if result.declined else result.label
