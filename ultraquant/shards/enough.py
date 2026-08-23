"""Read until you know enough, then stop.

A transformer spends its entire parameter space on every token. Asked
"what is 2 + 2" it does the same 9 billion multiply-accumulates it
does for a question at the edge of its competence, because a dense
forward pass has no way to do less. Cost is a property of the model,
not of the question.

That is not how recall works in anything that recalls. You reach for
what the question points at, and you stop when you know enough - and
the *stopping* is the part with no analogue in a dense model.

The library has had the first half of this for a long time: a
catalog that names which shards a question points at, a byte budget
that decides what stays resident, and a working set that prefetches
them. What it never had is the second half. `predict` then
`prefetch` is one shot: it reads what it guessed, and whether that
was too much or nowhere near enough, it reads exactly that.

**This module is the loop.** Experts are consulted in the order the
router ranks them, each read costing a page-in on a miss, and after
each one the answer's own evidence - §11.102's `log2(n) - H`, in
bits - decides whether to spend another. Three ways to stop:

* **confident** - evidence cleared the floor. The question is
  answered and the remaining shards are never touched.
* **budget** - the cap was reached. This is the transformer's only
  mode, made explicit and made a failure rather than a default.

**There used to be a third: "plateau"**, which stopped when the last
few reads bought less than `_STALL_BITS` between them. §11.104
measured only **11.8% of its plateaus as justified** - it abandoned
answerable questions nearly nine times in ten - and §11.107 swept it
before removing it:

===============  =======  ==========  =========  ===========
stall reads      reads    accuracy    plateaus   justified
===============  =======  ==========  =========  ===========
2 (as shipped)   2.56     0.9741      2.6%       14.3%
3                2.59     0.9815      1.9%       20.0%
5                2.63     1.0000      0.0%       100.0%
off              2.63     1.0000      0.0%       100.0%
===============  =======  ==========  =========  ===========

At every setting where it fires it fires wrongly, and at the setting
where it is justified it never fires - so "5" and "off" are the same
policy written two ways. It cost 2.6 points of accuracy to save 2.7%
of the reads. **A rule that is only ever wrong is deleted rather than
tuned**, and the numbers are here so nobody adds it back.

**Nothing here is free.** The loop trades reads for certainty, and
every one of those trades is measurable: reads spent, evidence
gained, and whether the answer changed. §11.104's gate measures it
against the two standard options - reading everything, which is what
a dense model does, and reading a fixed top-k, which is what
retrieval augmentation does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Consultation", "Recall", "recall_until_enough"]

#: Bits of evidence a read must buy to count as progress. Still used,
#: but only to decide whether the BEST answer so far has settled -
#: never to abandon a question. See the module note on the plateau
#: rule that was removed.
_STALL_BITS = 0.05


@dataclass
class Consultation:
    """One expert asked, and what asking it bought.

    Attributes:
        category: The expert consulted.
        label: The label it named.
        evidence: Bits it had, of the total the question is worth.
        gain: Bits this read added to the best so far - the number
            that decides whether reading is still working.
        was_resident: Whether the shard was already in RAM, so a read
            that cost nothing is not counted as though it did.
    """

    category: str
    label: str
    evidence: float
    gain: float
    was_resident: bool


@dataclass
class Recall:
    """An answer, what it cost, and why the reading stopped.

    Attributes:
        label: The answer, or None when nothing cleared the floor.
        category: Which expert produced it.
        evidence: Its evidence in bits.
        stopped: "confident" or "budget".
        consultations: Every expert asked, in order.
        available: How many the router offered.
    """

    label: str | None = None
    category: str | None = None
    evidence: float = 0.0
    stopped: str = "budget"
    consultations: list = field(default_factory=list)
    available: int = 0

    @property
    def reads(self) -> int:
        """Experts consulted - the cost this recall actually paid."""
        return len(self.consultations)

    @property
    def paid_reads(self) -> int:
        """Consultations that were cache misses, so genuinely read."""
        return sum(1 for c in self.consultations if not c.was_resident)

    @property
    def spared(self) -> int:
        """Experts the router offered and the loop never touched."""
        return max(0, self.available - self.reads)


def recall_until_enough(pool, categories: list, features: list,
                        floor: float, margin_floor: float = 0.0,
                        max_reads: int | None = None) -> Recall:
    """Consult experts in order, stopping as soon as one knows enough.

    Args:
        pool: An :class:`~ultraquant.experts.moe.ExpertPool`.
        categories: Candidate categories, best first - the router's
            ranking. The loop trusts the order and does not re-rank;
            re-ranking would require reading everything, which is the
            cost this exists to avoid.
        features: The input vector.
        floor: Bits of evidence that count as knowing enough.
        margin_floor: Gap to the runner-up also required. §11.102
            measured these catching different failures.
        max_reads: Hard cap. None means "as many as offered", which
            is the full-scan behaviour and is what the gate compares
            against.

    Returns:
        A :class:`Recall`. Its `label` is None unless something
        cleared both floors - a loop that runs out of budget has not
        answered, and saying so is the whole point.
    """
    recall = Recall(available=len(categories))
    cap = len(categories) if max_reads is None else min(max_reads,
                                                        len(categories))
    best_evidence = 0.0

    for category in categories[:cap]:
        # Whether this costs a read is decided BEFORE the call, since
        # asking is what makes it resident.
        resident = pool.cache.is_resident(pool.shard_id(category))
        try:
            label, result = pool.predict_uncertain(category, features,
                                                   floor, margin_floor)
        except (KeyError, ValueError):
            # No expert, or the wrong modality. Both mean this
            # candidate cannot answer; neither means the question is
            # unanswerable, so the loop moves on rather than refusing.
            continue
        gain = max(0.0, result.evidence - best_evidence)
        recall.consultations.append(
            Consultation(category=category, label=label,
                         evidence=result.evidence, gain=gain,
                         was_resident=resident))
        if result.evidence > best_evidence:
            best_evidence = result.evidence
            recall.evidence = result.evidence
            recall.category = category
            recall.label = None if result.declined else label

        # Stop when the BEST answer so far is good enough AND the last
        # read did not improve on it. Run one stopped at the first
        # expert that cleared the floor, and lost to a fixed top-k at
        # eleven of twelve settings for a reason that took a sweep to
        # see: first-past-the-post against best-of-k is not a contest
        # about cost, it is a contest about comparison, and stopping
        # at the first adequate answer throws the comparison away.
        # Confirming that nothing better follows costs one more read
        # and is what the constant was quietly buying.
        improved = gain >= _STALL_BITS
        if best_evidence >= floor and not improved:
            recall.evidence = best_evidence
            recall.stopped = "confident"
            return recall

    recall.stopped = "budget"
    return recall
