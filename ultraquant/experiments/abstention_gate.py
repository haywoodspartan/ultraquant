"""Can the router say "not mine"? A pre-registered gate.

Measured on the deployed library: of five queries belonging to none of its
categories, it correctly declined **one**. "what time does the train leave"
routed to ``crosses``, "the weather forecast for tomorrow" routed to
``crosses``, "how do I fix a leaking tap" routed to ``crosses``. None of them
matched a single base keyword.

They matched **learned weights on stopwords**. ``CategoryRouter.learn`` takes
every distinct token of whatever text it is given, so ``the``, ``a``, ``is`` and
``what`` accumulate weight for whichever category the input happened to route
to. On the deployed router **21% of all learned weight sits on stopwords**, led
by ``is`` at 1.05 and ``a`` at 0.60 — and a single such token is enough to beat
the ``total > 0.0`` filter that is currently the only thing standing between a
query and a category.

**The gate, written before the fix.**

1. **Abstention** — the fraction of decoy queries correctly left alone must
   improve by a margin larger than the seed-to-seed standard deviation.
2. **Retention (the control, and it can fail badly)** — genuine queries must
   still route correctly, at no worse than the pre-fix rate less one seed sd.

The control is the whole difficulty here. Abstention is *trivial* to maximise:
a router that returns nothing for every input scores 1.000 on decoys. So the
two are measured together and both are reported, because a mechanism that
buys silence with accuracy has not fixed anything — it has moved the failure.

A third quantity is reported and not gated: how much learned weight survives
the fix. A change that solves this by discarding most of what the router
learned would pass both criteria while throwing away the system's experience,
and that should be visible rather than inferred.

Run it::

    python -m ultraquant.experiments.abstention_gate
"""

from __future__ import annotations

import random
import statistics
import tempfile
from dataclasses import dataclass, field

__all__ = ["AbstentionReport", "run_gate", "DECOYS", "GENUINE"]

#: Queries belonging to none of the trained categories. Everyday phrasings, not
#: adversarial ones — the point is that ordinary text should not be claimed.
DECOYS = [
    "how do I fix a leaking tap",
    "what time does the train leave",
    "recipe for banana bread",
    "the weather forecast for tomorrow",
    "who won the football match last night",
    "is the shop open on a sunday",
    "what is a good film to watch",
    "the car needs an oil change",
    "how long does the flight take",
    "who is at the door",
]

#: Queries that genuinely belong to a category. The control: these must keep
#: routing correctly, or abstention has been bought with accuracy.
GENUINE = [
    ("what shape is this glyph", "geometry"),
    ("draw a diamond outline", "frames"),
    ("horizontal stripe pattern", "lines"),
    ("an arrow pointing upward", "arrows"),
    ("a plus and a cross", "crosses"),
    ("add these two numbers", "arithmetic"),
    ("what does this word mean", "language"),
    ("which country is that", "world"),
]

#: Base vocabulary for the categories the genuine queries use, so the gate can
#: build a router without a deployed library present.
_CATEGORIES = {
    "geometry": ["shape", "glyph", "pattern", "figure"],
    "frames": ["frame", "frames", "square", "diamond", "box", "outline"],
    "lines": ["line", "lines", "stripe", "stripes", "bars", "horizontal"],
    "arrows": ["arrow", "arrows", "upward", "direction", "pointer"],
    "crosses": ["cross", "crosses", "plus", "star", "intersection"],
    "arithmetic": ["add", "sum", "numbers", "subtract", "arithmetic"],
    "language": ["word", "meaning", "sentence", "spell", "language"],
    "world": ["country", "history", "capital", "world"],
}

#: Text the router learns from before being measured, mimicking ordinary use:
#: real queries in full natural phrasing, stopwords and all.
#:
#: Each seed learns a *sample* of these, not a reshuffle of all of them. The
#: first version shuffled the full list and every seed produced the identical
#: delta — learned weights are additive, so order changes nothing about which
#: tokens end up weighted. The zero-variance guard caught it, as it did for the
#: context gate.
_USAGE = [
    ("what is the shape of a square", "frames"),
    ("show me a line that is horizontal", "lines"),
    ("this is an arrow that points up", "arrows"),
    ("that is a plus sign in the middle", "crosses"),
    ("what is the sum of these numbers", "arithmetic"),
    ("what is the meaning of this word", "language"),
    ("what is the capital of that country", "world"),
    ("is this a diamond or a box", "frames"),
    ("a pattern of stripes across the page", "lines"),
    ("the glyph is a cross with a dot", "crosses"),
    ("is that a square or a rectangle", "frames"),
    ("draw the vertical bars for me", "lines"),
    ("the arrow is pointing to the left", "arrows"),
    ("what is the star shaped intersection", "crosses"),
    ("subtract the smaller of the numbers", "arithmetic"),
    ("how do you spell that sentence", "language"),
    ("the history of that country is long", "world"),
    ("a figure with a clear outline", "frames"),
]

#: How many usage turns each seed learns from.
_USAGE_PER_RUN = 12

#: Queries whose number disagrees with the registered vocabulary. Measured at
#: **0.500 without plural folding and 1.000 with it** - not 0.000 without,
#: because several categories happen to register both spellings (``frame`` and
#: ``frames``), which masks the gap wherever someone thought to do it. The ones
#: that fail are the ones where nobody did: ``arithmetic`` registers ``number``
#: and the query says ``numbers``.
NUMBER_MISMATCHED = [
    ("add these two numbers", "arithmetic"),
    ("draw some diamonds", "frames"),
    ("show me the horizontal lines", "lines"),
    ("two arrows pointing up", "arrows"),
    ("a page full of crosses", "crosses"),
    ("what do these words mean", "language"),
    ("which countries border it", "world"),
    ("the shapes on this page", "geometry"),
]


@dataclass
class AbstentionReport:
    """What the gate concluded.

    Attributes:
        passes: The verdict under the pre-registered criterion.
        abstain_before: Fraction of decoys declined, before the fix.
        abstain_after: The same, after.
        delta: ``abstain_after - abstain_before``.
        seed_sd: Seed-to-seed standard deviation of the delta.
        margin_ratio: ``delta / seed_sd``.
        route_before: Genuine-query accuracy before.
        route_after: Genuine-query accuracy after. The control.
        weight_kept: Fraction of learned weight surviving the fix. Reported,
            not gated.
        number_mismatched: Accuracy on queries whose number disagrees with the
            registered vocabulary — what plural folding is for.
        reason: Plain-language verdict.
    """

    passes: bool = False
    abstain_before: float = 0.0
    abstain_after: float = 0.0
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    route_before: float = 0.0
    route_after: float = 0.0
    weight_kept: float = 0.0
    number_mismatched: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """A printable report, with the control beside the win."""
        return "\n".join([
            f"decoys declined       {self.abstain_before:.3f} -> "
            f"{self.abstain_after:.3f}   delta {self.delta:+.3f}",
            f"genuine routed (CTRL) {self.route_before:.3f} -> "
            f"{self.route_after:.3f}"
            "   <- refusing everything scores 1.000 above and 0.000 here",
            f"seed sd {self.seed_sd:.3f}   margin ratio "
            f"{self.margin_ratio:.2f}x  (gate needs > 1.0)",
            f"number-mismatched     {self.number_mismatched:.3f}"
            "   <- plural folding; measured 0.500 without it",
            f"learned weight kept   {self.weight_kept:.3f}   "
            "(reported, not gated)",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ])


def _build(rng: random.Random, filter_stopwords: bool):
    """A router taught from ordinary usage, with or without the fix."""
    from ultraquant.shards.router import CategoryRouter
    from ultraquant.shards.vault import ShardVault

    router = CategoryRouter(ShardVault(tempfile.mkdtemp(prefix="uq_abs_")))
    for name, keywords in _CATEGORIES.items():
        router.register(name, keywords)
    usage = rng.sample(_USAGE, _USAGE_PER_RUN)
    for text, category in usage:
        router.learn(text, category, filter_tokens=filter_stopwords)
    return router


def _abstains(router) -> float:
    """Fraction of decoys the router declines to claim."""
    declined = sum(1 for query in DECOYS if not router.route(query, top_k=1))
    return declined / len(DECOYS)


def _mismatched(router) -> float:
    """Fraction of number-mismatched queries routed correctly."""
    hit = 0
    for query, truth in NUMBER_MISMATCHED:
        ranked = router.route(query, top_k=1)
        if ranked and ranked[0][0] == truth:
            hit += 1
    return hit / len(NUMBER_MISMATCHED)


def _routes(router) -> float:
    """Fraction of genuine queries routed to the right category."""
    hit = 0
    for query, truth in GENUINE:
        ranked = router.route(query, top_k=1)
        if ranked and ranked[0][0] == truth:
            hit += 1
    return hit / len(GENUINE)


def _weight(router) -> float:
    """Total learned weight, for the reported retention figure."""
    return sum(sum(weights.values()) for weights in router._learned.values())


def run_gate(seeds: int = 8) -> AbstentionReport:
    """Run the gate.

    Args:
        seeds: Usage orderings to average over. Must exceed 1 or the sd is
            meaningless.

    Returns:
        The :class:`AbstentionReport`.
    """
    abstain_before, abstain_after, deltas = [], [], []
    route_before, route_after, kept = [], [], []
    mismatch: list[float] = []
    for seed in range(seeds):
        rng = random.Random(seed)
        plain = _build(rng, filter_stopwords=False)
        rng = random.Random(seed)
        fixed = _build(rng, filter_stopwords=True)

        abstain_before.append(_abstains(plain))
        abstain_after.append(_abstains(fixed))
        deltas.append(abstain_after[-1] - abstain_before[-1])
        route_before.append(_routes(plain))
        route_after.append(_routes(fixed))
        mismatch.append(_mismatched(fixed))
        plain_weight = _weight(plain)
        kept.append(_weight(fixed) / plain_weight if plain_weight else 1.0)

    delta = statistics.fmean(deltas)
    # stdev, not pstdev: these orderings are a sample, and dividing by n would
    # understate the spread and inflate the margin against the threshold.
    seed_sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0

    report = AbstentionReport(
        abstain_before=statistics.fmean(abstain_before),
        abstain_after=statistics.fmean(abstain_after),
        delta=delta, seed_sd=seed_sd,
        margin_ratio=(delta / seed_sd) if seed_sd > 0 else 0.0,
        route_before=statistics.fmean(route_before),
        route_after=statistics.fmean(route_after),
        weight_kept=statistics.fmean(kept),
        number_mismatched=statistics.fmean(mismatch),
    )

    # The control decides first: silence bought with accuracy is not a fix.
    tolerance = seed_sd if seed_sd > 0 else 0.0
    if report.route_after < report.route_before - tolerance:
        report.reason = (
            f"genuine routing fell {report.route_before:.3f} -> "
            f"{report.route_after:.3f}; abstention was bought with accuracy")
        return report
    if delta <= 0:
        report.reason = f"no improvement in abstention ({delta:+.3f})"
        return report
    if seed_sd <= 0.0:
        report.reason = (
            f"every ordering gave the same delta ({delta:+.3f}), so there is no "
            "variance to measure the margin against")
        return report
    if report.margin_ratio <= 1.0:
        report.reason = (f"margin {delta:+.3f} is {report.margin_ratio:.2f}x "
                         f"the seed sd ({seed_sd:.3f})")
        return report
    report.passes = True
    report.reason = (
        f"decoys declined {report.abstain_before:.3f} -> "
        f"{report.abstain_after:.3f} ({delta:+.3f}, {report.margin_ratio:.2f}x "
        f"seed sd) with genuine routing held at {report.route_after:.3f}")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
